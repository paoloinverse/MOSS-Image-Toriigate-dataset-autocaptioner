"""Tab 2 - Inference Parameters.

* Raw llama.cpp passthrough args + server binary/host/port/split-mode.
* Binary row: 'Autodetect' (searches PATH/common build dirs, ranks CUDA >
  ROCm > Vulkan > CPU), 'Browse…' (pick an existing install) and
  'Auto-install / build' (guided clone+compile in a background thread with
  live log streaming).  The install flow is strictly staged: existing-binary
  check -> toolchain/GPU preflight (halts with copy-pasteable fix commands
  *before* any dialog) -> only then folder choice + confirmation -> build.
* Generation sliders (Temperature, Top-P, Top-K, Min-P); per-model profile
  defaults are re-applied whenever the selected model changes.
* "Thinking level" combo (Off/Low/Medium/High -> --reasoning-budget +
  reasoning_budget_tokens) plus a fine-grained Reasoning Budget slider.
* KV cache type dropdown populated from models.KV_CACHE_TYPES (incl. bf16).
* Start/Stop llama-server buttons; the computed resource plan (-c/-ngl) is
  previewed live so the user sees the dynamic KV sizing decision, capped by
  the per-model/per-quant hard limit from the model profile.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSlider, QSpinBox, QTextEdit,
                               QVBoxLayout, QWidget)

from app.context import AppContext
from app.core import llamacpp
from app.core.models import (KV_CACHE_TYPES, get_model_profile, read_arch_info,
                             resolve_mmproj)
from app.core.resources import ResourcePlanError, plan_resources


def _slider(row_label: str, vmin: float, vmax: float, value: float,
            factor: int = 100) -> tuple[QSlider, QLabel]:
    s = QSlider(Qt.Horizontal)
    s.setRange(int(vmin * factor), int(vmax * factor))
    s.setValue(int(value * factor))
    v = QLabel(str(value))
    v.setMinimumWidth(48)
    return s, v


class InferenceTab(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        st = ctx.settings
        self._last_plan = None
        self._last_file = None
        self._profiled_key: str | None = None   # model key whose defaults were applied
        self._installer: llamacpp.InstallRunner | None = None

        root = QVBoxLayout(self)

        # ---- Server ----------------------------------------------------------
        srv_box = QGroupBox("llama.cpp Server")
        form = QFormLayout(srv_box)
        # Binary row: field + Autodetect / Browse / guided install buttons so
        # nobody has to remember or type a path.
        self.binary_edit = QLineEdit(st.binary)
        self.binary_edit.setPlaceholderText(
            "path to llama-server (or just the name if it is on PATH)")
        self.binary_edit.textChanged.connect(
            lambda t: setattr(self.ctx.settings, "binary",
                              t.strip() or "llama-server"))
        # Every edit of the field must also reach ServerManager's discovery
        # (find_binary) immediately - before the first start(), server.config
        # is still None and would otherwise ignore the user's Binary: choice.
        self.binary_edit.textChanged.connect(
            lambda t: setattr(self.ctx.server, "preferred_binary",
                              t.strip() or "llama-server"))
        bin_row = QHBoxLayout()
        bin_row.addWidget(self.binary_edit, stretch=1)
        self.autodetect_btn = QPushButton("Autodetect")
        self.autodetect_btn.setToolTip(
            "Search PATH and common install/build locations; picks the best\n"
            "build (CUDA > ROCm > Vulkan > CPU). The chosen path is persisted.")
        self.browse_binary_btn = QPushButton("Browse…")
        self.browse_binary_btn.setToolTip(
            "Navigate to an existing llama-server binary (e.g. inside a\n"
            "llama.cpp build directory) - no typing required.")
        self.install_btn = QPushButton("Auto-install / build")
        self.install_btn.setToolTip(
            "Guided installer: clones llama.cpp and compiles llama-server in\n"
            "a background thread with live logs.  Runs a staged preflight\n"
            "first (build tools -> GPU hardware via PCI IDs -> vendor CUDA/\n"
            "ROCm/Vulkan toolchain) and halts with copy-pasteable fix\n"
            "commands before any dialog; only then asks for confirmation and\n"
            "an install folder.  Nothing heavy runs silently.")
        bin_row.addWidget(self.autodetect_btn)
        bin_row.addWidget(self.browse_binary_btn)
        bin_row.addWidget(self.install_btn)
        self.binary_status = QLabel("")
        self.binary_status.setWordWrap(True)
        self.binary_status.setStyleSheet("color: gray;")
        form.addRow("Binary:", bin_row)
        form.addRow(self.binary_status)
        self.host_edit = QLineEdit(st.host)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(st.port)
        self.split_combo = QComboBox()
        self.split_combo.addItems(["row", "layer", "graph"])
        form.addRow("Host:", self.host_edit)
        form.addRow("Port:", self.port_spin)
        form.addRow("Split mode (multi-GPU):", self.split_combo)
        # ---- Run-target model -------------------------------------------------
        # Explicit, always-current choice of which GGUF llama-server loads.
        # Previously Tab 2 silently reused whatever Tab 1 had selected *at
        # start-up*, so switching to the Qwen model there still launched (and
        # paired projectors for) ToriiGate.  'Follow Tab 1' keeps the old
        # behaviour; picking a model here overrides it for server launches
        # only - caption presets/payloads keep following Tab 1.
        self.run_model_combo = QComboBox()
        self.run_model_combo.setToolTip(
            "Which model llama-server actually loads.\n"
            "'Follow Tab 1 selection' mirrors the Tab 1 model/quant picker;\n"
            "choosing an entry here overrides it explicitly (persisted).\n"
            "Quant sub-menu lists the live-discovered Hub filenames.")
        self.run_quant_combo = QComboBox()
        self.run_quant_combo.setToolTip(
            "Exact GGUF file (quant) to launch - verbatim Hub filename,\n"
            "discovered live from the repository listing.")
        form.addRow("Model to run:", self.run_model_combo)
        form.addRow("Quant / file:", self.run_quant_combo)
        self.follow_label = QLabel("")
        self.follow_label.setWordWrap(True)
        self.follow_label.setStyleSheet("color: gray; font-family: monospace;")
        form.addRow(self.follow_label)
        # Flash-attention GUI switch.  Replaces the hardcoded '-af' flag that
        # used to be appended on every launch: '-af' was deprecated and has
        # been removed from llama.cpp, so current builds rejected it with
        # "invalid argument" and refused to start.  The server now maps this
        # choice onto the modern '--flash-attn auto|on|off' spelling (with a
        # probed fallback to the boolean '-fa' on slightly older builds).
        self.fa_combo = QComboBox()
        self.fa_combo.addItems(["auto", "on", "off"])
        self.fa_combo.setCurrentText(
            st.flash_attn if st.flash_attn in ("auto", "on", "off") else "auto")
        self.fa_combo.setToolTip(
            "Flash attention (--flash-attn auto|on|off).\n"
            "auto : let llama-server decide per model/backend (recommended)\n"
            "on   : force flash attention on (faster, lower KV memory)\n"
            "off  : disable flash attention\n"
            "The old '-af' alias is gone from llama.cpp - this switch uses\n"
            "the supported flag and adapts to the probed binary version.")
        form.addRow("Flash Attention:", self.fa_combo)
        root.addWidget(srv_box)

        # ---- Raw args ---------------------------------------------------------
        raw_box = QGroupBox("Llama.cpp Args (raw passthrough)")
        rb = QVBoxLayout(raw_box)
        self.extra_args = QTextEdit()
        self.extra_args.setPlaceholderText(
            "e.g. --flash-attn --parallel 4 --mlock  (appended verbatim)")
        self.extra_args.setFixedHeight(70)
        rb.addWidget(self.extra_args)
        root.addWidget(raw_box)

        # ---- Generation sliders -------------------------------------------------
        gen_box = QGroupBox("Generation Settings")
        gf = QFormLayout(gen_box)

        self.temp_slider, self.temp_val = _slider("Temp", 0.0, 2.0, st.temperature)
        self.top_p_slider, self.top_p_val = _slider("Top-P", 0.0, 1.0, st.top_p)
        self.top_k_slider, self.top_k_val = _slider("Top-K", 1, 200, st.top_k, 1)
        self.min_p_slider, self.min_p_val = _slider("Min-P", 0.0, 1.0, st.min_p)
        self.reason_slider, self.reason_val = _slider(
            "Reasoning Budget", 0, 16384, st.reasoning_budget, 1)
        self.reason_slider.setSingleStep(64)

        gf.addRow("Temperature:", self._row(self.temp_slider, self.temp_val))
        gf.addRow("Top-P:", self._row(self.top_p_slider, self.top_p_val))
        gf.addRow("Top-K:", self._row(self.top_k_slider, self.top_k_val))
        gf.addRow("Min-P:", self._row(self.min_p_slider, self.min_p_val))
        # Thinking level: named per-model preset (Off/Low/Medium/High) that
        # sets the reasoning budget; the slider below stays for fine control.
        self.thinking_combo = QComboBox()
        self.thinking_combo.setToolTip(
            "Model-specific thinking levels (Qwen3-family hybrid thinker).\n"
            "Sets --reasoning-budget on the server and reasoning_budget_tokens\n"
            "/ enable_thinking in the request payload.")
        self._populate_thinking_levels(st.thinking_level or st.reasoning_budget)
        gf.addRow("Thinking level:", self.thinking_combo)
        gf.addRow("Reasoning Budget (tokens):",
                  self._row(self.reason_slider, self.reason_val))
        hint = QLabel("Reasoning budget maps to --reasoning-budget / "
                      "reasoning_budget_tokens (Qwen3.8 thinking mode).")
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        gf.addRow(hint)
        root.addWidget(gen_box)

        # ---- KV cache -----------------------------------------------------------
        kv_box = QGroupBox("KV Cache")
        kf = QFormLayout(kv_box)
        self.cache_combo = QComboBox()
        # models.KV_CACHE_TYPES is the single source of truth (incl. bf16).
        self.cache_combo.addItems(list(KV_CACHE_TYPES))
        if st.cache_type in KV_CACHE_TYPES:
            self.cache_combo.setCurrentText(st.cache_type)
        else:
            self.cache_combo.setCurrentText("q8_0")
        kf.addRow("Cache type (-ctk/-ctv):", self.cache_combo)
        self.plan_label = QLabel("Plan: press Refresh Plan")
        self.plan_label.setWordWrap(True)
        self.plan_label.setStyleSheet("font-family: monospace;")
        kf.addRow(self.plan_label)
        row = QHBoxLayout()
        self.refresh_plan_btn = QPushButton("Refresh Plan")
        self.start_btn = QPushButton("Start llama-server")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        row.addWidget(self.refresh_plan_btn)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        kf.addRow(row)
        self.state_label = QLabel("Server: stopped")
        kf.addRow(self.state_label)
        root.addWidget(kv_box)

        root.addStretch(1)

        # ---- wiring --------------------------------------------------------------
        for sl, lbl, attr, div in (
                (self.temp_slider, self.temp_val, "temperature", 100),
                (self.top_p_slider, self.top_p_val, "top_p", 100),
                (self.min_p_slider, self.min_p_val, "min_p", 100)):
            sl.valueChanged.connect(
                lambda v, l=lbl, d=div, a=attr: self._set_param(a, v / d, l))
        self.top_k_slider.valueChanged.connect(
            lambda v: self._set_param("top_k", v, self.top_k_val))
        self.reason_slider.valueChanged.connect(
            lambda v: (self._set_param("reasoning_budget", v, self.reason_val),
                       self._sync_reasoning_to_slider()))
        self.thinking_combo.currentIndexChanged.connect(
            self._on_thinking_level_changed)
        self.cache_combo.currentTextChanged.connect(
            lambda t: setattr(self.ctx.settings, "cache_type", t))
        self.split_combo.currentTextChanged.connect(
            lambda t: setattr(self.ctx.settings, "split_mode", t))
        self.fa_combo.currentTextChanged.connect(
            lambda t: setattr(self.ctx.settings, "flash_attn", t))
        self.extra_args.textChanged.connect(
            lambda: setattr(self.ctx.settings, "extra_args",
                            self.extra_args.toPlainText()))
        self.host_edit.textChanged.connect(
            lambda t: setattr(self.ctx.settings, "host", t.strip() or "127.0.0.1"))
        self.port_spin.valueChanged.connect(
            lambda v: setattr(self.ctx.settings, "port", v))

        self.refresh_plan_btn.clicked.connect(self.refresh_plan)
        self.start_btn.clicked.connect(self.start_server)
        self.stop_btn.clicked.connect(lambda: self.ctx.server.stop())
        self.autodetect_btn.clicked.connect(self.autodetect_binary)
        self.browse_binary_btn.clicked.connect(self.browse_binary)
        self.install_btn.clicked.connect(self.autoinstall)
        self.ctx.server_state.connect(self.on_server_state)
        # Model switches must BOTH re-plan (per-model KV caps) and re-apply
        # the run-out-of-the-box profile defaults (sampling, thinking, cache).
        self.ctx.model_changed.connect(self._on_model_changed)
        self.ctx.binary_changed.connect(self._refresh_binary_status)
        self.ctx.registry_refreshed.connect(self._on_registry_refreshed)
        self.run_model_combo.currentIndexChanged.connect(
            self._on_run_model_changed)
        self.run_quant_combo.currentIndexChanged.connect(
            self._on_run_quant_changed)

        self._populate_run_models()
        self._profiled_key = self.ctx.run_model_key
        self._refresh_binary_status()

    # ------------------------------------------------------------------ plan
    def _profile(self):
        """ModelProfile of the run-target model (Tab 2 selection)."""
        return get_model_profile(self.ctx.run_model_key)

    def _on_registry_refreshed(self, key: str):
        if key == self.ctx.run_model_key or not self.ctx.settings.run_model_key:
            self._populate_run_models()

    # ---------------------------------------------------------- model profiles

    def _on_model_changed(self):
        """model_changed handler: re-populate per-model widgets + re-plan.

        The Tab 2 run-model dropdown is refreshed whenever the *follow* mode
        is active (or a live listing just landed), so switching models in
        Tab 1 immediately updates what llama-server will launch with - the
        stale "still thinks ToriiGate was selected" desync.

        Preset sampling defaults are only applied once per model *key* (not
        on every quant switch), so a user's manual slider tweaks survive
        quant changes - while the Thinking-level combo is always rebuilt to
        match the active profile and the plan is always recomputed with the
        new per-quant context cap.
        """
        key = self.ctx.run_model_key
        if not self.ctx.settings.run_model_key \
                or key == self.ctx.selected_model_key:
            self._populate_run_models()
        if key != self._profiled_key:
            self._profiled_key = key
            self._apply_profile_defaults()
            return
        self._populate_thinking_levels(
            self.ctx.settings.thinking_level
            or self.ctx.settings.reasoning_budget)
        self.refresh_plan()

    # ------------------------------------------------------- run-model picker
    def _populate_run_models(self):
        """Fill the explicit 'Model to run' dropdown from the registry.

        First entry = follow Tab 1; every registered model family is listed,
        and its quant sub-menu shows the *live-discovered* GGUF files (exact
        Hub filenames), so what Tab 2 launches can never silently lag behind
        a model switch made in Tab 1.
        """
        ctx = self.ctx
        want_key = ctx.run_model_key
        want_q = ctx.run_quant
        self.run_model_combo.blockSignals(True)
        self.run_quant_combo.blockSignals(True)
        self.run_model_combo.clear()
        self.run_model_combo.addItem("Follow Tab 1 selection", "")
        for key, spec in ctx.registry.items():
            self.run_model_combo.addItem(spec.display_name, key)
        idx = self.run_model_combo.findData(want_key if self.ctx.settings.run_model_key
                                            else "")
        self.run_model_combo.setCurrentIndex(max(0, idx))
        self.run_model_combo.blockSignals(False)
        self._populate_run_quants(preserve=(want_key, want_q))
        self._update_follow_label()

    def _populate_run_quants(self, preserve: "tuple[str, str] | None" = None):
        """Quant/file list for the current 'Model to run' choice."""
        ctx = self.ctx
        cur_key = ctx.run_model_key
        spec = ctx.registry.get(cur_key)
        want_q = ""
        if preserve and preserve[0] == cur_key:
            want_q = preserve[1]
        elif ctx.settings.run_quant and spec \
                and spec.file_for_quant(ctx.settings.run_quant) is not None:
            want_q = ctx.settings.run_quant
        elif cur_key == ctx.selected_model_key and spec \
                and spec.file_for_quant(ctx.selected_quant) is not None:
            want_q = ctx.selected_quant
        self.run_quant_combo.blockSignals(True)
        self.run_quant_combo.clear()
        if spec is not None:
            for f in spec.files:
                self.run_quant_combo.addItem(f"{f.quant}  [{f.filename}]",
                                             f.quant)
        idx = -1
        for i in range(self.run_quant_combo.count()):
            data = self.run_quant_combo.itemData(i)
            if want_q and data == want_q:
                idx = i
                break
        if idx < 0 and want_q and spec:
            alt = spec.file_for_quant(want_q)
            if alt is not None:
                for i in range(self.run_quant_combo.count()):
                    if self.run_quant_combo.itemData(i) == alt.quant:
                        idx = i
                        break
        self.run_quant_combo.setCurrentIndex(max(0, idx))
        q = self.run_quant_combo.currentData() or ""
        if q:
            ctx.settings.run_quant = q
        self.run_quant_combo.blockSignals(False)
        self._update_follow_label()

    def _on_run_model_changed(self, _index: int = 0):
        key = self.run_model_combo.currentData() or ""
        self.ctx.settings.run_model_key = key
        if key:
            spec = self.ctx.registry.get(key)
            # switching families invalidates the old quant choice unless it
            # also exists in the new repo
            q = self.ctx.settings.run_quant
            if spec is not None and q and spec.file_for_quant(q) is None:
                self.ctx.settings.run_quant = ""
        self._populate_run_models()
        self.ctx.save_settings()
        k = self.ctx.run_model_key
        if k != self._profiled_key:
            self._profiled_key = k
            self._apply_profile_defaults()
        else:
            self._populate_thinking_levels(
                self.ctx.settings.thinking_level
                or self.ctx.settings.reasoning_budget)
            self.refresh_plan()

    def _on_run_quant_changed(self, _index: int = 0):
        q = self.run_quant_combo.currentData()
        if q:
            self.ctx.settings.run_quant = q
            self.ctx.save_settings()
        self._update_follow_label()
        self.refresh_plan()

    def _update_follow_label(self):
        """Show exactly which main GGUF llama-server will launch with."""
        mf = self.ctx.run_model_file
        if mf is None:
            self.follow_label.setText(
                "Run target: none available - download a model in Tab 1.")
            return
        mode = ("explicit" if self.ctx.settings.run_model_key
                else "follows Tab 1")
        try:
            here = mf.local_path().is_file()
        except OSError:
            here = False
        self.follow_label.setText(
            f"Run target ({mode}): {mf.repo_id} / {mf.quant}\n"
            f"  file: {mf.filename}"
            + ("" if here else "   [NOT downloaded - see Tab 1]"))

    def _populate_thinking_levels(self, select):
        """Fill the Thinking-level combo from the active model profile.

        ``select`` is a level key (str) or a raw budget (int); when nothing
        matches exactly, the closest budget wins so persisted settings map
        onto the named levels.
        """
        prof = self._profile()
        self.thinking_combo.blockSignals(True)
        self.thinking_combo.clear()
        levels = prof.thinking_levels
        if not levels:
            # Unknown model: keep a single explicit Off entry; the fine-grained
            # Reasoning Budget slider still works for any chat template.
            self.thinking_combo.addItem("Off (no thinking)", "")
            self.thinking_combo.setEnabled(False)
            self.thinking_combo.blockSignals(False)
            return
        self.thinking_combo.setEnabled(True)
        for lvl in levels:
            self.thinking_combo.addItem(lvl.label, lvl.key)
        idx = 0
        if isinstance(select, str):
            for i, lvl in enumerate(levels):
                if lvl.key == select:
                    idx = i
                    break
        else:   # int budget from persisted settings -> nearest named level
            best = None
            for i, lvl in enumerate(levels):
                d = abs(lvl.budget - int(select or 0))
                if best is None or d < best[0]:
                    best = (d, i)
            idx = best[1] if best else 0
        self.thinking_combo.setCurrentIndex(idx)
        self.thinking_combo.blockSignals(False)

    def _on_thinking_level_changed(self, _index: int):
        lvl_key = self.thinking_combo.currentData() or ""
        budgets = {l.key: l.budget for l in self._profile().thinking_levels}
        budget = int(budgets.get(lvl_key, 0))
        self.ctx.settings.thinking_level = lvl_key
        # Clamp into the slider range before pushing the value through it so
        # the label + settings stay consistent with the widget state.
        self.reason_slider.blockSignals(True)
        clamped = max(self.reason_slider.minimum(),
                      min(self.reason_slider.maximum(), budget))
        self.reason_slider.setValue(clamped)
        self.reason_slider.blockSignals(False)
        self._set_param("reasoning_budget", clamped, self.reason_val)

    def _sync_reasoning_to_slider(self):
        """Keep the combo aligned when the budget slider is moved manually."""
        budgets = {l.key: l.budget for l in self._profile().thinking_levels}
        cur = self.thinking_combo.currentData() or ""
        val = int(self.reason_slider.value())
        if budgets.get(cur, 0) != val:
            self.ctx.settings.thinking_level = ""
            self.thinking_combo.blockSignals(True)
            for i in range(self.thinking_combo.count()):
                if self.thinking_combo.itemData(i) == "":
                    self.thinking_combo.setCurrentIndex(i)
                    break
            self.thinking_combo.blockSignals(False)

    def _apply_profile_defaults(self):
        """Run-out-of-the-box preset for the selected model (Tab 2).

        Applies sampling defaults, default KV-cache type and the default
        thinking level, then refreshes the plan (the per-quant context cap
        changed with the model).  Called on every model_changed.
        """
        prof = self._profile()
        st = self.ctx.settings
        # Sampling defaults -> sliders (which write back into settings).
        pairs = ((self.temp_slider, self.temp_val, "temperature",
                  prof.temperature, 100),
                 (self.top_p_slider, self.top_p_val, "top_p",
                  prof.top_p, 100),
                 (self.top_k_slider, self.top_k_val, "top_k",
                  prof.top_k, 1),
                 (self.min_p_slider, self.min_p_val, "min_p",
                  prof.min_p, 100))
        for sl, lbl, attr, value, div in pairs:
            sl.setValue(int(round(value * div)))
            self._set_param(attr, sl.value() / div, lbl)
        # Default cache type (kept only if still valid for llama.cpp).
        if prof.default_cache_type in KV_CACHE_TYPES:
            self.cache_combo.setCurrentText(prof.default_cache_type)
        # Default thinking level for this model.
        self._populate_thinking_levels(prof.default_thinking)
        self._on_thinking_level_changed(
            self.thinking_combo.currentIndex())
        if prof.note:
            self.ctx.log.emit(f"[profile] {self.ctx.selected_model_key}: "
                              + prof.note)
        self.refresh_plan()

    @staticmethod
    def _row(slider, label) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, stretch=1)
        h.addWidget(label)
        return w

    def _set_param(self, attr: str, value, label: QLabel):
        setattr(self.ctx.settings, attr, value)
        if attr == "reasoning_budget":
            label.setText(str(int(value)))
        elif isinstance(value, int):
            label.setText(str(value))
        else:
            label.setText(f"{value:.2f}")

    # ------------------------------------------------------------------ binary
    def _refresh_binary_status(self):
        """Show what the current Binary: field resolves to (and why not)."""
        path = self.binary_edit.text().strip() or "llama-server"
        best = llamacpp.pick_best(path)
        if best and best.get("preferred"):
            self.binary_status.setText(
                f"OK: {best['path']}  [backend {best['backend']}]"
                + (f"  {best['version']}" if best["version"] else ""))
            self.binary_status.setStyleSheet("color: #272;")
        elif best:
            self.binary_status.setText(
                f"Field '{path}' not runnable - autodetected instead:\n"
                f"{best['path']}  [backend {best['backend']}]"
                + (f"  {best['version']}" if best["version"] else ""))
            self.binary_status.setStyleSheet("color: #a60;")
        else:
            self.binary_status.setText(
                "No llama-server found. Press 'Autodetect' again, 'Browse…' to\n"
                "pick an existing build, or 'Auto-install / build' to compile one.\n"
                + llamacpp.platform_hint())
            self.binary_status.setStyleSheet("color: #a33;")

    def autodetect_binary(self):
        found = self.ctx.autodetect_binary(silent=False)
        if found:
            self.binary_edit.blockSignals(True)
            self.binary_edit.setText(found)
            self.binary_edit.blockSignals(False)
            self.ctx.save_settings()
            self.ctx.log.emit(f"[llama.cpp] autodetect adopted: {found}")
        else:
            self.ctx.report_error(
                "[llama.cpp] AUTODETECT FAILED: no runnable llama-server was "
                "found.\n" + llamacpp.describe_search() + "\n"
                "    fix    : use 'Auto-install / build' (guided compile), or\n"
                "             'Browse…' if you already have a llama.cpp build.",
                context="llama-server autodetect", dialog=True)
        self._refresh_binary_status()

    def browse_binary(self):
        start = self.binary_edit.text().strip()
        start_dir = os.path.dirname(start) if os.path.sep in start \
            else str(Path.home())
        p, _ = QFileDialog.getOpenFileName(
            self, "Select the llama-server binary", start_dir,
            "Executables (*);;All files (*)")
        if not p:
            return
        try:
            ok = Path(p).is_file() and os.access(p, os.X_OK)
        except OSError as exc:
            self.ctx.report_error(
                f"[llama.cpp] cannot inspect the picked file: "
                f"{type(exc).__name__}: {exc}",
                context=f"browsing for binary {p}", dialog=True)
            return
        if not ok:
            reply = QMessageBox.warning(
                self, "Not executable",
                f"'{p}' is not an executable file.\n\n"
                "Try to chmod +x it anyway and use this path?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.binary_edit.setText(p)   # fires textChanged -> ctx.settings.binary
        self.ctx.save_settings()
        self.ctx.binary_changed.emit()
        ver = llamacpp.probe_version(Path(p))
        self.ctx.log.emit(f"[llama.cpp] user-selected binary: {p}"
                          + (f"  [{ver}]" if ver else
                             "  (no --version output - still accepting it)"))
        self._refresh_binary_status()

    # ------------------------------------------------------- guided install
    def autoinstall(self):
        """Staged guided install - see app.core.llamacpp.InstallRunner.

        Order matters and is enforced here (nothing heavy, and no folder
        dialog, runs until every preflight stage passed):

        A. Existing-binary check: if a usable llama-server was already
           detected, ask whether to keep it or build a different copy.
        B. Preflight stages 1-4 (base tools -> GPU family via PCI IDs ->
           vendor toolchain -> backend choice).  Any failure halts with
           copy-pasteable fix commands *before* any dialog.
        C. Only then: folder picker + confirmation (with the disk estimate
           matching the validated backend), then the background build.
        """
        if self._installer is not None:
            self.ctx.log.emit("[install] an installation is already running - "
                              "see the '[install]' lines in the log.")
            return

        # ---- A) llama-server may already be present ----------------------
        existing = llamacpp.pick_best(self.binary_edit.text().strip())
        if existing:
            reply = QMessageBox.question(
                self, "llama-server already detected",
                f"A usable llama-server was found:\n\n  {existing['path']}\n"
                f"  backend: {existing['backend']}"
                + (f"\n  version: {existing['version']}"
                   if existing.get("version") else "")
                + "\n\nUse this existing binary (recommended), or continue to"
                "\nbuild a different copy in another location?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.Yes:
                self.binary_edit.setText(existing["path"])
                self.ctx.save_settings()
                self.ctx.binary_changed.emit()
                self.ctx.log.emit(
                    f"[install] kept existing llama-server: {existing['path']} "
                    f"(backend {existing['backend']}) - nothing was built.")
                self._refresh_binary_status()
                return
            # Explicitly wants another copy -> fall through to preflight.

        # ---- B) full preflight BEFORE any folder dialog ------------------
        probe = llamacpp.InstallRunner(llamacpp.suggested_install_root(),
                                       backend="auto")
        pf = probe.preflight()
        if not pf.ok:
            err = pf.error
            assert err is not None
            self.ctx.log.emit(err.report())
            detail = err.message + "\n    reason : " + err.detail \
                if err.detail else err.message
            fixes = "".join(f"\n    run    : $ {c}" for c in err.commands)
            notes = "".join(f"\n    note   : {m}" for m in err.manual)
            sugg = "".join(f"\n    alt    : {s}" for s in err.suggestions)
            self.ctx.report_error("[install] preflight failed - nothing was "
                                  "downloaded or compiled.\n" + detail
                                  + fixes + notes + sugg,
                                  context="guided llama.cpp install",
                                  dialog=True)
            return

        backend = pf.backend
        need_gb = probe.required_disk_gb()   # now matches the validated backend
        gpu_desc = ", ".join(sorted(pf.gpu_families)) or "none detected"

        # ---- C) only now: choose WHERE to install ------------------------
        folder = QFileDialog.getExistingDirectory(
            self, "Choose the llama.cpp install folder (cancel = default)",
            llamacpp.suggested_install_root())
        dest = folder or llamacpp.suggested_install_root()
        reply = QMessageBox.question(
            self, "Confirm guided llama.cpp install",
            "Preflight passed. The guided installer will now:\n"
            "  1. clone llama.cpp (~400 MB shallow git clone)\n"
            "  2. configure + compile llama-server (several minutes)\n\n"
            f"Install folder : {dest}\n"
            f"GPU hardware   : {gpu_desc}\n"
            f"Build backend  : {backend.upper()} ({pf.reason})\n"
            f"Free disk need : ~{need_gb:.0f} GiB\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            self.status_note = "Guided install cancelled by user."
            self.ctx.log.emit("[install] " + self.status_note)
            return

        # ---- D) run in background with live logs -------------------------
        runner = llamacpp.InstallRunner(dest, backend=backend)
        runner._preflight = pf          # reuse validation; _run re-checks anyway
        runner.log.connect(self.ctx.log.emit)
        runner.finished.connect(self._on_install_done)
        self.install_btn.setEnabled(False)
        self.install_btn.setText("Building… (click to cancel)")
        self._installer = runner
        runner.start()

    def _on_install_done(self, success: bool, message: str):
        runner, self._installer = self._installer, None
        del runner
        self.install_btn.setEnabled(True)
        self.install_btn.setText("Auto-install / build")
        if success:
            self.binary_edit.setText(message)   # fires textChanged -> settings
            self.ctx.save_settings()
            self.ctx.binary_changed.emit()
            self.ctx.log.emit(
                "[install] llama.cpp ready - Binary: updated and persisted. "
                "Press 'Start llama-server' when your model plan is set.")
        else:
            # message is the verbose AppError report from the worker thread.
            self.ctx.log.emit(message)
            self.ctx.report_error(message, context="guided llama.cpp install",
                                  dialog=True)
        self._refresh_binary_status()

    # ------------------------------------------------------------------ plan
    def compute_plan(self):
        """Build the resource plan; never raises - reports instead.

        Always plans for the Tab 2 *run target* (explicit 'Model to run'
        override, otherwise the Tab 1 selection) so the previewed -c/-ngl
        numbers match what Start will actually launch.

        Returns ``(plan, model_file, arch)`` or ``(None, None, None)`` after
        emitting a verbose error report through ``ctx.report_error``.
        """
        ctx = self.ctx
        spec = ctx.registry[ctx.run_model_key]
        mf = ctx.run_model_file
        if mf is None:
            ctx.report_error(
                "No GGUF entry for quantization "
                f"'{ctx.run_quant}' of '{spec.display_name}'.\n"
                "    fix    : pick another file in the 'Quant / file' dropdown "
                "(or press 'Refresh from Hub' in Tab 1).",
                context="resource planning", dialog=False)
            return None, None, None
        try:
            arch = read_arch_info(spec.hf_config_repo, mf.local_path())
            # Resolve this model's own projector (live repo listing first);
            # never a foreign repo's mmproj - see resolve_mmproj_companion.
            mm = mf.mmproj_local_path()
            # Per-model/per-quant KV-cache hard limit from the profile
            # (run-out-of-the-box presets); falls back to the global cap for
            # models without a profile entry.
            prof = get_model_profile(ctx.run_model_key)
            max_ctx = prof.max_context_by_quant.get(mf.quant or "")
            plan = plan_resources(ctx.hw, mf, arch,
                                  cache_type=ctx.settings.cache_type,
                                  mmproj_path=mm, max_context=max_ctx)
        except ResourcePlanError as exc:
            ctx.report_error(exc, context=f"planning resources for {mf.filename}",
                             dialog=True)
            return None, None, None
        except Exception as exc:
            ctx.report_error(exc, context=f"planning resources for {mf.filename}",
                             dialog=True)
            return None, None, None
        return plan, mf, arch

    def refresh_plan(self):
        plan, mf, arch = self.compute_plan()
        if plan is None:
            self.plan_label.setText("Plan: unavailable - see Log (Tab 5) for the error report.")
            self._last_plan = None
            self._last_file = None
            return None
        txt = f"[{arch.source}] {plan.summary()}"
        if plan.notes:
            txt += "\n" + "\n".join(plan.notes)
        self.plan_label.setText(txt)
        self._last_plan = plan
        self._last_file = mf
        return plan

    # ------------------------------------------------------------------ server
    def start_server(self):
        # Fast, friendly binary check before the (potentially slow) plan
        # computation - mirrors what the server preflight will say, but with
        # direct pointers to the three buttons that can fix it.
        if self.ctx.server.find_binary() is None:
            self.ctx.report_error(
                "[server] cannot start: no runnable llama-server found.\n"
                + llamacpp.describe_search() + "\n"
                "    fix    : 'Autodetect' (re-scan), 'Browse…' (pick an\n"
                "             existing build) or 'Auto-install / build'\n"
                "             (guided clone+compile) in the Server box above.",
                context="starting llama-server", dialog=True)
            self._refresh_binary_status()
            return
        plan, mf, _arch = self.compute_plan()
        if plan is None or mf is None:
            return
        ctx = self.ctx
        from app.core.server import ServerConfig
        mm_path = None
        # Resolve this model's *own* projector; refuse to pass a companion
        # that belongs to a different repo (the ToriiGate-mmproj-on-Qwen
        # crash): llama-server aborts at load time with an n_embd mismatch.
        p = mf.mmproj_local_path()
        if p is not None and p.is_file():
            try:
                bad = mf.mmproj_companion_error(p)
            except Exception:
                bad = None
            if bad:
                ctx.log.emit(
                    f"[server] WARNING: ignoring incompatible mmproj for "
                    f"{mf.filename}: {bad}\n"
                    f"    server would start text-only; run 'Download & "
                    f"Verify' in Tab 1 to fetch this model's own projector.")
            else:
                mm_path = str(p)
        cfg = ServerConfig(
            model=mf,
            mmproj_path=mm_path,
            plan=plan,
            host=ctx.settings.host,
            port=ctx.settings.port,
            split_mode=ctx.settings.split_mode,
            tensor_split=self.ctx.hw.tensor_split_string(),
            cache_type_k=ctx.settings.cache_type,
            cache_type_v=ctx.settings.cache_type,
            flash_attn=self.fa_combo.currentText().strip().lower() or "auto",
            reasoning_budget=int(ctx.settings.reasoning_budget),
            extra_args=self.extra_args.toPlainText().strip(),
            explicit_mode=ctx.explicit_mode,
            binary=self.binary_edit.text().strip() or "llama-server",
        )
        ctx.log.emit("[server] requested start with plan: " + plan.summary())
        # start() runs its own preflight checks and emits every problem into
        # the global log ("[server] PRE-FLIGHT ERROR: ...") - it returns
        # False without raising when something blocks the launch.
        ok = ctx.server.start(cfg)
        if not ok:
            ctx.log.emit(
                "[server] start aborted - see the PRE-FLIGHT ERROR lines above "
                "(also mirrored in Tab 5 'Log Output'). Nothing was launched.")

    def on_server_state(self, state: str):
        self.state_label.setText(f"Server: {state}")
        running = state in ("starting", "ready")
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
