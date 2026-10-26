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
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSlider,
                               QSpinBox, QTextEdit, QVBoxLayout, QWidget)

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
        # Initialise from the persisted setting (default is "layer" - see
        # InferenceSettings) so the GUI never silently diverges from what
        # build_args() would actually pass; unknown values fall back to it.
        if st.split_mode in ("row", "layer", "graph"):
            self.split_combo.setCurrentText(st.split_mode)
        else:
            self.split_combo.setCurrentText("layer")
            st.split_mode = "layer"
        form.addRow("Host:", self.host_edit)
        form.addRow("Port:", self.port_spin)
        form.addRow("Split mode (multi-GPU):", self.split_combo)
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
        # Verbose llama-server traffic logging: dumps the exact prompt sent
        # and the full raw reply (incl. streamed 'thinking') into the global
        # log for debugging odd responses.  Also settable via MOSS_VERBOSE_LOG=1.
        self.verbose_check = QCheckBox(
            "Verbose request/response logging (prompts + raw replies)")
        self.verbose_check.setToolTip(
            "Logs every llama-server request payload (system/user prompts,\n"
            "image redacted) and the complete response - including the model's\n"
            "streamed thinking (delta.reasoning_content) - to the main log.\n"
            "Equivalent to launching with MOSS_VERBOSE_LOG=1.")
        from app.core.api_client import verbose_log_enabled
        self.verbose_check.setChecked(verbose_log_enabled(st))
        self.verbose_check.toggled.connect(
            lambda on: setattr(self.ctx.settings, "verbose_logging", bool(on)))
        gf.addRow(self.verbose_check)
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

        self._refresh_binary_status()

    # ---------------------------------------------------------- model profiles
    def _profile(self):
        """ModelProfile of the currently selected model (Tab 1 key/quant)."""
        return get_model_profile(self.ctx.selected_model_key)

    def _on_model_changed(self):
        """model_changed handler: re-populate per-model widgets + re-plan.

        Preset sampling defaults are only applied once per model *key* (not
        on every quant switch), so a user's manual slider tweaks survive
        quant changes - while the Thinking-level combo is always rebuilt to
        match the active profile and the plan is always recomputed with the
        new per-quant context cap.
        """
        key = self.ctx.selected_model_key
        if key != self._profiled_key:
            self._profiled_key = key
            self._apply_profile_defaults()
            return
        self._populate_thinking_levels(
            self.ctx.settings.thinking_level
            or self.ctx.settings.reasoning_budget)
        self.refresh_plan()

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

        Returns ``(plan, model_file, arch)`` or ``(None, None, None)`` after
        emitting a verbose error report through ``ctx.report_error``.
        """
        ctx = self.ctx
        spec = ctx.active_model_spec
        mf = spec.file_for_quant(ctx.selected_quant)
        if mf is None:
            ctx.report_error(
                "No GGUF entry for quantization "
                f"'{ctx.selected_quant}' of '{spec.display_name}'.\n"
                "    fix    : pick another quant in Tab 1.",
                context="resource planning", dialog=False)
            return None, None, None
        try:
            arch = read_arch_info(spec.hf_config_repo, mf.local_path())
            mm = mf.mmproj_local_path() if mf.mmproj_filename else None
            # Per-model/per-quant KV-cache hard limit from the profile
            # (run-out-of-the-box presets); falls back to the global cap for
            # models without a profile entry.
            prof = get_model_profile(ctx.selected_model_key)
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
        if mf.mmproj_filename and mf.mmproj_local_path():
            p = mf.mmproj_local_path()
            if p.is_file():
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
