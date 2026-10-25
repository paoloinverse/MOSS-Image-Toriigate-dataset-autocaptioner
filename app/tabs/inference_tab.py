"""Tab 2 - Inference Parameters.

* Raw llama.cpp passthrough args + server binary/host/port/split-mode.
* Binary row: 'Autodetect' (searches PATH/common build dirs, ranks CUDA >
  ROCm > Vulkan > CPU), 'Browse…' (pick an existing install) and
  'Auto-install / build' (guided clone+compile in a background thread with
  live log streaming, explicit folder choice, disk/toolchain preflight).
* Generation sliders (Temperature, Top-P, Top-K, Min-P).
* "Reasoning Budget" slider -> --reasoning-budget / reasoning_budget_tokens.
* KV cache type dropdown (f16 / q8_0 / q4_0 …).
* Start/Stop llama-server buttons; the computed resource plan (-c/-ngl) is
  previewed live so the user sees the dynamic KV sizing decision.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSlider,
                               QSpinBox, QTextEdit, QVBoxLayout, QWidget)

from app.context import AppContext
from app.core import llamacpp
from app.core.models import read_arch_info, resolve_mmproj
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
            "a background thread with live logs. Asks for confirmation and\n"
            "an install folder first; nothing heavy runs silently.")
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
            "Reasoning Budget", 0, 8192, st.reasoning_budget, 1)
        self.reason_slider.setSingleStep(64)

        gf.addRow("Temperature:", self._row(self.temp_slider, self.temp_val))
        gf.addRow("Top-P:", self._row(self.top_p_slider, self.top_p_val))
        gf.addRow("Top-K:", self._row(self.top_k_slider, self.top_k_val))
        gf.addRow("Min-P:", self._row(self.min_p_slider, self.min_p_val))
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
        self.cache_combo.addItems(["f16", "q8_0", "q4_0", "q4_1", "fp8"])
        self.cache_combo.setCurrentText(st.cache_type)
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
            lambda v: self._set_param("reasoning_budget", v, self.reason_val))
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
        self.ctx.model_changed.connect(self.refresh_plan)
        self.ctx.binary_changed.connect(self._refresh_binary_status)

        self._refresh_binary_status()

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
        if self._installer is not None:
            self.ctx.log.emit("[install] an installation is already running - "
                              "see the '[install]' lines in Tab 5.")
            return
        # 1) Toolchain preflight BEFORE bothering the user with dialogs.
        runner_probe = llamacpp.InstallRunner(llamacpp.suggested_install_root())
        ok, missing = runner_probe.check_prerequisites()
        tool_msg = ("Build tools detected (git, cmake, g++)."
                    if ok else "MISSING build tools:\n  - " + "\n  - ".join(missing))
        # 2) Ask where to install (default folder pre-filled & editable).
        folder = QFileDialog.getExistingDirectory(
            self, "Choose the llama.cpp install folder (cancel = default)",
            llamacpp.suggested_install_root())
        dest = folder or llamacpp.suggested_install_root()
        reply = QMessageBox.question(
            self, "Confirm guided llama.cpp install",
            "The guided installer will now:\n"
            "  1. clone llama.cpp (~400 MB shallow git clone)\n"
            "  2. configure + compile llama-server (several minutes)\n\n"
            f"Install folder : {dest}\n"
            f"GPU backend    : auto-detect (CUDA/ROCm/Vulkan/CPU)\n"
            f"Free disk need : ~{runner_probe.required_disk_gb():.0f} GiB\n\n"
            f"{tool_msg}\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            self.status_note = "Guided install cancelled by user."
            self.ctx.log.emit("[install] " + self.status_note)
            return
        if not ok:
            self.ctx.report_error(
                "[install] cannot start the guided build - missing tools:\n  - "
                + "\n  - ".join(missing) + "\n"
                "    fix    : sudo apt install " +
                " ".join(m.split()[0] for m in missing) +
                "\n             then press 'Auto-install / build' again.",
                context="guided llama.cpp install", dialog=True)
            return
        # 3) Run in background with live logs.
        runner = llamacpp.InstallRunner(dest, backend="auto")
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
            plan = plan_resources(ctx.hw, mf, arch,
                                  cache_type=ctx.settings.cache_type,
                                  mmproj_path=mm)
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
            cache_type_k=ctx.settings.cache_type,
            cache_type_v=ctx.settings.cache_type,
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
