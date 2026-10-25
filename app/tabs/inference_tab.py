"""Tab 2 - Inference Parameters.

* Raw llama.cpp passthrough args + server binary/host/port/split-mode.
* Generation sliders (Temperature, Top-P, Top-K, Min-P).
* "Reasoning Budget" slider -> --reasoning-budget / reasoning_budget_tokens.
* KV cache type dropdown (f16 / q8_0 / q4_0 …).
* Start/Stop llama-server buttons; the computed resource plan (-c/-ngl) is
  previewed live so the user sees the dynamic KV sizing decision.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSlider, QSpinBox, QTextEdit,
                               QVBoxLayout, QWidget)

from app.context import AppContext
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

        root = QVBoxLayout(self)

        # ---- Server ----------------------------------------------------------
        srv_box = QGroupBox("llama.cpp Server")
        form = QFormLayout(srv_box)
        self.binary_edit = QLineEdit("llama-server")
        self.host_edit = QLineEdit(st.host)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(st.port)
        self.split_combo = QComboBox()
        self.split_combo.addItems(["row", "layer", "graph"])
        form.addRow("Binary:", self.binary_edit)
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
        self.ctx.server_state.connect(self.on_server_state)
        self.ctx.model_changed.connect(self.refresh_plan)

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
