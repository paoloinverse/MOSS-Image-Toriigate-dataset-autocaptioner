"""Main window: tabbed control area (left) + global Caption Editor side panel."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QMainWindow, QSplitter, QTabWidget)

from app import APP_NAME, __version__
from app.context import AppContext
from app.tabs.batch_tab import BatchTab
from app.tabs.explorer_tab import DatasetExplorerTab
from app.tabs.hardware_tab import HardwareModelsTab
from app.tabs.inference_tab import InferenceTab
from app.tabs.presets_tab import PresetsTab
from app.ui.side_panel import CaptionSidePanel


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.resize(1500, 900)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ---- tabs -------------------------------------------------------------
        self.tabs = QTabWidget()
        self.tab_hardware = HardwareModelsTab(ctx)
        self.tab_inference = InferenceTab(ctx)
        self.tab_presets = PresetsTab(ctx)
        self.tab_explorer = DatasetExplorerTab(ctx)
        self.tab_batch = BatchTab(ctx)
        self.tabs.addTab(self.tab_hardware, "1 · Hardware & Models")
        self.tabs.addTab(self.tab_inference, "2 · Inference Params")
        self.tabs.addTab(self.tab_presets, "3 · Presets & Prompts")
        self.tabs.addTab(self.tab_explorer, "4 · Dataset Explorer")
        self.tabs.addTab(self.tab_batch, "5 · Batch Processing")
        splitter.addWidget(self.tabs)

        # ---- global side panel ---------------------------------------------
        self.side_panel = CaptionSidePanel(ctx)
        splitter.addWidget(self.side_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1100, 400])

        # Focus changes in the explorer drive the side panel.
        ctx.image_focused.connect(self.side_panel.set_image)
        self.side_panel.caption_saved.connect(lambda p: ctx.log.emit(
            f"[panel] manual caption saved -> {p}"))

        # ---- startup config override ------------------------------------------
        # Tabs are built from the hardwired/persisted defaults, so if a
        # captioner_config.json was imported at AppContext construction we
        # must push that state into the widgets now (same refresh path Tab 3's
        # Import button uses).
        if getattr(ctx, "startup_config_source", None):
            try:
                self.tab_presets.reload_combos()
            except Exception as exc:
                ctx.report_error(exc, context="refreshing presets after "
                                 "startup config import", dialog=False)
            try:
                self.tab_inference.refresh_from_context()
            except Exception as exc:
                ctx.report_error(exc, context="refreshing inference tab after "
                                 "startup config import", dialog=False)
            try:
                self.tab_hardware.refresh_from_context()
            except Exception as exc:
                ctx.report_error(exc, context="refreshing hardware tab after "
                                 "startup config import", dialog=False)
            ctx.log.emit(
                f"[config] all tabs refreshed from startup config "
                f"({ctx.startup_config_source})")

        self.statusBar().showMessage("Ready.")
        ctx.server_state.connect(lambda s: self.statusBar().showMessage(
            f"llama-server: {s}"))

    # ------------------------------------------------------------------ lifecycle
    def closeEvent(self, event):  # noqa: N802
        try:
            if self.ctx.server.is_running():
                self.ctx.server.stop()
            if self.ctx.hw.isRunning():
                self.ctx.hw.stop()
        except Exception:
            pass
        super().closeEvent(event)
