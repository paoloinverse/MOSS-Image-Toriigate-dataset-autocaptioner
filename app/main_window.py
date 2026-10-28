"""Main window: tabbed control area (left) + global Caption Editor side panel."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QMainWindow, QScrollArea, QSizePolicy,
                               QSplitter, QTabWidget, QWidget)

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
        # Vertically resizable: a modest minimum height (instead of the large
        # implicit minimum the tab content would impose) lets the user shrink
        # and grow the window freely; anything that no longer fits is reachable
        # through the scroll areas below.
        self.resize(1500, 900)
        self.setMinimumHeight(400)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ---- tabs -------------------------------------------------------------
        self.tabs = QTabWidget()
        self.tab_hardware = HardwareModelsTab(ctx)
        self.tab_inference = InferenceTab(ctx)
        self.tab_presets = PresetsTab(ctx)
        self.tab_explorer = DatasetExplorerTab(ctx)
        self.tab_batch = BatchTab(ctx)

        # Each tab page is wrapped in a vertical QScrollArea: when the window
        # is shrunk below a tab's natural content height, a vertical scrollbar
        # appears instead of clipping widgets or forcing the window bigger.
        for tab, label in ((self.tab_hardware, "1 · Hardware & Models"),
                           (self.tab_inference, "2 · Inference Params"),
                           (self.tab_presets, "3 · Presets & Prompts"),
                           (self.tab_explorer, "4 · Dataset Explorer"),
                           (self.tab_batch, "5 · Batch Processing")):
            self.tabs.addTab(self._make_tab_scrollable(tab), label)
        splitter.addWidget(self.tabs)
        # ---- global side panel ---------------------------------------------
        # Same vertical-scroll treatment for the Caption Editor side panel.
        self.side_panel = CaptionSidePanel(ctx)
        splitter.addWidget(self._wrap_in_scroll_area(
            self.side_panel, stretch_width=True))
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

    # ------------------------------------------------------------- scrollable UI
    def _wrap_in_scroll_area(self, widget: QWidget, *,
                             stretch_width: bool = False) -> QScrollArea:
        """Wrap *widget* in a vertical-only QScrollArea.

        The horizontal scrollbar is suppressed so content always fits the
        available width; the vertical scrollbar appears as needed when the
        window is shrunk below the content's natural height.  With
        *stretch_width* the wrapped widget keeps expanding horizontally to the
        viewport (used by the side panel, which must still follow the splitter
        handle); otherwise its natural minimum width is preserved on the area.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        if stretch_width:
            # Let the panel keep stretching horizontally with the splitter.
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            scroll.setMinimumWidth(widget.minimumWidth())
        else:
            # Horizontal shrink is handled via the widgets' own minimum
            # widths; the scroll area adopts the content's minimum width.
            widget.setSizePolicy(QSizePolicy.MinimumExpanding,
                                 QSizePolicy.Minimum)
        scroll.setWidget(widget)
        return scroll

    def _make_tab_scrollable(self, tab: QWidget) -> QScrollArea:
        """Return a scroll area containing *tab*'s layout.

        QTabWidget reparents whatever is handed to addTab(), so instead of
        adding the tab first we move its top-level layout into a fresh inner
        widget and wrap that in a QScrollArea.  The now-empty *tab* object is
        kept as a child of the inner widget (a zero-footnote controller) so
        existing code holding it — e.g. ``MainWindow.tab_presets`` — keeps
        working and Python references stay valid.
        """
        inner = QWidget()
        layout = tab.layout()
        if layout is not None:
            inner.setLayout(layout)          # re-parents the layout to `inner`
        tab.setParent(inner)                 # keep the controller alive
        return self._wrap_in_scroll_area(inner)

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
