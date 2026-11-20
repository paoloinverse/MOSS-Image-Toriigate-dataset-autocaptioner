#!/usr/bin/env python3
"""Entry point for the Multi-Image Auto-Captioner.

Run with:  python -m app.main      (requires a display; use xvfb-run off-GUI)
"""
from __future__ import annotations

import logging
import sys


def _install_popup_diagnostics(app) -> None:
    from PySide6.QtCore import QEvent, QObject, Qt
    from PySide6.QtWidgets import QWidget

    logger = logging.getLogger("popup_diag")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler("dropdown_debug.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    class PopupDiagnosticFilter(QObject):
        def eventFilter(self, obj, event):
            if isinstance(obj, QWidget) and obj.isWindow():
                is_popup = bool(obj.windowFlags() & Qt.WindowType.Popup)
                if is_popup and event.type() in (
                    QEvent.Type.Show,
                    QEvent.Type.Move,
                    QEvent.Type.Resize,
                    QEvent.Type.Paint,
                ):
                    geom = obj.geometry()
                    global_pos = obj.mapToGlobal(obj.rect().topLeft())
                    screen = obj.screen()
                    screen_geom = screen.geometry().getRect() if screen else "NoScreen"
                    parent = obj.parentWidget()

                    logger.debug(
                        "Event=%s | Class=%s | ObjectName='%s' | Visible=%s "
                        "| UpdatesEnabled=%s | Opacity=%s | Geom=%s "
                        "| GlobalTopLeft=(%d, %d) | ScreenGeom=%s "
                        "| Translucent=%s | AppStyleSheetLen=%d | ObjStyleSheetLen=%d "
                        "| Flags=%s | ParentFlags=%s",
                        event.type().name,
                        obj.__class__.__name__,
                        obj.objectName(),
                        obj.isVisible(),
                        obj.updatesEnabled(),
                        obj.windowOpacity(),
                        geom.getRect(),
                        global_pos.x(),
                        global_pos.y(),
                        screen_geom,
                        obj.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground),
                        len(app.styleSheet()),
                        len(obj.styleSheet()),
                        obj.windowFlags(),
                        parent.windowFlags() if parent else None,
                    )
            return False

    diag_filter = PopupDiagnosticFilter(app)
    app.installEventFilter(diag_filter)


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from app import APP_NAME
    from app.context import AppContext
    from app.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    _install_popup_diagnostics(app)

    # Headless/VNC guard: if no physical output is connected to the X server
    # (KVM switched away), Qt sees a 0x0 screen and every combo popup
    # degenerates to an invisible 2x2 sliver at (0,0).  Re-place the popups
    # at their intended geometry in that (and only that) case.  No-op on a
    # normal desktop.
    from app.ui import popup_guard
    if popup_guard.install(app):
        print("[popup_guard] 0x0 screen detected (headless/VNC?): combo "
              "popups will be re-placed manually",
              file=sys.stderr, flush=True)

    ctx = AppContext()
    ctx.hw.start()                       # hardware management thread on launch
    ctx.log.emit("[app] starting " + APP_NAME)

    win = MainWindow(ctx)
    win.show()
    rc = app.exec()

    ctx.hw.stop()
    ctx.server.stop()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())