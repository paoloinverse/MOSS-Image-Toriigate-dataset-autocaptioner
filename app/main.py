#!/usr/bin/env python3
"""Entry point for the Multi-Image Auto-Captioner.

Run with:  python -m app.main      (requires a display; use xvfb-run off-GUI)
"""
from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from app import APP_NAME
    from app.context import AppContext
    from app.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

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
