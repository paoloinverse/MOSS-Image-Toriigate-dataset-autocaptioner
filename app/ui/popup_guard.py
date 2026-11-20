"""Guard for QComboBox popups on headless X sessions (0x0 screen).

Background
----------
When no physical output is connected to the X server (e.g. the KVM switch
is pointed at another machine and this box is driven over VNC), Xorg's
RANDR reports no active outputs.  Qt's xcb backend (RANDR >= 1.5) then
falls back to a *fake* screen whose geometry is never initialised:
``QXcbConnection::initializeScreensFromMonitor`` creates
``QXcbScreen(conn, virtualDesktop, nullptr)`` and ``setMonitor(nullptr)``
returns before touching the geometry, so the screen (named after the
display, e.g. ":0.0") is 0x0.

``QComboBox::showPopup()`` clamps the popup rect to the screen rect
(``listRect.setWidth(screen.width())``, ``qMin(listRect.height(),
screen.height())`` and clamping to the screen origin), so the popup
container ends up shown as a 2x2 pixel sliver at (0,0): the combo still
"opens" (keyboard navigation works, selections change) but the list is
invisible.

Fix
---
If ``QScreen.availableGeometry()`` (or ``geometry()``) is empty, an app-
wide event filter re-places every combo popup at its intended geometry:
``combo.width()`` x the list view's ``sizeHint()`` height, directly below
the combo -- flipping above it if a one-shot ``xrandr`` probe says the
list would otherwise run off the bottom of the physical screen.

Normal sessions are completely untouched: the filter bails out
immediately whenever the screen geometry is sane (re-checked per event,
so the guard disarms itself automatically if a monitor is reconnected).

Install in ``app/main.py``:  ``from app.ui import popup_guard; popup_guard.install(app)``
"""
from __future__ import annotations

import re
import subprocess

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PySide6.QtWidgets import QApplication, QComboBox, QWidget

__all__ = ["install", "screen_degenerate"]

# hot-path constants (the filter sees only Show/Move/Resize of top-levels)
_EV_SHOW = QEvent.Type.Show
_EV_MOVE = QEvent.Type.Move
_EV_RESIZE = QEvent.Type.Resize

_PHYSICAL_SIZE: tuple[int, int] | None = None
_PHYSICAL_PROBE_DONE = False
_CURRENT_GUARD: "_PopupGuard | None" = None


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------
def screen_degenerate(app: QApplication) -> bool:
    """True when Qt sees an unusable (0x0) primary screen.

    On a healthy X session this is False and the guard is a no-op.  On a
    headless session (all outputs disconnected / KVM switched away) Qt's
    fake fallback screen reports empty geometry and this returns True.
    """
    scr = app.primaryScreen()
    if scr is None:
        return False
    return scr.geometry().isEmpty() or scr.availableGeometry().isEmpty()


def physical_screen_size() -> tuple[int, int] | None:
    """Best-effort *physical* X screen size via xrandr (cached, one shot).

    Qt's own screen info is 0x0 in the degenerate case, so the real size
    can only be asked of the X server.  Returns None if xrandr is missing
    or unreadable; the caller then skips the flip-up logic.
    """
    global _PHYSICAL_SIZE, _PHYSICAL_PROBE_DONE
    if not _PHYSICAL_PROBE_DONE:
        _PHYSICAL_PROBE_DONE = True
        try:
            out = subprocess.run(["xrandr", "--query"], capture_output=True,
                                 text=True, timeout=3).stdout
            m = re.search(r"current\s+(\d+)\s*x\s*(\d+)", out)
            if m:
                _PHYSICAL_SIZE = (int(m.group(1)), int(m.group(2)))
        except Exception:
            pass
    return _PHYSICAL_SIZE


# --------------------------------------------------------------------------
# intended geometry
# --------------------------------------------------------------------------
def intended_popup_rect(combo: QComboBox, container: QWidget) -> QRect | None:
    """The geometry the popup *would* have on a sane screen.

    width  = max(combo.width(), list sizeHint width)
    height = the list view's sizeHint height (model-driven, screen-
             independent -- the container's own sizeHint is a bare QFrame's
             (2, 2) and must not be used)
    pos    = directly below the combo, flipped above / clamped to the
             physical screen when known.
    """
    view = combo.view()
    if view is None:
        return None
    w = max(combo.width(), view.sizeHint().width())
    h = view.sizeHint().height()
    if h < 8:                       # degenerate view state -> fall back to rows
        h = (view.sizeHintForRow(0) if combo.count() else 0) * combo.count()
    if w < 8 or h < 8:
        return None

    combo_tl = combo.mapToGlobal(QPoint(0, 0))
    rect = QRect(combo_tl.x(), combo_tl.y() + combo.height(), w, h)

    phys = physical_screen_size()
    if phys:
        pw, ph = phys
        if rect.right() > pw:
            rect.moveRight(pw)
        if rect.bottom() > ph:      # no room below -> open above the combo
            rect.moveTop(combo_tl.y() - h)
        if rect.bottom() > ph:
            rect.moveBottom(ph)     # taller than the screen: clamp
        if rect.top() < 0:
            rect.moveTop(0)
    return rect


# --------------------------------------------------------------------------
# the filter
# --------------------------------------------------------------------------
class _PopupGuard(QObject):
    """App-wide event filter: re-places combo popups when the screen is 0x0.

    Reacts to Show/Move/Resize of any top-level ``Qt.Popup`` window whose
    parent widget is a QComboBox (i.e. the QComboBoxPrivateContainer --
    exposed to Python as a bare QFrame).  Move/Resize are covered too,
    because Qt may re-apply its (degenerate) computed rect *after* show()
    (horizontal-scrollbar case).  Re-entry terminates: setGeometry only
    fires when the rect actually changes, and the rect is idempotent.
    """

    def __init__(self, app: QApplication,
                 degenerate=screen_degenerate) -> None:
        super().__init__(app)
        self._app = app
        self._degenerate = degenerate

    def eventFilter(self, obj: object, event) -> bool:
        try:
            t = event.type()
            if t not in (_EV_SHOW, _EV_MOVE, _EV_RESIZE):
                return False
            if not isinstance(obj, QWidget) or not obj.isWindow():
                return False
            if not (obj.windowFlags() & Qt.Popup):
                return False
            combo = obj.parentWidget()
            if not isinstance(combo, QComboBox):
                return False          # QMenu or any other popup: leave alone
            if not self._degenerate(self._app):
                return False          # healthy screen: stock Qt behaviour
            rect = intended_popup_rect(combo, obj)
            if rect is not None and obj.geometry() != rect:
                obj.setGeometry(rect)
        except Exception:
            pass                      # never break the popup over the guard
        return False


def install(app: QApplication, degenerate=screen_degenerate) -> bool:
    """Install the guard on *app*.

    Returns True if the screen was already degenerate at install time
    (i.e. the guard is expected to work immediately).
    """
    global _CURRENT_GUARD
    if _CURRENT_GUARD is not None:
        app.removeEventFilter(_CURRENT_GUARD)
    _CURRENT_GUARD = _PopupGuard(app, degenerate=degenerate)
    app.installEventFilter(_CURRENT_GUARD)
    return screen_degenerate(app)
