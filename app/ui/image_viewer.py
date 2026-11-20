"""Zoomable / pannable image viewer with a bounding-box overlay layer.

* Mouse wheel  -> zoom around the cursor position.
* Click+drag   -> pan (scroll-area style hand drag).
* Double click -> fit-to-window reset.
* Hover        -> the box under the cursor is highlighted and its text shown in
  a 50 %-transparent overlay panel (Tab 4's accuracy-review aid).

The overlay is painted in :meth:`drawForeground` from plain data
(:class:`app.core.boxes.AnnotatedBox`, in **image pixel** coordinates), so
zooming and panning never need a re-layout.  Two rules keep it readable at every
scale, and both are recomputed per paint rather than cached:

* strokes are **cosmetic** (``2 / scale`` scene units == ~2 screen pixels), and
* the tooltip is drawn in **viewport** coordinates with a **pixel** font size, so
  its text size never depends on how far the user has zoomed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QPainter, QPen,
                           QPixmap)

from app.core.boxes import LABEL_MODES, is_cjk, wrap_tokens
from PySide6.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView)

MIN_SCALE = 0.02
MAX_SCALE = 40.0
ZOOM_STEP = 1.25

#: tooltip typography / geometry (device-independent pixels)
TOOLTIP_FONT_MIN = 12
TOOLTIP_FONT_MAX = 28
TOOLTIP_FONT_RATIO = 0.12       # of the hovered box's smaller on-screen side
TOOLTIP_PAD = 8
TOOLTIP_MAX_W = 480.0
TOOLTIP_VIEWPORT_FRACTION = 0.5
TOOLTIP_ALPHA = 128             # "50 % transparent" overlay box
FILL_ALPHA = 46                 # ~18 % - keeps overlapping boxes separable
FILL_ALPHA_HOVER = 80

#: always-on chips (chronological number and/or text, drawn at each box corner)
LABEL_FONT_MIN = 10
LABEL_FONT_MAX = 18
LABEL_FONT_RATIO = 0.10         # of the box's smaller on-screen side
LABEL_ALPHA = 210               # near-opaque chip; the boxes must stay visible
LABEL_PAD = 3
LABEL_MAX_CHARS = 28
LABEL_MAX_BOXES = 200           # beyond this the chips stop being readable


@dataclass
class _PaintedBox:
    """Pre-resolved drawing state for one box (built once in :meth:`set_boxes`)."""
    rect: QRectF
    color: QColor
    dashed: bool = False
    text: str = ""
    norm: tuple = ()
    group_size: int = 1
    order: int = 0              # chronological index (for the number chips)
    source: object = None       # the originating AnnotatedBox


def _ink_for(color: QColor, alpha: int, backdrop: int = 128) -> QColor:
    """Black or white text for *color* composited at *alpha* over *backdrop*.

    Shared by the tooltip panel (50 % box colour over the scan) and the label
    chips (near-opaque box colour), so both stay readable whatever the md5 hue.
    """
    a = max(0, min(255, alpha)) / 255.0
    comp = [a * color.red() + (1 - a) * backdrop,
            a * color.green() + (1 - a) * backdrop,
            a * color.blue() + (1 - a) * backdrop]
    luma = 0.2126 * comp[0] + 0.7152 * comp[1] + 0.0722 * comp[2]
    return QColor(0, 0, 0) if luma > 128 else QColor(255, 255, 255)


class ZoomPanImageView(QGraphicsView):
    #: emitted with the overlay index when the user *clicks* a box (a click, not
    #: the end of a pan) - Tab 4 forwards it so the Caption Editor can select
    #: exactly that text.  -1 means "clicked empty space", boxes only.
    box_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pix_item)

        self._scale = 1.0
        self._dragging = False
        self._last_pos: QPoint | None = None

        # ---- overlay state ------------------------------------------------
        self._boxes: list[_PaintedBox] = []
        self._overlay_enabled = True
        self._hover = -1
        self._hover_pos: QPoint | None = None
        self._image_path: str | None = None
        self._last_tooltip_rect: QRectF | None = None   # introspection / tests
        self._labels_mode = "none"        # 'none' | 'number' | 'text'
        self._last_label_rects: list[QRectF] = []       # introspection / tests
        self._press_pos: QPoint | None = None           # click vs pan detector
        self._dragged = False

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)      # we implement panning manually
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setMouseTracking(True)
        # Panning by scrollbar / keyboard moves the image under a stationary
        # cursor, so hover has to be re-evaluated on scroll too.
        self.horizontalScrollBar().valueChanged.connect(self._on_scroll)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    # ------------------------------------------------------------------ image
    def set_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._image_path = path
        self._pix_item.setPixmap(pm)
        self._scene.setSceneRect(0, 0, pm.width(), pm.height())
        self.fit_to_window()
        return True

    def clear_image(self):
        self._pix_item.setPixmap(QPixmap())
        self._image_path = None
        self._scene.setSceneRect(0, 0, 1, 1)
        self.set_boxes([])

    def image_path(self) -> str | None:
        """Path currently displayed (None when empty)."""
        return self._image_path

    def image_size(self) -> tuple[int, int]:
        pm = self._pix_item.pixmap()
        return (pm.width(), pm.height()) if not pm.isNull() else (0, 0)

    # ----------------------------------------------------------------- overlay
    def set_boxes(self, boxes) -> None:
        """Install the overlay for the current image (list of AnnotatedBox)."""
        painted: list[_PaintedBox] = []
        for b in (boxes or []):
            x, y, w, h = b.rect
            color = QColor(*b.rgb)
            painted.append(_PaintedBox(
                rect=QRectF(x, y, max(float(w), 0.0), max(float(h), 0.0)),
                color=color,
                dashed=bool(getattr(b, "group_size", 1) > 1
                            and getattr(b, "group", 1) > 1),
                text=getattr(b, "text", "") or "",
                norm=getattr(b, "norm", ()),
                group_size=int(getattr(b, "group_size", 1) or 1),
                order=int(getattr(b, "order", 0) or 0),
                source=b))
        self._boxes = painted
        self._hover = -1
        self.viewport().update()

    def set_overlay_enabled(self, enabled: bool) -> None:
        if enabled == self._overlay_enabled:
            return
        self._overlay_enabled = bool(enabled)
        if not enabled:
            self._hover = -1
        self.viewport().update()

    def overlay_enabled(self) -> bool:
        return self._overlay_enabled

    def set_labels_mode(self, mode: str) -> None:
        """Always-on chips: ``'none'``, ``'number'`` (chronological) or ``'text'``."""
        mode = mode if mode in LABEL_MODES else "none"
        if mode == self._labels_mode:
            return
        self._labels_mode = mode
        self.viewport().update()

    def labels_mode(self) -> str:
        return self._labels_mode

    def last_label_rects(self) -> list:
        """Viewport-space rects of the chips drawn during the last paint."""
        return list(self._last_label_rects)

    def box_count(self) -> int:
        return len(self._boxes) if self._overlay_enabled else 0

    def hovered_index(self) -> int:
        return self._hover

    def set_hover_index(self, index: int) -> None:
        """Force the hovered box (drives the tooltip without a real pointer)."""
        self._hover = index if 0 <= index < len(self._boxes) else -1
        self.viewport().update()

    def simulate_pointer(self, viewport_pos) -> int:
        """Put the virtual cursor at a *viewport* position and re-hover.

        Offscreen there is no real pointer, so ``underMouse()`` is always False;
        parking a position here lets the normal hit-test path run unchanged (used
        by ``gui_check_bbox_overlay.py`` and handy for click-to-select later).
        """
        self._hover_pos = viewport_pos
        self._refresh_hover(viewport_pos)
        return self._hover

    def last_tooltip_rect(self) -> QRectF | None:
        """Viewport-space rect of the tooltip drawn during the last paint."""
        return self._last_tooltip_rect

    def box_at(self, viewport_pos: QPoint | None):
        """The AnnotatedBox under a *viewport* point (None when nothing hits).

        Hit-testing happens in **scene** coordinates, so this is correct at any
        pan offset and zoom level - which is what the offscreen GUI check drives.
        """
        if not self._overlay_enabled or not self._boxes:
            return None
        pos = viewport_pos if viewport_pos is not None else self._hover_pos
        if pos is None:
            return None
        hit = self._hit_test(self.mapToScene(pos))
        return self._boxes[hit].source if hit >= 0 else None

    def _hit_test(self, scene_pos: QPointF) -> int:
        """Topmost box containing *scene_pos*; nested boxes resolve to the
        smallest, so a face inside a body box is still selectable."""
        if not self._overlay_enabled or not self._boxes:
            return -1
        tol = 2.0 / max(self._scale, 1e-6)
        best, best_area = -1, float("inf")
        for i, b in enumerate(self._boxes):
            if not b.rect.adjusted(-tol, -tol, tol, tol).contains(scene_pos):
                continue
            area = max(1.0, b.rect.width() * b.rect.height())
            if area < best_area:
                best, best_area = i, area
        return best

    def _refresh_hover(self, viewport_pos: QPoint | None = None) -> None:
        """Re-evaluate the hovered box after any pan / zoom / resize / data change."""
        pos = viewport_pos or self._hover_pos
        if pos is None and self.underMouse():
            from PySide6.QtGui import QCursor
            pos = self.mapFromGlobal(QCursor.pos())
        idx = self._hit_test(self.mapToScene(pos)) if pos is not None else -1
        if idx != self._hover:
            self._hover = idx
            self.viewport().update()
        elif idx >= 0:
            self.viewport().update()      # tooltip geometry follows the box

    def _on_scroll(self, *_args) -> None:
        self._refresh_hover()

    # ------------------------------------------------------------------ zoom
    def _apply_scale(self, factor: float, anchor_point: QPoint | None = None):
        new_scale = max(MIN_SCALE, min(MAX_SCALE, self._scale * factor))
        if abs(new_scale - self._scale) < 1e-6:
            return
        real_factor = new_scale / self._scale
        self.scale(real_factor, real_factor)
        self._scale = new_scale
        if anchor_point is not None:
            # Keep the scene point under the cursor fixed while zooming.
            self.centerOn(self.mapToScene(anchor_point))
        self._refresh_hover()

    def wheelEvent(self, event):  # noqa: N802 (Qt naming)
        steps = event.angleDelta().y()
        if steps == 0:
            return
        factor = ZOOM_STEP if steps > 0 else 1.0 / ZOOM_STEP
        self._apply_scale(factor, event.position().toPoint())
        event.accept()

    def fit_to_window(self):
        if self._pix_item.pixmap().isNull():
            return
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        t = self.transform()
        self._scale = max(MIN_SCALE, min(MAX_SCALE, t.m11()))
        self._refresh_hover()

    # ------------------------------------------------------------- mouse events
    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._dragged = False
            self._press_pos = event.position().toPoint()
            self._last_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        pos = event.position().toPoint()
        self._hover_pos = pos
        if self._dragging and self._last_pos is not None:
            delta = pos - self._last_pos
            if abs(delta.x()) + abs(delta.y()) > 2:
                self._dragged = True      # a real pan, not a click
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            self._last_pos = pos
        self._refresh_hover(pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_pos = None
            self.unsetCursor()
            pos = event.position().toPoint()
            # A pan may have slid a different box under the stationary cursor.
            self._refresh_hover(pos)
            if not self._dragged:
                idx = self._hit_test(self.mapToScene(pos))
                if idx >= 0:
                    self.box_clicked.emit(idx)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.fit_to_window()

    def leaveEvent(self, event):  # noqa: N802
        self._hover_pos = None
        if self._hover != -1:
            self._hover = -1
            self.viewport().update()
        super().leaveEvent(event)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        # Window resize, splitter drag or side-panel collapse: the tooltip's wrap
        # width and clamp depend on the viewport, so re-evaluate.
        if self._boxes and self._overlay_enabled:
            self.viewport().update()
            self._refresh_hover()

    # ---------------------------------------------------------------- painting
    def drawForeground(self, painter: QPainter, rect: QRectF):  # noqa: N802
        super().drawForeground(painter, rect)
        boxes = getattr(self, "_boxes", None)
        if not getattr(self, "_overlay_enabled", False) or not boxes:
            # Nothing is drawn at all, so no stale tooltip/label geometry may
            # survive an off-toggle.
            self._last_tooltip_rect = None
            self._last_label_rects = []
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        base_w = max(1.0, 2.0 / max(self._scale, 1e-6))
        hover = self._hover
        order = [i for i in range(len(boxes)) if i != hover]
        if 0 <= hover < len(boxes):
            order.append(hover)                 # hovered box on top
        for i in order:
            b = boxes[i]
            hovered = (i == hover)
            stroke = base_w * (1.6 if hovered else 1.0)
            fill = QColor(b.color)
            fill.setAlpha(FILL_ALPHA_HOVER if hovered else FILL_ALPHA)
            if b.rect.width() <= 0.5 or b.rect.height() <= 0.5:
                self._paint_crosshair(painter, b, stroke, hovered)
                continue
            painter.fillRect(b.rect, fill)
            # Thin white/black halo: the md5 colour is kept exactly as derived,
            # but survives both white paper and dark artwork.
            for col, width in ((QColor(0, 0, 0), stroke + 2.0),
                               (QColor(255, 255, 255), stroke + 1.0)):
                painter.setPen(QPen(col, width))
                painter.drawRect(b.rect)
            pen = QPen(b.color, stroke)
            if b.dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(b.rect)
        painter.restore()
        self._paint_labels(painter)
        self._paint_tooltip(painter)

    def _paint_crosshair(self, painter, b, stroke, hovered):
        """Zero-area boxes still deserve a visible mark."""
        size = 6.0 / max(self._scale, 1e-6)
        cx, cy = b.rect.center().x(), b.rect.center().y()
        for col, width in ((QColor(0, 0, 0), stroke + 2.0), (b.color, stroke)):
            painter.setPen(QPen(col, width))
            painter.drawLine(QPointF(cx - size, cy), QPointF(cx + size, cy))
            painter.drawLine(QPointF(cx, cy - size), QPointF(cx, cy + size))

    # ---------------------------------------------------------------- tooltip
    def _group_texts(self, b: _PaintedBox) -> list[str]:
        """Every text sharing the hovered rectangle, numbered when several.

        Identical coordinates with *different* text are kept as separate boxes on
        purpose (plan §5), so the tooltip has to show them all.
        """
        if b.group_size > 1 and b.norm:
            members = [x for x in self._boxes if x.norm == b.norm]
            members.sort(key=lambda x: (x.text or ""))
            n = len(members)
            return [f"{i + 1}/{n}  {m.text or '(no text)'}"
                    for i, m in enumerate(members)]
        return [b.text or "(no text)"]

    def _wrap(self, text: str, max_w: float, fm: QFontMetricsF) -> list[str]:
        """Greedy wrap that also copes with CJK; over-long tokens are elided.

        Tokens come from :func:`app.core.boxes.wrap_tokens`: Latin words stay
        whole, every CJK glyph is its own token (those scripts write without
        spaces, so ``str.split()`` would see a whole sentence as one word and
        elide it), and ``\\n`` forces a line break.
        """
        lines: list[str] = []
        cur = ""
        for token in wrap_tokens(text):
            if token == "\n":
                lines.append(cur)
                cur = ""
                continue
            trial = cur + token
            if not cur or fm.horizontalAdvance(trial.rstrip()) <= max_w:
                cur = trial
            else:
                lines.append(cur.rstrip())
                cur = token
        if cur:
            lines.append(cur.rstrip())
        if not lines:
            lines = [""]
        return [fm.elidedText(l, Qt.TextElideMode.ElideRight, int(max_w))
                for l in lines]

    # ------------------------------------------------- vertical writing (CJK)
    @staticmethod
    def _want_vertical_cjk(texts, screen: QRectF, vp: QRectF) -> bool:
        """Stack glyphs top-to-bottom for CJK text on a tall, narrow box.

        Vertical writing is how CJK reads down a column, and a column label, a
        stamp or a book margin is exactly where a horizontal panel ends up much
        wider than the thing it describes.  Only fires when the text really is
        CJK-dominant *and* the box is a column, so Latin captions and wide boxes
        keep the ordinary wrapped panel.
        """
        joined = "".join(texts)
        letters = [c for c in joined if not c.isspace()]
        if not letters:
            return False
        if sum(1 for c in letters if is_cjk(c)) / len(letters) < 0.5:
            return False
        w, h = screen.width(), screen.height()
        return h >= 2.5 * max(1.0, w) and w < vp.width() * 0.5

    def _paint_vertical(self, painter, b, texts, fm, screen, vp) -> None:
        """The 50 %-transparent panel with its glyphs stacked in columns.

        Glyphs stay upright (which is how ideographs are written vertically) and
        the panel itself is axis-aligned, so the "never clipped by the
        viewport" rule is the same rectangle check as for horizontal text.
        Long text flows into further columns, right to left, as CJK does.
        """
        line_h = fm.lineSpacing()
        glyphs: list[str] = []
        for i, t in enumerate(texts):
            if i:
                glyphs.append("\n")        # each numbered group member: new column
            glyphs.extend(c for c in t if not c.isspace())
        if not glyphs:
            return
        per_col = max(1, int((vp.height() - 2 * TOOLTIP_PAD) // line_h))
        columns: list[list[str]] = [[]]
        for g in glyphs:
            if g == "\n":
                columns.append([])
            elif len(columns[-1]) >= per_col:
                columns.append([g])
            else:
                columns[-1].append(g)
        columns = [c for c in columns if c]
        if not columns:
            return
        col_w = max([fm.horizontalAdvance(c) for c in glyphs if c != "\n"]
                    or [float(TOOLTIP_FONT_MIN)])
        panel_w = col_w * len(columns) + 2 * TOOLTIP_PAD
        panel_h = min(float(vp.height()),
                      max(len(c) for c in columns) * line_h + 2 * TOOLTIP_PAD)

        # Prefer the box's right side (the columns then read back towards it),
        # fall back to its left, then to a centred clamp.
        x = screen.right() + 6.0
        if x + panel_w > vp.right():
            x = screen.left() - 6.0 - panel_w
        if x < float(vp.left()):
            x = max(float(vp.left()), min(screen.center().x() - panel_w / 2.0,
                                          float(vp.right()) - panel_w))
        y = screen.center().y() - panel_h / 2.0
        y = max(float(vp.top()), min(y, float(vp.bottom()) - panel_h))
        panel = QRectF(x, y, panel_w, panel_h)
        self._last_tooltip_rect = panel

        bg = QColor(b.color)
        bg.setAlpha(TOOLTIP_ALPHA)
        painter.fillRect(panel, bg)
        painter.setPen(QPen(b.color, 1))
        painter.drawRect(panel)
        painter.setPen(_ink_for(b.color, TOOLTIP_ALPHA))
        for ci, col in enumerate(columns):
            cx = panel.right() - TOOLTIP_PAD - (ci + 1) * col_w
            ty = panel.top() + TOOLTIP_PAD
            for ch in col:
                painter.drawText(QRectF(cx, ty, col_w, line_h),
                                 Qt.AlignmentFlag.AlignHCenter
                                 | Qt.AlignmentFlag.AlignVCenter, ch)
                ty += line_h

    # ------------------------------------------------------------------ labels
    def _label_text(self, b: _PaintedBox) -> str:
        num = str(b.order + 1)
        if self._labels_mode == "number":
            return num
        text = " ".join((b.text or "").split())
        if len(text) > LABEL_MAX_CHARS:
            text = text[:LABEL_MAX_CHARS - 1].rstrip() + "…"
        return f"{num}·{text}" if text else num

    def _paint_labels(self, painter: QPainter) -> None:
        """Chronological / text chips at each box corner, in **viewport** space.

        Like the tooltip this is drawn with a *pixel* font after
        ``resetTransform()``, so a chip is just as legible at 0.02x as at 40x;
        off-screen boxes are skipped, and past ``LABEL_MAX_BOXES`` the chips stop
        being readable and are dropped rather than smothering the scan.
        """
        self._last_label_rects = []
        if self._labels_mode == "none" or not self._boxes:
            return
        vp = self.viewport().rect()
        if vp.width() < 24 or vp.height() < 24:
            return
        painter.save()
        painter.resetTransform()
        for b in self._boxes[:LABEL_MAX_BOXES]:
            screen = self.mapFromScene(b.rect).boundingRect()
            if not screen.intersects(vp):
                continue
            side = min(screen.width(), screen.height())
            font_px = int(max(LABEL_FONT_MIN,
                              min(LABEL_FONT_MAX,
                                  round(LABEL_FONT_RATIO * side))))
            font = QFont(self.font())
            font.setPixelSize(font_px)
            painter.setFont(font)
            fm = QFontMetricsF(font)
            text = self._label_text(b)
            w = fm.horizontalAdvance(text) + 2 * LABEL_PAD
            h = fm.lineSpacing() + 2 * LABEL_PAD
            x = min(max(float(vp.left()), screen.left()),
                    max(float(vp.left()), vp.right() - w))
            y = screen.top() - h - 2                  # just above the box
            if y < vp.top():
                y = min(screen.top() + 2.0,
                        max(float(vp.top()), vp.bottom() - h))
            chip = QRectF(x, y, w, h)
            bg = QColor(b.color)
            bg.setAlpha(LABEL_ALPHA)
            painter.fillRect(chip, bg)
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawRect(chip)
            painter.setPen(_ink_for(b.color, LABEL_ALPHA, backdrop=64))
            painter.drawText(chip.adjusted(LABEL_PAD, 0, -LABEL_PAD, 0),
                             Qt.AlignmentFlag.AlignLeft
                             | Qt.AlignmentFlag.AlignVCenter, text)
            self._last_label_rects.append(chip)
        painter.restore()

    def _paint_tooltip(self, painter: QPainter) -> None:
        self._last_tooltip_rect = None
        if not (0 <= self._hover < len(self._boxes)):
            return
        b = self._boxes[self._hover]
        vp = self.viewport().rect()
        if vp.width() < 24 or vp.height() < 24:
            return
        painter.save()
        painter.resetTransform()            # -> viewport (device) coordinates
        screen = self.mapFromScene(b.rect).boundingRect()

        side = min(screen.width(), screen.height())
        font_px = int(max(TOOLTIP_FONT_MIN,
                          min(TOOLTIP_FONT_MAX,
                              round(TOOLTIP_FONT_RATIO * side))))
        font = QFont(self.font())
        font.setPixelSize(font_px)
        painter.setFont(font)
        fm = QFontMetricsF(font)

        texts = self._group_texts(b)
        if self._want_vertical_cjk(texts, screen, vp):
            self._paint_vertical(painter, b, texts, fm, screen, vp)
            painter.restore()
            return

        max_w = max(120.0, min(TOOLTIP_MAX_W,
                               vp.width() * TOOLTIP_VIEWPORT_FRACTION))
        lines: list[str] = []
        for t in texts:
            lines.extend(self._wrap(t, max_w, fm))
        line_h = fm.lineSpacing()
        text_w = max((fm.horizontalAdvance(l) for l in lines), default=0.0)
        panel_w = min(max_w, text_w) + 2 * TOOLTIP_PAD
        panel_h = line_h * len(lines) + 2 * TOOLTIP_PAD

        # Anchor on the box, flip to whichever side has room, then clamp inside
        # the viewport so the panel is never cut off.
        x = screen.center().x() - panel_w / 2.0
        y = screen.bottom() + 6
        if y + panel_h > vp.bottom():
            y = screen.top() - panel_h - 6
        if y < vp.top():
            y = max(vp.top(), screen.center().y() - panel_h / 2.0)
        x = max(float(vp.left()), min(x, vp.right() - panel_w))
        y = max(float(vp.top()), min(y, vp.bottom() - panel_h))
        panel = QRectF(x, y, panel_w, panel_h)
        self._last_tooltip_rect = panel

        bg = QColor(b.color)
        bg.setAlpha(TOOLTIP_ALPHA)          # the 50 % transparent overlay box
        painter.fillRect(panel, bg)
        painter.setPen(QPen(b.color, 1))
        painter.drawRect(panel)

        # Text stays opaque; pick black or white from the *composited* panel
        # colour (50 % box colour over mid-grey) so it reads on any scan.
        painter.setPen(_ink_for(b.color, TOOLTIP_ALPHA))
        ty = panel.top() + TOOLTIP_PAD
        for l in lines:
            painter.drawText(QRectF(panel.left() + TOOLTIP_PAD, ty,
                                    panel_w - 2 * TOOLTIP_PAD, line_h),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             l)
            ty += line_h
        painter.restore()
