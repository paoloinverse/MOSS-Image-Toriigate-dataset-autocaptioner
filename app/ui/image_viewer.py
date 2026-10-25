"""Zoomable / pannable image viewer built on QGraphicsView + QGraphicsPixmapItem.

* Mouse wheel  -> zoom around the cursor position.
* Click+drag   -> pan (scroll-area style hand drag).
* Double click -> fit-to-window reset.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView)

MIN_SCALE = 0.02
MAX_SCALE = 40.0
ZOOM_STEP = 1.25


class ZoomPanImageView(QGraphicsView):
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

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)      # we implement panning manually
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ image
    def set_image(self, path: str) -> bool:
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self._pix_item.setPixmap(pm)
        self._scene.setSceneRect(0, 0, pm.width(), pm.height())
        self.fit_to_window()
        return True

    def clear_image(self):
        self._pix_item.setPixmap(QPixmap())
        self._scene.setSceneRect(0, 0, 1, 1)

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

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._dragging and self._last_pos is not None:
            delta = event.position().toPoint() - self._last_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            self._last_pos = event.position().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_pos = None
            self.unsetCursor()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.fit_to_window()
