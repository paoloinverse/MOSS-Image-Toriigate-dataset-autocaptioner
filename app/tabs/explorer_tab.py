"""Tab 4 - Dataset Explorer.

* Directory picker + Recursive checkbox.
* QFileSystemModel-backed list filtered to image files (folder tree visible).
* Zoomable/pannable QGraphicsView preview of the focused image; focusing an
  image also drives the global Caption Editor side panel via ctx.image_focused.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QSortFilterProxyModel, Qt, QModelIndex
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QTreeView, QVBoxLayout, QWidget)
from PySide6.QtWidgets import QFileSystemModel

from app.context import AppContext
from app.ui.image_viewer import ZoomPanImageView

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")


class ImageFilterProxy(QSortFilterProxyModel):
    """Show directories plus files with image extensions."""

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex):  # noqa: N802
        model = self.sourceModel()
        idx = model.index(source_row, 0, source_parent)
        if not idx.isValid():
            return False
        file_info = model.fileInfo(idx)
        if file_info.isDir():
            return True
        return file_info.suffix().lower() in {e[1:] for e in IMAGE_EXTS}


class DatasetExplorerTab(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx

        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.pick_btn = QPushButton("Select Dataset Directory…")
        bar.addWidget(self.pick_btn)
        self.recursive_check = QCheckBox("Recursive")
        self.recursive_check.setChecked(ctx.recursive)
        bar.addWidget(self.recursive_check)
        self.dir_label = QLabel("(no directory)")
        self.dir_label.setStyleSheet("color: gray;")
        bar.addWidget(self.dir_label, stretch=1)
        self.count_label = QLabel("")
        bar.addWidget(self.count_label)
        root.addLayout(bar)

        body = QHBoxLayout()

        # ---- file tree -----------------------------------------------------
        left = QVBoxLayout()
        self.fs_model = QFileSystemModel()
        self.fs_model.setRootPath("")
        self.proxy = ImageFilterProxy()
        self.proxy.setSourceModel(self.fs_model)
        self.proxy.setSortCaseSensitivity(Qt.CaseInsensitive)
        self.tree = QTreeView()
        self.tree.setModel(self.proxy)
        self.tree.setColumnWidth(0, 320)
        self.tree.setHeaderHidden(True)
        self.tree.setSortingEnabled(True)
        left.addWidget(QLabel("Files:"))
        left.addWidget(self.tree, stretch=1)
        body.addLayout(left, stretch=1)

        # ---- viewer ----------------------------------------------------------
        right = QVBoxLayout()
        right.addWidget(QLabel("Preview (wheel = zoom, drag = pan, dbl-click = fit):"))
        self.viewer = ZoomPanImageView()
        right.addWidget(self.viewer, stretch=1)
        body.addLayout(right, stretch=2)

        root.addLayout(body, stretch=1)

        # ---- wiring ------------------------------------------------------------
        self.pick_btn.clicked.connect(self.pick_directory)
        self.recursive_check.toggled.connect(self.on_recursive_toggled)
        self.tree.clicked.connect(self.on_clicked)
        self.tree.doubleClicked.connect(self.on_clicked)

    # ------------------------------------------------------------------ slots
    def pick_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Dataset root directory",
                                             self.ctx.dataset_root or os.path.expanduser("~"))
        if d:
            self.set_directory(d)

    def set_directory(self, path: str):
        self.ctx.dataset_root = path
        try:
            idx = self.fs_model.setRootPath(path)
            root_idx = self.proxy.mapFromSource(idx)
            self.tree.setRootIndex(root_idx)
            self.tree.expand(root_idx)
            self.dir_label.setText(path)
        except Exception as exc:
            self.ctx.report_error(exc, context=f"opening dataset directory {path}",
                                  dialog=True)
            return
        n = self._count_images(path)
        self.count_label.setText(f"{n} image(s)")
        if n == 0:
            self.ctx.log.emit(
                f"[explorer] WARNING: '{path}' contains no images "
                f"({'recursive' if self.recursive_check.isChecked() else 'top level only'}). "
                "Tick 'Recursive' or pick a different folder.")

    def _count_images(self, path: str) -> int:
        rec = self.recursive_check.isChecked()
        total = 0
        try:
            walker = os.walk(path) if rec else [(path, [], os.listdir(path))]
            for dirpath, _dirs, files in walker:
                for fn in files:
                    if fn.lower().endswith(IMAGE_EXTS):
                        total += 1
        except OSError as exc:
            self.ctx.report_error(
                f"[explorer] cannot read dataset directory: {type(exc).__name__}: "
                f"{exc}\n    path   : {path}\n"
                "    fix    : check the folder still exists and is readable.",
                context="counting images", dialog=True)
        return total

    def on_recursive_toggled(self, checked: bool):
        self.ctx.recursive = checked
        if self.ctx.dataset_root:
            self.count_label.setText(
                f"{self._count_images(self.ctx.dataset_root)} image(s)")

    def on_clicked(self, proxy_index: QModelIndex):
        src = self.proxy.mapToSource(proxy_index)
        info = self.fs_model.fileInfo(src)
        if not info.isFile():
            return
        if not info.fileName().lower().endswith(IMAGE_EXTS):
            return
        path = info.absoluteFilePath()
        if self.viewer.set_image(path):
            self.ctx.focused_image = path
            self.ctx.image_focused.emit(path)
        else:
            # Verbose, non-fatal report: huge/tiff/corrupt files are common
            # in datasets; the app must degrade gracefully, never crash.
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
            except OSError:
                size_mb = -1
            self.ctx.log.emit(
                f"[explorer] preview unavailable for '{os.path.basename(path)}' "
                f"(Qt image decoder rejected it; file ~{size_mb:.1f} MiB).\n"
                "    note   : very large images and some .tiff variants are not "
                "decodable by Qt - captioning itself is unaffected.\n"
                "    fix    : convert to PNG/JPEG (e.g. 'convert in.png out.png') "
                "if a preview is required.")
