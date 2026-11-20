"""Tab 4 - Dataset Explorer.

* Directory picker + Recursive checkbox.
* QFileSystemModel-backed list filtered to image files (folder tree visible).
* Zoomable/pannable QGraphicsView preview of the focused image; focusing an
  image also drives the global Caption Editor side panel via ctx.image_focused.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QSortFilterProxyModel, Qt, QModelIndex
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QPushButton, QTreeView, QVBoxLayout,
                               QWidget)
from PySide6.QtWidgets import QFileSystemModel

from app.context import AppContext
from app.core.boxes import (DEFAULT_LAYOUT, DEFAULT_SCALE, LAYOUTS, SCALES,
                            preset_supports_boxes, resolve_and_extract)
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
        right.addWidget(QLabel("Preview (wheel = zoom, drag = pan, dbl-click = fit,"
                               " hover a box for its text):"))
        self.viewer = ZoomPanImageView()
        right.addWidget(self.viewer, stretch=1)
        body.addLayout(right, stretch=2)

        root.addLayout(body, stretch=1)

        # ---- bounding-box overlay controls --------------------------------------
        # The coordinate standard is the *user's* choice (it cannot be deduced
        # from a foreign file), so both dropdowns are exposed here and persisted
        # on the active preset - see app/core/boxes.py and plan §4.2.1.
        obar = QHBoxLayout()
        self.overlay_check = QCheckBox("Show bounding boxes")
        self.overlay_check.setToolTip(
            "Draw the JSON bounding boxes over the preview.\n"
            "Available for the MOSS-Image, OCR Transcript and Custom presets; "
            "disabled for tag/Markdown presets, which emit no geometry.")
        obar.addWidget(self.overlay_check)
        obar.addWidget(QLabel("Layout:"))
        self.layout_combo = QComboBox()
        for key, (label, _fn, origin) in LAYOUTS.items():
            self.layout_combo.addItem(label, key)
            self.layout_combo.setItemData(self.layout_combo.count() - 1, origin,
                                          Qt.ToolTipRole)
        self.layout_combo.setToolTip(
            "How to read a flat 4-number box. Corner-key objects and "
            "[[x1,y1],[x2,y2]] point pairs are recognised by shape and ignore "
            "this choice.")
        obar.addWidget(self.layout_combo)
        obar.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        for key, (label, _div) in SCALES.items():
            self.scale_combo.addItem(label, key)
        self.scale_combo.setToolTip(
            "Unit the coordinates are expressed in. 'Auto' guesses from the "
            "largest value - a convenience, not a guarantee.")
        obar.addWidget(self.scale_combo)
        obar.addWidget(QLabel("Colour:"))
        self.colour_combo = QComboBox()
        self.colour_combo.addItem("Bright (vivid)", "vivid")
        self.colour_combo.addItem("Raw md5 bytes", "raw")
        self.colour_combo.setToolTip(
            "How the md5 colour is turned into an RGB triple.\n"
            "Bright keeps the md5 hue but forces a light band, so no box is "
            "nearly black on a dark background; Raw is the literal first three "
            "digest bytes.")
        obar.addWidget(self.colour_combo)
        obar.addWidget(QLabel("Labels:"))
        self.labels_combo = QComboBox()
        self.labels_combo.addItem("None", "none")
        self.labels_combo.addItem("Numbers", "number")
        self.labels_combo.addItem("Text", "text")
        self.labels_combo.setToolTip(
            "Always-on chips at each box corner (hover still gives the full "
            "text).\nNumbers = chronological order as the JSON lists them; "
            "Text = number plus a short excerpt.")
        obar.addWidget(self.labels_combo)
        self.overlay_reset_btn = QPushButton("Reset")
        self.overlay_reset_btn.setToolTip(
            "Restore the app standard: ymin,xmin,ymax,xmax · 0-1000 normalized "
            "(the MOSS-Image / OCR Transcript prompt spec).")
        obar.addWidget(self.overlay_reset_btn)
        obar.addStretch(1)
        root.addLayout(obar)

        # Second overlay row: the power-user override, kept apart from the
        # per-dataset standard controls because it changes *what may be drawn*.
        o2 = QHBoxLayout()
        self.any_preset_check = QCheckBox(
            "Allow overlay for any preset (power user)")
        self.any_preset_check.setToolTip(
            "By default the overlay is offered for the MOSS-Image, OCR "
            "Transcript and Custom presets only.\nTick this to draw whatever "
            "geometry the JSON happens to carry even under the Booru / Markdown "
            "presets, whose prompts ask for tags or prose, not boxes.")
        o2.addWidget(self.any_preset_check)
        o2.addStretch(1)
        root.addLayout(o2)

        srow = QHBoxLayout()
        self.overlay_status = QLabel("")
        self.overlay_status.setStyleSheet("color: gray;")
        srow.addWidget(self.overlay_status, stretch=1)
        self.overlay_warning = QPushButton("")
        self.overlay_warning.setFlat(True)
        self.overlay_warning.setCursor(Qt.CursorShape.PointingHandCursor)
        self.overlay_warning.setStyleSheet(
            "QPushButton { color: #c58c00; text-align: left; border: none; }"
            "QPushButton:hover { text-decoration: underline; }")
        self.overlay_warning.setVisible(False)
        srow.addWidget(self.overlay_warning, stretch=2)
        root.addLayout(srow)
        self._warning_dismissed = False
        # Dismissals survive re-focusing the same image under the same standard
        # (otherwise an amber reminder you already waved away returns on every
        # click), but a new image / layout / scale shows it again.
        self._dismissed_warnings: set[tuple] = set()
        self._current_warning_key: tuple | None = None

        # ---- wiring ------------------------------------------------------------
        self.pick_btn.clicked.connect(self.pick_directory)
        self.recursive_check.toggled.connect(self.on_recursive_toggled)
        self.tree.clicked.connect(self.on_clicked)
        self.tree.doubleClicked.connect(self.on_clicked)
        self.overlay_check.toggled.connect(self._on_overlay_toggled)
        self.layout_combo.currentIndexChanged.connect(self._on_standard_changed)
        self.scale_combo.currentIndexChanged.connect(self._on_standard_changed)
        self.overlay_reset_btn.clicked.connect(self._reset_standard)
        self.colour_combo.currentIndexChanged.connect(self._on_presentation_changed)
        self.labels_combo.currentIndexChanged.connect(self._on_presentation_changed)
        self.any_preset_check.toggled.connect(self._on_any_preset_toggled)
        self.overlay_warning.clicked.connect(self._dismiss_warning)
        self.viewer.box_clicked.connect(self._on_box_clicked)
        ctx.preset_changed.connect(self._on_preset_changed)
        ctx.image_focused.connect(self._on_image_focused)
        self._suppress_focus_refresh = False
        self._sync_controls_from_preset()

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
            self._warning_dismissed = False
            self._update_overlay(path)
            # The overlay is already drawn for this click; the signal below is
            # for the side panel, so don't let it bounce back into a second
            # (redundant) resolve+parse of the same file.
            self._suppress_focus_refresh = True
            try:
                self.ctx.image_focused.emit(path)
            finally:
                self._suppress_focus_refresh = False
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
            self.viewer.set_boxes([])
            self._set_overlay_status("overlay skipped - no Qt-decodable preview")

    # ------------------------------------------------------- bounding-box overlay
    def _on_image_focused(self, path: str):
        """Re-draw when something *else* refreshed the focused image.

        A finished batch re-emits ``image_focused`` for the current image, so the
        overlay picks up the run it just produced without a second click.
        """
        if self._suppress_focus_refresh:
            return
        if path and self.viewer.image_path() == path:
            self._update_overlay(path)

    def _sync_controls_from_preset(self):
        """Load the coordinate standard stored on the active preset."""
        preset = self.ctx.active_preset
        st = self.ctx.settings
        layout = getattr(preset, "overlay_box_layout", DEFAULT_LAYOUT) or DEFAULT_LAYOUT
        scale = getattr(preset, "overlay_coord_scale", DEFAULT_SCALE) or DEFAULT_SCALE
        for combo, value in ((self.layout_combo, layout), (self.scale_combo, scale)):
            idx = combo.findData(value)
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)
        self.overlay_check.blockSignals(True)
        self.overlay_check.setChecked(bool(getattr(st, "overlay_enabled", True)))
        self.overlay_check.blockSignals(False)
        self.viewer.set_overlay_enabled(bool(getattr(st, "overlay_enabled", True)))
        # Presentation travels on the *preset* too, so "review this dataset with
        # numbered chips" exports and imports whole.  app_settings is only the
        # restart mirror; precedence matches the coordinate standard:
        # hardwired default -> settings mirror -> imported config.
        for combo, key, fallback in ((self.colour_combo, "overlay_colour_mode",
                                      "vivid"),
                                     (self.labels_combo, "overlay_labels",
                                      "none")):
            value = (getattr(preset, key, None) or getattr(st, key, None)
                     or fallback)
            idx = combo.findData(value)
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)
        self.viewer.set_labels_mode(self.labels_combo.currentData() or "none")
        any_p = getattr(preset, "overlay_any_preset", None)
        any_p = bool(getattr(st, "overlay_any_preset", False)) \
            if any_p is None else bool(any_p)
        st.overlay_any_preset = any_p          # keep the mirror honest
        self.any_preset_check.blockSignals(True)
        self.any_preset_check.setChecked(any_p)
        self.any_preset_check.blockSignals(False)
        self._warning_dismissed = False

    def _write_standard(self):
        """Persist the dropdowns onto the preset (+ settings mirror) - plan §4.2.1."""
        preset = self.ctx.active_preset
        layout = self.layout_combo.currentData() or DEFAULT_LAYOUT
        scale = self.scale_combo.currentData() or DEFAULT_SCALE
        preset.overlay_box_layout = layout
        preset.overlay_coord_scale = scale
        self.ctx.settings.overlay_box_layout = layout
        self.ctx.settings.overlay_coord_scale = scale
        self.ctx.save_settings()

    def _on_overlay_toggled(self, checked: bool):
        self.ctx.settings.overlay_enabled = bool(checked)
        self.ctx.save_settings()
        self.viewer.set_overlay_enabled(bool(checked))
        self._update_overlay()

    def _on_standard_changed(self, *_args):
        if self.sender() is self.overlay_check:
            return
        self._write_standard()
        self._warning_dismissed = False
        self._update_overlay()

    def _reset_standard(self):
        for combo, value in ((self.layout_combo, DEFAULT_LAYOUT),
                             (self.scale_combo, DEFAULT_SCALE)):
            idx = combo.findData(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)      # fires _on_standard_changed
        self.ctx.log.emit(
            "[overlay] coordinate standard reset to "
            "ymin,xmin,ymax,xmax · 0-1000 normalized.")

    def _on_preset_changed(self):
        # Gating and the stored standard both follow the active preset.
        self._sync_controls_from_preset()
        self._update_overlay()

    def _dismiss_warning(self):
        self._warning_dismissed = True
        if self._current_warning_key is not None:
            self._dismissed_warnings.add(self._current_warning_key)
        self.overlay_warning.setVisible(False)

    def _on_presentation_changed(self, *_args):
        """Colour mode / label chips: stored on the preset (+ settings mirror)."""
        preset = self.ctx.active_preset
        colour = self.colour_combo.currentData() or "vivid"
        labels = self.labels_combo.currentData() or "none"
        self.ctx.settings.overlay_colour_mode = colour
        self.ctx.settings.overlay_labels = labels
        preset.overlay_colour_mode = colour
        preset.overlay_labels = labels
        self.ctx.save_settings()
        self.viewer.set_labels_mode(labels)
        self._update_overlay()

    def _on_any_preset_toggled(self, _checked: bool):
        value = bool(self.any_preset_check.isChecked())
        self.ctx.active_preset.overlay_any_preset = value
        self.ctx.settings.overlay_any_preset = value
        self.ctx.save_settings()
        self._update_overlay()

    def _on_box_clicked(self, index: int):
        """Clicking a box asks the Caption Editor to select that very text."""
        src = self.viewer.box_at(None)
        if src is None or not self.ctx.focused_image:
            return
        self.ctx.box_clicked.emit(self.ctx.focused_image,
                                  getattr(src, "text", ""))

    def on_caption_saved(self, path: str):
        """A Caption Editor save is newer than the cached run -> re-resolve."""
        if not self.ctx.focused_image:
            return
        base = os.path.splitext(os.path.basename(self.ctx.focused_image))[0]
        if base and base in os.path.basename(path):
            self._update_overlay()

    def _set_overlay_status(self, text: str):
        self.overlay_status.setText(text)

    def _show_overlay_warning(self, text: str):
        key = (self.ctx.focused_image or "",
               self.layout_combo.currentData(),
               self.scale_combo.currentData(), text or "")
        self._current_warning_key = key
        if not text or key in self._dismissed_warnings:
            self.overlay_warning.setVisible(False)
            self.overlay_warning.setText("")
            return
        self.overlay_warning.setText("⚠ " + text)
        self.overlay_warning.setToolTip(
            "Click to dismiss for this session.\n\n"
            "Relative coordinates that are plausible but wrong cannot be "
            "detected automatically: if the boxes look clearly wrong, the "
            "source JSON most likely follows a different specification.")
        self.overlay_warning.setVisible(True)

    def _update_overlay(self, path: str | None = None):
        """Resolve + draw the boxes for *path* (default: the focused image)."""
        path = path or self.ctx.focused_image
        preset = self.ctx.active_preset
        allowed = preset_supports_boxes(
            preset, bool(getattr(self.ctx.settings, "overlay_any_preset", False)))
        for widget in (self.overlay_check, self.layout_combo, self.scale_combo,
                       self.overlay_reset_btn):
            widget.setEnabled(allowed)
        if not allowed:
            self.viewer.set_boxes([])
            self._set_overlay_status(
                f"Overlay off — active preset '{preset.display_name}' "
                "emits no boxes (tick 'Allow overlay for any preset' to try)")
            self._show_overlay_warning("")
            return
        if not path or self.viewer.image_size()[0] <= 0:
            self.viewer.set_boxes([])
            self._set_overlay_status("")
            self._show_overlay_warning("")
            return

        # resolve_and_extract uses the age-preferred source (latest run while
        # fresh, newer sidecar after that) but, if that one yields no boxes,
        # falls through to the other known source - so a freshly saved clean
        # file still shows its boxes instead of a broken in-session dump.
        img_w, img_h = self.viewer.image_size()
        src, res = resolve_and_extract(
            path, self.ctx.settings,
            getattr(self.ctx, "last_results", None),
            cache_time=getattr(self.ctx, "last_results_ts", {}).get(path),
            img_w=img_w, img_h=img_h,
            layout=self.layout_combo.currentData() or DEFAULT_LAYOUT,
            scale=self.scale_combo.currentData() or DEFAULT_SCALE,
            colour_mode=getattr(self.ctx.settings, "overlay_colour_mode",
                                "vivid"))
        if src is None:
            self.viewer.set_boxes([])
            self._set_overlay_status("no JSON sidecar for this image")
            self._show_overlay_warning("")
            return

        self.viewer.set_boxes(res.boxes)
        for line in res.warnings:
            self.ctx.log.emit(f"[overlay] {os.path.basename(path)}: {line}")
        if res.found or res.warnings:
            # Provenance: which section of which file each box came from.
            self.ctx.log.emit(
                f"[overlay] {os.path.basename(path)} \u2190 {src.name}: "
                f"{res.provenance()} (layout {res.layout_used}, "
                f"scale {res.scale_used})")

        if not res.parsed:
            self._set_overlay_status(
                f"{src.name}: JSON could not be parsed - see log")
            self._show_overlay_warning(res.warnings[0] if res.warnings else "")
            return
        n = len(res.boxes)
        if n == 0:
            self._set_overlay_status(f"{src.name}: JSON has no bounding boxes")
            self._show_overlay_warning("")
            return

        merged = f" ({res.merged} duplicate{'s' if res.merged != 1 else ''} merged)" \
            if res.merged else ""
        auto = (f" · auto\u2192{res.scale_used}"
                if self.scale_combo.currentData() == "auto"
                and res.scale_used != "auto" else "")
        status = f"Boxes: {n}{merged} · {src.name}{auto}"
        if src.kind == "session":
            status += " · latest run"
        elif src.alternatives:
            status += f" · other sidecars: {', '.join(src.alternatives[:3])}"
        self._set_overlay_status(status)

        # Warning: real extraction problems first, otherwise the honest reminder
        # that a non-default (or Custom-preset) standard was assumed.
        if res.warnings:
            self._show_overlay_warning("; ".join(res.warnings[:2]))
        elif getattr(preset, "key", "") == "custom" or \
                (self.layout_combo.currentData(),
                 self.scale_combo.currentData()) != (DEFAULT_LAYOUT, DEFAULT_SCALE):
            layout_label = LAYOUTS[self.layout_combo.currentData()][0]
            scale_label = SCALES[self.scale_combo.currentData()][0]
            self._show_overlay_warning(
                f"boxes drawn as {layout_label} · {scale_label} - if they look "
                "clearly wrong, the source JSON may follow a different "
                "specification (verify it externally or pick another "
                "layout/scale).")
        else:
            self._show_overlay_warning("")
