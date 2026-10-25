"""Global Caption Editor side panel.

Always visible; bound to whichever image is focused in the Dataset Explorer.
Shows file info (name, resolution, existing metadata), loads the sidecar
caption (.txt or .json), and lets the user overwrite it on disk manually.
"""
from __future__ import annotations

import json
import os

from PySide6.QtCore import Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QTextEdit, QVBoxLayout, QWidget)

from app.core.presets import load_existing_caption


class CaptionSidePanel(QWidget):
    """Right-hand global panel bound to the focused image (see spec §3)."""

    caption_saved = Signal(str)          # path of the written caption file

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_path: str | None = None
        self.setMinimumWidth(320)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        title = QLabel("Caption Editor")
        f = title.font()
        f.setBold(True)
        f.setPointSize(12)
        title.setFont(f)
        root.addWidget(title)

        self.info_label = QLabel("No image selected.")
        self.info_label.setWordWrap(True)
        root.addWidget(self.info_label)

        meta_box = QGroupBox("Existing metadata")
        mb_layout = QVBoxLayout(meta_box)
        self.meta_label = QLabel("-")
        self.meta_label.setWordWrap(True)
        self.meta_label.setTextInteractionFlags(json_text_flags())
        mb_layout.addWidget(self.meta_label)
        root.addWidget(meta_box)

        self.editor = QTextEdit()
        self.editor.setPlaceholderText("Caption (.txt) or MOSS JSON (.json) for the focused image…")
        root.addWidget(self.editor, stretch=1)

        btns = QHBoxLayout()
        self.save_btn = QPushButton("Save Caption")
        self.reload_btn = QPushButton("Reload")
        btns.addWidget(self.save_btn)
        btns.addWidget(self.reload_btn)
        root.addLayout(btns)

        self.status_label = QLabel("")
        root.addWidget(self.status_label)

        self.save_btn.clicked.connect(self.save_now)
        self.reload_btn.clicked.connect(self.reload)

    # ------------------------------------------------------------------ API
    def set_image(self, path: str | None):
        self.current_path = path
        if not path:
            self.info_label.setText("No image selected.")
            self.meta_label.setText("-")
            self.editor.clear()
            self.editor.setReadOnly(True)
            self.save_btn.setEnabled(False)
            return
        self.editor.setReadOnly(False)
        self.save_btn.setEnabled(True)
        name = os.path.basename(path)
        res = "?"
        img = QImage(path)
        if not img.isNull():
            res = f"{img.width()}x{img.height()}"
        try:
            size_kb = os.path.getsize(path) / 1024
        except OSError:
            size_kb = 0
        self.info_label.setText(f"{name}\nResolution: {res} | Size: {size_kb:.0f} KiB\nPath: {path}")
        kind, text = load_existing_caption(path)
        self.editor.setPlainText(text)
        self.meta_label.setText(self._describe_meta(kind, text))
        self.status_label.setText(f"Loaded existing .{kind}" if kind else "No caption file yet.")

    def reload(self):
        if self.current_path:
            self.set_image(self.current_path)

    def save_now(self):
        if not self.current_path:
            return
        text = self.editor.toPlainText().strip()
        target = os.path.splitext(self.current_path)[0]
        try:
            if text.startswith("{"):
                obj = json.loads(text)          # validate before writing
                with open(target + ".json", "w", encoding="utf-8") as fh:
                    json.dump(obj, fh, ensure_ascii=False, indent=2)
                written = target + ".json"
            else:
                with open(target + ".txt", "w", encoding="utf-8") as fh:
                    fh.write(text + "\n")
                written = target + ".txt"
            self.status_label.setText(f"Saved -> {os.path.basename(written)}")
            self.caption_saved.emit(written)
        except Exception as exc:
            self.status_label.setText(f"Save failed: {exc}")

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _describe_meta(kind: str, text: str) -> str:
        if not kind or not text.strip():
            return "-"
        if kind == "json":
            try:
                obj = json.loads(text)
                cap = obj.get("caption", {})
                parts = []
                for lvl in ("short", "medium", "long"):
                    v = cap.get(lvl)
                    if v:
                        parts.append(f"[{lvl}] {v[:120]}{'…' if len(v) > 120 else ''}")
                wm = obj.get("watermark") or []
                if wm:
                    parts.append(f"watermarks: {len(wm)}")
                b = obj.get("borders")
                if b:
                    parts.append(f"borders: {', '.join(map(str, b))}")
                return "\n".join(parts) or "-"
            except Exception:
                return text[:200]
        return text[:200] + ("…" if len(text) > 200 else "")


def json_text_flags():
    from PySide6.QtCore import Qt
    return Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
