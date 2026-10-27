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

    def __init__(self, ctx=None, parent=None):
        # Accept the AppContext as first positional argument (consistent with
        # the other tabs); QWidget itself only receives the parent.
        if ctx is not None and not hasattr(ctx, "log"):
            # Called as CaptionSidePanel(parent) with a QWidget — treat as parent.
            ctx, parent = None, ctx
        super().__init__(parent)
        self.ctx = ctx
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
        except json.JSONDecodeError as exc:
            msg = (f"Caption looks like JSON but is invalid: {exc}. "
                   "Fix the syntax or save it as plain text.")
            self.status_label.setText("Save failed: invalid JSON")
            if self.ctx is not None:
                self.ctx.report_error(msg, context="Caption Editor / Save",
                                      dialog=False)
            else:
                self.ctx_log(msg)
        except OSError as exc:
            msg = (f"Could not write caption file next to "
                   f"'{os.path.basename(self.current_path)}': {exc}. "
                   "Check disk space and folder permissions.")
            self.status_label.setText("Save failed: I/O error")
            if self.ctx is not None:
                self.ctx.report_error(msg, context="Caption Editor / Save",
                                      dialog=False)
            else:
                self.ctx_log(msg)
        except Exception as exc:
            self.status_label.setText(f"Save failed: {exc}")
            if self.ctx is not None:
                self.ctx.report_error(exc, context="Caption Editor / Save",
                                      dialog=False)

    def ctx_log(self, msg: str):
        if self.ctx is not None:
            self.ctx.log.emit(f"[panel] {msg}")

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
                    for w in wm:
                        if isinstance(w, dict):
                            box = w.get("box_2d")
                            tag = f" @{box}" if box else ""
                            parts.append(f"  · {w.get('label', '')}{tag}")
                b = obj.get("borders")
                if b:
                    parts.append(f"borders: {', '.join(map(str, b))}")
                # Any extra harvested section (bounding_boxes, characters,
                # background, texts, ...) — rendered generically.
                skip = {"caption", "watermark", "borders"}
                for key, val in obj.items():
                    if key in skip or not isinstance(val, list) or not val:
                        continue
                    parts.append(f"{key}: {len(val)}")
                    for r in val:
                        if isinstance(r, dict):
                            last = str(r.get("id", "")).rsplit(".", 1)[-1]
                            txt = r.get("text")
                            desc = f"{last}: {txt}" if txt else last
                            tag = ""
                            if r.get("box_2d"):
                                tag = f" @{r['box_2d']}"
                            elif isinstance(r.get("boxes"), dict):
                                tag = " " + ", ".join(
                                    f"{n}@{bx}" for n, bx in r["boxes"].items())
                            parts.append(f"  · {desc}{tag}")
                return "\n".join(parts) or "-"
            except Exception:
                return text[:200]
        return text[:200] + ("…" if len(text) > 200 else "")


def json_text_flags():
    from PySide6.QtCore import Qt
    return Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
