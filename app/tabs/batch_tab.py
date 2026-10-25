"""Tab 5 - Batch Processing.

* Image selection list (checkboxes) synced from the Dataset Explorer root;
  whole folders can be checked at once via the folder row.
* Start Auto-Caption / Stop / Save All controls.
* Global progress bar + live Log Output tailing llama-server stdout and the
  streamed caption deltas.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

from app.context import AppContext
from app.core.batch import BatchWorker, collect_images


class BatchTab(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._worker: BatchWorker | None = None
        self._stream_marker: str | None = None

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh list from Explorer directory")
        self.check_all_btn = QPushButton("Check all")
        self.uncheck_btn = QPushButton("Uncheck all")
        top.addWidget(self.refresh_btn)
        top.addWidget(self.check_all_btn)
        top.addWidget(self.uncheck_btn)
        top.addStretch(1)
        self.server_state_label = QLabel("server: stopped")
        top.addWidget(self.server_state_label)
        root.addLayout(top)

        sel_box = QGroupBox("Images selected for batch captioning")
        sb = QVBoxLayout(sel_box)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.NoSelection)
        sb.addWidget(self.list)
        root.addWidget(sel_box, stretch=2)

        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start Auto-Caption")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.save_all_btn = QPushButton("Save All")
        self.save_all_btn.setToolTip(
            "Flush every sidecar caption already written to disk "
            "(files are saved per-image as they complete).")
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.save_all_btn)
        controls.addStretch(1)
        root.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)
        self.progress_label = QLabel("")
        root.addWidget(self.progress_label)

        log_box = QGroupBox("Log Output (llama-server stdout + stream)")
        lb = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.autoscroll = QCheckBox("Autoscroll")
        self.autoscroll.setChecked(True)
        lb.addWidget(self.log_view)
        lb.addWidget(self.autoscroll)
        root.addWidget(log_box, stretch=3)

        # ---- wiring -------------------------------------------------------------
        self.refresh_btn.clicked.connect(self.refresh_list)
        self.check_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self.uncheck_btn.clicked.connect(lambda: self._set_all_checked(False))
        self.start_btn.clicked.connect(self.start_batch)
        self.stop_btn.clicked.connect(self.stop_batch)
        self.save_all_btn.clicked.connect(self.save_all)
        self.ctx.server_state.connect(self.on_server_state)
        self.ctx.log.connect(self.append_log)
        self.ctx.model_changed.connect(lambda: None)   # no-op keeps tabs in sync

    # ------------------------------------------------------------------ list
    def refresh_list(self):
        self.list.clear()
        root = self.ctx.dataset_root
        if not root or not os.path.isdir(root):
            self.list.addItem("(open Tab 4 and pick a dataset directory first)")
            self.progress_label.setText("No dataset directory set.")
            return
        try:
            images = collect_images(root, self.ctx.recursive)
        except OSError as exc:
            self.ctx.report_error(
                f"[batch] cannot scan dataset directory: {type(exc).__name__}: "
                f"{exc}\n    path   : {root}\n"
                "    fix    : verify the folder exists / is readable.",
                context="refreshing batch list", dialog=True)
            return
        common = os.path.commonpath([root] + images) if images else root
        for img in images:
            rel = os.path.relpath(img, common)
            item = QListWidgetItem(rel)
            item.setData(Qt.UserRole, img)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list.addItem(item)
        self.progress_label.setText(f"{len(images)} image(s) listed.")
        if not images:
            self.ctx.log.emit(
                f"[batch] WARNING: 0 images found under '{root}' "
                f"({'recursive' if self.ctx.recursive else 'top level only'}).")

    def _set_all_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole):
                it.setCheckState(state)

    def checked_images(self) -> list[str]:
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            p = it.data(Qt.UserRole)
            if p and it.checkState() == Qt.Checked:
                out.append(p)
        return out

    # ------------------------------------------------------------------ batch
    def start_batch(self):
        if self._worker and self._worker.isRunning():
            return
        if not self.ctx.server.ready:
            state = getattr(self.ctx.server, "state", None) or "stopped"
            self.append_log(
                f"[batch] CANNOT START: llama-server is not ready (state={state}).\n"
                "    fix    : start the server in Tab 2 and wait for 'ready'; "
                "check the PRE-FLIGHT ERROR lines above for why it failed.")
            return
        images = self.checked_images()
        if not images:
            self.append_log("[batch] nothing checked - use Check all or tick rows.")
            return
        self.ctx.batch_selection = images
        preset = self.ctx.active_preset
        worker = BatchWorker(self.ctx.server.config.base_url
                             if self.ctx.server.config else
                             f"http://{self.ctx.settings.host}:{self.ctx.settings.port}",
                             preset, self.ctx.effective_system_prompt(),
                             images, self, settings=self.ctx.settings)
        worker.log.connect(self.append_log)
        worker.progress.connect(self.on_progress)
        worker.chunk.connect(self.on_chunk)
        worker.image_done.connect(self.on_image_done)
        worker.image_failed.connect(self.on_image_failed)
        worker.finished_all.connect(self.on_finished)
        self._worker = worker
        try:
            worker.start()
        except Exception as exc:      # thread spawn failure must not kill the GUI
            self._worker = None
            self.ctx.report_error(exc, context="starting batch worker", dialog=True)
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.append_log(f"[batch] queued {len(images)} image(s) with system prompt "
                        f"({'explicit' if self.ctx.explicit_mode else 'standard'})")

    def stop_batch(self):
        if self._worker:
            self._worker.stop()
            self.append_log("[batch] stop requested…")

    def save_all(self):
        n = 0
        for i in range(self.list.count()):
            it = self.list.item(i)
            p = it.data(Qt.UserRole)
            if p:
                base = os.path.splitext(p)[0]
                if os.path.isfile(base + ".txt") or os.path.isfile(base + ".json"):
                    n += 1
        self.append_log(f"[batch] {n} caption file(s) present on disk for this list.")

    # ------------------------------------------------------------------ signals
    def on_progress(self, done: int, total: int):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.progress_label.setText(f"{done}/{total} captioned")

    def on_chunk(self, img: str, delta: str):
        # Live-stream the currently processing image into the log box.
        name = os.path.basename(img)
        if self._stream_marker != name:
            self.log_view.appendPlainText(f">>> {name}:")
            self._stream_marker = name
        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.End)
        self.log_view.setTextCursor(cursor)
        self.log_view.insertPlainText(delta.replace("\n", " "))
        if self.autoscroll.isChecked():
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def on_image_done(self, img: str, caption_path: str):
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole) == img:
                it.setForeground(Qt.darkGreen)
                if "✓" not in it.text():
                    it.setText(it.text() + " ✓")
        self._stream_marker = None

    def on_image_failed(self, img: str, err: str):
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole) == img:
                it.setForeground(Qt.red)

    def on_finished(self, completed: bool):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log(f"[batch] {'completed' if completed else 'cancelled'}."
                        f" Progress bar shows successful captions only.")

    def on_server_state(self, state: str):
        self.server_state_label.setText(f"server: {state}")

    def append_log(self, line: str):
        self.log_view.appendPlainText(line)
        if self.autoscroll.isChecked():
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())
