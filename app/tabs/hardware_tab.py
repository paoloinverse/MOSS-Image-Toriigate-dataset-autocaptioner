"""Tab 1 - Hardware & Models.

* GPU pool with per-GPU checkbox + tensor-split weight slider.
* Live VRAM (per GPU) and System RAM meters driven by HardwareManager.
* Model / quant dropdowns, a user-selectable model-storage location
  (default app cache vs Unsloth-desktop compatible folder) and the
  "Download & Verify" action backed by a QThread worker around
  huggingface_hub.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QProgressBar, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from app.core.hardware import GPUInfo, HardwareManager
from app.core.models import (AppError, STORAGE_DEFAULT, STORAGE_UNSLOTH,
                             ModelFile, StorageLocationError,
                             download_model_files, estimate_download_mb,
                             resolve_mmproj, verify_downloaded)
from app.context import AppContext


class RepoRefreshWorker(QThread):
    """Live re-discovery of a repo's GGUF filenames/quants off the GUI thread."""
    done = Signal(bool, str)          # ok, verbose message/report
    log = Signal(str)

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec

    def run(self):
        self.log.emit(f"[models] live lookup: listing GGUFs published by "
                      f"{self.spec.source_repo} ...")
        try:
            ok = self.spec.refresh(force=True)
        except Exception as exc:      # never crash the worker thread
            err = AppError.classify(exc, context=f"listing {self.spec.source_repo}")
            self.done.emit(False, err.report())
            return
        if ok:
            self.done.emit(True,
                           f"{self.spec.display_name}: discovered "
                           f"{len(self.spec.files)} GGUF(s) live from the Hub: "
                           + ", ".join(f"{f.quant} ({f.filename})"
                                       for f in self.spec.files))
        else:
            msg = self.spec.last_refresh_error or \
                "live listing unavailable - kept previously known files."
            self.done.emit(False, msg)


class DownloadWorker(QThread):
    progress = Signal(str, int, int)     # filename, done, total (-1 unknown)
    done = Signal(bool, str)
    log = Signal(str)

    def __init__(self, repo_id: str, filenames: list[str], parent=None):
        super().__init__(parent)
        self.repo_id = repo_id
        self.filenames = filenames

    def run(self):
        try:
            self.log.emit(f"[dl] queueing {len(self.filenames)} file(s) from "
                          f"{self.repo_id}")
            paths = download_model_files(self.repo_id, self.filenames)
            self.log.emit("[dl] downloads complete: " +
                          ", ".join(p.name for p in paths))
            self.done.emit(True, "Download finished.")
        except AppError as exc:
            # Verbose structured report (reason + fix hints) -> GUI dialog+log.
            self.done.emit(False, exc.report())
        except Exception as exc:
            err = AppError.classify(
                exc, context=f"worker downloading {self.filenames} from {self.repo_id}")
            self.done.emit(False, err.report())


class GpuRow(QWidget):
    """One GPU: checkbox, name label, VRAM bar and split-weight slider."""

    toggled = Signal(int, bool)         # gpu index, checked
    weight_changed = Signal(int, int)   # gpu index, weight

    def __init__(self, gpu: GPUInfo, parent=None):
        super().__init__(parent)
        self.gpu = gpu
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.check = QCheckBox()
        self.check.setChecked(gpu.selected)
        self.check.toggled.connect(
            lambda v: self.toggled.emit(gpu.index, v))
        lay.addWidget(self.check)

        self.name = QLabel(gpu.label())
        self.name.setMinimumWidth(280)
        lay.addWidget(self.name)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setFormat("VRAM %p% used")
        lay.addWidget(self.bar, stretch=1)

        lay.addWidget(QLabel("split"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(gpu.split_weight)
        self.slider.setMaximumWidth(120)
        self.slider.valueChanged.connect(
            lambda v: self.weight_changed.emit(gpu.index, v))
        lay.addWidget(self.slider)

    def update_gpu(self, gpu: GPUInfo):
        self.gpu = gpu
        self.name.setText(gpu.label())
        pct = int(100 * gpu.used_vram_mb / max(1, gpu.total_vram_mb))
        self.bar.setValue(pct)


class HardwareModelsTab(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.hw: HardwareManager = ctx.hw
        self._gpu_rows: dict[int, GpuRow] = {}
        self._worker: DownloadWorker | None = None

        root = QVBoxLayout(self)

        # ---- GPU Pool -------------------------------------------------------
        pool_box = QGroupBox("GPU Pool (select inference devices)")
        pl = QVBoxLayout(pool_box)
        self.gpu_list = QVBoxLayout()
        pl.addLayout(self.gpu_list)
        hint = QLabel("Sliders set the relative --tensor-split ratio per GPU.")
        hint.setStyleSheet("color: gray;")
        pl.addWidget(hint)
        root.addWidget(pool_box)

        # ---- Resource meters -------------------------------------------------
        res_box = QGroupBox("Resource Meters")
        rl = QVBoxLayout(res_box)
        self.safe_label = QLabel("Safe VRAM pool: —")
        rl.addWidget(self.safe_label)
        self.ram_bar = QProgressBar()
        self.ram_bar.setFormat("System RAM: %p% used")
        rl.addWidget(self.ram_bar)
        self.ram_label = QLabel("— / — GiB available")
        rl.addWidget(self.ram_label)
        root.addWidget(res_box)

        # ---- Model selector ---------------------------------------------------
        model_box = QGroupBox("Model")
        ml = QVBoxLayout(model_box)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for spec in ctx.registry.values():
            self.model_combo.addItem(spec.display_name, spec.key)
        row1.addWidget(self.model_combo, stretch=1)
        ml.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Quantization:"))
        self.quant_combo = QComboBox()
        self.quant_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        row2.addWidget(self.quant_combo, stretch=1)
        self.refresh_models_btn = QPushButton("Refresh from Hub")
        self.refresh_models_btn.setToolTip(
            "Live-lookup the repository file list on huggingface.co and rebuild\n"
            "this dropdown from the filenames/quants actually published there.\n"
            "Names are never composed locally, so renamed uploads (e.g. a\n"
            "lowercase '-bf16.gguf') are picked up automatically.")
        row2.addWidget(self.refresh_models_btn)
        ml.addLayout(row2)

        self.mmproj_label = QLabel("mmproj: —")
        self.mmproj_label.setWordWrap(True)
        self.mmproj_label.setStyleSheet("color: gray;")
        ml.addWidget(self.mmproj_label)

        # ---- Storage location --------------------------------------------------
        stor_box = QGroupBox("Storage Location (where GGUF models are kept / downloaded)")
        sl = QVBoxLayout(stor_box)

        srow0 = QHBoxLayout()
        self.storage_combo = QComboBox()
        self.storage_combo.addItem("Default app cache folder", STORAGE_DEFAULT)
        self.storage_combo.addItem("Unsloth-desktop compatible models folder",
                                   STORAGE_UNSLOTH)
        srow0.addWidget(self.storage_combo, stretch=1)
        sl.addLayout(srow0)

        srow1 = QHBoxLayout()
        self.unsloth_path_edit = QLineEdit()
        self.unsloth_path_edit.setPlaceholderText(
            "Path of the Unsloth Desktop models folder, e.g. ~/Unsloth/Desktop/models")
        self.unsloth_path_edit.textEdited.connect(self._on_unsloth_text_edited)
        self.browse_btn = QPushButton("Browse…")
        srow1.addWidget(self.unsloth_path_edit, stretch=1)
        srow1.addWidget(self.browse_btn)
        sl.addLayout(srow1)

        self.storage_info = QLabel("")
        self.storage_info.setWordWrap(True)
        self.storage_info.setStyleSheet("color: gray;")
        sl.addWidget(self.storage_info)
        self.apply_storage_btn = QPushButton("Apply storage location")
        sl.addWidget(self.apply_storage_btn)

        row3 = QHBoxLayout()
        self.download_btn = QPushButton("Download & Verify")
        self.dl_bar = QProgressBar()
        self.dl_bar.setRange(0, 0)
        self.dl_bar.setVisible(False)
        row3.addWidget(self.download_btn)
        row3.addWidget(self.dl_bar, stretch=1)
        ml.addLayout(row3)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        ml.addWidget(self.status_label)
        root.addWidget(stor_box)
        root.addWidget(model_box)

        root.addStretch(1)

        # ---- wiring ------------------------------------------------------------
        self.hw.gpus_updated.connect(self.on_gpus_updated)
        self.hw.ram_updated.connect(self.on_ram_updated)
        self.model_combo.currentIndexChanged.connect(self.refresh_quants)
        self.quant_combo.currentIndexChanged.connect(self.refresh_mmproj_label)
        self.download_btn.clicked.connect(self.start_download)
        self.refresh_models_btn.clicked.connect(self.refresh_from_hub)
        self.storage_combo.currentIndexChanged.connect(self._update_storage_widgets)
        self.browse_btn.clicked.connect(self.browse_unsloth_folder)
        self.apply_storage_btn.clicked.connect(self.apply_storage)
        self.ctx.storage_changed.connect(self.refresh_storage_ui)
        # Hydrate widgets from the persisted configuration (app_settings.json).
        idx = self.storage_combo.findData(self.ctx.storage.mode)
        self.storage_combo.setCurrentIndex(max(0, idx))
        self.unsloth_path_edit.setText(self.ctx.storage.unsloth_root)
        self._update_storage_widgets()
        self.refresh_storage_ui()
        self.refresh_quants()
        # Late-bound live listing: if start-up fell back to the seed files
        # (offline), pull the real Hub filenames now without blocking the UI.
        for spec in self.ctx.registry.values():
            if not spec.live:
                self._start_repo_refresh(spec, startup=True)
                break

    # ------------------------------------------------------------------ GPU UI
    def on_gpus_updated(self, gpus: list[GPUInfo]):
        seen = set()
        for g in gpus:
            seen.add(g.index)
            row = self._gpu_rows.get(g.index)
            if row is None or row.gpu.name != g.name:
                if row is not None:
                    self.gpu_list.removeWidget(row)
                    row.deleteLater()
                row = GpuRow(g)
                row.toggled.connect(self.on_gpu_toggled)
                row.weight_changed.connect(self.on_gpu_weight)
                self._gpu_rows[g.index] = row
                self.gpu_list.addWidget(row)
            row.update_gpu(g)
        for idx in list(self._gpu_rows):
            if idx not in seen:
                row = self._gpu_rows.pop(idx)
                self.gpu_list.removeWidget(row)
                row.deleteLater()
        total = self.hw.safe_vram_total_mb
        n = len(self.hw.selected_gpus)
        ts = self.hw.tensor_split_string() or "-"
        spill = "" if n > 1 else ""
        self.safe_label.setText(
            f"Safe VRAM pool: {total/1024:.1f} GiB across {n} GPU(s) "
            f"| tensor-split: {ts}{spill}")

    def on_gpu_toggled(self, index: int, checked: bool):
        for g in self.hw.gpus:
            if g.index == index:
                g.selected = checked
        self.ctx.log.emit(f"[hw] GPU {index} {'added to' if checked else 'removed from'} "
                          "inference pool")

    def on_gpu_weight(self, index: int, weight: int):
        for g in self.hw.gpus:
            if g.index == index:
                g.split_weight = weight

    # ------------------------------------------------------------------ RAM UI
    def on_ram_updated(self, avail_mib: int, total_mib: int):
        """Slot for HardwareManager.ram_updated (values arrive in MiB)."""
        total_mib = max(1, int(total_mib))
        avail_mib = max(0, min(int(avail_mib), total_mib))
        used_pct = int(100 * (total_mib - avail_mib) / total_mib)
        # QProgressBar.setValue() takes a 32-bit int; clamp so no upstream
        # arithmetic slip can ever raise a libshiboken OverflowError here.
        self.ram_bar.setValue(max(0, min(100, used_pct)))
        self.ram_label.setText(
            f"{avail_mib / 1024:.1f} / {total_mib / 1024:.1f} GiB available "
            "(spilling target when weights exceed VRAM)")

    # ------------------------------------------------------------------ models
    def refresh_quants(self):
        """Populate the quant dropdown from the LIVE-discovered file list.

        Each entry shows the parsed quant tag plus the exact Hub filename, so
        what the user selects is literally what gets downloaded - no locally
        composed names anywhere in the pipeline.
        """
        key = self.model_combo.currentData()
        self.ctx.selected_model_key = key
        spec = self.ctx.registry[key]
        prev_q = self.quant_combo.currentText()
        self.quant_combo.blockSignals(True)
        self.quant_combo.clear()
        for f in spec.files:
            self.quant_combo.addItem(f"{f.quant}  [{f.filename}]", f.quant)
        default_q = ""
        if prev_q and spec.file_for_quant(prev_q) is not None:
            default_q = prev_q                    # keep selection across refresh
        elif "Q8_0" in spec.quants():
            default_q = "Q8_0"
        elif any(q.upper().replace("_", "-") == "Q8-0" for q in spec.quants()):
            default_q = next(q for q in spec.quants()
                             if q.upper().replace("_", "-") == "Q8-0")
        elif spec.quants():
            default_q = spec.quants()[0]
        idx = self.quant_combo.findText(default_q) \
            if default_q else -1
        if idx < 0:
            for i in range(self.quant_combo.count()):
                if self.quant_combo.itemData(i) == default_q:
                    idx = i
                    break
        self.quant_combo.setCurrentIndex(max(0, idx))
        self.quant_combo.blockSignals(False)
        self.ctx.selected_quant = self.quant_combo.currentData() or ""
        if not spec.live and spec.last_refresh_error:
            self.status_label.setText(
                "Offline fallback: showing bundled seed filenames "
                "(live Hub lookup failed). Press 'Refresh from Hub' to retry.")
        self.refresh_mmproj_label()
        self.ctx.model_changed.emit()

    # ----------------------------------------------------- live Hub discovery
    def _start_repo_refresh(self, spec, startup: bool = False):
        if getattr(self, "_repo_worker", None) is not None \
                and self._repo_worker.isRunning():
            return
        self._repo_worker = RepoRefreshWorker(spec, self)
        self._repo_worker.log.connect(self.ctx.log.emit)
        self._repo_worker.done.connect(
            lambda ok, msg, s=spec: self.on_repo_refreshed(ok, msg, s))
        self.refresh_models_btn.setEnabled(False)
        self.refresh_models_btn.setText("Looking up…")
        self._repo_worker.start()
        if startup:
            self.ctx.log.emit(
                f"[models] start-up listing of '{spec.source_repo}' was "
                "unavailable; retrying the live lookup in the background.")

    def refresh_from_hub(self):
        """Manual trigger: re-query the Hub and rebuild the quant dropdown."""
        self._start_repo_refresh(self.ctx.active_model_spec)

    def on_repo_refreshed(self, ok: bool, msg: str, spec):
        self.refresh_models_btn.setEnabled(True)
        self.refresh_models_btn.setText("Refresh from Hub")
        self.ctx.log.emit(msg if ok else f"[models] {msg}")
        if not ok and not getattr(spec, "live", False):
            self.status_label.setText("Live Hub lookup failed - see Log (Tab 5).")
            return
        if self.model_combo.currentData() == spec.key:
            self.refresh_quants()
        else:
            self.ctx.model_changed.emit()

    def current_file(self):
        spec = self.ctx.active_model_spec
        return spec.file_for_quant(self.quant_combo.currentData()
                                   or self.quant_combo.currentText())

    def refresh_mmproj_label(self):
        mf = self.current_file()
        self.ctx.selected_quant = (self.quant_combo.currentData()
                                   or self.quant_combo.currentText())
        if mf is None:
            self.mmproj_label.setText("mmproj: —")
            return
        spec = self.ctx.active_model_spec
        state = []
        try:
            # Fuzzy companion lookup: finds Unsloth-style names
            # ('mmproj-Model-Q8_0.gguf') as well as our default
            # ('Model-Q8_0.mmproj.gguf'); adopts the discovered name.
            mp_path, mm_name = mf.resolve_mmproj_companion()
            if mp_path is None:
                mm_name = mm_name or resolve_mmproj(spec, mf) or "(none found)"
        except Exception as exc:      # e.g. unexpected Hub API breakage
            mp_path, mm = None, "(lookup failed)"
            self.ctx.report_error(exc, context=f"auto-detecting mmproj for {spec.source_repo}")
        else:
            mm = mm_name or "(none found)"
        try:
            gguf_here = mf.local_path().is_file()
        except OSError:
            gguf_here = False
        if gguf_here:
            state.append(f"GGUF cached ({mf.local_path().parent})")
        if mp_path is not None:
            state.append(f"mmproj cached ({mp_path.parent})")
        elif mm != "(none found)":
            state.append("mmproj NOT found locally - would be downloaded")
        self.mmproj_label.setText(f"mmproj: {mm}" +
                                  (f"   [{', '.join(state)}]" if state else ""))
        self.ctx.model_changed.emit()

    # ---------------------------------------------------------- storage location
    def _update_storage_widgets(self):
        """Path row only makes sense in Unsloth mode; cache path is fixed."""
        unsloth = self.storage_combo.currentData() == STORAGE_UNSLOTH
        self.unsloth_path_edit.setEnabled(unsloth)
        self.browse_btn.setEnabled(unsloth)

    def _on_unsloth_text_edited(self, _text: str):
        """Live preview of the typed folder without persisting anything yet."""
        self._preview_storage(STORAGE_UNSLOTH, self.unsloth_path_edit.text())

    def browse_unsloth_folder(self):
        start = self.unsloth_path_edit.text().strip() or str(Path.home())
        d = QFileDialog.getExistingDirectory(
            self, "Select the Unsloth Desktop models folder", start)
        if d:
            self.unsloth_path_edit.setText(d)
            self._preview_storage(STORAGE_UNSLOTH, d)

    def _preview_storage(self, mode: str, unsloth_root: str):
        """Non-destructive autodetect summary shown under the selector."""
        from app.core.models import StorageManager
        probe = StorageManager(mode=mode, unsloth_root=unsloth_root)
        lines = [probe.describe()]
        if mode == STORAGE_UNSLOTH and unsloth_root.strip():
            p = Path(os.path.expanduser(unsloth_root.strip()))
            try:
                exists, isdir = p.exists(), p.is_dir()
            except OSError as exc:
                # Exotic filesystems (FUSE/network mounts, failing drives) can
                # raise on stat(); report it instead of claiming "missing".
                exists = isdir = False
                lines.append(f"WARNING: cannot inspect this path ({type(exc).__name__}: "
                             f"{exc}). Check that the volume is mounted/readable.")
            if not exists:
                lines.append("Directory does not exist yet - it will be created "
                             "automatically when you apply / download.")
            elif not isdir:
                lines.append("WARNING: the path exists but is not a directory.")
            spec = self.ctx.active_model_spec
            mf = self.current_file()
            if mf is not None:
                gguf = probe.find_existing(spec.source_repo, mf.filename)
                # Fuzzy companion search via the probe storage (handles both
                # 'Model-Q8_0.mmproj.gguf' and Unsloth's
                # 'mmproj-Model-Q8_0.gguf' naming).
                from app.core.models import STORAGE as _ACTIVE
                mp_path, mm_name = None, None
                try:
                    _saved_mode, _saved_root = (_ACTIVE.mode, _ACTIVE.unsloth_root)
                    _ACTIVE.mode, _ACTIVE.unsloth_root = mode, unsloth_root
                    try:
                        mp_path, mm_name = mf.resolve_mmproj_companion()
                    finally:
                        _ACTIVE.mode, _ACTIVE.unsloth_root = _saved_mode, _saved_root
                except Exception:
                    mp_path, mm_name = None, mf.mmproj_filename
                if gguf is not None:
                    lines.append(f"Required model re-use OK: {mf.filename} found at "
                                 f"{gguf} - no download needed.")
                    if mp_path is not None:
                        lines.append(f"mmproj re-use OK: {mp_path.name}")
                    else:
                        lines.append(f"mmproj '{mm_name or '?'}' NOT found "
                                     "in this folder - it would still be downloaded.")
                else:
                    lines.append(f"'{mf.filename}' not found here yet - it would be "
                                 "downloaded into this folder on 'Download & Verify'.")
        self.storage_info.setText("\n".join(lines))

    def refresh_storage_ui(self):
        """Re-render the preview from the currently active (persisted) config."""
        idx = self.storage_combo.findData(self.ctx.storage.mode)
        self.storage_combo.blockSignals(True)
        self.storage_combo.setCurrentIndex(max(0, idx))
        self.storage_combo.blockSignals(False)
        self.unsloth_path_edit.blockSignals(True)
        self.unsloth_path_edit.setText(self.ctx.storage.unsloth_root)
        self.unsloth_path_edit.blockSignals(False)
        self._update_storage_widgets()
        self._preview_storage(self.ctx.storage.mode, self.ctx.storage.unsloth_root)

    def apply_storage(self):
        mode = self.storage_combo.currentData()
        root = self.unsloth_path_edit.text().strip()
        try:
            self.ctx.set_storage_location(mode, root)
        except AppError as exc:
            # Verbose report (reason + fixes) -> global log + modal dialog.
            self.ctx.report_error(exc, context="applying storage location",
                                  dialog=True)
            return
        except Exception as exc:
            self.ctx.report_error(exc, context="applying storage location",
                                  dialog=True)
            return
        self.refresh_storage_ui()
        self.refresh_mmproj_label()
        self.status_label.setText("Storage location saved: " +
                                  self.ctx.storage.describe())

    # ------------------------------------------------------------------ download
    def start_download(self):
        if self._worker and self._worker.isRunning():
            self.status_label.setText("Download already in progress…")
            return
        mf = self.current_file()
        if mf is None:
            self.status_label.setText("Select a quantization first.")
            return
        spec = self.ctx.active_model_spec
        names = [mf.filename]
        # Fuzzy companion lookup first: if the projector already exists in
        # the selected storage under a non-default name (Unsloth layout),
        # adopt that name instead of queueing a duplicate download.
        try:
            mp_path, mm_name = mf.resolve_mmproj_companion()
        except Exception as exc:
            mp_path, mm_name = None, mf.mmproj_filename
            self.ctx.report_error(exc, context="mmproj detection before download")
        if mp_path is not None:
            self.ctx.log.emit(f"[dl] mmproj re-use OK: {mp_path}")
        mm = mm_name or resolve_mmproj(spec, mf)
        if mm:
            mf.mmproj_filename = mm
            names.append(mm)

        # ---- heavy-download guard ----------------------------------------
        # GGUFs are routinely 10-30 GiB.  Never pull them silently: estimate
        # the size from Hub metadata, show it, and require confirmation.
        # Files already present in the selected storage (e.g. downloaded by
        # Unsloth Desktop) are skipped automatically - subtract them from the
        # confirmation total so the user sees what actually transfers.
        est_mb = estimate_download_mb(spec.source_repo, names)
        if est_mb <= 0:
            est_mb = mf.estimated_size_mb   # fall back to bits-per-weight math
        pending: list[str] = []
        reused_mb = 0.0
        for n in names:
            hit = self.ctx.storage.find_existing(spec.source_repo, n)
            if hit is not None:
                try:
                    reused_mb += hit.stat().st_size / (1024 * 1024)
                except OSError:
                    pass
            else:
                pending.append(n)
        est_mb = max(0.0, est_mb - reused_mb)
        est_gb = est_mb / 1024.0

        reuse_note = ""
        if reused_mb > 0:
            reuse_note = (f"\nAlready available in the selected folder "
                          f"(re-used, no download): ~{reused_mb / 1024:.1f} GiB\n")

        if not pending:
            QMessageBox.information(
                self, "Nothing to download",
                f"All required files are already present in\n"
                f"'{self.ctx.storage.primary_download_dir(spec.source_repo)}'\n\n"
                f"Files: {', '.join(names)}\n\nUse 'Download & Verify' again "
                "after deleting a file to force a re-transfer.")
            self.refresh_mmproj_label()
            return

        reply = QMessageBox.question(
            self, "Confirm large download",
            f"About to download ~{est_gb:.1f} GiB into\n"
            f"{self.ctx.storage.primary_download_dir(spec.source_repo)}\n"
            f"{reuse_note}\nFiles to transfer: {', '.join(pending)}\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            self.status_label.setText("Download cancelled by user (no bytes transferred).")
            return

        self.dl_bar.setVisible(True)
        self.download_btn.setEnabled(False)
        self.status_label.setText(f"Downloading {len(pending)} file(s), ~{est_gb:.1f} GiB…")
        w = DownloadWorker(spec.source_repo, names, self)
        w.log.connect(self.ctx.log.emit)
        w.done.connect(self.on_download_done)
        self._worker = w
        w.start()

    def on_download_done(self, ok: bool, msg: str):
        self.dl_bar.setVisible(False)
        self.download_btn.setEnabled(True)
        self.status_label.setText(msg.splitlines()[0])
        mf = self.current_file()
        if ok and mf is not None:
            good, detail = verify_downloaded(mf)
            self.status_label.setText(("Verified: " if good else "Verify FAILED: ")
                                      + detail)
            self.ctx.log.emit(f"[verify] {detail}")
            if not good:
                self.ctx.report_error(detail, context="post-download verification",
                                      dialog=True)
        elif not ok:
            # msg carries the verbose AppError report from the worker
            self.ctx.log.emit(msg)
            self.ctx.report_error(msg, context="model download", dialog=True)
        self.refresh_mmproj_label()


def download_dir_for(repo_id: str) -> str:
    """Active destination folder for new downloads of ``repo_id``."""
    from app.core.models import STORAGE
    return str(STORAGE.primary_download_dir(repo_id))
