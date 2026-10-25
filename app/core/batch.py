"""Batch captioning worker.

Runs an asyncio loop on a dedicated QThread: iterates the selected images,
streams each response to the GUI log, writes sidecar captions (.json for
MOSS / .txt for Booru), updates progress, and honours a Stop request.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.core.api_client import CaptionClient
from app.core.presets import (CaptionPreset, load_existing_caption,
                              save_caption)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def collect_images(root: str, recursive: bool, include_exts=IMAGE_EXTS) -> list[str]:
    base = Path(root)
    if not base.is_dir():
        return []
    it = base.rglob("*") if recursive else base.glob("*")
    files = sorted(str(p) for p in it
                   if p.is_file() and p.suffix.lower() in include_exts)
    return files


class BatchWorker(QThread):
    """Sequentially captions ``images`` against the running llama-server."""

    progress = Signal(int, int)             # done, total
    image_started = Signal(str)
    chunk = Signal(str, str)                # (image path, streamed delta)
    image_done = Signal(str, str)           # (image path, saved caption path)
    image_failed = Signal(str, str)         # (image path, error message)
    log = Signal(str)
    finished_all = Signal(bool)             # completed vs cancelled

    def __init__(self, base_url: str, preset: CaptionPreset,
                 system_prompt: str, images: list[str], parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.preset = preset
        self.system_prompt = system_prompt
        self.images = list(images)
        self.cancel_event = threading.Event()

    def stop(self):
        self.cancel_event.set()

    # ------------------------------------------------------------------ Qt
    def run(self):  # QThread entry point
        try:
            ok = asyncio.run(self._run_async())
        except Exception as exc:
            self.log.emit(f"[batch] fatal error: {exc}")
            ok = False
        self.finished_all.emit(ok and not self.cancel_event.is_set())

    async def _run_async(self) -> bool:
        client = CaptionClient(self.base_url)
        total = len(self.images)
        fmt = self.preset.output_format
        done = 0
        self.log.emit(f"[batch] starting: {total} image(s), preset "
                      f"'{self.preset.display_name}' -> *.{fmt}")
        for i, img in enumerate(self.images, start=1):
            if self.cancel_event.is_set():
                self.log.emit("[batch] stopped by user.")
                break
            self.progress.emit(done, total)
            self.image_started.emit(img)
            name = Path(img).name
            try:
                def on_chunk(delta: str, _img=img):
                    self.chunk.emit(_img, delta)

                text = await client.caption(
                    img, self.preset, self.system_prompt,
                    on_chunk=on_chunk, cancel_event=self.cancel_event)
                if self.cancel_event.is_set():
                    break
                if not text.strip():
                    raise RuntimeError("empty response from server")
                target = save_caption(img, text, fmt)
                done += 1
                self.image_done.emit(img, target)
                self.log.emit(f"[batch] ({i}/{total}) {name} -> "
                              f"{Path(target).name}")
            except Exception as exc:
                self.image_failed.emit(img, str(exc))
                self.log.emit(f"[batch] ({i}/{total}) FAILED {name}: {exc}")
        self.progress.emit(done, total)
        return not self.cancel_event.is_set()
