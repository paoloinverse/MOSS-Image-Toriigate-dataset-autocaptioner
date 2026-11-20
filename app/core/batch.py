"""Batch captioning worker.

Runs an asyncio loop on a dedicated QThread: iterates the selected images,
streams each response to the GUI log, writes sidecar captions (.json for
MOSS / .txt for Booru), updates progress, and honours a Stop request.

Transient server errors (503 "Loading model" after a crash+restart, or
connection refused during process recovery) are handled by polling the
server's /health endpoint until it is ready again, then retrying the same
image.  This prevents a mid-batch server restart from cascading into
cannot-fail errors for every remaining image.
"""
from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path

import httpx
from PySide6.QtCore import QThread, Signal

from app.core.api_client import CaptionClient
from app.core.presets import (CaptionPreset, load_existing_caption,
                              save_caption)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# Transient-error detection: these patterns in a RuntimeError from
# CaptionClient.caption() indicate the server is recovering (crashed +
# restarted, model reloading) and will accept requests again shortly.
_TRANSIENT_PATTERNS = re.compile(
    r"Loading model|HTTP 50[23]|ConnectError|connection refused|"
    r"Connection refused|RemoteDisconnected|Server disconnected",
    re.IGNORECASE,
)
# How long to wait for the server to come back after a transient error.
_RETRY_HEALTH_TIMEOUT_S = 300.0     # 5 min max (large models on slow disks)
_RETRY_HEALTH_POLL_S = 5.0          # poll interval


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
    image_result = Signal(str, str)         # (image path, raw model JSON) -
                                            # feeds ctx.last_results for Tab 4
    image_failed = Signal(str, str)         # (image path, error message)
    log = Signal(str)
    finished_all = Signal(bool)             # completed vs cancelled

    def __init__(self, base_url: str, preset: CaptionPreset,
                 system_prompt: str, images: list[str], parent=None,
                 settings=None, verbose: bool = False, api_key: str = "",
                 model_key: str | None = None,
                 external_model: str = "", external: bool = False):
        super().__init__(parent)
        self.base_url = base_url
        self.preset = preset
        self.system_prompt = system_prompt
        self.images = list(images)
        # When True (the active preset has include_image_filename=True), each
        # request's effective system prompt gets the image basename appended
        # so the model can echo it back in its structured output.
        self._include_filename = bool(
            getattr(preset, "include_image_filename", False))
        # Live Tab 2 InferenceSettings (temperature/top_k/... + thinking
        # budget); None keeps the pure preset behaviour.
        self.settings = settings
        # Resolved run-target registry key at start time: gates family-
        # specific payload extras in api_client.build_payload (plan §4.5).
        self.model_key = model_key or ""
        # Bearer token from the Tab 2 'API Key:' field ('' -> env fallback).
        self.api_key = api_key or ""
        # Verbose llama-server conversation logging: full prompts sent,
        # streamed answers AND thinking/reasoning deltas land on self.log.
        self.verbose = verbose
        self.external_model = external_model or ""
        self.external = bool(external)
        self.cancel_event = threading.Event()
        # Static per-run tag for the always-on [batch] log lines: makes the
        # upscaling state visible without verbose mode (the per-image dims +
        # timing line is emitted by api_client.caption via log_cb there).
        self._upscale_tag = ""
        if settings is not None and getattr(settings, "upscale_enabled", False):
            try:
                _sx = int(getattr(settings, "upscale_scale_x", 2) or 1)
                _sy = int(getattr(settings, "upscale_scale_y", 2) or 1)
            except (TypeError, ValueError):
                _sx = _sy = 1
            if _sx > 1 or _sy > 1:
                from app.core.upscale import METHODS as _UP_METHODS
                _m = str(getattr(settings, "upscale_method", "lanczos"))
                _mlabel = _UP_METHODS.get(_m, (_m, ""))[0]
                self._upscale_tag = f"  [upscale {_mlabel} {_sx}x/{_sy}x]"

    def _health_url(self) -> str:
        path = "/v1/models" if self.external else "/health"
        return self.base_url.rstrip("/") + path

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

    async def _wait_for_server_healthy(self) -> bool:
        """Poll /health until the server accepts requests again.

        Called after a transient error (503 'Loading model', connection
        refused) so the batch can resume instead of cascading failures.
        Returns True when the server is healthy, False on timeout or cancel.
        """
        url = self._health_url()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        elapsed = 0.0
        attempt = 0
        while elapsed < _RETRY_HEALTH_TIMEOUT_S:
            if self.cancel_event.is_set():
                return False
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=5.0) as hc:
                    resp = await hc.get(url, headers=headers)
                    if resp.status_code < 500:
                        # 200 or even 401 (auth required but server alive)
                        self.log.emit(
                            f"[batch] server recovered after "
                            f"~{elapsed:.0f}s ({attempt} probe(s)).")
                        return True
            except Exception:
                pass  # connection refused / DNS / timeout — keep polling
            await asyncio.sleep(_RETRY_HEALTH_POLL_S)
            elapsed += _RETRY_HEALTH_POLL_S
        self.log.emit(
            f"[batch] WARNING: server did NOT recover within "
            f"{_RETRY_HEALTH_TIMEOUT_S:.0f}s ({attempt} probes).")
        return False

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        """True if the exception indicates a recoverable server issue."""
        return bool(_TRANSIENT_PATTERNS.search(str(exc)))

    async def _verbose_health_check(self, name: str, i: int, total: int):
        """Lightweight per-image health poll (verbose mode only).

        Hits /health with a short timeout and logs a single compact line.
        Gives a continuous liveness heartbeat in the log during long batches.
        Zero disk writes — output goes through self.log signal only.
        """
        import time as _time
        url = self._health_url()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = _time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=3.0) as hc:
                resp = await hc.get(url, headers=headers)
                ms = (_time.monotonic() - t0) * 1000
                if resp.status_code < 500:
                    self.log.emit(
                        f"[llama] ({i}/{total}) health OK ({ms:.0f}ms) "
                        f"before {name}")
                else:
                    self.log.emit(
                        f"[llama] ({i}/{total}) health WARNING HTTP "
                        f"{resp.status_code} ({ms:.0f}ms) before {name}")
        except Exception as e:
            ms = (_time.monotonic() - t0) * 1000
            self.log.emit(
                f"[llama] ({i}/{total}) health FAIL ({ms:.0f}ms): "
                f"{type(e).__name__} before {name}")

    async def _run_async(self) -> bool:
        client = CaptionClient(self.base_url, api_key=self.api_key)
        total = len(self.images)
        fmt = self.preset.output_format
        # Custom companion-file naming (Tab 2 'Companion caption files'): the
        # sidecar is written as '<image_name><save_suffix>.<save_ext>'; an
        # empty extension follows the preset format (json/txt).  settings may be
        # None in headless/test use -> fall back to the plain defaults.
        _s = self.settings
        save_suffix = getattr(_s, "caption_save_suffix", "") if _s else ""
        save_ext = getattr(_s, "caption_save_ext", "") if _s else ""
        save_keep = bool(getattr(_s, "caption_keep_image_ext", False)) if _s else False
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
            # Verbose per-image health poll: confirms the server is alive
            # before each request and gives a continuous liveness heartbeat
            # in the log (especially useful during long batches).
            if self.verbose:
                await self._verbose_health_check(name, i, total)
            try:
                def on_chunk(delta: str, _img=img):
                    self.chunk.emit(_img, delta)

                def on_log(line: str):
                    self.log.emit(line)

                # Per-image system prompt: inject the basename so the model
                # can report it in its output (OCR Transcript 'image_filename').
                eff_sp = self.system_prompt
                if self._include_filename:
                    eff_sp += f'\n\n"image_filename": "{name}"'
                text = await client.caption(
                    img, self.preset, eff_sp,
                    on_chunk=on_chunk, cancel_event=self.cancel_event,
                    settings=self.settings,
                    model_key=self.model_key or None,
                    model_id=self.external_model or None,
                    log_cb=on_log if self.verbose else None)
                if self.cancel_event.is_set():
                    break
                if not text.strip():
                    raise RuntimeError("empty response from server")
                target = save_caption(img, text, fmt, suffix=save_suffix,
                                      ext=save_ext, keep_image_ext=save_keep)
                done += 1
                self.image_result.emit(img, text)
                self.image_done.emit(img, target)
                self.log.emit(f"[batch] ({i}/{total}) {name} -> "
                              f"{Path(target).name}{self._upscale_tag}")
            except Exception as exc:
                # ---- transient server error: wait & retry ---------------
                if self._is_transient_error(exc) and not self.cancel_event.is_set():
                    self.log.emit(
                        f"[batch] ({i}/{total}) {name}: transient server "
                        f"error detected — waiting for llama-server to recover…\n"
                        f"    reason: {str(exc)[:200]}")
                    recovered = await self._wait_for_server_healthy()
                    if recovered:
                        # Retry the same image once the server is back.
                        try:
                            text = await client.caption(
                                img, self.preset, eff_sp,
                                on_chunk=on_chunk,
                                cancel_event=self.cancel_event,
                                settings=self.settings,
                                model_key=self.model_key or None,
                                model_id=self.external_model or None,
                                log_cb=on_log if self.verbose else None)
                            if text.strip():
                                target = save_caption(
                                    img, text, fmt, suffix=save_suffix,
                                    ext=save_ext, keep_image_ext=save_keep)
                                done += 1
                                self.image_result.emit(img, text)
                                self.image_done.emit(img, target)
                                self.log.emit(
                                    f"[batch] ({i}/{total}) {name} -> "
                                    f"{Path(target).name}"
                                    f"{self._upscale_tag} (recovered after retry)")
                                continue
                        except Exception as exc2:
                            self.image_failed.emit(img, str(exc2))
                            self.log.emit(
                                f"[batch] ({i}/{total}) FAILED {name} "
                                f"(after server recovery): {exc2}")
                            continue
                # ---- permanent error or retry exhausted ------------------
                self.image_failed.emit(img, str(exc))
                self.log.emit(f"[batch] ({i}/{total}) FAILED {name}: {exc}")
        self.progress.emit(done, total)
        return not self.cancel_event.is_set()
