"""Async OpenAI-compatible client for llama-server's /v1/chat/completions.

Uses ``httpx`` (async).  Images are base64 data-URLs embedded in multimodal
message content, exactly as llama-server's compatibility endpoint expects.

Payload extras per spec:
* ``enable_thinking: true``           -> MOSS "Thinking Mode"
* ``reasoning_budget_tokens: N``      -> Qwen3.8 reasoning budget
* streaming responses are forwarded chunk-by-chunk to a callback so the GUI
  can show live output.
* thinking-capable models stream their chain-of-thought in
  ``delta.reasoning_content`` (llama.cpp chat API); it is captured separately
  from the answer and surfaced through the verbose log / ``last_reasoning``.
* verbose request/response logging (prompt sent, full raw reply) is emitted
  via the optional ``log`` sink; enable it with ``MOSS_VERBOSE_LOG=1`` or the
  ``AppContext.settings.verbose_logging`` flag.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from typing import Awaitable, Callable, Optional

import httpx

from app.core.presets import CaptionPreset, GenerationParams

ChunkCb = Optional[Callable[[str], None]]
LogCb = Optional[Callable[[str], None]]


def verbose_log_enabled(settings=None) -> bool:
    """Global opt-in for the verbose request/response transcript."""
    if settings is not None and getattr(settings, "verbose_logging", None) is not None:
        return bool(getattr(settings, "verbose_logging"))
    return os.environ.get("MOSS_VERBOSE_LOG", "").strip().lower() in ("1", "true", "yes", "on")


def _redact_payload(payload: dict) -> dict:
    """Copy of *payload* with base64 image blobs replaced by a size summary
    so the verbose log stays readable."""
    def scrub(content):
        if isinstance(content, list):
            out = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = ((part.get("image_url") or {}).get("url")) or ""
                    head = url.split(",", 1)[0]
                    nbytes = len(url) * 3 // 4 if "," in url else 0
                    out.append({"type": "image_url",
                                "image_url": f"<{head} ~{nbytes // 1024} KiB>"})
                else:
                    out.append(part)
            return out
        return content

    red = dict(payload)
    red["messages"] = [{"role": m.get("role", ""),
                        "content": scrub(m.get("content"))}
                       for m in payload.get("messages", [])]
    return red


def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_payload(image_path: str, preset: CaptionPreset, system_prompt: str,
                  max_tokens: int | None = None,
                  settings=None) -> dict:
    params: GenerationParams = preset.params
    # Tab 2 live settings win over the preset when provided (AppContext's
    # InferenceSettings); this keeps server-side --reasoning-budget and the
    # request-side thinking flags consistent with what the user picked.
    reasoning_budget = int(getattr(settings, "reasoning_budget",
                                   params.reasoning_budget) or 0)
    enable_thinking = bool(params.enable_thinking) or reasoning_budget > 0
    payload: dict = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": preset.user_prompt},
                {"type": "image_url",
                 "image_url": {"url": image_to_data_url(image_path)}},
            ]},
        ],
        "temperature": float(getattr(settings, "temperature",
                                     params.temperature)),
        "top_p": float(getattr(settings, "top_p", params.top_p)),
        "max_tokens": max_tokens or params.max_tokens,
        "stream": True,
    }
    # llama-server accepts sampling extras at top level of the request body.
    payload["top_k"] = int(getattr(settings, "top_k", params.top_k))
    payload["min_p"] = float(getattr(settings, "min_p", params.min_p))
    if enable_thinking:
        payload["enable_thinking"] = True
    if reasoning_budget > 0:
        payload["reasoning_budget_tokens"] = reasoning_budget
    return payload


class CaptionClient:
    """Thin async wrapper around one running llama-server instance."""

    def __init__(self, base_url: str, timeout: float = 600.0,
                 log_cb: LogCb = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        #: Optional sink for the verbose request/response transcript
        #: (usually AppContext.log).  Nothing is logged unless verbose mode
        #: is enabled via settings.verbose_logging / MOSS_VERBOSE_LOG.
        self.log_cb = log_cb
        #: Chain-of-thought captured from the most recent request
        #: (delta.reasoning_content), empty when the model doesn't think.
        self.last_reasoning: str = ""
        #: Stop reason of the most recent completion ("stop", "length", ...).
        self.last_finish_reason: str = ""

    def _log(self, msg: str):
        if self.log_cb is not None:
            try:
                self.log_cb(msg)
            except Exception:
                pass    # logging must never break a request

    async def caption(self, image_path: str, preset: CaptionPreset,
                      system_prompt: str, on_chunk: ChunkCb = None,
                      cancel_event=None, settings=None) -> str:
        """Stream a chat completion and return the full assistant text.

        ``on_chunk(text)`` receives each answer delta for live GUI log
        tailing.  ``cancel_event`` (threading.Event/asyncio-friendly bool
        check) allows the batch runner to abort mid-request.  ``settings``
        (AppContext's InferenceSettings) overrides sampling/thinking values
        with the live Tab 2 state so requests match the running server
        configuration.  When verbose logging is on, the exact prompt sent and
        the full raw reply (answer + thinking) are emitted to ``self._log``.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = build_payload(image_path, preset, system_prompt,
                                settings=settings)
        verbose = verbose_log_enabled(settings)
        if verbose:
            self._log("[request] POST " + url)
            self._log("[request] payload (image data redacted):\n"
                      + json.dumps(_redact_payload(payload), indent=2,
                                   ensure_ascii=False))
        chunks: list[str] = []
        reasoning: list[str] = []
        self.last_reasoning = ""
        self.last_finish_reason = ""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise RuntimeError(
                        f"llama-server HTTP {resp.status_code}: {body[:500]}")
                async for line in resp.aiter_lines():
                    if cancel_event is not None and getattr(cancel_event, "is_set",
                                                            lambda: False)():
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    fr = choice.get("finish_reason")
                    if fr:
                        self.last_finish_reason = str(fr)
                    d = choice.get("delta")
                    if isinstance(d, dict):
                        # llama.cpp streams chain-of-thought separately in
                        # delta.reasoning_content for thinking-capable models;
                        # keep it out of the caption but log it in verbose mode.
                        rc = d.get("reasoning_content")
                        if isinstance(rc, str) and rc:
                            reasoning.append(rc)
                        delta = d.get("content")
                    else:
                        delta = None
                    if not delta:
                        delta = choice.get("text") or ""
                    if not isinstance(delta, str):
                        delta = ""
                    if delta:
                        chunks.append(delta)
                        if on_chunk:
                            on_chunk(delta)
        text = "".join(chunks)
        self.last_reasoning = "".join(reasoning)
        if verbose:
            if self.last_reasoning:
                self._log("[response] thinking (reasoning_content):\n"
                          + self.last_reasoning)
            self._log(f"[response] answer ({len(text)} chars, "
                      f"finish_reason={self.last_finish_reason or 'n/a'}):\n"
                      + (text if text else "<empty>"))
        return text
