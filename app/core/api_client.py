"""Async OpenAI-compatible client for llama-server's /v1/chat/completions.

Uses ``httpx`` (async).  Images are base64 data-URLs embedded in multimodal
message content, exactly as llama-server's compatibility endpoint expects.

Payload extras per spec:
* ``enable_thinking: true``           -> MOSS "Thinking Mode"
* ``reasoning_budget_tokens: N``      -> Qwen3.8 reasoning budget
* streaming responses are forwarded chunk-by-chunk to a callback so the GUI
  can show live output.

Verbose conversation logging
----------------------------
When ``log_cb`` is supplied (or the module-level logger is enabled with
``set_verbose(True)``), every exchange with llama-server is dumped in full:

* the complete request payload — system prompt, user prompt, sampling
  parameters, thinking flags — with base64 image data elided;
* every streamed assistant delta, including *thinking/reasoning* content
  (``delta.reasoning_content`` / ``delta.reasoning``, which llama-server
  emits for thinking-capable models) tagged ``[thinking]`` separately from
  the visible answer;
* finish reason, usage stats and any error body.

The raw SSE lines themselves are also recorded (again with base64 elided) so
the exact wire traffic can be reconstructed from the log.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from typing import Awaitable, Callable, Optional

import httpx

from app.core.presets import CaptionPreset, GenerationParams

ChunkCb = Optional[Callable[[str], None]]
LogCb = Optional[Callable[[str], None]]

# Module-level fallback sink + verbosity switch (used when a caller does not
# pass an explicit ``log_cb``).  Default OFF keeps normal runs quiet; enable
# with set_verbose(True) or by configuring this logger at DEBUG level.
logger = logging.getLogger("app.llama.conversation")
_verbose_enabled = False


def set_verbose(enabled: bool = True) -> None:
    """Globally enable/disable verbose llama-server conversation logging."""
    global _verbose_enabled
    _verbose_enabled = bool(enabled)
    if enabled:
        logger.setLevel(logging.DEBUG)


def _emit(log_cb: LogCb, line: str) -> None:
    """Route one log line to the per-call callback and/or module logger."""
    if log_cb is not None:
        try:
            log_cb(line)
        except Exception:
            pass    # logging must never break inference
    elif _verbose_enabled:
        logger.debug(line)


# ------------------------------------------------------------------ redaction
_B64_RE = re.compile(r"(data:[^;,]+;base64,)([A-Za-z0-9+/=]{32,})")


def _redact(obj):
    """Deep-copy *obj*, replacing base64 payloads with a short summary.

    Keeps logs readable (a single image can be several MB of base64) while
    still showing exactly what was sent: MIME type, byte size and a tiny
    fingerprint of the blob.
    """
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    if isinstance(obj, str):
        def _sub(m: re.Match) -> str:
            head, b64 = m.group(1), m.group(2)
            approx_bytes = len(b64) * 3 // 4
            return f"{head}{b64[:12]}…<{approx_bytes} bytes elided>"
        return _B64_RE.sub(_sub, obj)
    return obj


def _dump(obj) -> str:
    return json.dumps(_redact(obj), indent=2, ensure_ascii=False)


def _format_messages(messages: list) -> list[str]:
    """Human-readable rendering of the chat messages actually being sent."""
    lines: list[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            lines.append(f"  [{i}] {role}:")
            for ln in content.splitlines() or [""]:
                lines.append(f"        {ln}")
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    parts.append(f"text: {part.get('text', '')!r}")
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    m = _B64_RE.match(url)
                    if m:
                        approx = len(m.group(2)) * 3 // 4
                        parts.append(f"image_url: <base64 ~{approx} bytes>")
                    else:
                        parts.append(f"image_url: {url[:80]}")
                else:
                    parts.append(f"{ptype}: …")
            lines.append(f"  [{i}] {role}: " + " | ".join(parts))
        else:
            lines.append(f"  [{i}] {role}: {content!r}")
    return lines


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
                 api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Optional bearer token.  Priority: explicit ``api_key`` argument
        # (the GUI passes the Tab 2 'API Key:' field through AppContext) ->
        # MOSS_API_KEY / APPCAP_API_KEY env vars (kept for headless / legacy
        # setups) -> no auth at all, which matches a server started without
        # --api-key (the localhost default).
        if api_key is None:
            api_key = (os.environ.get("MOSS_API_KEY")
                       or os.environ.get("APPCAP_API_KEY") or "")
        self.api_key = api_key or ""

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def caption(self, image_path: str, preset: CaptionPreset,
                      system_prompt: str, on_chunk: ChunkCb = None,
                      cancel_event=None, settings=None,
                      log_cb: LogCb = None) -> str:
        """Stream a chat completion and return the full assistant text.

        ``on_chunk(text)`` receives each delta for live GUI log tailing.
        ``cancel_event`` (threading.Event/asyncio-friendly bool check) allows
        the batch runner to abort mid-request.  ``settings`` (AppContext's
        InferenceSettings) overrides sampling/thinking values with the live
        Tab 2 state so requests match the running server configuration.
        ``log_cb(line)`` enables *verbose conversation logging*: the full
        request payload (base64 images elided), every streamed content and
        thinking/reasoning delta, raw SSE traffic, finish reason and usage
        stats are reported line-by-line.  When omitted, lines go to the
        module logger if :func:`set_verbose` was enabled.
        """
        if log_cb is None and not _verbose_enabled:
            log_cb = None    # stay completely quiet unless asked
        url = f"{self.base_url}/v1/chat/completions"
        payload = build_payload(image_path, preset, system_prompt,
                                settings=settings)

        # ---- verbose: dump the exact conversation being sent ----------------
        _emit(log_cb, f"[llama] >>> POST {url}  "
                      f"(preset '{preset.display_name}', image "
                      f"'{image_path}')")
        for ln in _format_messages(payload["messages"]):
            _emit(log_cb, f"[llama] >>> {ln}")
        sampling = {k: v for k, v in payload.items() if k != "messages"}
        _emit(log_cb, "[llama] >>> params: " + json.dumps(
            _redact(sampling), ensure_ascii=False))

        chunks: list[str] = []          # visible answer text
        think_chunks: list[str] = []    # reasoning / thinking text
        n = 0                           # SSE data-event counter
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload,
                                     headers=self._headers()) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    _emit(log_cb, f"[llama] <<< HTTP {resp.status_code} "
                                  f"ERROR (body follows):")
                    for ln in body.splitlines():
                        _emit(log_cb, f"[llama] <<<   {ln}")
                    raise RuntimeError(
                        f"llama-server HTTP {resp.status_code}: {body[:500]}")
                _emit(log_cb, f"[llama] <<< HTTP {resp.status_code}, "
                              f"streaming response:")
                async for line in resp.aiter_lines():
                    if cancel_event is not None and getattr(cancel_event, "is_set",
                                                            lambda: False)():
                        _emit(log_cb, "[llama] !!! cancelled by user "
                                      "mid-stream")
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        _emit(log_cb, "[llama] <<< [DONE]")
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        _emit(log_cb, f"[llama] <<< raw (unparsed): {data[:200]}")
                        continue
                    n += 1
                    # Raw wire line, redacted — full audit trail of the SSE.
                    _emit(log_cb, f"[llama] <<< #{n} raw: "
                                  f"{_dump(obj).replace(chr(10), ' ')}")
                    choice = (obj.get("choices") or [{}])[0]
                    d = choice.get("delta") or {}
                    # Thinking / reasoning deltas (llama-server exposes them
                    # as delta.reasoning_content; some builds use
                    # delta.reasoning).  Logged separately, never mixed into
                    # the returned caption text.
                    think = d.get("reasoning_content") or d.get("reasoning") or ""
                    if think:
                        think_chunks.append(think)
                        _emit(log_cb, f"[llama] <<< #{n} [thinking] "
                                      f"{think.replace(chr(10), ' ')}")
                    delta = d.get("content") or choice.get("text") or ""
                    if delta:
                        _emit(log_cb, f"[llama] <<< #{n} [answer] "
                                      f"{delta.replace(chr(10), ' ')}")
                        chunks.append(delta)
                        if on_chunk:
                            on_chunk(delta)
                    if choice.get("finish_reason"):
                        _emit(log_cb, f"[llama] <<< finish_reason="
                                      f"{choice['finish_reason']}")
                    if obj.get("usage"):
                        _emit(log_cb, "[llama] <<< usage: " + json.dumps(
                            obj["usage"], ensure_ascii=False))

        answer = "".join(chunks)
        # ---- verbose: final summary of the exchange -------------------------
        _emit(log_cb, f"[llama] === response complete: {n} chunk(s); "
                      f"thinking {len(''.join(think_chunks))} chars, "
                      f"answer {len(answer)} chars")
        if think_chunks:
            _emit(log_cb, "[llama] === FULL THINKING:")
            for ln in "".join(think_chunks).splitlines():
                _emit(log_cb, f"[llama] ===   {ln}")
        _emit(log_cb, "[llama] === FULL ANSWER:")
        for ln in (answer or "(empty)").splitlines():
            _emit(log_cb, f"[llama] ===   {ln}")
        return answer
