"""Async OpenAI-compatible client for llama-server's /v1/chat/completions.

Uses ``httpx`` (async).  Images are base64 data-URLs embedded in multimodal
message content, exactly as llama-server's compatibility endpoint expects.

Payload extras per spec:
* ``enable_thinking: true``           -> MOSS "Thinking Mode"
* ``reasoning_budget_tokens: N``      -> Qwen3.8 reasoning budget
* streaming responses are forwarded chunk-by-chunk to a callback so the GUI
  can show live output.
"""
from __future__ import annotations

import base64
import mimetypes
from typing import Awaitable, Callable, Optional

import httpx

from app.core.presets import CaptionPreset, GenerationParams

ChunkCb = Optional[Callable[[str], None]]


def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_payload(image_path: str, preset: CaptionPreset, system_prompt: str,
                  max_tokens: int | None = None) -> dict:
    params: GenerationParams = preset.params
    payload: dict = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": preset.user_prompt},
                {"type": "image_url",
                 "image_url": {"url": image_to_data_url(image_path)}},
            ]},
        ],
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": max_tokens or params.max_tokens,
        "stream": True,
    }
    # llama-server accepts sampling extras at top level of the request body.
    payload["top_k"] = params.top_k
    payload["min_p"] = params.min_p
    if params.enable_thinking:
        payload["enable_thinking"] = True
    if params.reasoning_budget > 0:
        payload["reasoning_budget_tokens"] = params.reasoning_budget
    return payload


class CaptionClient:
    """Thin async wrapper around one running llama-server instance."""

    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def caption(self, image_path: str, preset: CaptionPreset,
                      system_prompt: str, on_chunk: ChunkCb = None,
                      cancel_event=None) -> str:
        """Stream a chat completion and return the full assistant text.

        ``on_chunk(text)`` receives each delta for live GUI log tailing.
        ``cancel_event`` (threading.Event/asyncio-friendly bool check) allows
        the batch runner to abort mid-request.
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload = build_payload(image_path, preset, system_prompt)
        chunks: list[str] = []
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
                        import json
                        obj = json.loads(data)
                    except Exception:
                        continue
                    delta = ""
                    choice = (obj.get("choices") or [{}])[0]
                    d = choice.get("delta") or {}
                    delta = d.get("content") or choice.get("text") or ""
                    if delta:
                        chunks.append(delta)
                        if on_chunk:
                            on_chunk(delta)
        return "".join(chunks)
