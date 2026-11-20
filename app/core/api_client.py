"""Async OpenAI-compatible client for llama-server's /v1/chat/completions.

Uses ``httpx`` (async).  Images are base64 data-URLs embedded in multimodal
message content, exactly as llama-server's compatibility endpoint expects.

Payload extras per spec (Qwen3.8-family compliance, Unsloth-patched chat
template; every knob is sent PER REQUEST so Tab 2 changes apply to the next
turn without restarting llama-server):
* ``chat_template_kwargs.enable_thinking``   -> master thinking switch
* ``chat_template_kwargs.reasoning_effort``  -> "xhigh" | "medium" | "low"
  (exactly what the template supports distinctly; the silent 'high'->'xhigh'
  alias is deliberately never emitted).  Omitted when thinking is off so a
  strict template build cannot raise a Jinja error on an unusable value.
* ``chat_template_kwargs.preserve_thinking`` -> keep earlier  blocks
  verbatim in the history (recommended: prevents the model being confused
  by old thinking blocks after a mid-conversation effort change).
* top-level ``reasoning_budget``             -> hard token cap enforced by
  the llama-server sampler (forces the closing  when reached).
  -1 / 0-budget-unlimited means "omit the cap"; thinking disabled sends 0
  so the server immediately closes any reasoning block.
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

External servers (OpenAI-compatible)
------------------------------------
``external_url`` / ``CaptionClient`` also work against any *external*
OpenAI-compatible server (llama.cpp on another box, the Unsloth studio
server, vLLM, …).  Two things that used to be flaky are now handled:

* **URL spelling.**  Users paste different things - the bare origin
  (``http://192.168.1.5:8888``), the API prefix (``.../v1``) or even the
  full chat endpoint (``.../v1/chat/completions``).  All of them mean the
  same server; :func:`normalize_base_url` strips them to the bare origin so
  this client can append its own ``/v1/...`` paths.  The user is only ever
  asked for the server address.
* **Model selection.**  The ``model`` field is NOT required.  By default
  the app assumes there is a single loaded model: :func:`fetch_models`
  reads ``/v1/models`` (with the Bearer key) and
  :func:`resolve_external_model` sends that model's exact ``id`` when
  exactly one is loaded, otherwise the literal ``"default"`` - which the
  server routes to whatever model is currently loaded.  The user can still
  force a specific id (Tab 2 'External server model:').
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

#: Literal model id used when no single loaded model can be determined.
#: llama.cpp / Unsloth-studio servers route ``"default"`` to the model that
#: is currently loaded, so an unqualified chat request still works.
DEFAULT_MODEL_ID = "default"

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


# ------------------------------------------------------------------ external servers

def normalize_base_url(url: str) -> str:
    """Normalise any accepted endpoint spelling to a bare server base URL.

    Users paste different things for the *same* server - the bare origin
    (``http://192.168.1.5:8888``), the API prefix (``.../v1``) or the full
    chat endpoint (``.../v1/chat/completions``).  This strips everything
    back down to the origin (plus any non-``/v1`` path prefix, e.g. a
    reverse-proxy mount like ``/api``), no trailing slash, so callers can
    append their own ``/v1/...`` paths safely.  A missing scheme defaults
    to ``http://``.  Pure string work - never raises.
    """
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        u = "http://" + u
    u = u.rstrip("/")
    # A pasted full chat endpoint: cut back to the (prefix +) origin.
    while u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]
    # Trailing '/v1' is the API prefix, not part of the base (repeats too).
    while u.endswith("/v1"):
        u = u[: -len("/v1")]
    return u.rstrip("/")


def fetch_models(base_url: str, api_key: str = "", timeout: float = 10.0):
    """GET ``{base}/v1/models`` on an OpenAI-compatible server.

    Returns ``(models, error)``: ``models`` is the raw ``data`` list from
    the server (each entry carries at least ``id``; llama.cpp / Unsloth
    studio servers additionally report ``loaded`` / ``status`` / ``quant``),
    or ``[]`` on failure.  ``error`` is "" on success, else a one-line
    human-readable reason.  The Bearer token is sent when ``api_key`` is
    non-empty (the listing is auth-gated on servers started with
    ``--api-key``).  Never raises.
    """
    import urllib.error as _ue
    import urllib.request as _ur
    base = normalize_base_url(base_url)
    if not base:
        return [], "empty server URL"
    url = base + "/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = _ur.Request(url, headers=headers)
        with _ur.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
        obj = json.loads(body)
        data = obj.get("data") if isinstance(obj, dict) else None
        if not isinstance(data, list):
            return [], "unrecognized /v1/models response (no 'data' list)"
        return data, ""
    except _ue.HTTPError as e:
        if e.code in (401, 403):
            return [], (f"HTTP {e.code}: "
                        + ("server rejected the configured API key"
                           if api_key else "server requires an API key"))
        return [], f"HTTP {e.code} from {url}"
    except Exception as e:                     # noqa: BLE001 - never raise
        return [], f"{type(e).__name__}: {e}"


def loaded_models(models) -> list:
    """Entries of a ``/v1/models`` list that count as *loaded/available*.

    llama.cpp / Unsloth-studio servers report this explicitly via
    ``loaded: true`` and/or ``status.value == 'loaded'``.  Plain OpenAI
    servers (vLLM, the real API) carry no load flag at all - every listed
    model is available there, so entries without any load info count as
    loaded too.  Entries without an ``id`` are skipped.
    """
    out = []
    for m in models or []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        status = m.get("status")
        status_val = status.get("value") if isinstance(status, dict) else None
        if m.get("loaded") is None and status_val is None:
            available = True          # no load info -> treat as available
        else:
            available = (m.get("loaded") is True) or status_val == "loaded"
        if available:
            out.append(m)
    return out


def resolve_external_model(base_url: str, api_key: str = "", explicit: str = "",
                           log_cb: LogCb = None, timeout: float = 5.0) -> str:
    """Pick the ``model`` field for chat requests to an EXTERNAL server.

    Policy (the model simply defaults to *what is loaded*):

    * an explicit choice (``explicit``, from Tab 2's 'External server
      model' combo) wins - sent verbatim, no network round-trip;
    * otherwise ``/v1/models`` is queried (Bearer key included) and, when
      EXACTLY ONE model is loaded, that model's exact ``id`` is used;
    * otherwise - none listed, several loaded, or the listing failed - the
      literal ``"default"`` is returned, which llama.cpp / Unsloth-studio
      servers route to the currently loaded model.

    Every decision is reported line-by-line through ``log_cb`` (when given)
    so the GUI log shows *which* model the batch will talk to.  Never
    raises; always returns a usable non-empty string.
    """
    def _log(line: str) -> None:
        if log_cb is not None:
            try:
                log_cb(line)
            except Exception:
                pass                 # logging must never break the flow

    exp = (explicit or "").strip()
    if exp:
        _log(f"[ext] using explicit server model: '{exp}'")
        return exp
    models, err = fetch_models(base_url, api_key, timeout=timeout)
    if err:
        _log(f"[ext] model listing failed ({err}) - sending model='" +
             f"{DEFAULT_MODEL_ID}' (the server's currently loaded model).")
        return DEFAULT_MODEL_ID
    loaded = loaded_models(models)
    if len(loaded) == 1:
        mid = str(loaded[0].get("id"))
        _log(f"[ext] /v1/models lists exactly one loaded model -> using "
             f"'{mid}' ({len(models)} model(s) registered server-side).")
        return mid
    if loaded:
        ids = ", ".join(str(m.get("id")) for m in loaded)
        _log(f"[ext] /v1/models lists {len(loaded)} loaded models "
             f"({ids}) - sending model='{DEFAULT_MODEL_ID}'; pick one "
             "explicitly in Tab 2 'External server model:' to be sure.")
        return DEFAULT_MODEL_ID
    _log(f"[ext] /v1/models lists {len(models)} model(s), none marked "
         f"loaded - sending model='{DEFAULT_MODEL_ID}'.")
    return DEFAULT_MODEL_ID


def build_payload(image_path: str, preset: CaptionPreset, system_prompt: str,
                  max_tokens: int | None = None,
                  settings=None, model_key: str | None = None,
                  model_id: str | None = None,
                  image_data_url: str | None = None) -> dict:
    """Build the /v1/chat/completions payload (see module docstring).

    ``model_key`` is the resolved registry key of the run target (Tab 2's
    explicit override or the Tab 1 selection in follow mode).  It drives the
    family gate below: profiles with ``supports_reasoning_kwargs=False``
    (GLM5.3, generic/unknown custom repos) get a plain sampling-only payload.
    Empty keys keep the legacy full Qwen-family behaviour for headless and
    pre-gating callers (see tests/test_smoke.py::test_payload).

    ``model_id`` is the wire value of the ``model`` field for EXTERNAL
    servers (a ``/v1/models`` id or the literal ``'default'`` - see
    :func:`resolve_external_model`).  When set it is sent in the payload AND
    replaces ``model_key`` as the family-gate input: the Qwen-only extras
    are gated on the EXTERNAL model's family (``detect_family`` heuristics),
    not on the local Tab 1/2 registry selection.

    ``image_data_url`` (Tab 2 "Upscale Before Analysis"): a pre-built base64
    data URL for an in-RAM upscaled copy of the image.  When given it is
    embedded verbatim instead of re-reading ``image_path``; all legacy
    callers omit it, so raw file bytes are used and payloads stay
    byte-identical to the pre-feature app.
    """
    params: GenerationParams = preset.params
    # Tab 2 live settings win over the preset when provided (AppContext's
    # InferenceSettings); this keeps server-side --reasoning-budget and the
    # request-side thinking flags consistent with what the user picked.
    # ``-1`` is the app-wide "unlimited" sentinel: never forward a negative
    # cap to llama-server (the sampler would reject it) - just omit the
    # override so the server default applies.
    budget_raw = getattr(settings, "reasoning_budget", None)
    if budget_raw is None:
        budget_raw = params.reasoning_budget
    try:
        reasoning_budget = int(budget_raw or 0)
    except (TypeError, ValueError):
        reasoning_budget = 0
    unlimited = reasoning_budget < 0
    # Effort level ("xhigh" | "medium" | "low"; "" = off).  Fall back to the
    # nearest named level for legacy configs that only persisted an int
    # budget, so old captioner_config.json files keep working unchanged.
    effort = (getattr(settings, "thinking_level", None) or "").strip().lower()
    if not effort:
        from app.core.models import get_model_profile
        prof = get_model_profile(getattr(settings, "run_model_key", "") or "")
        if unlimited:
            cand = next((l.key for l in prof.thinking_levels if l.budget < 0),
                        "")
        else:
            pairs = [(abs(l.budget - reasoning_budget), l.key)
                     for l in prof.thinking_levels if l.key and l.budget > 0]
            cand = min(pairs)[1] if pairs else ""
        # Only trust the derived key when the budget actually matches it;
        # otherwise leave effort empty (pure numeric-cap behaviour).
        lvl = next((l for l in prof.thinking_levels if l.key == cand), None)
        if lvl is not None and (lvl.budget == reasoning_budget or
                                (unlimited and lvl.budget < 0)):
            effort = cand
    enable_flag = getattr(settings, "enable_thinking", None)
    if enable_flag is None:
        # No Tab 2 settings (headless / legacy callers): honour the preset.
        enable_thinking = bool(params.enable_thinking) or reasoning_budget > 0
    else:
        enable_thinking = bool(enable_flag) and (bool(effort) or unlimited
                                                 or reasoning_budget > 0)
    preserve = bool(getattr(settings, "preserve_thinking", True))
    payload: dict = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": preset.user_prompt},
                # Tab 2 upscaling: a pre-built (upscaled) data URL wins;
                # otherwise the original file bytes, exactly as before.
                {"type": "image_url",
                 "image_url": {"url": (image_data_url
                                       if image_data_url is not None
                                       else image_to_data_url(image_path))}},
            ]},
        ],
        "temperature": float(getattr(settings, "temperature",
                                     params.temperature)),
        "top_p": float(getattr(settings, "top_p", params.top_p)),
        # Explicit argument wins; then the live Tab 2 'Max Tokens' spinbox
        # (settings.max_tokens); finally the per-preset GenerationParams
        # default.  Before the GUI control existed this line was
        # ``max_tokens or params.max_tokens`` only, so the saved/loaded
        # setting could never take effect from the interface.
        "max_tokens": int(max_tokens or getattr(settings, "max_tokens", None)
                          or params.max_tokens),
        "stream": True,
    }
    # llama-server accepts sampling extras at top level of the request body.
    payload["top_k"] = int(getattr(settings, "top_k", params.top_k))
    payload["min_p"] = float(getattr(settings, "min_p", params.min_p))
    # Presence / repetition penalties (Tab 2 spinboxes win; per-preset default
    # otherwise).  Both default to llama.cpp's own values, so an untouched
    # config produces an identical request to before.
    payload["presence_penalty"] = float(
        getattr(settings, "presence_penalty", params.presence_penalty))
    payload["repetition_penalty"] = float(
        getattr(settings, "repetition_penalty", params.repetition_penalty))
    # EXTERNAL servers only: the resolved /v1/models id (or literal
    # 'default') travels as the standard OpenAI 'model' field.  Local
    # llama-server runs ignore it (one loaded model), so it is omitted
    # entirely when not set - legacy payloads stay byte-identical.
    if model_id:
        payload["model"] = str(model_id)

    # ---- family gate (generalized model support, plan §4.5) --------------
    # Non-Qwen families must NOT receive Qwen-only extras: sending
    # enable_thinking / reasoning_effort / preserve_thinking into a foreign
    # Jinja chat template is at best silently ignored and at worst a hard
    # error.  Empty/legacy keys resolve to the neutral profile whose flag
    # defaults to True, so headless callers keep today's payload exactly.
    from app.core.models import get_model_profile, profile_for_model_name
    if model_id:
        # EXTERNAL server: the run target is the model id sent on the wire,
        # not the local registry selection - gate on its detected family.
        _prof = profile_for_model_name(model_id)
    else:
        _key = model_key or getattr(settings, "run_model_key", "") or ""
        _prof = get_model_profile(_key)
    if not _prof.supports_reasoning_kwargs:
        return payload                     # plain sampling payload only

    # ---- Qwen3.8-family reasoning block ----------------------------------
    # chat_template_kwargs reaches the Jinja template only when the server
    # runs with --jinja (see server.build_args / llamacpp.probe_capabilities).
    ctk: dict = {"enable_thinking": enable_thinking,
                 "preserve_thinking": preserve}
    # reasoning_effort ONLY when thinking is on: the Qwen3.8 template rejects
    # an effort value combined with enable_thinking=False, and 'high' is
    # never emitted (the Unsloth patch silently aliases it onto 'xhigh').
    if enable_thinking and effort:
        ctk["reasoning_effort"] = effort
    payload["chat_template_kwargs"] = ctk
    if enable_thinking:
        if reasoning_budget > 0:
            # Top-level hard cap enforced by the llama-server sampler: it
            # force-closes  once N thinking tokens were generated.
            payload["reasoning_budget"] = reasoning_budget
        # unlimited (-1): omit the cap entirely -> server default applies.
    else:
        # Thinking off: budget 0 forces the model to close any reasoning
        # block immediately instead of relying on the template alone.
        payload["reasoning_budget"] = 0
    return payload


class CaptionClient:
    """Thin async wrapper around one running llama-server instance.

    Works against the local server AND external OpenAI-compatible servers
    (see the module docstring): the base URL accepts the bare origin, a
    trailing ``/v1`` or a full ``/v1/chat/completions`` paste - all are
    normalised away, and the Bearer key (when set) is sent with every
    request, including the ``/v1/models`` listing.
    """

    def __init__(self, base_url: str, timeout: float = 600.0,
                 api_key: str | None = None):
        # Normalise any pasted spelling (bare origin / .../v1 /
        # .../v1/chat/completions) down to the bare base this client
        # appends its own /v1/... paths to.
        self.base_url = normalize_base_url(base_url)
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

    def fetch_models(self, timeout: float = 10.0):
        """``/v1/models`` listing for THIS client's server (sync).

        Returns ``(models, error)`` - see :func:`fetch_models`.  Synchronous
        on purpose: the GUI calls it from a QThread, never the event loop.
        """
        return fetch_models(self.base_url, self.api_key, timeout=timeout)

    async def caption(self, image_path: str, preset: CaptionPreset,
                      system_prompt: str, on_chunk: ChunkCb = None,
                      cancel_event=None, settings=None,
                      model_key: str | None = None,
                      model_id: str | None = None,
                      log_cb: LogCb = None) -> str:
        """Stream a chat completion and return the full assistant text.

        ``model_key`` is the resolved registry key of the run target; it gates
        family-specific payload extras (models with
        ``supports_reasoning_kwargs=False`` get sampling-only payloads).

        ``model_id`` (EXTERNAL servers only) is the wire value of the
        ``model`` field - a ``/v1/models`` id or the literal ``'default'``
        (see :func:`resolve_external_model`).  Omitted for local servers,
        which use their one loaded model anyway.

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

        Tab 2 upscaling (``settings.upscale_enabled``): when active with a
        factor > 1 on either axis, the image is upscaled IN RAM and its
        re-encoded bytes are embedded instead of the original file; any
        failure falls back to the original.  See :mod:`app.core.upscale`.
        """
        if log_cb is None and not _verbose_enabled:
            log_cb = None    # stay completely quiet unless asked
        # ---- Tab 2 "Upscale Before Analysis" (in RAM, per request) --------
        # Resizes the image with the selected method/factors and embeds the
        # re-encoded result in the payload; the source file is never touched.
        # Off / 1x/1x: nothing happens and the payload stays byte-identical
        # to the pre-feature behaviour.
        image_data_url: str | None = None
        if settings is not None and getattr(settings, "upscale_enabled", False):
            _method = str(getattr(settings, "upscale_method", "lanczos"))
            try:
                _sx = int(getattr(settings, "upscale_scale_x", 2) or 1)
                _sy = int(getattr(settings, "upscale_scale_y", 2) or 1)
            except (TypeError, ValueError):
                _sx = _sy = 1
            if _sx > 1 or _sy > 1:
                from app.core.upscale import upscale_to_data_url
                res = upscale_to_data_url(image_path, _method, _sx, _sy)
                if res is not None:
                    image_data_url, report = res
                    _emit(log_cb, f"[upscale] {image_path}: {report} "
                                  "(in RAM; original file untouched)")
                else:
                    _emit(log_cb, f"[upscale] {image_path}: upscaling "
                                  "unavailable or failed (Pillow missing / "
                                  "undecodable image?) - sending the "
                                  "original image")
        url = f"{self.base_url}/v1/chat/completions"
        payload = build_payload(image_path, preset, system_prompt,
                                settings=settings, model_key=model_key,
                                model_id=model_id,
                                image_data_url=image_data_url)

        # ---- verbose: dump the exact conversation being sent ----------------
        _emit(log_cb, f"[llama] >>> POST {url}  "
                      f"(model '{payload.get('model', '(server default)')}', "
                      f"preset '{preset.display_name}', image "
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
