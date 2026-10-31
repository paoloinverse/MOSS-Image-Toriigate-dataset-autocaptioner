"""Model registry + GGUF/HuggingFace helpers.

Supported models (per spec):

1. Primary  : ``Minthy/ToriiGate-0.5`` – GGUFs hosted at
              ``DraconicDragon/ToriiGate-0.5-GGUF``.  Quants: BF16, Q8_0.
              mmproj: ``ToriiGate-0.5-Q8_0.mmproj.gguf``.
2. Alternate: ``huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF``.
              Quants: Q8_0, Q6_K, UD-IQ4_XS.  The multimodal projector is
              auto-detected from the repo file list (any ``mmproj*.gguf``
              whose quant matches the chosen base quant when possible).

Storage locations (user-selectable, persisted in ``app_settings.json``):

* ``default``   – the app cache folder ``~/.cache/app_models/`` (overridable
                  with the ``APPCAP_CACHE_DIR`` env var).  Disk-space is
                  checked before any bytes hit the wire.
* ``unsloth``   – an Unsloth Desktop compatible folder supplied by the user.
                  The directory is auto-created when missing; already-present
                  GGUFs are detected and re-used instead of re-downloading.
                  Unsloth Desktop stores its downloads under ``<root>/models/``
                  (and legacy ``<root>/llama-cpp/``), so those subfolders are
                  scanned automatically.

Error-management policy for this module: **never** attempt huge allocations
or blind multi-GB downloads inside tests or background threads without an
explicit user action.  All network/disk failures surface as
:class:`AppError` carrying a verbose, human-readable report that
the GUI can show verbatim (reason + remediation hints).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Structured, verbose error type shared by model/backend operations
# ---------------------------------------------------------------------------
class AppError(Exception):
    """Actionable error for downloads / verification / architecture lookup.

    ``report()`` renders a multi-line message designed for the GUI log and
    message boxes: what failed, why, and concrete steps to fix it.
    """

    prefix = "[models]"      # subclasses may override the log tag

    def __init__(self, message: str, detail: str = "", suggestions=None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.suggestions: list[str] = list(suggestions or [])

    def report(self) -> str:
        lines = [f"{self.prefix} ERROR: {self.message}"]
        if self.detail:
            lines.append(f"    reason : {self.detail}")
        for s in self.suggestions:
            lines.append(f"    fix    : {s}")
        return "\n".join(lines)

    @classmethod
    def classify(cls, exc: Exception, context: str = "") -> "AppError":
        """Map any low-level exception to a typed, verbose error."""
        import socket
        name = type(exc).__name__
        text = f"{name}: {exc}"
        lower = text.lower()
        base_detail = f"{context} -> {text}" if context else text

        net_hints = [
            "Check your internet connection (ping a host, try a browser).",
            "If behind a proxy, export HF_ENDPOINT / HTTPS_PROXY before launch.",
            "Retry later - Hugging Face Hub may be rate-limiting or down.",
        ]
        if isinstance(exc, ImportError) or "modulenotfound" in lower or "no module" in lower:
            return cls("Missing dependency",
                       base_detail,
                       ["Install requirements: pip install -r requirements.txt",
                        "The 'huggingface_hub' package is needed for downloads."])
        if isinstance(exc, (ConnectionError, socket.gaierror, TimeoutError)) or \
                any(k in lower for k in ("connection", "timed out", "dns",
                                         "network", "ssl", "proxy")):
            return cls("Network failure while contacting Hugging Face Hub",
                       base_detail, net_hints)
        if "401" in lower or "unauthorized" in lower or "gated" in lower or "403" in lower:
            return cls("Repository access denied (gated/private repo)",
                       base_detail,
                       ["Accept the model licence on huggingface.co in a browser.",
                        "Run 'hf auth login' (or set HF_TOKEN) for gated repos."])
        if "404" in lower or "not found" in lower or "does not exist" in lower:
            return cls("File or repository not found on the Hub",
                       base_detail,
                       ["The repo layout may have changed; verify filenames in the "
                        "registry (app/core/models.py)."])
        if isinstance(exc, OSError) and (
                "space" in lower or exc.errno == errno_ENOSPC()):
            free = 0
            try:
                free = shutil.disk_usage(default_cache_dir()).free // (1024 ** 3)
            except OSError:
                pass
            return cls("Insufficient disk space for download",
                       base_detail,
                       [f"Free space on '{default_cache_dir()}': ~{free} GiB.",
                        "Delete unused GGUFs there, or switch the storage "
                        "location to an Unsloth-desktop compatible folder on a "
                        "larger volume (Tab 1 -> Storage Location)."])
        return cls("Model operation failed", base_detail,
                   ["Re-run 'Download & Verify' in Tab 1.",
                    "If the problem persists, copy this whole message into the "
                    "bug report."])


# Backwards-compatible alias (older code/tests may still import this name).
ModelOperationError = AppError


class StorageLocationError(AppError):
    """Raised when the configured model-storage location is unusable."""

    prefix = "[storage]"


def errno_ENOSPC() -> int:
    import errno
    return errno.ENOSPC


# ---------------------------------------------------------------------------
# Persistent application settings (~/.config/moss-captioner/app_settings.json)
# ---------------------------------------------------------------------------
SETTINGS_PATH = Path(os.environ.get(
    "APPCAP_SETTINGS",
    os.path.join(os.environ.get("XDG_CONFIG_HOME",
                                os.path.expanduser("~/.config")),
                 "moss-captioner", "app_settings.json")))

#: Default filename of the optional startup configuration override.  When
#: this file exists (see :func:`find_startup_config`) it is imported at
#: launch and OVERRIDES the hardwired preset defaults - see
#: ``AppContext._load_startup_config_override`` in app/context.py.
STARTUP_CONFIG_NAME = "captioner_config.json"


def find_startup_config(name: str = STARTUP_CONFIG_NAME) -> str | None:
    """Locate the optional startup config file for this launch.

    Search order (first existing file wins):

    1. ``$APPCAP_CONFIG`` — explicit path; either the JSON file itself or a
       directory containing *name*.
    2. *name* in the current working directory.
    3. *name* in the project root (the folder holding ``run.sh`` / the ``app``
       package) — where Tab 3's 'Export full application config' default
       filename lands when exported next to the app.

    Returns the path string, or ``None`` when no candidate exists.
    """
    env = os.environ.get("APPCAP_CONFIG", "").strip()
    candidates: list[Path] = []
    if env:
        p = Path(env).expanduser()
        candidates.append(p if p.suffix else p / name)
    proj_root = Path(__file__).resolve().parent.parent.parent
    candidates += [Path.cwd() / name, proj_root / name]
    for p in candidates:
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return None


def load_app_settings() -> dict:
    """Read the persisted settings; corrupt/unreadable files never crash.

    Returns a dict; on any problem an empty dict is returned and the issue is
    printed to stderr (surfaced in the GUI log by callers).
    """
    try:
        if SETTINGS_PATH.is_file():
            with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
            print(f"[settings] {SETTINGS_PATH} does not contain a JSON object; "
                  "ignoring it.", flush=True)
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[settings] could not read {SETTINGS_PATH}: "
              f"{type(exc).__name__}: {exc}", flush=True)
    return {}


def save_app_settings(data: dict) -> None:
    """Atomically persist settings; failures raise StorageLocationError."""
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, SETTINGS_PATH)
    except OSError as exc:
        raise StorageLocationError(
            "Could not save application settings",
            detail=f"{type(exc).__name__}: {exc} (path: {SETTINGS_PATH})",
            suggestions=["Check permissions of ~/.config/moss-captioner/",
                         "Ensure the partition holding it has free space."]
        ) from exc


def default_cache_dir() -> Path:
    """Default app-managed model cache (env-overridable for portability)."""
    return Path(os.environ.get(
        "APPCAP_CACHE_DIR",
        os.path.expanduser("~/.cache/app_models")))


#: Deprecated alias - always resolves to the *default* cache.  Runtime code
#: must go through ``STORAGE`` (see StorageManager) to honour the user's
#: selected storage location.
CACHE_DIR = default_cache_dir()

# Quantization name -> approximate bits per weight (for size estimation when
# the remote file metadata is unavailable).
QUANT_BITS = {
    "BF16": 16.0, "F16": 16.0, "Q8_0": 8.5, "Q6_K": 6.6,
    "Q5_K_M": 5.7, "Q5_K": 5.77, "Q4_K_M": 4.8, "Q4_K": 4.83,
    "UD-IQ4_XS": 4.4, "IQ4_XS": 4.25,
    "Q3_K_M": 3.9, "Q3_K": 3.96, "Q2_K": 2.6,
}


# ---------------------------------------------------------------------------
# Live Hugging Face Hub lookups (file names are DISCOVERED, never composed)
# ---------------------------------------------------------------------------
#: repo_id -> {"files": [...], "sizes": {...}, "ts": monotonic-time}
_REPO_LOOKUP_CACHE: dict[str, dict] = {}
REPO_LOOKUP_TTL = 300.0          # seconds; refresh on demand afterwards


def _lookup_entry(repo_id: str, force: bool = False) -> dict:
    """Fetch (and cache) the live file list + sizes of an HF GGUF repo.

    Filenames come verbatim from the Hub - the app no longer composes or
    restricts them.  Raises AppError (via :meth:`AppError.classify`) when the
    repo cannot be listed so callers can show a verbose report.
    """
    ent = _REPO_LOOKUP_CACHE.get(repo_id)
    if ent is not None and not force:
        import time as _t
        if _t.monotonic() - ent["ts"] < REPO_LOOKUP_TTL:
            return ent
    try:
        from huggingface_hub import HfApi
        info = HfApi().repo_info(repo_id, files_metadata=True)
        files = [s.rfilename for s in (info.siblings or [])]
        sizes = {s.rfilename: int(s.size or 0) for s in (info.siblings or [])}
    except ImportError as exc:
        raise AppError.classify(exc, context=f"listing '{repo_id}'") from exc
    except Exception as exc:
        raise AppError.classify(exc, context=f"listing '{repo_id}'") from exc
    import time as _t
    ent = {"files": files, "sizes": sizes, "ts": _t.monotonic()}
    _REPO_LOOKUP_CACHE[repo_id] = ent
    return ent


#: Quant span anchored to the END of a filename: optional family prefix
#: (GSQ-RCO / UD-DW / UD), base quant (BF16/F16/F32, Q4_K, IQ3_XS, ...) and an
#: optional variant suffix (_P / _XL / _L / -MTP).  The anchor matters: the
#: old pattern stopped before such suffixes, so six '...-Q*_K_P.gguf' files in
#: a HauhauCS Qwen3.6 repo all parsed to the bare tag 'P', and selection (which
#: keys on the tag) resolved to whichever of them came first.
_QUANT_TAIL_RE = re.compile(
    r"(?:^|[._-])"
    r"((?:GSQ-RCO[-_]?|UD-DW[-_]?|UD[-_]?)?"
    r"(?:BF16|F16|F32"
    r"|Q\d+(?:_\d)?(?:_[A-Z]{1,4})?"
    r"|I[MQ]\d+(?:_[A-Z]{1,4})?)"
    r"(?:[-_]?(?:XL|P|MTP|L|M|S))?)",
    re.IGNORECASE)


def _quant_tag_from_name(name: str) -> str:
    """Extract the quant tag embedded in a real Hub filename.

    Primary rule: a quant span anchored to the END of the stem - family
    prefix (``GSQ-RCO`` / ``UD-DW`` / ``UD``), base quant (``Q8_0``,
    ``Q4_K_P``, ``IQ3_XS`` ...) and optional variant suffix (``_P``, ``_XL``,
    ``_L``, ``-MTP``) - so suffixed quants keep their full identity instead
    of collapsing to a bare 'P'/'XL'/'L' shared by several files.  When the
    quant sits mid-name, a left-to-right scan for the first GGUF-quant-like
    underscore token is used; last resort is the last underscore token.

    This is *parsing* an existing name, not composing one.  Tags are a
    best-effort display key - :func:`_disambiguate_quant_tags` guarantees they
    stay unique within one repository listing.
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    # Tail-anchored match first (standard names end with the quant), then a
    # mid-name search, then the token scan / last-token fallbacks.
    m = re.search(_QUANT_TAIL_RE.pattern + r"$", stem, re.IGNORECASE) \
        or _QUANT_TAIL_RE.search(stem)
    if m:
        return m.group(1).upper()
    toks = stem.split("_")
    for i, t in enumerate(toks):
        if re.fullmatch(r"(?:UD-?)?(?:BF16|F16|F32|Q\d(\d)?|[IQ][M]?Q?\d+[A-Z]*|"
                        r"[A-Z]{2,4})", t, re.IGNORECASE) and \
                re.search(r"\d|BF16|F16", t):
            return "_".join(toks[i:]).upper()
    return (toks[-1] if toks else stem).upper()


def _disambiguate_quant_tags(
        entries: "list[tuple[str, str, int]]"
) -> "list[tuple[str, str, int]]":
    """Make quant tags UNIQUE within one repository listing.

    ``entries`` are the ``(filename, quant, size_bytes)`` triples from
    :func:`discover_repo_ggufs`.  The best-effort parser can assign one tag to
    several files (e.g. two '...-Q4_K.gguf' uploads with different model
    names, or any scheme this grammar does not know).  Selection, persistence
    and download planning all key on the tag, so a collision would silently
    resolve to the FIRST file carrying it.  Colliding tags are therefore
    extended leftwards over the filename's own ``[_-]`` units - 'K_P' ->
    'Q2_K_P', 'XL' -> 'Q2_K_XL' - until unique; candidates that contain a
    digit are preferred (they read as quant names) and the full stem is the
    last resort.  The first file of a colliding group keeps its parsed tag,
    later ones extend.  Pure function of the listing: completely deterministic
    and repo-scoped, so it copes with any future naming scheme.
    """
    n = len(entries)
    if n < 2:
        return entries
    tags = [t for _fn, t, _sz in entries]

    def _unique_among_others(cand: str, i: int) -> bool:
        up = cand.upper()
        return all(up != tags[j].upper() for j in range(n) if j != i)

    for i in range(n):
        fn, tag, _sz = entries[i]
        if tags.count(tag) == 1:
            continue                          # unique as-is
        seen_earlier = sum(1 for j in range(i)
                           if tags[j].upper() == tag.upper())
        if seen_earlier == 0:
            continue                          # first of the group: keep it
        stem = os.path.splitext(os.path.basename(fn))[0]
        su, tu = stem.upper(), tag.upper()
        units: list[tuple[str, str]] = []     # (leading separator, text)
        for u in re.finditer(r"(?:^|([_.-]))([^_.-]+)", stem):
            units.append((u.group(1) or "", u.group(2)))
        cands: list[str] = []
        if tu and su.endswith(tu):
            for k in range(1, len(units) + 1):
                span = "".join(s + t for s, t in units[-k:])
                span = span.lstrip("_.-")
                if span and span.upper() != tag.upper():
                    cands.append(span)
        new_tag = None
        for cand in cands:                    # digit-bearing candidates first
            if re.search(r"\d", cand) and _unique_among_others(cand, i):
                new_tag = cand
                break
        if new_tag is None:
            for cand in cands:
                if _unique_among_others(cand, i):
                    new_tag = cand
                    break
        if new_tag is None:
            new_tag = stem                    # last resort: full stem
        tags[i] = new_tag
    return [(entries[i][0], tags[i], entries[i][2]) for i in range(n)]


def discover_repo_ggufs(repo_id: str, force_refresh: bool = False
                        ) -> "list[tuple[str, str, int]]":
    """Live ``(filename, quant, size_bytes)`` list of main GGUFs in a repo.

    ``mmproj`` files are excluded (they are companions, not base models).
    Every ``*.gguf`` the repo publishes is offered - there are **no
    restrictions on allowed quants**.  Returns [] only when the listing
    fails after the offline fallback (callers should surface it).
    """
    try:
        ent = _lookup_entry(repo_id, force=force_refresh)
    except AppError:
        return []
    out = []
    for f in ent["files"]:
        low = f.lower()
        if not low.endswith(".gguf") or "mmproj" in low:
            continue
        out.append((f, _quant_tag_from_name(f), ent["sizes"].get(f, 0)))
    # Tags must be unique per repo: selection/persistence/download planning
    # key on them, and a shared tag would resolve to the first file only.
    return _disambiguate_quant_tags(out)


def find_live_filename(repo_id: str, wanted: str) -> "str | None":
    """Resolve a hardwired/default filename to its real counterpart on the
    Hub (case-insensitive match over the live listing; exact match wins).

    Used so legacy registry seeds (e.g. the old ``ToriiGate-0.5-BF16.gguf``)
    automatically adopt renamed files (e.g. ``...-bf16.gguf``) instead of
    404-ing mid-download.  Returns ``wanted`` unchanged when offline.
    """
    try:
        files = _lookup_entry(repo_id)["files"]
    except AppError:
        return wanted
    if wanted in files:
        return wanted
    low = wanted.lower()
    for f in files:
        if f.lower() == low:
            return f
    return wanted


def resolve_mmproj_live(repo_id: str, quant: str,
                        force_refresh: bool = False) -> "str | None":
    """Pick the projector companion for ``quant`` from the LIVE repo listing.

    Priority: same-quant mmproj > F16/BF16 mmproj > any single mmproj.
    Returns None when the repo has none or the Hub is unreachable.
    """
    try:
        ent = _lookup_entry(repo_id, force=force_refresh)
    except AppError:
        return None
    mmprojs = [f for f in ent["files"]
               if "mmproj" in f.lower() and f.lower().endswith(".gguf")]
    if not mmprojs:
        return None
    want = (quant or "").upper().replace("-", "")
    for m in mmprojs:
        mu = m.upper().replace("-", "")
        if want and want in mu:
            return m
    for m in mmprojs:
        if "F16" in m.upper() or "BF16" in m.upper():
            return m
    return mmprojs[0]


# ---------------------------------------------------------------------------
# Arbitrary Hugging Face repo ids (generalized model support)
# ---------------------------------------------------------------------------
#: A valid plain repo id: exactly two segments of letters/digits/._- .
REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def normalize_repo_id(raw: str) -> str:
    """Strip whitespace and a pasted Hub URL down towards ``org/name``.

    'https://huggingface.co/org/name' (scheme optional, optional trailing
    path segments kept for the validator to reject with a hint) becomes
    'org/name...'.  Anything else is returned stripped but otherwise
    untouched - :func:`validate_repo_id` produces the actionable error.
    """
    s = (raw or "").strip()
    if "://" in s:
        s = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", s)
    s = re.sub(r"^www\.", "", s, flags=re.IGNORECASE)
    m = re.match(r"(?:huggingface\.co/)?(.*)$", s, re.IGNORECASE)
    if m:
        s = m.group(1)
    return s.strip()


def validate_repo_id(raw: str) -> str:
    """Validate a Hugging Face repo id; returns the normalised ``org/name``.

    Raises :class:`AppError` with remediation hints for anything that is not
    exactly two non-empty segments of letters/digits/._-  (pasted URLs are
    stripped first, then rejected with a 'did you mean' hint when their path
    starts with a plausible repo id).
    """
    rid = normalize_repo_id(raw)
    if REPO_ID_RE.fullmatch(rid):
        return rid
    parts = [p for p in rid.split("/")]
    detail_bits: list[str] = []
    if len(parts) != 2:
        detail_bits.append(f"expected exactly two segments 'org/name' "
                           f"(got {len([p for p in parts if True])})")
    elif not all(p.strip() for p in parts):
        detail_bits.append("empty segment (no leading/trailing slash or double slashes)")
    else:
        detail_bits.append("illegal characters "
                           "(allowed: letters, digits, '.', '_', '-')")
    suggestions = ["Use the plain repository id shown on huggingface.co, e.g. "
                   "'DraconicDragon/ToriiGate-0.5-GGUF'.",
                   "Branch/revision syntax (org/name@rev) is not supported - "
                   "plain repo ids only."]
    if "://" in (raw or ""):
        head = normalize_repo_id(raw).split("/")[:2]
        if len(head) == 2 and all(head):
            hint = head[0] + "/" + head[1]
            suggestions.append(
                f"Did you mean '{hint}'? (URLs are not needed - paste the repo id)")
    raise AppError("Invalid Hugging Face repository id",
                   f"'{raw}' is not a valid 'org/name' repo id - "
                   + "; ".join(detail_bits),
                   suggestions)


def detect_family(repo_id: str, hf_config: dict | None = None) -> str:
    """Auto-detect the model family of a repo id (pure, unit-testable).

    Ordered substring heuristics over the lowercased repo id - first match
    wins.  The Qwen3.8 flash/next variant is deliberately tested BEFORE plain
    'qwen…3.8' so Flash-Next repos get their own profile entry.  When the
    name is uninformative, ``config.json``'s ``model_type`` provides a hint;
    when nothing matches the repo falls back to the neutral ``generic``
    family (neutral profile - the model still runs with manual sampling).
    """
    rid = (repo_id or "").lower()

    def _qwen(minor: str) -> bool:
        return re.search(r"qwen[._-]?3[._-]?" + minor, rid) is not None

    if "torii" in rid:
        return "toriigate"
    if _qwen("8") and ("flash" in rid or "next" in rid):
        return "qwen3.8-flash-next"          # precedence over plain 3.8
    for fam, minor in (("qwen3.5", "5"), ("qwen3.6", "6"), ("qwen3.8", "8")):
        if _qwen(minor):
            return fam
    if re.search(r"glm[._-]?5[._-]?3", rid) is not None:
        return "glm5.3"

    # Name uninformative -> cross-check config.json model_type (if fetched).
    mt = str(((hf_config or {}).get("model_type") or "")).lower()
    if mt:
        for fam, pat in (("qwen3.8-flash-next",
                          r"(flash|next).*qwen.*3[._-]?8|qwen.*3[._-]?8.*(flash|next)"),
                         ("qwen3.5", r"qwen.*3[._-]?5"),
                         ("qwen3.6", r"qwen.*3[._-]?6"),
                         ("qwen3.8", r"qwen.*3[._-]?8"),
                         ("glm5.3", r"glm.*5[._-]?3|5[._-]?3.*glm")):
            if re.search(pat, mt):
                return fam
    return "generic"


# ---------------------------------------------------------------------------
# Per-model run-out-of-the-box profiles
# ---------------------------------------------------------------------------
#: Valid llama.cpp ``--cache-type-k/-v`` values surfaced in Tab 2 (incl. bf16).
KV_CACHE_TYPES = ("f16", "bf16", "q8_0", "fp8", "q4_0", "iq4_nl")


#: Hard token cap of the fine-grained Reasoning Budget control in Tab 2.
#: A level/budget of -1 means "unlimited": no server-side cap and no
#: per-request ``reasoning_budget`` override (see api_client.build_payload).
REASONING_BUDGET_MAX = 16384
#: Sentinel meaning "no cap" for every reasoning-budget knob in the app.
REASONING_BUDGET_UNLIMITED = -1


@dataclass(frozen=True)
class ThinkingLevel:
    """One named reasoning/thinking option for a model family.

    ``key`` is the exact value handed to the Qwen3.8 chat template as
    ``chat_template_kwargs.reasoning_effort`` ("" = thinking disabled).
    The Unsloth-patched template silently maps ``'high'`` onto ``'xhigh'``;
    exposing ``high`` here would hide that aliasing, so the selectable set
    mirrors exactly what the template supports distinctively:
    ``xhigh`` / ``medium`` / ``low``.
    """
    key: str                 # "" = disabled | "low" | "medium" | "xhigh"
    label: str               # GUI text
    budget: int              # reasoning-budget tokens (-1 = unlimited)


@dataclass(frozen=True)
class ModelProfile:
    """Run-out-of-the-box configuration preset for one supported model.

    Holds the model-specific defaults Tab 2 used to lack: recommended KV
    cache type, per-quant context hard limits (the KV-cache size cap),
    thinking-level options and sane sampling defaults.
    """
    max_context_by_quant: dict = field(default_factory=dict)
    default_cache_type: str = "q8_0"
    thinking_levels: tuple = ()
    default_thinking: str = ""
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    preserve_thinking_default: bool = True
    #: Whether this family accepts Qwen-style ``chat_template_kwargs``
    #: (enable_thinking / reasoning_effort / preserve_thinking) plus the
    #: top-level ``reasoning_budget`` in chat payloads.  False for non-Qwen
    #: families (GLM5.3, generic): api_client.build_payload() omits that whole
    #: block so a foreign Jinja template never sees Qwen-only kwargs.
    supports_reasoning_kwargs: bool = True
    note: str = ""


# Qwen3.8-family (Unsloth-patched chat template) effort levels.  Ordered
# from most to least thinking so the combo reads Deep -> Brief -> Off;
# budgets *descend* towards "Off" (-1 = unlimited is the largest amount of
# thinking, 0 disables it).  The persisted-int nearest-match logic in
# api_client._resolve_effort special-cases the -1 sentinel, so tuple order
# carries no meaning there beyond GUI display.
_QWEN_THINKING_LEVELS = (
    ThinkingLevel("xhigh", "Deep / Maximum  (xhigh — unrestricted thinking)",
                  REASONING_BUDGET_UNLIMITED),
    ThinkingLevel("medium", "Standard / Balanced  (medium — ~4K reasoning tokens)",
                  4096),
    ThinkingLevel("low", "Brief / Fast  (low — ~1K reasoning tokens)", 1024),
    ThinkingLevel("", "Off (no thinking)", 0),
)

#: Static *family* defaults; ``max_context_by_quant`` is only consulted for
#: quants whose live context cap is unknown (per-quant entries are now
#: discovered dynamically via :func:`model_profile_for`).
_MODEL_FAMILY_DEFAULTS: dict[str, ModelProfile] = {
    "toriigate": ModelProfile(
        max_context_by_quant={},
        default_cache_type="q8_0",
        thinking_levels=_QWEN_THINKING_LEVELS,
        default_thinking="low",
        temperature=1.0, top_p=0.95, top_k=40, min_p=0.05,
        preserve_thinking_default=True,
        note="ToriiGate-0.5 is a Qwen3-VL-based captioner (Unsloth-patched "
             "Qwen3.8-family chat template): reasoning_effort xhigh/medium/"
             "low + enable_thinking/preserve_thinking go per-request via "
             "chat_template_kwargs (server needs --jinja); the budget maps "
             "to --reasoning-budget / per-request reasoning_budget."),
    "qwen3.8-27b": ModelProfile(
        max_context_by_quant={},
        default_cache_type="q8_0",
        thinking_levels=_QWEN_THINKING_LEVELS,
        default_thinking="xhigh",
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        preserve_thinking_default=True,
        note="Qwen3.8-family hybrid thinker: enable_thinking / "
             "reasoning_effort ('xhigh' | 'medium' | 'low'; 'high' is only a "
             "silent alias of 'xhigh' and never sent) / preserve_thinking go "
             "in chat_template_kwargs per request (--jinja required); the "
             "token cap uses --reasoning-budget on the server plus a "
             "top-level reasoning_budget override per request."),
    # ---- generalized open-weight families (auto-detected repo ids) --------
    # Detected via detect_family() for ANY user-added Hugging Face repo; the
    # values marked 'verify' must be confirmed against each family's model
    # card / chat template before shipping defaults are considered final.
    "qwen3.5": ModelProfile(
        thinking_levels=_QWEN_THINKING_LEVELS, default_thinking="xhigh",
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        preserve_thinking_default=True,
        note="Qwen3.5-family hybrid thinker: same chat_template_kwargs "
             "contract as Qwen3.8 (verify against the model card)."),
    "qwen3.6": ModelProfile(
        thinking_levels=_QWEN_THINKING_LEVELS, default_thinking="xhigh",
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        preserve_thinking_default=True,
        note="Qwen3.6-family hybrid thinker: same chat_template_kwargs "
             "contract as Qwen3.8 (verify against the model card)."),
    # Plain (non-Flash) Qwen3.8 repos auto-detected by detect_family(): they
    # must land here - NOT in the neutral fallback - or acceptance criterion 3
    # breaks (no thinking levels, T1.0 sampling instead of the family's T0.6).
    "qwen3.8": ModelProfile(
        thinking_levels=_QWEN_THINKING_LEVELS, default_thinking="xhigh",
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        preserve_thinking_default=True,
        note="Qwen3.8-family hybrid thinker: same chat_template_kwargs "
             "contract as Qwen3.5/3.6 (verify against the model card)."),
    "qwen3.8-flash-next": ModelProfile(
        thinking_levels=_QWEN_THINKING_LEVELS, default_thinking="xhigh",
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        preserve_thinking_default=True,
        note="Qwen3.8-Flash/Next: Qwen3.8 chat_template_kwargs contract "
             "(verify; consider a lower default budget if the model card says "
             "so)."),
    "glm5.3": ModelProfile(
        thinking_levels=(), default_thinking="",
        temperature=0.7, top_p=0.95, top_k=40, min_p=0.05,
        preserve_thinking_default=True,
        supports_reasoning_kwargs=False,
        note="GLM5.3: conservative sampling; NO Qwen chat_template_kwargs / "
             "reasoning_budget are sent (foreign chat template)."),
    "generic": ModelProfile(
        thinking_levels=(), default_thinking="",
        supports_reasoning_kwargs=False,
        note="Unknown model family: neutral defaults, user-set sampling only; "
             "no Qwen-specific payload extras."),
}

#: Rough VRAM budget (GiB) allowed to the KV cache when deriving per-quant
#: context caps from the live file size - keeps big quants from OOMing while
#: letting small ones stretch further.  ToriiGate's published caps
#: (BF16->16K @ ~1 GiB, Q8_0->32K @ ~0.6 GiB) match this budget closely.
_KV_BUDGET_GIB = 0.8


def _arch_from_config_json(data: dict) -> "ArchInfo | None":
    """Extract ArchInfo fields from a parsed HF ``config.json`` dict."""
    try:
        text_cfg = data.get("text_config") or {}
        hidden = int(text_cfg.get("hidden_size") or data.get("hidden_size") or 0)
        layers = int(text_cfg.get("num_hidden_layers")
                     or data.get("num_hidden_layers") or 0)
        kv_heads = int(text_cfg.get("num_key_value_heads")
                       or data.get("num_key_value_heads") or 0)
        head_dim = int(text_cfg.get("head_dim") or 0)
        if head_dim == 0 and hidden and kv_heads:
            head_dim = hidden // kv_heads
        if hidden and layers and kv_heads and head_dim:
            return ArchInfo(hidden_size=hidden, num_layers=layers,
                            num_kv_heads=kv_heads, head_dim=head_dim)
    except (TypeError, ValueError):
        pass
    return None


def _hf_config_fields(spec: "ModelSpec") -> "ArchInfo | None":
    """Fetch arch fields from the model repo's config.json (cached)."""
    import time as _t
    ent = _CONFIG_ARCH_CACHE.get(spec.key)
    if ent is not None and _t.monotonic() - ent[0] < REPO_LOOKUP_TTL:
        return ent[1]
    arch = None
    repos = [r for r in (spec.hf_config_repo, spec.source_repo) if r]
    for repo in repos:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=repo, filename="config.json",
                                cache_dir=str(default_cache_dir() / "hf_meta"))
            with open(p, "r", encoding="utf-8") as fh:
                arch = _arch_from_config_json(json.load(fh))
        except Exception:
            arch = None
        if arch is not None:
            break
    _CONFIG_ARCH_CACHE[spec.key] = (_t.monotonic(), arch)
    return arch


#: spec.key -> (monotonic ts, ArchInfo | None)
_CONFIG_ARCH_CACHE: dict[str, tuple] = {}


_QUANT_SUFFIX_RE = re.compile(r"[-_](?:XL|P|MTP|L|M|S)$")


def _quant_bits_for(quant: str, filename: str) -> float:
    """Bits per weight for size estimates of a (possibly suffixed) quant.

    Tries the exact tag, then the tag without a variant suffix ('Q4_K_P' ->
    'Q4_K'), and repeats both with the tag re-parsed from the filename.  So
    K-P / -XL / -L uploads estimate at their base quant instead of the 8.5-bit
    unknown-quant default.  Unknown quants still default to 8.5 (conservative).
    """
    cands: list[str] = []
    for raw in (quant, _quant_tag_from_name(filename)):
        u = raw.upper()
        if u not in cands:
            cands.append(u)
        stripped = _QUANT_SUFFIX_RE.sub("", u)
        if stripped != u and stripped not in cands:
            cands.append(stripped)
    for c in cands:
        if c in QUANT_BITS:
            return QUANT_BITS[c]
    return 8.5


def _live_max_context(spec: "ModelSpec", f: "ModelFile") -> int:
    """Derive a per-quant context cap from the LIVE file size + arch math.

    No hardwired quant list: any GGUF the repo publishes gets a cap computed
    from its real byte size (weights leave ``_KV_BUDGET_GIB`` for the KV
    cache inside the profile's total VRAM budget).  Falls back to the family
    static entry, then to nothing (global cap applies).
    """
    bits = _quant_bits_for(f.quant, f.filename)
    params_b = f.params_billions or 0.0
    gib = 0.0
    try:
        sz = _lookup_entry(f.repo_id)["sizes"].get(f.filename, 0)
    except AppError:
        sz = 0
    if sz > 0:
        gib = sz / (1024 ** 3)
        params_b = params_b or round(sz * 8 / bits / 1e9, 2)
    elif params_b > 0:
        gib = params_b * 1e9 * bits / 8 / (1024 ** 3)
    if params_b <= 0 or gib <= 0:
        return 0
    total_budget_gib = max(_KV_BUDGET_GIB, 2.0 - gib)   # family weight budget guard
    kv_gib = max(0.05, total_budget_gib * 0.5)
    arch = _hf_config_fields(spec)
    mmproj_gib = 0.35
    free_gib = max(0.05, total_budget_gib - gib - mmproj_gib)
    kv_gib = min(kv_gib, free_gib)
    if arch is None:
        return 0
    tok_cap = int(kv_gib * (1024 ** 3) / arch.kv_bytes_per_token("q8_0"))
    return max(4096, tok_cap // 1024 * 1024)


def model_profile_for(key: str | None, quant: str | None = None,
                      spec: "ModelSpec | None" = None) -> ModelProfile:
    """Profile for a registry key, with the per-quant context cap resolved
    against the LIVE Hub listing (sizes/arch) instead of a fixed table.

    Profile resolution is FAMILY-FIRST: ``spec.family`` (the auto-detected
    repo family such as 'qwen3.8' or 'glm5.3') wins, then the raw registry
    key (legacy built-in keys), else the fully neutral profile.  This keeps
    the two hardwired models working exactly as before while any user-added
    custom spec maps onto its detected family's defaults.
    """
    if spec is None and key:
        spec = _active_registry().get(key)
    fam = getattr(spec, "family", "") or ""     # non-spec stand-ins -> ""
    base = (_MODEL_FAMILY_DEFAULTS.get(fam)
            or _MODEL_FAMILY_DEFAULTS.get(key or "", ModelProfile()))
    caps = dict(base.max_context_by_quant)
    if spec is not None and quant:
        f = spec.file_for_quant(quant)
        if f is not None:
            cap = _live_max_context(spec, f)
            if cap:
                caps[quant] = cap
    return ModelProfile(max_context_by_quant=caps,
                        default_cache_type=base.default_cache_type,
                        thinking_levels=base.thinking_levels,
                        default_thinking=base.default_thinking,
                        temperature=base.temperature, top_p=base.top_p,
                        top_k=base.top_k, min_p=base.min_p,
                        preserve_thinking_default=base.preserve_thinking_default,
                        supports_reasoning_kwargs=base.supports_reasoning_kwargs,
                        note=base.note)


def get_model_profile(key: str | None, quant: str | None = None,
                      spec: "ModelSpec | None" = None) -> ModelProfile:
    """Profile for a registry key; neutral fallback for unknown models.

    When ``quant`` (and optionally the live ``spec``) are given, the
    per-quant context cap is included exactly as configured by the family
    defaults; live-discovered caps come via :func:`model_profile_for`.

    Registry-resolvable keys (built-ins AND user-added custom repos) always
    get their family profile - even when no quant/caps are known yet.  Only
    keys that exist in neither the registry nor the family table degrade to
    a fully neutral profile (legacy behaviour for arbitrary/empty keys, e.g.
    headless callers without Tab 2 settings).
    """
    prof = model_profile_for(key, quant, spec)
    if not key:
        return prof                    # neutral default (headless / legacy)
    resolved = spec or _active_registry().get(key)
    if resolved is None and key not in _MODEL_FAMILY_DEFAULTS \
            and not prof.max_context_by_quant:
        return ModelProfile()          # truly unknown model: fully neutral
    return prof


# ---------------------------------------------------------------------------
# Storage-location management (default app cache vs Unsloth-desktop folder)
# ---------------------------------------------------------------------------
STORAGE_DEFAULT = "default"
STORAGE_UNSLOTH = "unsloth"

#: Subfolders scanned inside an Unsloth Desktop data directory.  Unsloth UI
#: keeps its GGUF downloads under ``models/`` (newer builds also use
#: ``llama-cpp/``); both are probed so existing downloads are re-used.
UNSLOTH_SUBDIRS = ("models", "llama-cpp")


class StorageManager:
    """Remembers where GGUFs live / should be downloaded.

    Two user-selectable modes (Tab 1 -> "Storage Location"):

    * ``STORAGE_DEFAULT`` – ``~/.cache/app_models/<repo-id>/``.  The location
      is displayed to the user and free disk space is checked before any
      download starts; a shortage raises :class:`StorageLocationError` with
      remediation hints instead of failing mid-transfer.
    * ``STORAGE_UNSLOTH`` – a user-supplied Unsloth Desktop compatible folder.
      Missing directories are created automatically; required models already
      present anywhere inside the tree are detected and re-used (no download).
    """

    def __init__(self, mode: str = STORAGE_DEFAULT, unsloth_root: str = ""):
        self.mode: str = mode if mode in (STORAGE_DEFAULT, STORAGE_UNSLOTH) \
            else STORAGE_DEFAULT
        self.unsloth_root: str = unsloth_root or ""

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"mode": self.mode, "unsloth_root": self.unsloth_root}

    @classmethod
    def from_dict(cls, d: dict | None) -> "StorageManager":
        d = d or {}
        return cls(mode=str(d.get("mode", STORAGE_DEFAULT)),
                   unsloth_root=str(d.get("unsloth_root", "") or ""))

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _usable(p: Path) -> bool:
        """Directory that exists (or can be created) and accepts writes."""
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p.is_dir() and os.access(p, os.W_OK)
        except OSError:
            return False

    def unsloth_search_roots(self) -> list[Path]:
        """Candidate scan roots for an Unsloth data folder (best first)."""
        root = Path(os.path.expanduser(self.unsloth_root)) if self.unsloth_root else None
        if root is None:
            return []
        roots = [root]
        for sub in UNSLOTH_SUBDIRS:
            cand = root / sub
            # NOTE: plain .is_dir() raises OSError on exotic filesystems
            # (FUSE/network mounts, permission-transit errors); treating that
            # as "folder missing" was bug #1 - an existing (even empty)
            # 'models' subdir got reported as non-existent.  Swallow only
            # filesystem access errors here; anything else must surface.
            try:
                ok = cand.is_dir()
            except OSError:
                ok = False
            if ok:
                roots.append(cand)
        return roots

    # ------------------------------------------------------------ public API
    def ensure_ready(self) -> tuple[Path, list[str]]:
        """Validate/auto-create the active location.

        Returns ``(primary_dir, warnings)``.  Raises
        :class:`StorageLocationError` when the location cannot be used at all
        (never silently falls back - the GUI shows the verbose report).
        """
        warnings: list[str] = []
        if self.mode == STORAGE_DEFAULT:
            dest = default_cache_dir()
            try:
                dest.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StorageLocationError(
                    "Cannot create the default model cache folder",
                    detail=f"{type(exc).__name__}: {exc} (path: {dest})",
                    suggestions=["Check permissions on ~/.cache/",
                                 "Or switch to an Unsloth-desktop folder on another volume."]
                ) from exc
            if not os.access(dest, os.W_OK):
                raise StorageLocationError(
                    "Default model cache folder is not writable",
                    detail=f"path: {dest}",
                    suggestions=[f"chmod u+w '{dest}' or change ownership.",
                                 "Or select an Unsloth-desktop folder elsewhere."])
            return dest, warnings

        # ---- Unsloth-compatible mode --------------------------------------
        if not self.unsloth_root.strip():
            raise StorageLocationError(
                "Unsloth storage selected but no folder path configured",
                detail="The 'Unsloth desktop models folder' field is empty.",
                suggestions=["Pick the folder in Tab 1 -> Storage Location "
                             "(e.g. ~/Unsloth/Desktop/models), or switch back "
                             "to the default cache."])
        root = Path(os.path.expanduser(self.unsloth_root.strip()))
        try:
            existed = root.exists()
            is_dir = root.is_dir()
        except OSError as exc:
            raise StorageLocationError(
                "Cannot inspect the Unsloth models folder",
                detail=f"{type(exc).__name__}: {exc} (path: {root})",
                suggestions=["The path lives on a filesystem that refused the "
                             "stat() call (unmounted/failing drive, stale "
                             "network mount, or permission-transit error).",
                             "Verify the volume is mounted and readable.",
                             "Or pick a different folder / switch back to the "
                             "default cache location."]) from exc
        if not existed:
            try:
                root.mkdir(parents=True, exist_ok=True)
                warnings.append(f"[storage] created missing folder: {root}")
            except OSError as exc:
                raise StorageLocationError(
                    "Cannot create the Unsloth models folder",
                    detail=f"{type(exc).__name__}: {exc} (path: {root})",
                    suggestions=["Verify the path (typos, unmounted drive?).",
                                 "Ensure the parent directory is writable.",
                                 "Or switch back to the default cache location."]
                ) from exc
        elif not is_dir:
            raise StorageLocationError(
                "Unsloth models path exists but is not a directory",
                detail=f"path: {root}",
                suggestions=["Remove/rename the conflicting file, or pick a "
                             "different folder in Tab 1."])
        if not os.access(root, os.W_OK):
            raise StorageLocationError(
                "Unsloth models folder is not writable",
                detail=f"path: {root}",
                suggestions=["Fix ownership/permissions (chown/chmod), or run "
                             "from the account that owns the Unsloth install.",
                             "Read-only reuse is not supported while downloads "
                             "may still be needed."])
        if not existed:
            warnings.append(f"[storage] new empty Unsloth folder at {root}; "
                            "downloads will land here.")
        return root, warnings

    def search_dirs(self, repo_id: str) -> list[Path]:
        """Ordered candidate folders where this repo's files may live."""
        slug = repo_id.replace("/", "--")
        dirs: list[Path] = []
        if self.mode == STORAGE_UNSLOTH and self.unsloth_root.strip():
            for r in self.unsloth_search_roots():
                for sub in (slug, repo_id):          # --slug and literal forms
                    d = r / sub
                    if d not in dirs:
                        dirs.append(d)
        d = default_cache_dir() / slug
        if d not in dirs:
            dirs.append(d)
        return dirs

    def primary_download_dir(self, repo_id: str) -> Path:
        """Where new downloads for ``repo_id`` should be written."""
        if self.mode == STORAGE_UNSLOTH and self.unsloth_root.strip():
            return Path(os.path.expanduser(self.unsloth_root.strip())) / \
                repo_id.replace("/", "--")
        return default_cache_dir() / repo_id.replace("/", "--")

    def find_existing(self, repo_id: str, filename: str) -> Path | None:
        """Locate an already-downloaded copy (Unsloth reuse + own cache)."""
        for d in self.search_dirs(repo_id):
            cand = d / filename
            try:
                if cand.is_file():
                    return cand
            except OSError:
                # Exotic/FUSE filesystems can raise on stat(); treat this
                # candidate as absent and keep looking rather than aborting.
                continue
        if self.mode == STORAGE_UNSLOTH and self.unsloth_root.strip():
            # Last resort: recursive scan of the Unsloth tree (some builds
            # nest by quant shard folders).  We walk directories manually
            # instead of rglob("**/name") because several FUSE-backed mounts
            # (exFAT/NTFS under /run/media) do not implement the recursive
            # glob protocol and raise ENOTSUP - previously that made present
            # files look "missing".  Bounded depth keeps the scan snappy.
            import os as _os
            for r in self.unsloth_search_roots():
                try:
                    base_depth = str(r).rstrip("/").count("/")
                    for dirpath, dirnames, filenames_ in _os.walk(
                            r, topdown=True, onerror=None, followlinks=False):
                        if dirpath.count("/") - base_depth >= 4:
                            dirnames[:] = []      # prune deeper levels
                        if filename in filenames_:
                            p = Path(dirpath) / filename
                            try:
                                if p.is_file():
                                    return p
                            except OSError:
                                continue
                except OSError:
                    continue
        return None

    def required_files_status(self, spec: "ModelSpec",
                              model_file: "ModelFile | None" = None):
        """Autodetect whether the models a spec needs are already available.

        Returns ``(present, missing, bytes_present)`` where ``present`` /
        ``missing`` are lists of ``(filename, path_or_None)`` tuples covering
        every quant of ``spec`` (or just ``model_file`` when given), including
        each file's mmproj companion.  Used by the GUI to report re-use hits
        (e.g. files downloaded earlier by Unsloth Desktop) and gaps.
        """
        present: list[tuple[str, Path]] = []
        missing: list[str] = []
        total_bytes = 0
        files = [model_file] if model_file is not None else list(spec.files.values())
        seen: set[str] = set()
        for mf in files:
            names = [mf.filename]
            # Use the fuzzy companion resolver: Unsloth folders often store
            # the projector under a *different* name than our hardwired
            # default (e.g. 'mmproj-Model-Q8_0.gguf' vs
            # 'Model-Q8_0.mmproj.gguf'); matching only the fixed name made
            # present files look "missing" after switching storage location.
            try:
                _, mm_name = mf.resolve_mmproj_companion()
            except Exception:
                mm_name = mf.mmproj_filename
            if mm_name:
                names.append(mm_name)
            for n in names:
                if n in seen:
                    continue
                seen.add(n)
                hit = self.find_existing(spec.source_repo, n)
                if hit is not None:
                    try:
                        total_bytes += hit.stat().st_size
                    except OSError:
                        pass
                    present.append((n, hit))
                else:
                    missing.append(n)
        return present, missing, total_bytes

    def describe(self) -> str:
        """Human-readable one-liner of the active location for the GUI."""
        if self.mode == STORAGE_UNSLOTH:
            root = self.unsloth_root.strip() or "(not configured)"
            n = 0
            exists = None          # None = unknown (stat failed)
            if self.unsloth_root.strip():
                p = Path(os.path.expanduser(root))
                try:
                    exists = p.is_dir()
                    n = sum(1 for _ in p.glob("**/*.gguf"))
                except OSError:
                    exists = None
                    n = -1
            if exists is False:
                extra = " - folder missing; it will be created on first use"
            elif exists is None:
                extra = (" - WARNING: the filesystem refused to inspect this "
                         "path (unmounted/failing drive?)")
            elif n > 0:
                extra = f" - {n} GGUF file(s) found for re-use"
            else:
                extra = ""
            return f"Unsloth desktop folder: {root}{extra}"
        d = default_cache_dir()
        try:
            free_gb = shutil.disk_usage(d if d.is_dir() else d.parent).free / 1024 ** 3
            state = f"{free_gb:.1f} GiB free"
        except OSError:
            state = "space unknown"
        return f"Default app cache: {d} ({state})"

    # ------------------------------------------------------ downloaded listing
    def _repo_slug_map(self) -> dict:
        """Reverse map: directory-name forms -> owning :class:`ModelSpec`.

        Used to turn a file's folder name (the repo slug, e.g.
        ``DraconicDragon--ToriiGate-0.5-GGUF``) back into a human-friendly
        repository display name for the downloaded-files list.
        """
        m: dict = {}
        for s in _all_specs():
            rid = s.source_repo
            for form in (rid, rid.replace("/", "--"), rid.replace("/", "-"),
                         rid.split("/")[-1]):
                m.setdefault(form.lower(), s)
        return m

    def list_downloaded(self) -> list[dict]:
        """Scan the active storage location for every downloaded ``*.gguf``.

        Returns a list of dicts (one per file) with keys ``repo`` (display
        name or folder slug), ``repo_id``, ``filename``, ``quant``,
        ``size_bytes`` and ``path``.  Files are de-duplicated by resolved
        path and sorted by repo then filename so the GUI list is stable.
        The walk is depth-bounded (matching :meth:`find_existing`) so huge
        or FUSE-backed trees stay snappy and never raise.
        """
        import os as _os
        slug_map = self._repo_slug_map()
        seen: set = set()
        out: list = []

        def _repo_for(path):
            """(display_name, repo_id) inferred from ancestor folder names."""
            d = path.parent
            for _ in range(4):
                spec = slug_map.get(d.name.lower())
                if spec is not None:
                    return spec.display_name, spec.source_repo
                parent = d.parent
                if parent == d:
                    break
                d = parent
            return path.parent.name, ""

        def _walk(root):
            try:
                base_depth = str(root).rstrip("/").count("/")
            except Exception:
                return
            try:
                for dirpath, dirnames, filenames_ in _os.walk(
                        root, topdown=True, onerror=None, followlinks=False):
                    if dirpath.count("/") - base_depth >= 5:
                        dirnames[:] = []      # prune deep levels
                    for fn in filenames_:
                        if not fn.lower().endswith(".gguf"):
                            continue
                        p = Path(dirpath) / fn
                        try:
                            key = str(p.resolve())
                        except OSError:
                            key = str(p)
                        if key in seen:
                            continue
                        seen.add(key)
                        try:
                            size = p.stat().st_size
                        except OSError:
                            size = 0
                        disp, rid = _repo_for(p)
                        out.append({
                            "repo": disp,
                            "repo_id": rid,
                            "filename": fn,
                            "quant": _quant_tag_from_name(fn),
                            "size_bytes": size,
                            "path": str(p),
                        })
            except OSError:
                return

        if self.mode == STORAGE_UNSLOTH and self.unsloth_root.strip():
            root = Path(os.path.expanduser(self.unsloth_root.strip()))
            try:
                if root.is_dir():
                    _walk(root)
            except OSError:
                pass
        else:
            d = default_cache_dir()
            try:
                if d.is_dir():
                    _walk(d)
            except OSError:
                pass
        out.sort(key=lambda e: (e["repo"].lower(), e["filename"].lower()))
        return out


# Process-wide active storage configuration (hydrated from app_settings.json
# at startup; mutated by Tab 1 and read by every path resolver below).
STORAGE = StorageManager.from_dict(load_app_settings().get("storage"))


def set_storage(mode: str, unsloth_root: str = "") -> StorageManager:
    """Update + persist the global storage configuration."""
    global STORAGE
    STORAGE = StorageManager(mode=mode, unsloth_root=unsloth_root)
    try:
        settings = load_app_settings()
        settings["storage"] = STORAGE.to_dict()
        save_app_settings(settings)
    except AppError:
        raise       # caller reports verbosely; config stays in-memory anyway
    return STORAGE


@dataclass
class ModelFile:
    """One concrete GGUF choice inside a repository.

    ``filename`` is the name exactly as published on the Hub - discovered by
    :func:`discover_repo_ggufs` at runtime, never composed from a template.
    ``params_billions`` is only a last-resort seed for size estimates when
    both the local file and the live Hub metadata are unavailable.
    """
    filename: str            # e.g. "ToriiGate-0.5-Q8_0.gguf" (verbatim Hub name)
    quant: str               # e.g. "Q8_0" (parsed back out of the filename)
    repo_id: str             # HF repo hosting it
    params_billions: float = 0.0   # rough param count for size estimate
    mmproj_filename: str | None = None   # resolved companion (live lookup)

    @property
    def estimated_size_mb(self) -> float:
        """Best available size estimate: real local file > live Hub metadata
        > bits-per-weight math over the seeded param count."""
        try:
            p = self.local_path()
            if p.is_file():
                return p.stat().st_size / (1024 * 1024)
        except OSError:
            pass
        try:
            sz = _lookup_entry(self.repo_id)["sizes"].get(self.filename, 0)
        except AppError:
            sz = 0
        if sz > 0:
            return sz / (1024 * 1024)
        bits = _quant_bits_for(self.quant, self.filename)
        if self.params_billions <= 0:
            self.params_billions = max(0.5, round(sz / 1e9, 2)) if sz else 0.5
        return self.params_billions * 1e9 * bits / 8 / (1024 * 1024)

    def local_path(self) -> Path:
        """Best available local copy (Unsloth reuse wins), else canonical
        destination inside the configured storage location."""
        hit = STORAGE.find_existing(self.repo_id, self.filename)
        if hit is not None:
            return hit
        return STORAGE.primary_download_dir(self.repo_id) / self.filename

    def resolve_mmproj_companion(self) -> "tuple[Path | None, str | None]":
        """Locate this file's vision projector on disk.

        Returns ``(path, filename)`` - ``path`` is an existing file when one
        can be found, ``filename`` is the best-guess companion name for
        reporting/downloading even when nothing was found locally.  The
        fallback *name* comes from a live lookup of the repo listing (see
        :func:`resolve_mmproj`) - never from a composed template.

        Robust to real-world Unsloth layouts where the *projector* is stored
        under a different name than the hardwired default (e.g. Unsloth
        ships ``mmproj-ToriiGate-0.5-Q8_0.gguf`` while our seed registry
        expects ``ToriiGate-0.5-Q8_0.mmproj.gguf``).  Search order:

        1. the remembered ``mmproj_filename`` via ``find_existing``,
        2. any already-present ``*mmproj*.gguf`` inside this repo's folders
           in the active storage (preferring same-quant, then F16/BF16),
        3. recursive scan of the Unsloth tree (same preference),
        4. live repo listing for the companion name (download planning).
        """
        # 1. exact expected name
        if self.mmproj_filename:
            hit = STORAGE.find_existing(self.repo_id, self.mmproj_filename)
            if hit is not None:
                return hit, self.mmproj_filename

        want = self.quant.upper()

        def _pick(cands: list[Path]) -> Path | None:
            ggufs = [p for p in cands
                     if "mmproj" in p.name.lower()
                     and p.name.lower().endswith(".gguf")]
            for m in ggufs:                       # same quant first
                if want and want in m.name.upper():
                    return m
            for m in ggufs:                       # then full-precision
                if "F16" in m.name.upper() or "BF16" in m.name.upper():
                    return m
            return ggufs[0] if ggufs else None

        # 2. same folder(s) as the main GGUF / repo dirs in active storage
        dirs: list[Path] = []
        main = STORAGE.find_existing(self.repo_id, self.filename)
        if main is not None:
            dirs.append(main.parent)
        for d in STORAGE.search_dirs(self.repo_id):
            if d not in dirs:
                dirs.append(d)
        cands: list[Path] = []
        for d in dirs:
            try:
                if d.is_dir():
                    cands.extend(sorted(d.glob("*.gguf")))
            except OSError:
                continue
        hit = _pick(cands)
        if hit is not None:
            return hit, hit.name

        # 3. bounded recursive scan of the Unsloth tree (different nesting).
        #    STRICTLY scoped to this file's own repository folder (slug or
        #    literal repo-id path segments): an unscoped whole-tree scan used
        #    to hand back *another* model's projector whenever this repo had
        #    none locally - e.g. ToriiGate's ``ToriiGate-0.5-Q8_0.mmproj.gguf``
        #    (n_embd 2560) got attached to the Qwen3.8-27B main GGUF
        #    (n_embd 5120) and llama-server died with
        #    "mtmd_init_from_file: mismatch between text model and mmproj".
        if STORAGE.mode == STORAGE_UNSLOTH and STORAGE.unsloth_root.strip():
            import os as _os
            slug = self.repo_id.replace("/", "--")
            parts = [p for p in self.repo_id.split("/") if p]
            all_ggufs: list[Path] = []
            for r in STORAGE.unsloth_search_roots():
                try:
                    base_depth = str(r).rstrip("/").count("/")
                    for dirpath, dirnames, fnames in _os.walk(
                            r, topdown=True, followlinks=False):
                        rel_parts = set(
                            Path(dirpath).relative_to(r).parts)
                        if slug not in rel_parts and not set(parts) & rel_parts:
                            dirnames[:] = []      # not our repo -> prune
                            continue
                        if dirpath.count("/") - base_depth >= 4:
                            dirnames[:] = []
                        for fn in fnames:
                            if fn.lower().endswith(".gguf"):
                                all_ggufs.append(Path(dirpath) / fn)
                except (OSError, ValueError):
                    continue
            hit = _pick(all_ggufs)
            if hit is not None:
                return hit, hit.name

        # 4. nothing on disk - report only the ADOPTED/remembered name (if any).
        # Deliberately NO live Hub lookup in this method: it is called from UI
        # paths that must stay local (downloaded-files list, follow label,
        # server preflight), and plan_mmproj_download() performs its own
        # single live pass - a live fallback in both places queried the same
        # repo twice per uncached run.  Discovery belongs exclusively to
        # resolve_mmproj()/plan_mmproj_download().
        return None, self.mmproj_filename

    def mmproj_companion_error(self, companion_path: Path) -> str | None:
        """Return a human-readable incompatibility reason for ``companion_path``.

        Guards against the field-reported "wrong projector" crash where a
        leftover *other-model* mmproj (e.g. ToriiGate's ``..._Q8_0.mmproj.gguf``,
        n_embd 2560) got auto-attached to a different main GGUF (Qwen3.8-27B,
        n_embd 5120) and llama-server aborted with::

            E mtmd_init_from_file: mismatch between text model (n_embd = 5120)
              and mmproj (n_embd = 2560)
            hint: you may be using wrong mmproj

        Checks, in order:
        1. filename cross-reference - if the companion name contains another
           registered repo's model slug it cannot belong to this model;
        2. header metadata - both GGUFs must expose ``general.architecture``
           and those architectures must match.

        Returns ``None`` when the companion looks compatible (or cannot be
        checked); otherwise a message naming both files and the mismatch.
        """
        try:
            mp_arch = _read_gguf_metadata(Path(companion_path))
        except Exception:
            mp_arch = None
        # 1. cross-repo name sniffing (works even without readable headers).
        #    Real-world projector names carry the *model* slug but usually not
        #    the "-GGUF" repo suffix ("ToriiGate-0.5-Q8_0.mmproj.gguf"), so the
        #    raw repo slug alone misses; strip trailing quant/format tags to
        #    get a distinctive base name as well.
        def _base_of(slug: str) -> str:
            import re as _re
            s = slug.lower()
            prev = None
            while s != prev:                      # peel e.g. '-gguf', '.gguf'
                prev = s
                s = _re.sub(r"[-._ ]?(gguf|safetensors)$", "", s)
                s = _re.sub(r"[-._ ](mtp|rco|gsq|abliterated|unsloth|"
                            r"hf|v\d+(\.\d+)*)$", "", s)
            return s.strip("-._ ")

        mp_name = Path(companion_path).name.lower()
        own_slug = self.repo_id.split("/")[-1].lower()
        own_base = _base_of(own_slug)
        for spec in _all_specs():
            other_repo = spec.source_repo.lower()
            if other_repo == self.repo_id.lower():
                continue
            other_slug = other_repo.split("/")[-1].lower()
            other_base = _base_of(other_slug)
            hit = ((other_slug and other_slug in mp_name)
                   or (len(other_base) >= 4 and other_base in mp_name))
            own_hit = ((own_slug and own_slug in mp_name)
                       or (len(own_base) >= 4 and own_base in mp_name))
            if hit and not own_hit:
                return (f"'{Path(companion_path).name}' belongs to repository "
                        f"'{spec.source_repo}', not '{self.repo_id}'")
        # 2. architecture comparison via GGUF headers
        try:
            main_arch = _read_gguf_metadata(self.local_path())
        except Exception:
            main_arch = None
        if main_arch is not None and mp_arch is not None:
            a = (main_arch.source or "").replace("gguf:", "").strip().lower()
            b = (mp_arch.source or "").replace("gguf:", "").strip().lower()
            if a and b and a != b:
                return (f"text model architecture '{a}' does not match mmproj "
                        f"architecture '{b}'")
        return None

    def mmproj_local_path(self) -> Path | None:
        path, name = self.resolve_mmproj_companion()
        if name is not None:
            # adopt the discovered/expected name so downstream code
            # (verify_downloaded, server launch, status labels) stays aligned
            self.mmproj_filename = name
        return path

    def is_downloaded(self) -> bool:
        try:
            ok = self.local_path().is_file()
        except OSError:
            ok = False
        if ok:
            # resolve_mmproj_companion searches the whole active storage, so
            # it finds Unsloth-named projectors even when mmproj_filename is
            # still unset (BF16 quants) or mismatched.
            mp, _name = self.resolve_mmproj_companion()
            ok = mp is not None and mp.is_file()
        return ok


@dataclass
class ModelSpec:
    """A selectable model family whose quant list is discovered LIVE.

    ``seed_files`` are fallback entries used only when the Hub cannot be
    reached (offline start-up).  At runtime :meth:`refresh` replaces the
    file list with whatever GGUFs the repository actually publishes - every
    quant, no allow-list - so renamed files (e.g. lowercase ``-bf16.gguf``)
    are picked up automatically instead of 404-ing mid-download.
    """
    key: str                       # internal id used in configs
    display_name: str
    source_repo: str
    hf_config_repo: str | None     # repo whose config.json describes arch
    seed_files: list[ModelFile] = field(default_factory=list)
    files: list[ModelFile] = field(default_factory=list)
    mmproj_candidates: list[str] = field(default_factory=list)
    abliterated: bool = False      # safe to inject explicit-mode triggers
    vision: bool = True
    family: str = "generic"        # detect_family() result; profile key
    last_refresh_error: str = ""   # verbose report of the failed live lookup
    refreshed_at: float = 0.0      # monotonic timestamp of last refresh
    live: bool = False             # True once a live listing populated files

    def __post_init__(self):
        if not self.files:
            self.files = list(self.seed_files)

    # ------------------------------------------------------------- live list
    def refresh(self, force: bool = False) -> bool:
        """Re-discover quants/filenames from the live Hub listing.

        Returns True when the registry now reflects the Hub.  On failure the
        previous list (seeds offline) is kept and a verbose report is stored
        in ``last_refresh_error``; nothing ever raises so start-up / dropdown
        population stays safe.  Honours the module-level lookup cache TTL
        unless ``force``.
        """
        import time as _t
        if self.refreshed_at and not force \
                and _t.monotonic() - self.refreshed_at < REPO_LOOKUP_TTL:
            return self.live
        try:
            ggufs = discover_repo_ggufs(self.source_repo, force_refresh=force)
        except AppError as exc:
            self.last_refresh_error = exc.report()
            return self.live
        if not ggufs:
            self.last_refresh_error = (
                f"[models] WARNING: live file listing of '{self.source_repo}' "
                "returned no GGUFs (offline or gated?) - falling back to the "
                "bundled seed filenames.")
            return self.live
        params = max((f.params_billions for f in self.seed_files), default=0.0)
        old_by_name = {f.filename.lower(): f for f in self.files}
        new: list[ModelFile] = []
        for fn, quant, _size in ggufs:
            prev = old_by_name.get(fn.lower())
            new.append(ModelFile(
                filename=fn, quant=quant, repo_id=self.source_repo,
                params_billions=params,
                # keep an already-adopted local mmproj name; otherwise the
                # companion is resolved live on demand (resolve_mmproj)
                mmproj_filename=prev.mmproj_filename if prev else None))
        self.files = new
        self.live = True
        self.last_refresh_error = ""
        self.refreshed_at = _t.monotonic()
        return True

    # ------------------------------------------------------------ accessors
    def quants(self) -> list[str]:
        return [f.quant for f in self.files]

    def file_for_quant(self, quant: str) -> ModelFile | None:
        """Match by displayed tag first, then case/punctuation-insensitively
        (live tags normalise ``Q8_0`` names to ``Q8-0`` etc.)."""
        for f in self.files:
            if f.quant == quant:
                return f
        norm = (quant or "").upper().replace("_", "-")
        for f in self.files:
            if f.quant.upper().replace("_", "-") == norm:
                return f
        return None


#: Registry keys of the two hardwired models - everything else in the
#: registry is a user-added custom repository (persisted 'custom_repos').
BUILTIN_SPEC_KEYS = frozenset({"toriigate", "qwen3.8-27b"})


def build_spec_from_repo(repo_id: str, label: str = "") -> ModelSpec:
    """Create a live :class:`ModelSpec` from an arbitrary HF repo id.

    Validates the id (:func:`validate_repo_id`), lists the repository on the
    Hub and REFUSES to create a spec when it publishes no main GGUF file -
    'repo exists but contains no *.gguf files' (safetensors-only repos are
    not supported) - so the registry never gains a dead entry.  The quant
    list comes verbatim from the live listing (**no allow-list**),
    ``config.json`` is fetched for family detection, and ``abliterated`` is
    set heuristically when the repo id or any filename contains
    'abliterated'/'uncensored' (keeps Tab 3 Explicit Mode working).

    The spec key is the deterministic repo slug (``org/name`` ->
    ``org--name``) - the same scheme StorageManager uses for on-disk folders,
    so registry keys and storage directories stay 1:1.

    Raises :class:`AppError` (verbose, classified) when the id is invalid or
    the repo cannot be listed; raises a plain AppError with remediation hints
    when no GGUFs are published.
    """
    rid = validate_repo_id(repo_id)
    _lookup_entry(rid)                     # existence + access (401/403 classified)
    ggufs = discover_repo_ggufs(rid)       # cached listing; [] when no GGUFs
    if not ggufs:
        raise AppError(
            f"Repository '{rid}' exists but contains no *.gguf files",
            "safetensors-only (or dataset/empty) repositories are not "
            "supported - this app runs GGUF models through llama-server.",
            ["Pick a repo that publishes .gguf quantizations, e.g. one of the "
             "'-GGUF' repos on huggingface.co."])
    cfg = _fetch_hf_config(rid)            # may be None (no config.json); fine
    family = detect_family(rid, cfg)
    names_lower = [fn.lower() for fn, _q, _sz in ggufs]
    abliterated = (("abliterated" in rid or "uncensored" in rid)
                   or any(("abliterated" in n or "uncensored" in n)
                          for n in names_lower))
    key = rid.lower().replace("/", "--")   # 1:1 with StorageManager dir naming
    display = (label or "").strip() or rid.split("/")[-1]
    files = [ModelFile(filename=fn, quant=q, repo_id=rid, params_billions=0.0)
             for fn, q, _sz in ggufs]
    import time as _t
    return ModelSpec(key=key, display_name=display, source_repo=rid,
                     hf_config_repo=rid, seed_files=[], files=files,
                     abliterated=abliterated, family=family,
                     refreshed_at=_t.monotonic(), live=True)


def build_registry(refresh: bool = True) -> dict[str, ModelSpec]:
    """Supported-model catalogue: built-ins + user-added custom repos.

    Only *repositories* are hardwired here; the concrete GGUF filenames and
    available quants are looked up live from the Hugging Face Hub when the
    registry is built (``refresh=True``, the default).  The seed entries act
    strictly as offline fallbacks.  User-added repositories persisted in
    ``app_settings.json['custom_repos']`` are merged after the built-ins,
    deduplicated by source repo, and created OFFLINE-SAFE: no network access
    happens here beyond the existing live refresh - custom specs start with
    empty file lists that the late-bound background listing fills in.
    """
    torii_repo = "DraconicDragon/ToriiGate-0.5-GGUF"
    qwen_repo = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"

    torii = ModelSpec(
        key="toriigate",
        display_name="ToriiGate-0.5 (MOSS captioner)",
        source_repo=torii_repo,
        hf_config_repo="Minthy/ToriiGate-0.5",
        seed_files=[
            ModelFile("ToriiGate-0.5-BF16.gguf", "BF16", torii_repo, 0.5),
            ModelFile("ToriiGate-0.5-Q8_0.gguf", "Q8_0", torii_repo, 0.5,
                      mmproj_filename="ToriiGate-0.5-Q8_0.mmproj.gguf"),
        ],
        mmproj_candidates=["ToriiGate-0.5-Q8_0.mmproj.gguf"],
        family="toriigate",                # explicit: family-first profile lookup
    )

    qwen = ModelSpec(
        key="qwen3.8-27b",
        display_name="Huihui-Qwen3.8-27B-abliterated",
        source_repo=qwen_repo,
        hf_config_repo="huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF",
        seed_files=[
            ModelFile("Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf", "Q8_0", qwen_repo, 27),
            ModelFile("Huihui-Qwen3.8-27B-abliterated-Q6_K.gguf", "Q6_K", qwen_repo, 27),
            ModelFile("Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf",
                      "UD-IQ4_XS", qwen_repo, 27),
        ],
        # Resolved dynamically against the live repo listing (see resolve_mmproj).
        mmproj_candidates=[
            "mmproj-Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf",
            "mmproj-F16.gguf",
        ],
        abliterated=True,
        family="qwen3.8-27b",              # explicit: family-first profile lookup
    )
    specs = {torii.key: torii, qwen.key: qwen}

    # ---- merge persisted user-added repos (offline-safe) ------------------
    data = load_app_settings()
    crepos = data.get("custom_repos") if isinstance(data, dict) else None
    if isinstance(crepos, list):
        for entry in crepos:
            if not isinstance(entry, dict):
                continue
            repo_raw = str(entry.get("repo", "")).strip()
            if not repo_raw:
                continue
            try:
                norm = validate_repo_id(repo_raw)
            except AppError as exc:
                print(f"[models] skipping invalid persisted custom repo "
                      f"{repo_raw!r}: {exc.message}", flush=True)
                continue
            if any(s.source_repo.lower() == norm.lower()
                   for s in specs.values()):
                continue                    # dedupe by source_repo (built-ins win)
            label = str(entry.get("label", "") or "").strip()
            key = norm.lower().replace("/", "--")
            if key in specs:
                continue
            try:
                specs[key] = ModelSpec(
                    key=key,
                    display_name=label or norm.split("/")[-1],
                    source_repo=norm,
                    hf_config_repo=norm,
                    family=detect_family(norm),
                    abliterated=("abliterated" in norm.lower()
                                 or "uncensored" in norm.lower()),
                )            # empty file list; live listing fills it later
            except Exception as exc:        # never break registry construction
                print(f"[models] could not create custom spec for {norm!r}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
    if refresh:
        for spec in specs.values():
            spec.refresh(force=True)
    return specs


# ---------------------------------------------------------------------------
# Unified application configuration (export / import)
# ---------------------------------------------------------------------------
#: Current schema version of the single-file config bundle.  v3 adds the
#: model's Hugging Face repo id and the 'custom_repos' list; v1/v2 bundles
#: import unchanged (see import_full_config).
CONFIG_VERSION = 3


def export_full_config(ctx) -> dict:
    """One JSON-serialisable dict covering **every** configurable piece.

    Sections: captioning presets (Tab 3), inference settings incl. the
    llama-server binary path (Tab 2), storage location (Tab 1) and the
    selected model/quant (Tab 1).  Dataset paths are deliberately excluded -
    they belong to the session, not to a shareable profile.
    """
    from app.core.presets import export_config
    presets_payload = json.loads(export_config(
        ctx.presets, ctx.active_preset_key, ctx.explicit_mode))
    return {
        "version": CONFIG_VERSION,
        "app": "moss-toriigate-autocaptioner",
        "presets": presets_payload,
        "settings": ctx.settings.to_dict(),
        "storage": ctx.storage.to_dict(),
        "model": {"key": ctx.selected_model_key,
                  "quant": ctx.selected_quant,
                  "repo": getattr(ctx, "selected_repo_id", "")},
        "custom_repos": [{"repo": s.source_repo, "label": s.display_name}
                         for k, s in ctx.registry.items()
                         if k not in BUILTIN_SPEC_KEYS],
    }


def _coerce(value, target_type, default):
    """Best-effort type coercion for imported scalars; keeps *default* on junk.

    Booleans are honoured only for bool fields: JSON ``true``/``false``
    landing on an int/float setting (e.g. ``port``, ``top_k``,
    ``max_tokens`` — a frequent accident when hand-editing configs that mix
    thinking flags with sampling values) used to coerce to ``1``/``0`` and
    silently mangle the generation settings.  Such mismatches now keep the
    current/default value so :func:`import_full_config` can warn about it.
    Numeric strings still coerce normally (legacy configs store numbers as
    text); numeric types are never accepted for bool fields — only real
    booleans or their textual ``true``/``false`` spellings.
    """
    try:
        if target_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str) \
                    and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            if isinstance(value, str):
                raise TypeError(f"cannot map string {value!r} to bool")
            return default              # numbers never flip a bool field
        if isinstance(value, bool):
            return default              # bools never fill numeric/str fields
        if target_type in (int,):
            if isinstance(value, str):
                return int(value.strip(), 10)
            return int(value)
        if target_type in (float,):
            return float(value)
        if target_type is str:
            return str(value)
    except (TypeError, ValueError):
        return default
    return value


def import_full_config(ctx, raw: str) -> list[str]:
    """Apply a config bundle produced by :func:`export_full_config`.

    Accepts both the unified v2 bundle and legacy Tab-3-only files (v1 with
    just presets) so old exports keep working.  Unknown/invalid keys are
    skipped with a warning instead of aborting the whole import.
    Returns the list of human-readable warnings (possibly empty).
    """
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("the file does not contain a JSON object")
    warnings: list[str] = []

    # ---- presets (also covers legacy v1 files where presets are top-level)
    pload = data.get("presets")
    if (isinstance(pload, dict) and "presets" in pload
            and data.get("scope") != "presets"):
        from app.core.presets import import_config
        presets, active, explicit = import_config(json.dumps(pload))
        ctx.presets = presets
        ctx.active_preset_key = active
        ctx.explicit_mode = explicit
    elif "presets" in data or "active_preset" in data:
        from app.core.presets import import_config
        presets, active, explicit = import_config(raw)   # legacy Tab-3 format
        ctx.presets = presets
        ctx.active_preset_key = active
        ctx.explicit_mode = explicit
        warnings.append("legacy preset-only config imported (v1); "
                        "re-export to get the unified all-tabs bundle.")
    else:
        warnings.append("no 'presets' section found - captioning presets left "
                        "unchanged.")

    # ---- inference settings (Tab 2)
    st = data.get("settings")
    if isinstance(st, dict):
        fields = ctx.settings.__dataclass_fields__
        for k, v in st.items():
            if k not in fields:
                warnings.append(f"ignored unknown setting key '{k}'")
                continue
            cur = getattr(ctx.settings, k)
            new = _coerce(v, type(cur), cur)
            if new == cur and type(new) is type(cur) and v is not None \
                    and not isinstance(v, type(cur)) \
                    and str(v) != str(cur):
                # Coercion refused the value (bool into a numeric field,
                # garbage string, ...): keep the live value and say so
                # instead of silently mangling generation settings.
                warnings.append(f"setting '{k}': value {v!r} is not a valid "
                                f"{type(cur).__name__}; kept current "
                                f"{cur!r}.")
            setattr(ctx.settings, k, new)
    elif st is None and ctx.presets:
        # Legacy preset-only files (v1 / hand-written captioner_config.json)
        # carry no 'settings' block at all.  Without syncing, Tab 2 would
        # keep showing hardwired defaults while every request actually ran
        # with the freshly imported preset params — the classic "generation
        # settings get mangled after import" report.  Seed the shared knobs
        # from the active preset so GUI state matches what gets sent.
        try:
            p = ctx.presets.get(ctx.active_preset_key) \
                or next(iter(ctx.presets.values()), None)
        except Exception:
            p = None
        if p is not None:
            gp = p.params
            mapped = ("temperature", "top_p", "top_k", "min_p",
                      "enable_thinking", "reasoning_budget", "max_tokens")
            for name in mapped:
                if name in ctx.settings.__dataclass_fields__:
                    setattr(ctx.settings, name, getattr(gp, name))
            warnings.append("no 'settings' section - inference parameters "
                            "synced from the active preset's generation "
                            "params.")
        else:
            warnings.append("no 'settings' section - inference parameters "
                            "left unchanged.")
    else:
        warnings.append("no 'settings' section - inference parameters left "
                        "unchanged.")

    # ---- storage location (Tab 1)
    stor = data.get("storage")
    if isinstance(stor, dict):
        mode = str(stor.get("mode", ""))
        root = str(stor.get("unsloth_root", "") or "")
        if mode in (STORAGE_DEFAULT, STORAGE_UNSLOTH):
            try:
                set_storage(mode, root)
            except AppError as exc:
                warnings.append("storage section present but applying it "
                                f"failed: {exc.message}")
        else:
            warnings.append(f"invalid storage mode '{mode}' - storage left "
                            "unchanged.")

    # ---- user-added HF repositories (v3) ----------------------------------
    crepos = data.get("custom_repos")
    if isinstance(crepos, list):
        for entry in crepos:
            if not isinstance(entry, dict):
                continue
            repo_raw = str(entry.get("repo", "")).strip()
            label = str(entry.get("label", "") or "").strip()
            if not repo_raw:
                continue
            try:
                norm = validate_repo_id(repo_raw)
            except AppError:
                warnings.append(f"custom_repos: invalid repo id {repo_raw!r} - skipped.")
                continue
            if any(s.source_repo.lower() == norm.lower()
                   for s in ctx.registry.values()):
                continue                    # already known (built-in or added)
            try:
                spec = build_spec_from_repo(norm, label)
            except AppError as exc:
                warnings.append(
                    f"custom repo '{norm}' could not be verified right now "
                    f"({exc.message}) - it will appear once the Hub is reachable.")
                continue
            ctx.registry[spec.key] = spec

    # ---- model selection (Tab 1)
    mdl = data.get("model")
    if isinstance(mdl, dict):
        key = str(mdl.get("key", ""))
        quant = str(mdl.get("quant", ""))
        repo_raw = str(mdl.get("repo", "") or "").strip()
        # v3 bundles name the repository explicitly: when the key is unknown
        # but a valid, reachable repo id accompanies it, (re)create the spec
        # so an exported profile reproduces the exact model on another
        # machine.  Invalid/unreachable repos only warn - never abort.
        if key and key not in ctx.registry and repo_raw:
            try:
                norm = validate_repo_id(repo_raw)
            except AppError:
                warnings.append(f"model section carries an invalid repo id "
                                f"{repo_raw!r} - ignored.")
                norm = ""
            if norm and not any(s.source_repo.lower() == norm.lower()
                                for s in ctx.registry.values()):
                try:
                    added = build_spec_from_repo(norm)
                    ctx.registry[added.key] = added
                except AppError as exc:
                    warnings.append(
                        f"model repo '{norm}' could not be verified "
                        f"({exc.message}) - model selection left unchanged.")
        if key in ctx.registry:
            ctx.selected_model_key = key
            spec = ctx.registry[key]
            if quant and quant in spec.quants():
                ctx.selected_quant = quant
            else:
                ctx.selected_quant = ("Q8_0" if "Q8_0" in spec.quants()
                                      else spec.quants()[0])
                warnings.append(f"quant '{quant}' not valid for model '{key}' "
                                f"- using '{ctx.selected_quant}'.")
        elif key:
            warnings.append(f"unknown model key '{key}' - model selection "
                            "left unchanged.")
    return warnings


# Process-wide registry.  ModelFile.resolve_mmproj_companion() needs to look
# up a file's owning spec for fallback naming; AppContext builds its own copy
# (mutable presets live there), so we expose a shared read-only view here.
# Built from the cached live Hub listing when reachable, seed files otherwise
# (the import must never block on or crash because of the network).
REGISTRY: dict[str, ModelSpec] = build_registry()

#: The live AppContext once a GUI session exists.  Module-level helpers that
#: resolve profiles by key consult ITS registry first (see below) so repos
#: added at runtime through Tab 1 get their detected family profile instead
#: of degrading to the neutral one.  Set from ``AppContext.__init__``; stays
#: None in headless/test processes, which then use REGISTRY as before.
_ACTIVE_CTX: "object | None" = None


def _active_registry() -> dict[str, ModelSpec]:
    """The registry key lookups should consult: the active context's when a
    GUI session is running (it holds runtime-added custom repos), else the
    import-time module REGISTRY."""
    ctx = _ACTIVE_CTX
    if ctx is not None and getattr(ctx, "registry", None):
        return dict(ctx.registry)
    return REGISTRY


def _all_specs() -> list[ModelSpec]:
    """All known specs, preferring the live AppContext registry if present."""
    try:
        return list(_active_registry().values())
    except Exception:
        pass
    return list(REGISTRY.values())


# --------------------------------------------------------------------------
# Hugging Face access (guarded so unit tests can run without network)
# --------------------------------------------------------------------------
def list_repo_files(repo_id: str) -> list[str]:
    """Return the file list of an HF repo, or [] when offline/unavailable.

    Never raises: callers treat [] as "auto-detection unavailable" and fall
    back to hardwired candidates; the GUI logs a warning instead.
    """
    try:
        from huggingface_hub import list_repo_files
        return list(list_repo_files(repo_id))
    except Exception as exc:
        # Offline / gated / transient - report via stderr for the tailing log
        # but keep the app running (spec: startup must never crash on this).
        print(f"[models] list_repo_files({repo_id!r}) unavailable: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return []


def resolve_mmproj(spec: ModelSpec, chosen: ModelFile) -> str | None:
    """Pick the compatible multimodal projector for ``chosen``.

    Live-first: the companion name is taken verbatim from the repository's
    current Hub listing (:func:`resolve_mmproj_live`), never composed.
    Priority: explicit fixed filename > same-quant mmproj in the live repo >
    any F16/BF16 mmproj > first candidate from the (offline) listing >
    hardwired seed candidates.
    """
    if chosen.mmproj_filename:
        return chosen.mmproj_filename
    live = resolve_mmproj_live(spec.source_repo, chosen.quant)
    if live:
        return live
    files = list_repo_files(spec.source_repo)
    mmprojs = [f for f in files if "mmproj" in f.lower() and f.lower().endswith(".gguf")]
    if not mmprojs:
        return spec.mmproj_candidates[0] if spec.mmproj_candidates else None
    want = chosen.quant.upper()
    for m in mmprojs:
        if want in m.upper():
            return m
    for m in mmprojs:
        if "F16" in m.upper() or "BF16" in m.upper():
            return m
    return mmprojs[0]


# ---------------------------------------------------------------------------
# Unified vision-projector (mmproj) planning - single source of truth used by
# BOTH Tab 1 ("Download & Verify") and Tab 2 ("Fetch mmproj" / start gate).
# Every decision here is model-agnostic: ToriiGate-style projectors
# ("<Model>-Q8_0.mmproj.gguf", same repo as the GGUF) and Qwen-style ones
# ("mmproj-<Model>-<quant>.gguf", sometimes published in a *separate*
# companion repo) go through exactly the same discovery pipeline.
# ---------------------------------------------------------------------------
#: Optional per-repo override: repos whose multimodal projector lives in a
#: different Hugging Face repository than the main GGUFs.  Empty by default -
#: every other repo is resolved purely from its own live listing.
MMPROJ_REPO_OVERRIDE: dict[str, str] = {}


def spec_for_repo(repo_id: str) -> "ModelSpec | None":
    """The registry :class:`ModelSpec` owning ``repo_id`` (either direction)."""
    rid = (repo_id or "").lower()
    for s in _all_specs():
        if s.source_repo.lower() == rid or s.key == repo_id:
            return s
    return None


def mmproj_repos_to_try(spec: "ModelSpec | None", mf: "ModelFile") -> list[str]:
    """Repos that may host ``mf``'s projector, main repo first.

    Includes the optional companion-repo override (for layouts where the
    projector is published separately) and de-duplicates.
    """
    repos: list[str] = []
    if spec is not None:
        ov = (MMPROJ_REPO_OVERRIDE.get(spec.key)
              or MMPROJ_REPO_OVERRIDE.get(spec.source_repo))
        if ov and ov not in repos:
            repos.append(ov)
    if mf.repo_id not in repos:
        repos.append(mf.repo_id)
    return repos


def plan_mmproj_download(mf: "ModelFile", *, force_refresh: bool = False,
                         log_cb=None) -> "tuple[bool, str, Path | None]":
    """Decide what must happen to make ``mf``'s vision projector available.

    Returns ``(needed, message, local_path)``:

    * ``(False, "...found...", path)`` - the projector is already cached in
      the active storage (Unsloth re-use included); nothing to download.
    * ``(True, "<exact Hub filename>", None)`` - not on disk; the message is
      the verbatim live-Hub filename to fetch (never a composed guess).  The
      discovered name is adopted onto ``mf.mmproj_filename`` so downstream
      verify/preflight stays aligned.
    * ``(False, "<unavailability report>", None)`` - no projector could be
      located locally or on the Hub (offline, gated, or the repo genuinely
      publishes none).  The message explains why and what to do next.

    Resolution order (identical for both tabs):
    1. fuzzy local scan (:meth:`ModelFile.resolve_mmproj_companion`),
    2. live listing of the main repo (force-refreshable),
    3. live listing of any configured companion repo,
    4. offline fallback: last remembered name / bundled seed candidates.
    """
    def note(msg: str):
        if log_cb:
            log_cb(msg)

    # 1. already on disk?  (adopt Unsloth-style names as a side effect)
    path, name = mf.resolve_mmproj_companion()
    if path is not None and path.is_file():
        if name:
            mf.mmproj_filename = name
        note(f"[mmproj] re-use OK: {path}")
        return False, f"re-using cached projector: {path}", path

    spec = spec_for_repo(mf.repo_id)
    tried: list[str] = []

    # 2./3. live lookups over every candidate repo (main + companions)
    for repo in mmproj_repos_to_try(spec, mf):
        if repo in tried:
            continue                    # companion == main repo: query once
        tried.append(repo)
        live = resolve_mmproj_live(repo, mf.quant, force_refresh=force_refresh)
        if not live:
            continue
        # A live hit may still match a file sitting in another storage dir
        # under a slightly different name - check before queueing bytes.
        hit = STORAGE.find_existing(repo, live)
        if hit is not None and hit.is_file():
            mf.mmproj_filename = hit.name
            note(f"[mmproj] re-use OK: {hit}")
            return False, f"re-using cached projector: {hit}", hit
        mf.mmproj_filename = live
        note(f"[mmproj] '{mf.filename}' -> projector '{live}' "
             f"(live listing of '{repo}')")
        return True, live, None

    # 4. offline fallbacks - only trustworthy when we already know the name
    if mf.mmproj_filename:
        note(f"[mmproj] Hub unreachable; falling back to remembered name "
             f"'{mf.mmproj_filename}'.")
        return True, mf.mmproj_filename, None
    if spec is not None and spec.mmproj_candidates:
        cand = spec.mmproj_candidates[0]
        mf.mmproj_filename = cand
        note(f"[mmproj] Hub unreachable; using bundled candidate '{cand}'.")
        return True, cand, None
    note("[mmproj] WARNING: no projector found locally and none could be "
         f"looked up on the Hub (tried: {', '.join(tried) or mf.repo_id}). "
         "Check your connection and press 'Refresh from Hub'.")
    return (False,
            "vision projector unavailable: not cached locally and no "
            f"mmproj*.gguf found in '{' / '.join(tried) or mf.repo_id}' "
            "(offline, gated, or the repo publishes none)",
            None)


def ensure_mmproj_downloaded(mf: "ModelFile", log_cb=None) -> "tuple[bool, str]":
    """Blocking: make ``mf``'s projector present, downloading it if needed.

    Shared by Tab 1's "Download & Verify" worker and Tab 2's on-demand
    "Fetch mmproj" action.  Returns ``(ok, message)``; on success the message
    names the local file, failures carry the verbose report.
    """
    needed, msg, path = plan_mmproj_download(mf, log_cb=log_cb)
    if not needed:
        if path is not None:
            return True, msg
        return False, msg          # unavailability report
    try:
        paths = download_model_files(mf.repo_id, [msg])
    except AppError as exc:
        if log_cb:
            log_cb(exc.report())
        return False, exc.report()
    except Exception as exc:
        err = AppError.classify(exc, context=f"downloading mmproj for "
                                            f"'{mf.filename}'")
        if log_cb:
            log_cb(err.report())
        return False, err.report()
    got = paths[-1] if paths else None
    if got is not None:
        mf.mmproj_filename = Path(got).name
    ok = got is not None and Path(got).is_file()
    return ok, (f"projector ready: {got}" if ok
                else "mmproj download produced no usable file")


def estimate_download_mb(repo_id: str, filenames: list[str]) -> float:
    """Best-effort total download size in MiB using Hub metadata only.

    Returns 0.0 when the Hub is unreachable; callers must then ask the user
    to confirm rather than silently pulling tens of gigabytes.
    """
    total = 0.0
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.repo_info(repo_id, files_metadata=True)
        sizes = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
        for fn in filenames:
            total += sizes.get(fn, 0) / (1024 * 1024)
    except Exception as exc:
        print(f"[models] size lookup for {repo_id} failed: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return 0.0
    return total


def preflight_download_check(repo_id: str, filenames: list[str],
                             est_total_mb: float | None = None):
    """Validate storage location + disk space BEFORE any bytes hit the wire.

    Raises :class:`AppError` / :class:`StorageLocationError` with a verbose
    report when the download cannot possibly succeed, so the GUI can tell the
    user exactly what to fix instead of failing mid-transfer after minutes of
    bandwidth.  In Unsloth mode this also auto-creates the folder when it is
    missing (reported as a warning).

    Returns ``(free_mb, est_total_mb)``.
    """
    if est_total_mb is None:
        est_total_mb = estimate_download_mb(repo_id, filenames)

    # ---- storage location (auto-create / validate) -------------------------
    try:
        _root, warnings = STORAGE.ensure_ready()
    except StorageLocationError:
        raise
    except Exception as exc:
        raise AppError.classify(exc, context=f"preparing storage location "
                                            f"'{STORAGE.describe()}'") from exc
    for w in warnings:
        print(w, flush=True)          # surfaced in the GUI log by the caller

    dest_dir = STORAGE.primary_download_dir(repo_id)
    try:
        probe_dir = dest_dir if dest_dir.is_dir() else _root
        probe_dir.mkdir(parents=True, exist_ok=True)
        free_mb = shutil.disk_usage(probe_dir).free / (1024 * 1024)
    except OSError as exc:
        raise StorageLocationError(
            "Cannot prepare the model download directory",
            detail=f"{type(exc).__name__}: {exc} (path: {dest_dir})",
            suggestions=[f"Ensure '{_root}' is on a writable volume.",
                         "Check permissions of the selected storage folder."]) from exc

    # Writability smoke test (tiny file, removed immediately).
    try:
        fd, tmp = tempfile.mkstemp(dir=str(probe_dir), prefix=".preflight-")
        os.write(fd, b"x")
        os.close(fd)
        os.unlink(tmp)
    except OSError as exc:
        raise StorageLocationError(
            "Model storage directory is not writable",
            detail=f"{type(exc).__name__}: {exc} (path: {probe_dir})",
            suggestions=["Fix ownership/permissions of the selected storage "
                         "folder (Tab 1 -> Storage Location)."]) from exc

    if est_total_mb > 0:
        needed_mb = est_total_mb * 1.05          # 5% slack for temp/LFS overhead
        if needed_mb > free_mb:
            raise AppError(
                "Not enough disk space for this download",
                detail=(f"{repo_id}: {', '.join(filenames)} needs ~"
                        f"{est_total_mb / 1024:.1f} GiB but only "
                        f"{free_mb / 1024:.1f} GiB is free on {probe_dir}."),
                suggestions=[
                    "Delete unused GGUFs from the current storage folder.",
                    "Choose a smaller quantization (e.g. UD-IQ4_XS instead of Q8_0).",
                    "Switch the storage location (Tab 1) to a folder on a larger "
                    "volume - e.g. an Unsloth-desktop compatible directory.",
                    "If the models were already downloaded elsewhere (Unsloth), "
                    "point that folder here to re-use them without downloading."])
    return free_mb, est_total_mb


def download_model_files(repo_id: str, filenames: list[str],
                         progress_cb=None) -> list[Path]:
    """Blocking download of ``filenames`` from ``repo_id`` into the
    configured storage location (default cache or Unsloth folder).

    Runs ONLY from an explicit user action ("Download & Verify"); performs a
    disk-space/writability preflight first and converts every Hub failure
    into an :class:`AppError` with remediation hints.

    Files already present anywhere in the storage tree (e.g. downloaded by
    Unsloth Desktop) are skipped and reported as re-used.

    Every requested name is first reconciled against the LIVE Hub listing
    (:func:`find_live_filename`): renamed files - e.g. ``ToriiGate-0.5-BF16``
    republished as lowercase ``ToriiGate-0.5-bf16.gguf`` - are corrected to
    their real filename before any request goes out, so stale composed names
    can never produce a mid-download 404.

    ``progress_cb(filename, done_bytes, total_bytes)`` is invoked by the
    caller-side worker thread.  Returns the local paths written/re-used.
    """
    # ---- live filename reconciliation (never trust composed names) --------
    resolved: list[str] = []
    for fn in filenames:
        real = find_live_filename(repo_id, fn)
        if real != fn:
            print(f"[models] '{fn}' not published under that exact name; "
                  f"using live Hub filename '{real}' instead.", flush=True)
        resolved.append(real)
    filenames = resolved

    try:
        preflight_download_check(repo_id, filenames)
    except AppError:
        raise
    except Exception as exc:
        raise AppError.classify(exc, context="download preflight") from exc

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise AppError.classify(exc, context="import huggingface_hub") from exc

    dest_dir = STORAGE.primary_download_dir(repo_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for fn in filenames:
        existing = STORAGE.find_existing(repo_id, fn)
        if existing is not None:
            # Re-use (Unsloth download or previous app download) - no bytes.
            print(f"[models] re-using existing file: {existing}", flush=True)
            out.append(existing)
            continue
        try:
            p = hf_hub_download(
                repo_id=repo_id,
                filename=fn,
                local_dir=str(dest_dir),
            )
            out.append(Path(p))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise AppError.classify(
                exc, context=f"downloading '{fn}' from '{repo_id}'") from exc
    return out


def verify_downloaded(model_file: ModelFile) -> tuple[bool, str]:
    """Post-download sanity check: existence + plausible size (>50% estimate).

    ``huggingface_hub`` already validates LFS SHA256 during download; this is
    the extra local guard requested by the spec ("Download & Verify").
    Also detects leftover *.incomplete files from interrupted transfers.
    """
    p = model_file.local_path()
    if not p.is_file():
        leftovers = sorted(str(x.name) for x in p.parent.glob("*.incomplete")) \
            if p.parent.is_dir() else []
        hint = (f" Interrupted transfer leftovers: {', '.join(leftovers)}."
                if leftovers else "")
        return False, (f"missing file: {p}.{hint} Re-run 'Download & Verify' "
                       "or delete the .incomplete files manually.")
    actual_mb = p.stat().st_size / (1024 * 1024)
    est_mb = model_file.estimated_size_mb
    if actual_mb < est_mb * 0.5:
        return False, (f"{p.name}: size {actual_mb:.0f} MiB far below "
                       f"expected ~{est_mb:.0f} MiB (corrupt/incomplete?) - "
                       "delete the file and re-download.")
    if model_file.mmproj_filename:
        mp = model_file.mmproj_local_path()
        if mp is None or not mp.is_file():
            return False, (f"missing mmproj: {model_file.mmproj_filename} - "
                           "the vision projector is required for image input; "
                           "re-run 'Download & Verify'.")
    return True, f"{p.name}: OK ({actual_mb:.0f} MiB)"


# --------------------------------------------------------------------------
# config.json reading for KV-cache math
# --------------------------------------------------------------------------
@dataclass
class ArchInfo:
    hidden_size: int = 4096
    num_layers: int = 32
    num_kv_heads: int = 8          # GQA/MQA aware
    head_dim: int | None = None
    source: str = "default"

    @property
    def kv_bytes_per_token_f16(self) -> int:
        """K+V bytes for one token at 16-bit storage.

        2 (K and V) * layers * kv_heads * head_dim * dtype_size
        """
        hd = self.head_dim or (self.hidden_size // max(1, self._heads_guess()))
        return 2 * self.num_layers * self.num_kv_heads * hd * 2

    def _heads_guess(self) -> int:
        return max(1, self.num_kv_heads)

    def kv_bytes_per_token(self, cache_type: str) -> int:
        """Per-token KV footprint after quantisation.

        Supported llama.cpp cache types: f16 / bf16 (both 16-bit), fp8,
        q8_0, q4_0, q4_1, iq4_nl.  Unknown names fall back to the f16 size
        (conservative - plans more VRAM usage, never less).
        """
        base = self.kv_bytes_per_token_f16
        factor = {
            "f16": 1.0, "bf16": 1.0,
            "q8_0": 0.5625,   # 8.5 bits effective incl. block scales
            "fp8": 0.53,
            "q4_0": 0.265,    # 4.5 bits effective
            "q4_1": 0.289,
            "iq4_nl": 0.26,
        }.get((cache_type or "f16").lower(), 1.0)
        return int(base * factor)


def read_arch_info(repo_id: str | None, local_gguf: Path | None = None) -> ArchInfo:
    """Best-effort architecture discovery.

    1. If a local GGUF exists, parse its KV metadata (llama.cpp format).
    2. Else fetch ``config.json`` from the HF Hub and map common fields
       (Qwen3-style GQA/MQA included).
    3. Fall back to conservative defaults.
    """
    if local_gguf and local_gguf.is_file():
        info = _read_gguf_metadata(local_gguf)
        if info:
            return info
    if repo_id:
        cfg = _fetch_hf_config(repo_id)
        if cfg:
            return _arch_from_config(cfg)
    return ArchInfo(source="fallback-defaults")


def _fetch_hf_config(repo_id: str) -> dict | None:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _arch_from_config(cfg: dict) -> ArchInfo:
    text_cfg = cfg.get("text_config", cfg)   # multimodal wrappers nest under text_config

    def pick(*names, default=None):
        for n in names:
            if n in text_cfg and text_cfg[n] is not None:
                return text_cfg[n]
        return default

    hidden = int(pick("hidden_size", "d_model", "n_embd", default=4096))
    layers = int(pick("num_hidden_layers", "n_layer", "num_layers", default=32))
    heads = int(pick("num_attention_heads", "n_head", default=32))
    kv_heads = int(pick("num_key_value_heads", "num_kv_heads", default=heads))
    head_dim = pick("head_dim", None)
    head_dim = int(head_dim) if head_dim else hidden // max(1, heads)
    return ArchInfo(hidden_size=hidden, num_layers=layers,
                    num_kv_heads=max(1, kv_heads), head_dim=head_dim,
                    source=f"config.json:{cfg.get('model_type', '?')}")


_GGUF_META_KEYS = {
    "general.architecture": "arch",
    "block_count": "num_layers", "embedding_length": "hidden_size",
    "attention.head_count": "heads", "attention.head_count_kv": "kv_heads",
    "attention.key_length": "head_dim",
}


def _read_gguf_metadata(path: Path) -> ArchInfo | None:
    """Parse just enough of the GGUF header to obtain the KV metadata."""
    try:
        import struct
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
            ver = struct.unpack("<I", fh.read(4))[0]
            n_tensors = struct.unpack("<Q", fh.read(8))[0]
            n_kv = struct.unpack("<Q", fh.read(8))[0]

            def rstr() -> str:
                ln = struct.unpack("<Q", fh.read(8))[0]
                return fh.read(ln).decode("utf-8", "replace")

            def rval(t: int):
                # Minimal scalar reader; arrays/objects skipped conservatively.
                if t == 0: return struct.unpack("<B", fh.read(1))[0]
                if t == 1: return struct.unpack("<b", fh.read(1))[0]
                if t == 2: return struct.unpack("<H", fh.read(2))[0]
                if t == 3: return struct.unpack("<h", fh.read(2))[0]
                if t == 4: return struct.unpack("<I", fh.read(4))[0]
                if t == 5: return struct.unpack("<i", fh.read(4))[0]
                if t == 6: return struct.unpack("<f", fh.read(4))[0]
                if t == 7: return struct.unpack("<Q", fh.read(8))[0]
                if t == 8: return struct.unpack("<q", fh.read(8))[0]
                if t == 9: return rstr()
                if t == 10:  # array: element type + count, skip elements
                    et = struct.unpack("<I", fh.read(4))[0]
                    cnt = struct.unpack("<Q", fh.read(8))[0]
                    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,8:8}
                    if et in sizes:
                        fh.read(sizes[et] * cnt)
                    elif et == 9:
                        for _ in range(cnt):
                            rstr()
                    return None
                raise EOFError(f"unsupported GGUF type {t} (ver {ver})")

            meta: dict[str, object] = {}
            for _ in range(n_kv):
                k = rstr()
                t = struct.unpack("<I", fh.read(4))[0]
                v = rval(t)
                if v is not None:
                    meta[k] = v
    except Exception:
        return None

    arch = str(meta.get("general.architecture", ""))
    def get(suffix, default=None):
        for k, v in meta.items():
            if k.endswith("." + suffix):
                return v
        return default

    hidden = int(get("embedding_length", 4096))
    layers = int(get("block_count", 32))
    heads = int(get("head_count", 32))
    kv_heads = int(get("head_count_kv", heads))
    head_dim = get("key_length")
    head_dim = int(head_dim) if head_dim else hidden // max(1, heads)
    return ArchInfo(hidden_size=hidden, num_layers=layers,
                    num_kv_heads=max(1, kv_heads), head_dim=head_dim,
                    source=f"gguf:{arch or 'unknown'}")
