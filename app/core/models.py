"""Model registry + GGUF/HuggingFace helpers.

Supported models (per spec):

1. Primary  : ``Minthy/ToriiGate-0.5`` – GGUFs hosted at
              ``DraconicDragon/ToriiGate-0.5-GGUF``.  Quants: BF16, Q8_0.
              mmproj: ``ToriiGate-0.5-Q8_0.mmproj.gguf``.
2. Alternate: ``huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF``.
              Quants: Q8_0, Q6_K, UD-IQ4_XS.  The multimodal projector is
              auto-detected from the repo file list (any ``mmproj*.gguf``
              whose quant matches the chosen base quant when possible).

All files are downloaded into ``~/.cache/app_models/`` via ``huggingface_hub``
with SHA verification handled by the hub client itself (etag / LFS checksums)
plus a local size sanity check performed after download ("verify").

Error-management policy for this module: **never** attempt huge allocations
or blind multi-GB downloads inside tests or background threads without an
explicit user action.  All network/disk failures surface as
:class:`ModelOperationError` carrying a verbose, human-readable report that
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
class ModelOperationError(Exception):
    """Actionable error for downloads / verification / architecture lookup.

    ``report()`` renders a multi-line message designed for the GUI log and
    message boxes: what failed, why, and concrete steps to fix it.
    """

    def __init__(self, message: str, detail: str = "", suggestions=None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.suggestions: list[str] = list(suggestions or [])

    def report(self) -> str:
        lines = [f"[models] ERROR: {self.message}"]
        if self.detail:
            lines.append(f"    reason : {self.detail}")
        for s in self.suggestions:
            lines.append(f"    fix    : {s}")
        return "\n".join(lines)

    @classmethod
    def classify(cls, exc: Exception, context: str = "") -> "ModelOperationError":
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
                "space" in lower or isinstance(exc, OSError) and exc.errno == errno_ENOSPC()):
            free = 0
            try:
                free = shutil.disk_usage(CACHE_DIR.parent).free // (1024 ** 3)
            except OSError:
                pass
            return cls("Insufficient disk space for download",
                       base_detail,
                       [f"Free space on {CACHE_DIR}: ~{free} GiB available.",
                        "Delete unused GGUFs from ~/.cache/app_models/ or set a "
                        "larger volume via the CACHE_DIR environment override."])
        return cls("Model operation failed", base_detail,
                   ["Re-run 'Download & Verify' in Tab 1.",
                    "If the problem persists, copy this whole message into the "
                    "bug report."])


def errno_ENOSPC() -> int:
    import errno
    return errno.ENOSPC

CACHE_DIR = Path(os.path.expanduser("~/.cache/app_models"))

# Quantization name -> approximate bits per weight (for size estimation when
# the remote file metadata is unavailable).
QUANT_BITS = {
    "BF16": 16.0, "F16": 16.0, "Q8_0": 8.5, "Q6_K": 6.6,
    "Q5_K_M": 5.7, "Q4_K_M": 4.8, "UD-IQ4_XS": 4.4, "IQ4_XS": 4.25,
    "Q3_K_M": 3.9, "Q2_K": 2.6,
}


@dataclass
class ModelFile:
    """One concrete GGUF choice inside a repository."""
    filename: str            # e.g. "ToriiGate-0.5-Q8_0.gguf"
    quant: str               # e.g. "Q8_0"
    repo_id: str             # HF repo hosting it
    params_billions: float   # rough param count for size estimate
    mmproj_filename: str | None = None   # fixed companion if known

    @property
    def estimated_size_mb(self) -> float:
        bits = QUANT_BITS.get(self.quant, 8.5)
        return self.params_billions * 1e9 * bits / 8 / (1024 * 1024)

    def local_path(self) -> Path:
        return CACHE_DIR / self.repo_id.replace("/", "--") / self.filename

    def mmproj_local_path(self) -> Path | None:
        if not self.mmproj_filename:
            return None
        return CACHE_DIR / self.repo_id.replace("/", "--") / self.mmproj_filename

    def is_downloaded(self) -> bool:
        ok = self.local_path().is_file()
        if ok and self.mmproj_filename:
            ok = self.mmproj_local_path() is not None and self.mmproj_local_path().is_file()
        return ok


@dataclass
class ModelSpec:
    """A selectable model family with its quant dropdown entries."""
    key: str                       # internal id used in configs
    display_name: str
    source_repo: str
    hf_config_repo: str | None     # repo whose config.json describes arch
    files: list[ModelFile] = field(default_factory=list)
    mmproj_candidates: list[str] = field(default_factory=list)
    abliterated: bool = False      # safe to inject explicit-mode triggers
    vision: bool = True

    def quants(self) -> list[str]:
        return [f.quant for f in self.files]

    def file_for_quant(self, quant: str) -> ModelFile | None:
        for f in self.files:
            if f.quant == quant:
                return f
        return None


def build_registry() -> dict[str, ModelSpec]:
    """Hardwired supported-model catalogue."""
    torii_repo = "DraconicDragon/ToriiGate-0.5-GGUF"
    qwen_repo = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"

    torii = ModelSpec(
        key="toriigate",
        display_name="ToriiGate-0.5 (MOSS captioner)",
        source_repo=torii_repo,
        hf_config_repo="Minthy/ToriiGate-0.5",
        files=[
            ModelFile("ToriiGate-0.5-BF16.gguf", "BF16", torii_repo, 0.5),
            ModelFile("ToriiGate-0.5-Q8_0.gguf", "Q8_0", torii_repo, 0.5,
                      mmproj_filename="ToriiGate-0.5-Q8_0.mmproj.gguf"),
        ],
        mmproj_candidates=["ToriiGate-0.5-Q8_0.mmproj.gguf"],
    )

    qwen = ModelSpec(
        key="qwen3.8-27b",
        display_name="Huihui-Qwen3.8-27B-abliterated",
        source_repo=qwen_repo,
        hf_config_repo="huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF",
        files=[
            ModelFile("Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf", "Q8_0", qwen_repo, 27),
            ModelFile("Huihui-Qwen3.8-27B-abliterated-Q6_K.gguf", "Q6_K", qwen_repo, 27),
            ModelFile("Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf",
                      "UD-IQ4_XS", qwen_repo, 27),
        ],
        # Resolved dynamically against the repo listing (see resolve_mmproj).
        mmproj_candidates=[
            "mmproj-Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf",
            "mmproj-F16.gguf",
        ],
        abliterated=True,
    )
    return {torii.key: torii, qwen.key: qwen}


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

    Priority: explicit fixed filename > same-quant mmproj in the repo >
    any F16 mmproj > first mmproj candidate.
    """
    if chosen.mmproj_filename:
        return chosen.mmproj_filename
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
    """Validate disk space + writability BEFORE any bytes hit the wire.

    Raises :class:`ModelOperationError` with a verbose report when the
    download cannot possibly succeed, so the GUI can tell the user exactly
    what to fix instead of failing mid-transfer after minutes of bandwidth.
    """
    if est_total_mb is None:
        est_total_mb = estimate_download_mb(repo_id, filenames)

    dest_dir = CACHE_DIR / repo_id.replace("/", "--")
    try:
        probe_dir = dest_dir if dest_dir.is_dir() else CACHE_DIR.parent
        probe_dir.mkdir(parents=True, exist_ok=True)
        free_mb = shutil.disk_usage(probe_dir).free / (1024 * 1024)
    except OSError as exc:
        raise ModelOperationError(
            "Cannot prepare the model cache directory",
            detail=f"{type(exc).__name__}: {exc} (path: {dest_dir})",
            suggestions=[f"Ensure {CACHE_DIR} is on a writable volume.",
                         "Check permissions: chmod/chown the ~/.cache tree."]) from exc

    # Writability smoke test (tiny file, removed immediately).
    try:
        fd, tmp = tempfile.mkstemp(dir=str(probe_dir), prefix=".preflight-")
        os.write(fd, b"x")
        os.close(fd)
        os.unlink(tmp)
    except OSError as exc:
        raise ModelOperationError(
            "Model cache directory is not writable",
            detail=f"{type(exc).__name__}: {exc} (path: {probe_dir})",
            suggestions=["Fix ownership/permissions of ~/.cache/app_models."]) from exc

    if est_total_mb > 0:
        needed_mb = est_total_mb * 1.05          # 5% slack for temp/LFS overhead
        if needed_mb > free_mb:
            raise ModelOperationError(
                "Not enough disk space for this download",
                detail=(f"{repo_id}: {', '.join(filenames)} needs ~"
                        f"{est_total_mb / 1024:.1f} GiB but only "
                        f"{free_mb / 1024:.1f} GiB is free on {probe_dir}."),
                suggestions=[
                    "Delete unused GGUFs under ~/.cache/app_models/.",
                    "Choose a smaller quantization (e.g. UD-IQ4_XS instead of Q8_0).",
                    "Point the cache at a larger volume before retrying."])
    return free_mb, est_total_mb


def download_model_files(repo_id: str, filenames: list[str],
                         progress_cb=None) -> list[Path]:
    """Blocking download of ``filenames`` from ``repo_id`` into CACHE_DIR.

    Runs ONLY from an explicit user action ("Download & Verify"); performs a
    disk-space/writability preflight first and converts every Hub failure
    into a :class:`ModelOperationError` with remediation hints.

    ``progress_cb(filename, done_bytes, total_bytes)`` is invoked by the
    caller-side worker thread.  Returns the local paths written.
    """
    try:
        preflight_download_check(repo_id, filenames)
    except ModelOperationError:
        raise
    except Exception as exc:
        raise ModelOperationError.classify(exc, context="download preflight") from exc

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ModelOperationError.classify(exc, context="import huggingface_hub") from exc

    dest_dir = CACHE_DIR / repo_id.replace("/", "--")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for fn in filenames:
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
            raise ModelOperationError.classify(
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
        """Per-token KV footprint after quantisation (f16/q8_0/q4_0/fp8)."""
        base = self.kv_bytes_per_token_f16
        factor = {
            "f16": 1.0, "bf16": 1.0,
            "q8_0": 0.5625,   # 8.5 bits effective incl. block scales
            "fp8": 0.53,
            "q4_0": 0.265,    # 4.5 bits effective
            "q4_1": 0.289,
            "iq4_nl": 0.26,
        }.get(cache_type.lower(), 1.0)
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
