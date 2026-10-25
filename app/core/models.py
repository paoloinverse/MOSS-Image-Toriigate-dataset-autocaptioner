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
    "Q5_K_M": 5.7, "Q4_K_M": 4.8, "UD-IQ4_XS": 4.4, "IQ4_XS": 4.25,
    "Q3_K_M": 3.9, "Q2_K": 2.6,
}


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
        reporting/downloading even when nothing was found locally.

        Robust to real-world Unsloth layouts where the *projector* is stored
        under a different name than the hardwired default (e.g. Unsloth
        ships ``mmproj-ToriiGate-0.5-Q8_0.gguf`` while our registry expects
        ``ToriiGate-0.5-Q8_0.mmproj.gguf``).  Search order:

        1. the fixed ``mmproj_filename`` via ``find_existing``,
        2. any already-present ``*mmproj*.gguf`` inside this repo's folders
           in the active storage (preferring same-quant, then F16/BF16),
        3. recursive scan of the Unsloth tree (same preference),
        4. fallback candidate name from the spec (for download planning).
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

        # 3. bounded recursive scan of the Unsloth tree (different nesting)
        if STORAGE.mode == STORAGE_UNSLOTH and STORAGE.unsloth_root.strip():
            import os as _os
            all_ggufs: list[Path] = []
            for r in STORAGE.unsloth_search_roots():
                try:
                    base_depth = str(r).rstrip("/").count("/")
                    for dirpath, dirnames, fnames in _os.walk(
                            r, topdown=True, followlinks=False):
                        if dirpath.count("/") - base_depth >= 4:
                            dirnames[:] = []
                        for fn in fnames:
                            if fn.lower().endswith(".gguf"):
                                all_ggufs.append(Path(dirpath) / fn)
                except OSError:
                    continue
            hit = _pick(all_ggufs)
            if hit is not None:
                return hit, hit.name

        # 4. nothing on disk - report the expected name so downloads fetch it
        expected = self.mmproj_filename
        if expected is None:
            specs = _all_specs()
            spec = next((s for s in specs if s.source_repo == self.repo_id), None)
            if spec is not None:
                expected = resolve_mmproj(spec, self)
        return None, expected

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


# Process-wide registry.  ModelFile.resolve_mmproj_companion() needs to look
# up a file's owning spec for fallback naming; AppContext builds its own copy
# (mutable presets live there), so we expose a shared read-only view here.
REGISTRY: dict[str, ModelSpec] = build_registry()


def _all_specs() -> list[ModelSpec]:
    """All known specs, preferring the live AppContext registry if present."""
    try:
        import app.core.models as _m
        ctx = getattr(_m, "_ACTIVE_CTX", None)
        if ctx is not None and getattr(ctx, "registry", None):
            return list(ctx.registry.values())
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

    ``progress_cb(filename, done_bytes, total_bytes)`` is invoked by the
    caller-side worker thread.  Returns the local paths written/re-used.
    """
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
