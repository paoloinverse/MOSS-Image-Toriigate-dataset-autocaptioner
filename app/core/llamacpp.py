"""llama.cpp discovery, guided installation and autoconfiguration.

The app is only a *manager* around ``llama-server``; the binary itself is
external.  Historically users had to know where their build lived and type
the full path into Tab 2 - error-prone when it is not on PATH.  This module:

* :func:`discover_binaries`  - exhaustive autodetection (PATH, common install
  prefixes, per-user build dirs, flatpak/snap spillovers) with version probing.
* :func:`pick_best_binary`   - ranks discoveries (explicit user choice wins,
  then CUDA > ROCm/Vulkan > plain-CPU builds).
* :class:`InstallRunner`     - QObject worker that clones + compiles llama.cpp
  in a background thread with live log streaming, so the GUI never freezes
  during the (minutes-long) build.  Its :meth:`preflight` enforces the staged
  order: base build tools first, GPU family via PCI enumeration (/sys/bus/pci)
  second, vendor toolchain (nvcc / ROCm / Vulkan SDK) third; only when every
  stage passes does the GUI ask for an install folder and start the clone ->
  configure -> compile sequence.  Missing toolchains HALT with copy-pasteable
  distro commands instead of silently degrading to CPU-only builds.
  Everything heavy is behind an explicit user click with a confirmation
  dialog - nothing here downloads or compiles silently.

No huge allocations, no network access at import time; every failure surfaces
as :class:`~app.core.models.AppError` with a verbose reason + fix report.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from app.core.models import AppError

#: Binary name we look for everywhere.
BINARY_NAME = "llama-server"

#: Default folder used by the guided installer when the user does not pick one.
DEFAULT_INSTALL_ROOT = Path(os.environ.get(
    "APPCAP_LLAMA_DIR", str(Path.home() / ".local" / "share" /
                            "moss-captioner" / "llama.cpp")))


def _exe(name: str) -> str:
    """Platform-aware executable suffix."""
    return name + (".exe" if os.name == "nt" else "")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def candidate_paths() -> list[Path]:
    """Every place we are willing to look for ``llama-server``, best first."""
    exe = _exe(BINARY_NAME)
    home = Path.home()
    raw: list[Path] = []

    # 1. PATH entries (shutil.which handles this too, but listing them gives
    #    the user a verbose "where did you look" answer).
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d.strip():
            raw.append(Path(d) / exe)

    # 2. In-tree / conventional locations.
    cwd = Path.cwd()
    raw += [
        cwd / exe,
        cwd / "bin" / exe,
        cwd / "build" / "bin" / exe,
    ]

    # 3. Common system installs (apt packages, manual make installs).
    raw += [
        Path("/usr/local/bin") / exe,
        Path("/usr/bin") / exe,
        Path("/opt/llama.cpp/bin") / exe,
        Path("/opt/llama.cpp/build/bin") / exe,
    ]

    # 4. Per-user build/install folders people commonly use.
    for sub in ("llama.cpp/build/bin", "llama.cpp/bin", "llama.cpp",
                "llama-cpp/build/bin", "src/llama.cpp/build/bin",
                ".local/bin", ".local/share/moss-captioner/llama.cpp/build/bin",
                "Desktop/llama.cpp/build/bin", "Downloads/llama.cpp/build/bin"):
        raw.append(home / sub / exe)

    # 5. Any sibling checkout next to this repository (dev setups).
    repo_root = Path(__file__).resolve().parents[2]
    for parent in {repo_root.parent, home}:
        for name in ("llama.cpp", "llama-cpp"):
            raw.append(parent / name / "build" / "bin" / exe)
            raw.append(parent / name / "bin" / exe)

    # De-duplicate while preserving order; drop unreadable paths quietly
    # (exotic mounts can raise on stat()).
    seen: set[str] = set()
    out: list[Path] = []
    for p in raw:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        try:
            if p.is_file() and os.access(p, os.X_OK):
                out.append(p)
        except OSError:
            continue
    return out


def probe_version(path: Path, timeout: float = 4.0) -> str:
    """Run ``<path> --version`` and return its first line ('' on failure)."""
    try:
        r = subprocess.run([str(path), "--version"], capture_output=True,
                           text=True, timeout=timeout)
        blob = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        for line in blob.splitlines():
            line = line.strip()
            if line:
                return line[:160]
    except Exception:
        pass
    return ""


#: Cache of per-binary capability probes (help-output based).  Keyed by
#: (resolved path, mtime_ns) so recompiles at the same path invalidate it.
_CAP_CACHE: dict[tuple[str, int], dict[str, bool]] = {}


def _run_help(path: str | Path, timeout: float = 8.0) -> str:
    """Combined stdout+stderr of ``<path> --help`` ('' when unprovable)."""
    try:
        r = subprocess.run([str(path), "--help"], capture_output=True,
                           text=True, timeout=timeout)
        return (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception:
        return ""


#: Cache for accepts_flag() results, keyed like _CAP_CACHE.
_FLAG_CACHE: dict[tuple[str, int], dict[str, bool]] = {}


def accepts_flag(path: str | Path, flag: str, value: str = "",
                 timeout: float = 8.0) -> bool | None:
    """Ask the binary itself whether it accepts ``flag`` (optional ``value``).

    Runs ``<path> <flag> [<value>] --help``: llama.cpp parses its arguments
    *before* printing help, so an unsupported option exits early with
    ``error: invalid argument: ...`` instead of the usage text.  This is far
    more reliable than grepping ``--help`` output, which does not always
    list every accepted flag (e.g. ``--api-key`` was assumed present on a
    build that rejected it with 'invalid argument' + exit code 1, blocking
    every launch).

    Returns True/False from the actual parse result, or None when the check
    could not be performed (missing binary, timeout...).  Callers must
    decide their own conservative fallback for None.
    """
    key: tuple[str, int] | None = None
    try:
        p = Path(str(path))
        key = (str(p.resolve()), p.stat().st_mtime_ns)
    except OSError:
        key = None
    ck = (flag, value)
    if key is not None:
        cached = _FLAG_CACHE.get(key)
        if cached and ck in cached:
            return cached[ck]
    argv = [str(path), flag]
    if value != "":
        argv.append(value)
    argv.append("--help")
    try:
        r = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout)
        blob = ((r.stdout or "") + "\n" + (r.stderr or "")).lower()
    except Exception:
        return None
    # Rejection markers used by common_getopt / gpt4all's arg parser:
    #   'error: invalid argument: --api-key'
    #   'unrecognized option '--api-key''
    low_flag = flag.lower()
    rejected = (("invalid argument" in blob and low_flag in blob)
                or ("unrecognized" in blob and low_flag in blob)
                or ("unknown option" in blob and low_flag in blob))
    if rejected:
        result = False
    elif "usage" in blob or "options" in blob or "--help" in blob:
        # Help actually printed => the parse survived our flag.
        result = True
    else:
        result = None
    if key is not None and result is not None:
        _FLAG_CACHE.setdefault(key, {})[ck] = result
    return result


def _build_info_lines(path: str | Path) -> str:
    """Version/build-info output lines that list compiled features."""
    try:
        r = subprocess.run([str(path), "--version"], capture_output=True,
                           text=True, timeout=8.0)
        blob = (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception:
        return ""
    # llama.cpp prints 'feature: BLAS OPENMP CUDA ...' / 'CUDA RR = ...'
    # style lines; keep only those (the plain version line is not useful).
    return "\n".join(l for l in blob.splitlines() if ":" in l).lower()


def probe_capabilities(path: str | Path) -> dict[str, bool]:
    """Introspect a llama-server binary before launching it.

    Fixes two classes of launch failures seen in the wild:

    * ``error: invalid argument: --api`` - old llama.cpp builds never had
      ``--api-key`` (and some builds print exactly this message); passing
      ``--api`` blindly crashed the server with exit code 1.
    * ``no usable GPU found, --gpu-layers option will be ignored`` - a build
      compiled without CUDA silently ran CPU-only.

    Returns ``{"has_api_key": bool, "has_single_turn": bool,
               "has_reasoning_budget": bool, "has_jinja": bool,
               "flash_attn_auto": bool, "flash_attn_on_off": bool,
               "cuda": bool, "gpu": bool}``.  Every check is conservative:
    when the binary cannot be probed at all we assume the *modern* flags the
    help text hints at exist, but optional security/exotic flags
    (``--api-key``, ``-sp``) default to absent - launching without them is
    always safe, passing them to a build that rejects them exits 1.
    ``gpu`` is True when *any* GPU backend (CUDA/ROCm/Vulkan/Metal)
    was detected.  ``has_jinja`` gates ``--jinja``, which is REQUIRED for
    llama-server to forward per-request ``chat_template_kwargs`` (the
    Qwen3.8 reasoning knobs: enable_thinking / reasoning_effort /
    preserve_thinking) into the model's Jinja chat template.

    Flash-attention spelling matters: ``-af`` / ``--flash-attn`` (plain
    boolean) was deprecated and then *removed* from llama.cpp - current
    builds reject it with "invalid argument" and exit 1, which is exactly
    what used to block launches here.  The supported spellings today are
    ``--flash-attn auto|on|off`` (very recent builds; ``auto`` is the new
    default) or the older ``--flash-attn`` / ``-fa`` boolean flags.  We
    detect which one the probed binary understands so :mod:`app.core.server`
    can pass the right thing (or nothing, for ancient builds).
    """
    p = Path(str(path))
    key: tuple[str, int] | None = None
    try:
        key = (str(p.resolve()), p.stat().st_mtime_ns)
    except OSError:
        key = None
    if key is not None and key in _CAP_CACHE:
        return _CAP_CACHE[key]

    help_txt = _run_help(p).lower()
    info = _build_info_lines(p)
    if not help_txt and not info:
        caps = {"has_api_key": False, "has_single_turn": False,
                "has_reasoning_budget": True, "has_jinja": True,
                "flash_attn_auto": True, "flash_attn_on_off": False,
                "cuda": False, "gpu": False}
    else:
        # Grep the usage text first (cheap), then *verify by parsing*: some
        # builds accept a flag without listing it in --help, and others
        # reject flags the help mentions.  accepts_flag() runs
        # `<binary> <flag> [<value>] --help` and watches for
        # 'error: invalid argument' - the same code path that killed launches
        # with '--api-key' / '-sp' on builds lacking them.  When the binary
        # cannot be probed at all (None) fall back to the help-text guess.
        def verified(flag: str, value: str, help_guess: bool) -> bool:
            actual = accepts_flag(p, flag, value)
            return help_guess if actual is None else actual

        has_api_key = verified("--api-key", "", "--api-key" in help_txt)
        # '-sp' (--single-turn) takes an optional string; probe it with an
        # empty value exactly like server.build_args passes it.
        has_sp = verified("-sp", "", "-sp," in help_txt
                          or "--single-turn" in help_txt)
        has_rb = verified("--reasoning-budget", "1",
                          "reasoning-budget" in help_txt)
        # --jinja: required so per-request chat_template_kwargs (Qwen3.8
        # reasoning knobs) reach the model's Jinja template instead of being
        # ignored by the built-in simplified one.  Boolean flag -> probed
        # with an empty value, exactly like build_args passes it.
        has_jinja = verified("--jinja", "", "--jinja" in help_txt)
        # Flash-attention spelling (see docstring): modern llama.cpp builds
        # take `--flash-attn auto|on|off`; older ones only the boolean
        # `--flash-attn` / `-fa`; ancient ones have nothing.  Detect from the
        # --help text; when the help is unparseable assume the modern form so
        # we never re-introduce the removed '-af' flag.
        fa_auto = False
        fa_on_off = False
        if "--flash-attn" in help_txt:
            idx = help_txt.find("--flash-attn")
            window = help_txt[idx:idx + 120]
            fa_auto = ("{auto" in window.replace(" ", "")
                       or "auto," in window
                       or re.search(r"\bauto\b.*\bon\b.*\boff\b", window)
                           is not None)
            if not fa_auto:
                fa_on_off = ("-fa" in help_txt or "{on" in window.replace(" ", "")
                             or "on, off" in window or "bool" in window)
        if not fa_auto and not fa_on_off:
            # No usable hint in --help: be conservative *forward* - assume the
            # current auto|on|off spelling rather than the removed boolean.
            fa_auto = True
        # GPU detection: prefer the explicit feature list printed by
        # `llama-server --version` ("ggml CUDA_blas", "feature: ... CUDA").
        # Fall back to linked libraries via ldd (catches older builds whose
        # version banner omits the feature list).  ROCm/Vulkan/Metal count as
        # GPU support too - any of them means -ngl/-sm/-ts are honoured.
        cuda = ("cuda" in info) or ("cuda" in help_txt)
        if not cuda:
            ldd = shutil.which("ldd")
            if ldd:
                try:
                    r = subprocess.run([ldd, str(p)], capture_output=True,
                                       text=True, timeout=8.0)
                    out = ((r.stdout or "") + (r.stderr or "")).lower()
                    cuda = ("libcudart" in out or "libcuda.so" in out
                            or "cublas" in out)
                except Exception:
                    pass
        gpu_any = (cuda or "rocm" in info or "hip" in info or "vulkan" in info
                   or "metal" in info or "vulkan" in help_txt
                   or "metal" in help_txt)
        caps = {"has_api_key": has_api_key, "has_single_turn": has_sp,
                "has_reasoning_budget": has_rb, "has_jinja": has_jinja,
                "flash_attn_auto": fa_auto, "flash_attn_on_off": fa_on_off,
                "cuda": cuda, "gpu": gpu_any}
    if key is not None:
        _CAP_CACHE[key] = caps
    return caps


def classify_backend(version_line: str, path: Path) -> str:
    """Rough 'cuda' | 'rocm' | 'vulkan' | 'metal' | 'cpu' guess."""
    blob = (version_line + " " + str(path)).lower()
    if "cuda" in blob:
        return "cuda"
    if "rocm" in blob or "hip" in blob:
        return "rocm"
    if "vulkan" in blob:
        return "vulkan"
    if "metal" in blob:
        return "metal"
    return "cpu"


def discover_binaries() -> list[dict]:
    """All usable llama-server binaries with metadata.

    Returns dicts: ``{path, version, backend, score}`` sorted best-first.
    """
    found: list[dict] = []
    seen_real: set[str] = set()
    for p in candidate_paths():
        try:
            real = str(p.resolve())
        except OSError:
            real = str(p)
        if real in seen_real:
            continue
        seen_real.add(real)
        ver = probe_version(p)
        backend = classify_backend(ver, p)
        score = {"cuda": 4, "rocm": 3, "vulkan": 2, "metal": 2, "cpu": 1}[backend]
        # Builds living under our own installer get a small bonus: they were
        # configured deliberately for this app.
        if ".local/share/moss-captioner" in real or "APPCAP_LLAMA_DIR" in real:
            score += 0.5
        found.append({"path": str(p), "version": ver,
                      "backend": backend, "score": score})
    found.sort(key=lambda d: (-d["score"], d["path"]))
    return found


def describe_search() -> str:
    """Multi-line summary of where we looked - used in error reports."""
    lines = ["searched:",
             f"  - PATH ({len(os.environ.get('PATH', '').split(os.pathsep))} entries)",
             "  - ./ , ./bin/ , ./build/bin/ (current directory)",
             "  - /usr/local/bin, /usr/bin, /opt/llama.cpp[/build]/bin",
             f"  - ~/{'llama.cpp/build/bin'} and other per-user build folders",
             f"  - guided-installer default: {DEFAULT_INSTALL_ROOT}/build/bin"]
    return "\n".join(lines)


def pick_best(preferred: str = "") -> dict | None:
    """Best discovered binary; ``preferred`` (Tab 2 field) wins when valid."""
    preferred = (preferred or "").strip()
    if preferred and preferred != BINARY_NAME:
        p = Path(os.path.expanduser(preferred))
        try:
            ok = p.is_file() and os.access(p, os.X_OK)
        except OSError:
            ok = False
        if not ok and "/" not in preferred:
            w = shutil.which(preferred)
            if w:
                p, ok = Path(w), True
        if ok:
            ver = probe_version(p)
            return {"path": str(p), "version": ver,
                    "backend": classify_backend(ver, p), "score": 99,
                    "preferred": True}
    cands = discover_binaries()
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# Missing-package detection -> copy-pasteable fix commands
# ---------------------------------------------------------------------------
#: Well-known toolchain / GPU-SDK packages, keyed by distro family.  Used to
#: turn "nvcc / g++ / cmake / vulkan.h is missing" into ready-to-run command
#: lines instead of a cryptic mid-compile failure.
_FIX_PACKAGES: dict[str, dict[str, list[str]]] = {
    "cuda": {
        "apt": ["nvidia-cuda-toolkit"],
        "dnf": ["gcc-c++", "cuda-toolkit"],
        "pacman": ["cuda"],
        "zypper": ["cuda-toolkit"],
    },
    # A CUDA *compiler* (nvcc) can be present while the math libraries are
    # not: minimal toolkit installs (or NVIDIA-repo 'cuda-compiler-13-x'
    # packages) ship nvcc but no libcublas headers/libs, so cmake finds the
    # toolkit yet the generate step dies with
    #   Target "ggml-cuda" links to: CUDA::cublas ... but the target was
    #   not found.
    # NOTE: there is deliberately NO static package list here for apt.
    # Version-suffixed names such as 'libcublas-dev-13-1' (Ubuntu multiverse)
    # or 'cuda-toolkit-13-1' (NVIDIA apt repo) change with *every* CUDA
    # release, and unversioned spellings do not exist in every repo - a
    # hard-coded list would rot the moment CUDA updates.  For apt the fix
    # command is resolved dynamically at runtime instead: search the local
    # apt cache for the versioned -dev packages matching the installed nvcc
    # (see InstallRunner._apt_cublas_candidates / _cuda_libs_fix_command),
    # falling back to a self-explanatory 'apt-cache search' recipe that the
    # user can run themselves.  The other managers keep best-effort lists.
    "cuda_libs": {
        "dnf": ["cuda-cublas-devel", "cuda-curand-devel",
                "cuda-cudnn-devel"],
        "pacman": ["cblas", "libcublas"],
        "zypper": ["libcublas-devel", "libcurand-devel"],
    },
    "vulkan": {
        "apt": ["libvulkan-dev", "vulkan-tools", "glslang-tools"],
        "dnf": ["vulkan-devel", "glslang", "vulkan-tools"],
        "pacman": ["vulkan-devel", "glslang"],
        "zypper": ["vulkan-devel", "glslang-tools", "vulkan-tools"],
    },
    "rocm": {
        "apt": ["hip-runtime-amd", "hip-dev"],
        "dnf": ["hip-runtime-amd", "hip-dev"],
        "pacman": ["hip-runtime-amd"],
        "zypper": ["rocm-hip-runtime"],
    },
    "buildtools": {
        "apt": ["git", "cmake", "build-essential"],
        "dnf": ["git", "cmake", "gcc", "gcc-c++", "make"],
        "pacman": ["git", "cmake", "gcc", "make"],
        "zypper": ["git", "cmake", "gcc", "gcc-c++", "make"],
    },
}


def detect_pkg_manager() -> str | None:
    """Return 'apt' | 'dnf' | 'pacman' | 'zypper' (best guess) or None."""
    if sys.platform == "darwin":
        return "brew"
    if shutil.which("apt-get") or shutil.which("apt"):
        return "apt"
    if shutil.which("dnf") or shutil.which("yum"):
        return "dnf"
    if shutil.which("pacman"):
        return "pacman"
    if shutil.which("zypper"):
        return "zypper"
    # Fall back to os-release when the manager itself is not on PATH.
    try:
        txt = Path("/etc/os-release").read_text(errors="replace").lower()
    except OSError:
        return None
    if any(d in txt for d in ("debian", "ubuntu", "mint", "pop")):
        return "apt"
    if any(d in txt for d in ("fedora", "rhel", "centos", "rocky", "alma")):
        return "dnf"
    if any(d in txt for d in ("arch", "manjaro", "endeavour")):
        return "pacman"
    if "suse" in txt or "opensuse" in txt:
        return "zypper"
    return None


#: Ubuntu-family ids in /etc/os-release ID / ID_LIKE (copy-pasteable apt lines
#: are only promised for these; other distros get per-manager commands too).
_UBUNTU_FAMILY = ("ubuntu", "xubuntu", "kubuntu", "lubuntu", "mint", "pop",
                  "zorin", "elementary", "linuxmint")


def is_ubuntu_family() -> bool:
    """True when the OS identifies as Ubuntu (or an Ubuntu derivative)."""
    try:
        txt = Path("/etc/os-release").read_text(errors="replace").lower()
    except OSError:
        return False
    ids: list[str] = []
    for m in re.finditer(r'^(?:ID|ID_LIKE)=(.*)$', txt, re.M):
        ids += m.group(1).strip('"').split()
    return any(i in _UBUNTU_FAMILY for i in ids)


def _install_verb(mgr: str) -> str:
    return {"apt": "sudo apt install",
            "dnf": "sudo dnf install",
            "yum": "sudo dnf install",
            "pacman": "sudo pacman -S",
            "zypper": "sudo zypper install",
            "brew": "brew install"}.get(mgr, "your package manager to install")


def fix_command(group: str, mgr: str | None = None) -> str:
    """Copy-pasteable privileged command line that installs a package group.

    ``group`` is a key of :data:`_FIX_PACKAGES` ('cuda', 'vulkan', 'rocm',
    'buildtools').  When the distro's package manager cannot be determined a
    generic hint is produced instead of a bogus command.
    """
    mgr = mgr or detect_pkg_manager()
    pkgs = _FIX_PACKAGES.get(group, {})
    if mgr and mgr in pkgs:
        return f"{_install_verb(mgr)} {' '.join(pkgs[mgr])}"
    if mgr == "brew":  # no GPU SDKs on macOS; generic fallback below
        return f"brew install {' '.join(pkgs.get('apt', []))}"
    listed = ", ".join(sorted({p for lst in pkgs.values() for p in lst}))
    return (f"install these packages with your distro's package manager "
            f"(sudo …): {listed}")


def manual_hint(group: str) -> str:
    """Software names to install manually when no package manager was found."""
    listed = ", ".join(sorted({p for lst in _FIX_PACKAGES.get(group, {}).values()
                               for p in lst}))
    return (f"No supported package manager detected - install these manually "
            f"for your distro: {listed}")


# ---------------------------------------------------------------------------
# Future-proof CUDA math-library fix commands (apt cache lookup)
# ---------------------------------------------------------------------------
#: NVIDIA CUDA libraries whose *development* packages ggml-cuda links
#: against.  cublas is mandatory; curand/cudnn are optional but ship in the
#: same versioned family, so we search for them too and include whichever
#: the user's repos actually carry.
_CUDA_LIB_STEMS = ("cublas", "curand", "cudnn")


def _apt_cache_search(patterns: list[str]) -> set[str]:
    """Package names from `apt-cache search --names-only` for regex patterns.

    Returns an empty set when apt-cache is unavailable or fails; every
    exception is swallowed because this runs on a UI error path where any
    answer beats none.
    """
    found: set[str] = set()
    exe = shutil.which("apt-cache")
    if not exe:
        return found
    try:
        proc = subprocess.run([exe, "search", "--names-only"] + patterns,
                              capture_output=True, text=True, timeout=10.0)
    except Exception:
        return found
    if proc.returncode != 0 and not proc.stdout:
        return found
    for line in proc.stdout.splitlines():
        name = line.strip().split(" ", 1)[0].strip()
        # 'apt-cache search' prints 'NAME - short description'; plain names
        # only ever contain lowercase letters/digits/+-.  Anything else
        # (e.g. 'No matches') is discarded.
        if name and re.fullmatch(r"[a-z0-9][a-z0-9+.\-]*", name):
            found.add(name)
    return found


def _cuda_version_dots(nvcc_ver: str) -> str:
    """Normalise an nvcc version ('13.1', '13.1.2') to 'MAJOR.MINOR'."""
    m = re.match(r"(\d+)\.(\d+)", nvcc_ver or "")
    return f"{m.group(1)}.{m.group(2)}" if m else ""


def _cuda_version_dashes(nvcc_ver: str) -> str:
    """Normalise an nvcc version to the repo suffix 'MAJOR-MINOR'."""
    d = _cuda_version_dots(nvcc_ver)
    return d.replace(".", "-") if d else ""


def apt_cuda_libs_fix_command(nvcc_ver: str = "",
                              search_fn=None) -> tuple[list[str], list[str]]:
    """Resolve copy-pasteable apt commands for the missing cuBLAS dev libs.

    Never hard-codes version-suffixed package names: those change with every
    CUDA release (libcublas-dev-13-1 today, libcublas-dev-14-0 tomorrow),
    and unversioned spellings do not exist in every repo.  Instead we query
    the *local apt cache* for candidates that match the installed toolkit
    version, so the suggestion is always a package the user's own repos can
    actually deliver.

    Candidate families, tried in order of precision:
      1. Ubuntu multiverse split packages:   libcublas-dev[-X.Y(-Z)]
      2. NVIDIA-repo metapackage:            cuda-toolkit-X.Y[-Z]
      3. NVIDIA-repo component packages:     cuda-libraries-dev-X-Y[-Z]

    Returns ``(commands, notes)``: ready-to-run lines first, then plain
    guidance strings (empty when nothing concrete could be resolved).  When
    the cache yields no exact match the user gets a self-explanatory
    ``apt-cache search`` recipe instead of a guessed name that would fail.
    """
    search = search_fn or _apt_cache_search
    dashes = _cuda_version_dashes(nvcc_ver)          # e.g. '13-1'
    dots = _cuda_version_dots(nvcc_ver)              # e.g. '13.1'

    # Regex families used both to query the apt cache and to filter its
    # answers.  Every pattern is *anchored* and pinned to the installed
    # major.minor (or major-minor) whenever the version is known - never a
    # blind wildcard that could suggest a package from a different CUDA
    # series than the toolkit in use.
    lib_dev_pats = tuple(rf"lib{stem}-dev([-_.]\d[^ ]*)?"
                         for stem in _CUDA_LIB_STEMS)
    tk_pats = ((rf"cuda-toolkit-{re.escape(dashes)}([-_.]\d[^ ]*)?",)
               if dashes else ())
    comp_pats = ((rf"cuda-libraries-dev-{re.escape(dashes)}([-_.]\d[^ ]*)?",)
                 if dashes else ())
    pats: list[str] = []
    for p in (*lib_dev_pats, *tk_pats, *comp_pats):
        pats.append("^" + p + "$")
    if not dashes:   # unknown nvcc version: accept any versioned spelling
        for stem in _CUDA_LIB_STEMS:
            pats.append(rf"^lib{stem}-dev([-_.]\d[^ ]*)?$")

    hits = search(pats)

    def pick(prefixes: tuple[str, ...]) -> list[str]:
        sel = [n for n in hits
               if any(re.fullmatch(p, n) for p in prefixes)]
        return sorted(sel, key=lambda s: (len(s), s))[:3]

    commands: list[str] = []
    notes: list[str] = []
    libdev = pick(lib_dev_pats)
    toolkit = pick(tk_pats)
    comp = pick(comp_pats)

    # The *smallest* set that definitely satisfies ggml-cuda is the mandatory
    # cuBLAS dev package(s); curand/cudnn are optional in llama.cpp's CMake,
    # so list them only when they share a family with cublas (i.e. installing
    # one command pulls everything).  A fat multi-hundred-MB metapackage is
    # never suggested as the primary line when a lean split package exists.
    must = [n for n in libdev if "cublas" in n]
    if must:
        commands.append(f"sudo apt install {' '.join(must)}")
    elif libdev:
        commands.append(f"sudo apt install {' '.join(libdev)}")
    alt = ([f"sudo apt install {' '.join(toolkit)}"] if toolkit else
           [f"sudo apt install {' '.join(comp)}"] if comp else [])
    if alt:
        commands.append(alt[0] + "   # alternative: full toolkit metapackage")

    if not commands:
        # Nothing in the local cache matches.  Do NOT guess a versioned name
        # (it would rot or simply not exist); hand the user the discovery
        # recipe plus the two canonical package families to look for.
        probe = "apt-cache search -n '^lib(cublas|curand|cudnn)-dev'"
        if dashes:
            probe += f" ; apt-cache search -n 'cuda-(toolkit|libraries-dev)-{re.escape(dashes)}'"
            notes.append(
                f"No package matching CUDA {dots or dashes} was found in "
                "your apt cache. Run the search command above; if it lists "
                "nothing, either enable the universe/multiverse components "
                "(Ubuntu ships libcublas-dev-* there) or add the NVIDIA CUDA "
                "apt repository (it provides cuda-toolkit-"
                f"{dashes}/cuda-libraries-dev-{dashes}).")
        else:
            notes.append(
                "Could not determine your CUDA version from nvcc, and the "
                "apt cache search returned nothing. Run the search command "
                "above to find the 'libcublas-dev*' package matching your "
                "toolkit, then sudo apt install it.")
        commands.append(probe + "   # then: sudo apt install <the -dev package(s) it lists>")
    else:
        notes.append(
            "These names were resolved live from your apt cache against the "
            "installed nvcc version, so they exist in your repos today. If "
            "apt reports 'Unable to locate', run: sudo apt update && retry, "
            "or use the alternative command listed next.")
    return commands, notes


def _cuda_libs_fix_command(mgr: str | None, nvcc_ver: str,
                           search_fn=None) -> tuple[list[str], list[str]]:
    """Fix commands + notes for the 'cuda_libs' group, per package manager.

    apt gets the dynamic cache-resolved spelling (future-proof across CUDA
    releases); other managers fall back to their static best-effort lists.
    ``search_fn`` lets callers (and tests) inject a custom apt-cache query.
    Returns ``(commands, notes)``, mirroring :func:`apt_cuda_libs_fix_command`.
    """
    if mgr == "apt":
        return apt_cuda_libs_fix_command(nvcc_ver, search_fn=search_fn)
    return ([fix_command("cuda_libs", mgr)],
            ["Package names above come from the app's built-in table for your "
             "package manager; verify with its search command if apt/dnf "
             "reports 'Unable to locate'."])


# ---------------------------------------------------------------------------
# Preflight: base build tools, GPU family (PCI enumeration), vendor toolchains
# ---------------------------------------------------------------------------
#: PCI base class for display controllers (0x03).  Only its *graphics*
#: programming interfaces count as GPUs: 0x0300 VGA-compatible, 0x0302
#: 3D controller (the class modern discrete NVIDIA/AMD cards report) and
#: 0x0380 other display controller.  The sibling subclass 0x04xx is
#: *Multimedia audio* - every discrete GPU ships an HDMI/DP audio function
#: (e.g. an RTX 5060 Ti exposes 01:00.0 class 0x030000 + 01:00.1 class
#: 0x040300) and motherboards expose a 0x040300 HDA controller under the
#: chipset vendor ID.  Matching on the whole 0x03 prefix was a bug: it let
#: those audio endpoints masquerade as GPUs, so machines with e.g. an AMD
#: Threadripper PCIe virtual-HDMI function (0x1022:0x145a, class 0x038000)
#: were tagged 'amd' and the preflight demanded a ROCm toolchain that the
#: user's NVIDIA-only box could never satisfy.
_PCI_GRAPHICS_IFACES = ("0x0300", "0x0302", "0x0380")
#: Well-known GPU vendor PCI IDs: 0x10de NVIDIA, 0x1002 AMD/ATI,
#: 0x8086 Intel (iGPU - usable through the Vulkan backend).
_GPU_VENDOR_IDS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}


def enumerate_gpu_families() -> set[str]:
    """Hardware-level GPU vendor enumeration - *not* toolkit based.

    Walks ``/sys/bus/pci/devices/*/class`` + ``vendor`` and maps **graphics**
    display controllers (PCI class 0x0300/0x0302/0x0380 - see
    :data:`_PCI_GRAPHICS_IFACES` for why 0x04xx audio functions are excluded)
    to 'nvidia' | 'amd' | 'intel'.  This works even when no driver, CUDA
    toolkit or ROCm stack is installed (which is precisely the case where the
    old ``shutil.which('nvcc')``-style probing silently degraded the guided
    build to CPU-only).  Non-Linux / unreadable sysfs yields an empty set;
    callers then fall back to toolkit-based detection.
    """
    families: set[str] = set()
    base = Path("/sys/bus/pci/devices")
    try:
        devs = sorted(base.iterdir()) if base.is_dir() else []
    except OSError:
        return families
    for d in devs:
        try:
            cls = (d / "class").read_text().strip().lower()
            if not cls.startswith(_PCI_GRAPHICS_IFACES):
                continue
            ven = (d / "vendor").read_text().strip().lower()
            fam = _GPU_VENDOR_IDS.get(ven)
            if fam:
                families.add(fam)
        except OSError:
            continue
    return families


class PreflightError(AppError):
    """Raised by :meth:`InstallRunner.preflight` - carries ready-to-run fixes.

    ``commands`` holds copy-pasteable shell lines (they need sudo/root, so we
    only *offer* them); ``manual`` holds software names for systems without a
    supported package manager.  The Tab 2 handler renders both verbatim.
    """

    prefix = "[install]"

    def __init__(self, message: str, detail: str = "", suggestions=None,
                 commands=None, manual=None):
        super().__init__(message, detail=detail, suggestions=suggestions)
        self.commands: list[str] = list(commands or [])
        self.manual: list[str] = list(manual or [])

    def report(self) -> str:
        base = super().report()
        if self.commands:
            base += ("\n    run    : copy-paste into a terminal (needs "
                     "sudo/root):"
                     + "".join(f"\n             $ {c}" for c in self.commands))
        for hint in self.manual:
            base += f"\n    note   : {hint}"
        return base


class PreflightResult:
    """Outcome of :meth:`InstallRunner.preflight` (all cheap checks, no I/O).

    Attributes:
        ok: True when a build can proceed.
        error: :class:`PreflightError` describing what blocks the build.
        gpu_families: hardware-enumerated vendors ('nvidia'/'amd'/'intel').
        backend: validated backend to build ('cuda'|'rocm'|'vulkan'|'cpu').
        reason: one-line human summary of how ``backend`` was chosen.
    """

    def __init__(self):
        self.ok = False
        self.error: PreflightError | None = None
        self.gpu_families: set[str] = set()
        self.backend = "cpu"
        self.reason = ""

    @property
    def commands(self) -> list[str]:
        return self.error.commands if self.error else []

    @property
    def manual(self) -> list[str]:
        return self.error.manual if self.error else []


#: Disk footprint per backend (GiB): shallow clone ~0.4 + build tree.  CUDA
#: compiles far larger objects (ptxas embeds per-arch cubins) than CPU.
_DISK_GB_BY_BACKEND = {"cuda": 8.0, "rocm": 5.0, "vulkan": 4.0, "cpu": 3.0}


# ---------------------------------------------------------------------------
# Guided installation (clone + cmake build) - runs inside a QThread
# ---------------------------------------------------------------------------
class InstallRunner(QObject):
    """Background clone+compile of llama.cpp with streamed logs.

    Signals:
        log(str)         - progress lines for the GUI log box
        finished(bool, str) - success flag + final message (report on failure)

    Usage contract (Tab 2 'Auto-install / build'): the GUI calls
    :meth:`preflight` *before* asking the user anything; only when it returns
    ``ok`` does it prompt for the install folder and start :meth:`start`.
    ``preflight`` never raises - inspect ``result.ok`` / ``result.error``.
    """

    log = Signal(str)
    finished = Signal(bool, str)

    GIT_URL = "https://github.com/ggml-org/llama.cpp.git"

    def __init__(self, dest_root: str | Path = DEFAULT_INSTALL_ROOT,
                 backend: str = "auto", parent=None):
        super().__init__(parent)
        self.dest = Path(os.path.expanduser(str(dest_root)))
        self.backend = backend          # auto | cuda | rocm | vulkan | cpu
        self._cancel = threading.Event()
        self._preflight: PreflightResult | None = None
        #: CUDA toolkit root discovered during the last preflight() - reused
        #: by _stream()'s error translation so a mid-configure cuBLAS failure
        #: can name the exact cuda-toolkit-X-Y metapackage to install.
        self._cuda_root: Path | None = None
        self._configure_retried = False

    # ------------------------------------------------------------- public
    def cancel(self):
        self._cancel.set()
        self.log.emit("[install] cancellation requested - stopping after the "
                      "current build step finishes.")

    # -------------------------------------------------------- preflight
    def preflight(self) -> PreflightResult:
        """Run the full staged pre-install validation *without* side effects.

        Stage order (each stage halts hard before the next one runs, so the
        user is never asked for a folder while the toolchain is broken):

        1. Base build tools: git, cmake (>= 3.14) and a C++ compiler.
           Missing -> stop with distro-specific copy-pasteable commands.
        2. GPU family by PCI enumeration (/sys/bus/pci), NOT by toolkit
           presence - a bare NVIDIA machine with no CUDA SDK still gets
           detected as 'nvidia' instead of silently building CPU-only.
        3. Vendor toolchain for the detected family (nvcc for NVIDIA,
           hipcc/ROCm for AMD, full Vulkan SDK for Intel/AMD iGPUs).
           Missing -> stop with guidance; NO silent degradation to CPU.
        4. Backend selection + disk estimate based on the validated backend.

        The result is cached on ``self._preflight`` and reused by :meth:`run`
        (re-call explicitly to refresh after the user installs packages).
        """
        res = PreflightResult()
        err = self._check_base_tools()
        if err:
            res.error = err
            self._preflight = res
            return res

        # -- Stage 2: hardware GPU family (PCI IDs, driver-independent) -----
        hw = enumerate_gpu_families()
        has_nv_driver = self._has_nvidia_driver()
        rocm_present = self._has_rocm()
        # Driver-truth override (mirrors Tab 1's NVML-based detection): when
        # the NVIDIA kernel driver demonstrably owns at least one GPU, any
        # 'amd'/'intel' family that PCI enumeration added comes from *other*
        # silicon - a chipset virtual-display function, a dead legacy ATI
        # card, or an integrated GPU the BIOS disabled in favour of the
        # discrete card.  Such phantom families must not hijack backend
        # selection or demand a ROCm toolchain this NVIDIA-only box can
        # never satisfy (the exact false-positive seen with RTX cards whose
        # platform exposes an AMD-family display function).
        amd_real = self._amdgpu_driver_loaded()
        if has_nv_driver and "nvidia" in hw:
            if not amd_real:
                hw.discard("amd")
                hw.discard("intel")
        # On non-Linux (or empty sysfs) fall back to toolkit/driver evidence.
        elif not hw:
            if has_nv_driver:
                hw.add("nvidia")
            if rocm_present:
                hw.add("amd")
        # A loaded ROCm *kernel driver* is stronger proof of a real AMD GPU
        # than the /opt/rocm toolkit directory alone (users install that on
        # NVIDIA boxes too, e.g. for HIP-SDK builds).  Add 'amd' when amdgpu
        # is bound to a device so an iGPU/dGPU that sysfs enumeration missed
        # still gets its toolchain demanded - and never removed later.
        if amd_real:
            hw.add("amd")
        res.gpu_families = hw

        # -- Stage 3: vendor-specific toolchain must match the hardware -----
        cuda_root = self._find_cuda_root()
        self._cuda_root = cuda_root         # also used by _stream() error hints
        has_nvcc = self._has_nvcc(cuda_root)
        # nvcc alone is NOT a complete CUDA toolkit: minimal installs (the
        # NVIDIA-repo 'cuda-compiler-*' metapackage, distro split packages)
        # ship the compiler but no math libraries.  llama.cpp's ggml-cuda
        # links CUDA::cublas / curand / cudnn, so cmake *configures* fine and
        # then dies in the generate step with
        #   Target "ggml-cuda" links to: CUDA::cublas ... not found.
        # Verify the linkable dev libraries during preflight instead of
        # letting the user discover it after a multi-minute download.
        missing_cuda_libs: list[str] = []
        if has_nvcc and cuda_root is not None:
            missing_cuda_libs = self._missing_cuda_libs(cuda_root)
        rocm_root = self._find_rocm_root()
        has_hipcc = (shutil.which("hipcc") is not None
                     or (rocm_root / "bin" / _exe("hipcc")).is_file())
        vulkan_ready = self._has_vulkan()
        want = self.backend if self.backend != "auto" else None

        blocked: list[tuple[str, str]] = []     # (reason, fix package group)
        forced_cpu = os.environ.get("APPCAP_LLAMA_FORCE_CPU") == "1"
        if "nvidia" in hw and not has_nvcc and not forced_cpu:
            # Hardware present but no CUDA compiler anywhere -> HALT.  The old
            # behaviour silently built a CPU-only binary whose -ngl flags were
            # ignored; users only noticed hours later via out-of-RAM runs.
            detail = ("no kernel driver detected either (nvidia-smi / "
                      "/proc/driver/nvidia absent), so the GPU is unusable "
                      "as-is" if not has_nv_driver else
                      "the NVIDIA driver is loaded, only the toolkit is "
                      "missing")
            blocked.append((
                f"NVIDIA GPU(s) present but no CUDA compiler (nvcc) found - "
                f"{detail}", "cuda"))
        elif has_nvcc and any(n in missing_cuda_libs
                              for n in self._CUDA_REQUIRED_LIBS):
            # nvcc is there, cuBLAS dev files are not: cmake would find
            # 'CUDAToolkit' but generate would die on CUDA::cublas.  HALT now,
            # with the exact packages to install, instead of failing twice
            # after a multi-minute download+configure.  (Missing curand/cudnn
            # alone do NOT block - those imported targets are optional in
            # llama.cpp's CMake; they only appear in the fix command.)
            blocked.append((
                f"CUDA toolkit at {cuda_root} is incomplete: nvcc works but "
                f"the development libraries ggml-cuda links against are "
                f"missing ({', '.join(missing_cuda_libs)}). A minimal "
                "'cuda-compiler' install is not enough - llama.cpp needs "
                "cuBLAS dev files too.", "cuda_libs"))
        if "amd" in hw and not forced_cpu:
            # Only HALT when the AMD family is backed by *real usable AMD
            # compute hardware*: a driver-owned card (amdgpu loaded) with no
            # HIP compiler.  'amd' can also enter `hw` through the toolkit-
            # only fallback (a leftover /opt/rocm directory on an NVIDIA box)
            # or as PCI-enumerated silicon whose kernel driver is absent; in
            # both cases ROCm cannot work here, so instead of blocking the
            # (perfectly valid) CUDA/CPU build we drop 'amd' from the family
            # set - it must not hijack backend selection further down either.
            if not has_hipcc:
                cuda_can_serve = ("nvidia" in hw and has_nvcc) or \
                                 self._has_nvidia_smi()
                if amd_real and not cuda_can_serve:
                    blocked.append((
                        "AMD GPU(s) present but no ROCm/HIP compiler found "
                        "(hipcc missing, /opt/rocm/bin/hipcc absent)", "rocm"))
                else:
                    # Either phantom AMD silicon (no amdgpu driver bound), or
                    # a real-but-secondary AMD display chip on an NVIDIA box
                    # where CUDA already covers inference: dropping 'amd' here
                    # mirrors Tab 1's NVML truth instead of demanding a ROCm
                    # stack this machine will never use.
                    hw.discard("amd")
                    res.gpu_families = hw
            elif not amd_real:
                # hipcc exists (e.g. real ROCm install) but nothing AMD is
                # actually drivable here -> phantom family from the toolkit
                # fallback; keep it out of the report and the choice logic.
                hw.discard("amd")
                res.gpu_families = hw
        if want == "vulkan" and not vulkan_ready and not forced_cpu:
            blocked.append((
                "Vulkan backend requested but the development SDK is "
                "incomplete (needs libvulkan.so, vulkan/vulkan.h, glslc)",
                "vulkan"))

        if blocked:
            groups = list(dict.fromkeys(g for _, g in blocked))
            mgr = detect_pkg_manager()
            # 'cuda_libs' is resolved dynamically (apt cache lookup pinned to
            # the installed nvcc version) because its package names are
            # version-suffixed and change with every CUDA release; every
            # other group uses its static per-manager list.
            cmds: list[str] = []
            cuda_notes: list[str] = []
            for g in groups:
                if g == "cuda_libs":
                    ver = (self._cuda_toolkit_version(cuda_root)
                           if cuda_root is not None else "")
                    # route *all* managers through the helper so the apt path
                    # always uses this runner's (injectable) cache lookup and
                    # non-apt managers get their static list + a note.
                    c, n = _cuda_libs_fix_command(mgr, ver,
                                                  search_fn=self._apt_cache_search)
                    cmds += c
                    cuda_notes += n
                else:
                    cmds.append(fix_command(g, mgr))
            manuals = ([manual_hint(g) for g in groups] if mgr is None
                       else ([] if is_ubuntu_family() else
                             [f"Package manager detected: {mgr}\n"
                              "(the exact package names above may differ on "
                              "your distro - verify with your package "
                              "manager's search command)"]))
            sug = ["Install the packages above (requires sudo/root), then "
                   "press 'Auto-install / build' again."]
            if "cuda_libs" in groups:
                sug.append("You do NOT need to reinstall CUDA itself - nvcc "
                           "is fine; only the math-library dev packages are "
                           "missing. No need to delete the build folder: "
                           "the installer re-runs cmake automatically.")
                sug += cuda_notes
            sug += ["Or point the 'Binary:' field at an existing llama-server "
                    "('Autodetect' / 'Browse…') and skip building entirely.",
                    "Only if you really want a CPU-only build: set "
                    "APPCAP_LLAMA_FORCE_CPU=1 before launching."]
            res.error = PreflightError(
                "Vendor GPU toolchain incomplete - refusing to build",
                detail="; ".join(r for r, _ in blocked)
                       + ". Building CPU-only would silently ignore -ngl, "
                         "so the install was stopped before any download.",
                suggestions=sug,
                commands=cmds,
                manual=manuals if mgr is None else [])
            self._preflight = res
            return res

        # -- Stage 4: choose the backend -------------------------------------
        # NOTE: rocm_present is deliberately NOT used here - it merely means
        # "/opt/rocm exists or hipcc on PATH", which can be a leftover SDK
        # directory on a machine without any AMD GPU (or an iGPU that must
        # not hijack a CUDA build).  The validated ROCm toolchain from stage
        # 3 (has_hipcc) plus real AMD hardware in `hw` is what counts.
        if os.environ.get("APPCAP_LLAMA_FORCE_CPU") == "1":
            res.backend, res.reason = "cpu", "CPU forced via APPCAP_LLAMA_FORCE_CPU=1"
        elif want:
            res.backend, res.reason = want, f"explicitly requested {want.upper()}"
        elif "nvidia" in hw and has_nvcc:
            res.backend, res.reason = "cuda", "NVIDIA GPU + CUDA toolkit (nvcc)"
        elif "amd" in hw and has_hipcc:
            res.backend, res.reason = "rocm", "AMD GPU + ROCm/HIP toolchain (hipcc)"
        elif hw and vulkan_ready:
            res.backend, res.reason = "vulkan", "GPU present, Vulkan SDK complete"
        elif not hw:
            res.backend, res.reason = "cpu", "no GPU detected in /sys/bus/pci"
        else:
            # Unreachable: `blocked` above already stops this case.
            res.error = PreflightError(
                f"No usable toolchain for the detected GPU family "
                f"({', '.join(sorted(hw))})",
                detail="Stage-3 verification unexpectedly passed anyway.",
                suggestions=["Report this as a bug - include the '[install]' "
                             "log lines."])
            self._preflight = res
            return res
        res.ok = True
        self._preflight = res
        return res

    # ------------------------------------------------------ stage-1 helper
    def _check_base_tools(self) -> "PreflightError | None":
        """Verify git, cmake (>=3.14) and a C++ compiler; no heavy work."""
        mgr = detect_pkg_manager()
        missing: list[str] = []
        for tool in ("git", "cmake"):
            if shutil.which(tool) is None:
                missing.append(tool)
        # Accept either g++ or clang++ as the C++ compiler.
        if shutil.which("g++") is None and shutil.which("clang++") is None:
            missing.append("g++ (or clang++)")
        if missing:
            cmds = [fix_command("buildtools", mgr)] if mgr else []
            return PreflightError(
                "Base build tools missing: " + ", ".join(missing),
                detail=("llama.cpp is compiled from source, so git, cmake and "
                        "a C++ compiler are mandatory before anything else."),
                suggestions=(["Install them with the command above, then "
                              "press 'Auto-install / build' again."] if mgr
                             else ["Install git, cmake and a C++ compiler "
                                   "manually for your system, then retry."]),
                commands=cmds,
                manual=[] if mgr else [manual_hint("buildtools")])
        v = self._run_capture(["cmake", "--version"])
        m = re.search(r"(\d+)\.(\d+)", v)
        if m and (int(m.group(1)), int(m.group(2))) < (3, 14):
            return PreflightError(
                f"cmake {m.group(0)} is too old (llama.cpp needs >= 3.14)",
                detail="newer cmake required for the GGML_* build options",
                suggestions=["Upgrade cmake, e.g.: pip install --user "
                             "--upgrade cmake  (then make sure ~/.local/bin "
                             "is on PATH), or use your distro's updated "
                             "package / Kitware's APT repo."])
        return None

    #: Backwards-compatible alias (older call sites / tests).
    def check_prerequisites(self) -> tuple[bool, list[str]]:
        """Legacy two-value form of :meth:`preflight` (stage 1 only)."""
        err = self._check_base_tools()
        if err is None:
            return True, []
        items = [f"{t}  (fix: {c})" for t, c in
                 zip(self._missing_tool_names(), err.commands or ["see hint"])]
        return False, items

    @staticmethod
    def _missing_tool_names() -> list[str]:
        out = []
        for tool in ("git", "cmake"):
            if shutil.which(tool) is None:
                out.append(tool)
        if shutil.which("g++") is None and shutil.which("clang++") is None:
            out.append("g++ (or clang++)")
        return out

    def required_disk_gb(self) -> float:
        """Free space needed for the *validated* backend (default: worst case).

        Called after :meth:`preflight` in the GUI flow, so the confirmation
        dialog shows the number that matches the backend actually about to be
        built (CUDA trees dwarf CPU ones); before preflight it conservatively
        reports the largest value.
        """
        if self._preflight is not None and self._preflight.ok:
            return _DISK_GB_BY_BACKEND.get(self._preflight.backend, 3.0)
        return max(_DISK_GB_BY_BACKEND.values())

    # ------------------------------------------------------------- worker
    def start(self):
        """Kick off the build in a daemon thread (call from the GUI thread)."""
        t = threading.Thread(target=self.run, daemon=True,
                             name="llamacpp-install")
        t.start()

    def run(self):
        try:
            self._run()
        except AppError as exc:
            self.finished.emit(False, exc.report())
        except Exception as exc:
            err = AppError.classify(exc, context="guided llama.cpp install")
            self.finished.emit(False, err.report())

    def _run(self):
        self.log.emit(f"[install] guided llama.cpp install -> {self.dest}")

        # ---- preflight (stages 1-4) ---------------------------------------
        # Re-run it here even though the GUI already did: this thread is also
        # the safety net for any caller that skipped the dialog-side checks,
        # and it re-validates the environment after possible user changes.
        pf = self.preflight()
        if not pf.ok:
            assert pf.error is not None
            raise pf.error
        backend = pf.backend
        cuda_root = self._find_cuda_root()   # reused for -DCMAKE_CUDA_COMPILER
        self.log.emit(f"[install] backend: {backend} "
                      f"({pf.reason}; GPU families seen: "
                      f"{', '.join(sorted(pf.gpu_families)) or 'none'})")

        # Disk-space preflight BEFORE any bytes hit the wire; the estimate now
        # matches the backend that was just validated (CUDA needs ~8 GiB).
        probe_dir = self.dest.parent if self.dest.parent.exists() else Path("/")
        try:
            free_gb = shutil.disk_usage(probe_dir).free / 1024 ** 3
        except OSError:
            free_gb = -1.0
        need = self.required_disk_gb()
        if 0 <= free_gb < need:
            raise AppError(
                f"Not enough free disk space to build llama.cpp "
                f"({free_gb:.1f} GiB free, ~{need:.1f} GiB needed)",
                detail=f"target volume: {probe_dir}",
                suggestions=["Free up space, or choose another install folder "
                             "with the 'Folder…' picker.",
                             "A CPU-only build needs less than a CUDA one."])

        src = self.dest / "src"
        git_exe = shutil.which("git")
        assert git_exe
        if (src / ".git").is_dir():
            self.log.emit("[install] existing source tree - updating via "
                          "'git pull --ff-only'")
            self._stream([git_exe, "-C", str(src), "pull", "--ff-only"])
        elif src.exists() and any(src.iterdir()):
            raise AppError(
                "Install folder exists but is not a llama.cpp git checkout",
                detail=f"path: {src}",
                suggestions=["Pick an empty/new folder, or delete/rename the "
                             "conflicting directory and retry."])
        else:
            src.parent.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"[install] cloning llama.cpp (shallow) into {src} …")
            self._stream([git_exe, "clone", "--depth", "1", self.GIT_URL,
                          str(src)])
        if self._cancel.is_set():
            raise AppError("Installation cancelled by user",
                           detail="git stage completed; build not started.")

        build_flags = {
            "cuda": ["-DGGML_CUDA=ON"],
            "rocm": ["-DGGML_HIP=ON"],
            "vulkan": ["-DGGML_VULKAN=ON"],
            "cpu": [],
        }[backend]
        # Explicitly DISABLE every GPU backend that was NOT selected.
        # llama.cpp's CMake auto-enables backends whose SDK merely happens
        # to be present (e.g. GGML_VULKAN defaults to ON when Vulkan is
        # available - and on recent trees even without a full SDK), which
        # made a pure CUDA configure abort with
        #   "Could NOT find Vulkan (missing: Vulkan_LIBRARY
        #    Vulkan_INCLUDE_DIR glslc)".
        _BACKEND_SWITCHES = {
            "cuda": "GGML_CUDA",
            "rocm": "GGML_HIP",
            "vulkan": "GGML_VULKAN",
        }
        for name, var in _BACKEND_SWITCHES.items():
            if name != backend:
                build_flags.append(f"-D{var}=OFF")

        build_dir = src / "build"
        jobs = max(1, (os.cpu_count() or 4) // 2)
        configure_cmd = ["cmake", "-S", str(src), "-B", str(build_dir),
                        "-DCMAKE_BUILD_TYPE=Release"] + build_flags
        if backend == "cuda" and cuda_root and \
                not shutil.which("nvcc"):
            # nvcc exists only inside the CUDA toolkit root: hand it to cmake.
            configure_cmd.append(f"-DCMAKE_CUDA_COMPILER="
                                 f"{cuda_root / 'bin' / 'nvcc'}")
        self.log.emit(f"[install] configuring ({' '.join(build_flags) or 'CPU-only'}) …")
        try:
            self._stream(configure_cmd)
        except AppError:
            # A stale CMakeCache.txt from an earlier attempt makes cmake
            # replay detection with the previous flags; wipe the build dir
            # and retry exactly once before giving up.
            if self._configure_retried:
                raise
            self._configure_retried = True
            if not (build_dir / "CMakeCache.txt").exists():
                raise
            self.log.emit("[install] configure failed - clearing the "
                          "stale CMake cache and retrying once …")
            shutil.rmtree(build_dir, ignore_errors=True)
            self._stream(configure_cmd)
        if self._cancel.is_set():
            raise AppError("Installation cancelled by user",
                           detail="configure completed; compile skipped.")
        self.log.emit(f"[install] compiling llama-server with {jobs} job(s) - "
                      "this can take several minutes on first run …")
        self._stream(["cmake", "--build", str(build_dir), "--config", "Release",
                      "-j", str(jobs), "--target", "llama-server"])

        bin_path = build_dir / "bin" / _exe(BINARY_NAME)
        try:
            present = bin_path.is_file() and os.access(bin_path, os.X_OK)
        except OSError:
            present = False
        if not present:
            raise AppError(
                "Build finished but llama-server was not produced",
                detail=f"expected: {bin_path}",
                suggestions=["Scroll up in the log for the compiler error.",
                             "Common causes: GPU SDK missing (CUDA toolkit / "
                             "ROCm), RAM exhaustion during parallel compile "
                             "(retry with fewer jobs), or cmake too old.",
                             "Missing packages can be installed with e.g. '"
                             + fix_command("buildtools") + "' or '"
                             + fix_command("cuda") + "'."])
        ver = probe_version(bin_path)
        caps = probe_capabilities(bin_path)
        self.log.emit(f"[install] SUCCESS: {bin_path}"
                      + (f"  [{ver}]" if ver else ""))
        # ---- verify the compiled binary really has the requested backend --
        # caps keys are cuda/gpu only, so map the chosen backend onto them.
        gpu_ok = {"cuda": caps["cuda"], "rocm": caps["gpu"],
                  "vulkan": caps["gpu"]}.get(backend, True)
        if backend in ("cuda", "rocm", "vulkan") and not gpu_ok:
            self.log.emit(
                f"[install] WARNING: the compiled llama-server does NOT list "
                f"{backend.upper()} support (see the 'ggml ...' feature lines "
                "above). It will run CPU-only and ignore -ngl.")
            self.log.emit(
                f"[install] hint: install/repair the {backend.upper()} "
                "toolchain, e.g. '" + fix_command(backend) +
                "' (or the CUDA toolkit from developer.nvidia.com), then "
                "press 'Auto-install / build' again; the existing checkout "
                "is re-used so only the compile step repeats.")
        elif backend == "cuda":
            self.log.emit("[install] verified: binary advertises CUDA support "
                          "- GPU offload (-ngl) will work.")
        self.finished.emit(True, str(bin_path))

    # ------------------------------------------------------------- helpers
    # ------------------------------------------------------------- CUDA libs
    #: Libraries ggml-cuda links via CMake's FindCUDAToolkit imported targets.
    #: curand/cudnn are *optional* in llama.cpp (GGML_CUDA_CUDNN defaults to
    #: OFF) - a missing one there is only a warning, so we list them as fix
    #: packages but never block the build on them alone.  cublas is mandatory:
    #: without it cmake configure aborts at generate time with
    #: "Target 'ggml-cuda' links to: CUDA::cublas but the target was not
    #: found", which is exactly what this preflight now catches up-front.
    _CUDA_REQUIRED_LIBS = ("libcublas",)
    _CUDA_OPTIONAL_LIBS = ("libcurand", "libcudnn")

    def _cuda_lib_dirs(self, cuda_root: Path) -> list[Path]:
        """Library search dirs of a toolkit root (incl. the per-target tree)."""
        dirs = [cuda_root / "lib64", cuda_root / "lib"]
        t = cuda_root / "targets"
        try:
            if t.is_dir():
                dirs += sorted((d for d in t.glob("*/lib")), reverse=True)
        except OSError:
            pass
        return [d for d in dirs if d.is_dir()]

    def _has_cuda_dev_lib(self, name: str, cuda_root: Path | None) -> bool:
        """True when <name>.so (dev symlink or plain file) is linkable.

        Checks the toolkit root first, then the system loader cache - distro
        split packages (e.g. Debian's libcublas-dev) install into /usr/lib
        instead of /usr/local/cuda.
        """
        roots = [cuda_root] if cuda_root else []
        for r in roots:
            for d in self._cuda_lib_dirs(r):
                if (d / f"{name}.so").exists() or \
                        (d / "stubs" / f"{name}.so").exists():
                    return True
        try:
            out = subprocess.run(["ldconfig", "-p"], capture_output=True,
                                 text=True, timeout=5.0).stdout
        except Exception:
            return False
        needle = f"{name}.so"
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(needle + " "):   # 'libcublas.so (x64) => …'
                path = line.split("=>", 1)[-1].strip()
                # The runtime-only library is NOT enough to link against;
                # accept it only if the dev symlink itself resolves.
                try:
                    if path and Path(path).is_file():
                        return True
                except OSError:
                    pass
        return False

    def _missing_cuda_libs(self, cuda_root: Path) -> list[str]:
        """Names of required/optional CUDA math libs absent from the toolkit."""
        missing = [n for n in self._CUDA_REQUIRED_LIBS
                   if not self._has_cuda_dev_lib(n, cuda_root)]
        opt = [n for n in self._CUDA_OPTIONAL_LIBS
               if not self._has_cuda_dev_lib(n, cuda_root)]
        return missing + opt

    def _cuda_toolkit_version(self, cuda_root: Path) -> str:
        """Toolkit version string ('13.1') from nvcc, or '' when unknown."""
        nvcc = cuda_root / "bin" / _exe("nvcc")
        if not nvcc.is_file():
            return ""
        out = self._run_capture([str(nvcc), "--version"])
        m = re.search(r"release\s+(\d+\.\d+)", out)
        return m.group(1) if m else ""

    #: Cache lookup used for the dynamic 'cuda_libs' fix command.  Bound as
    #: a *class attribute* (not an @staticmethod) so tests - and callers with
    #: exotic setups - can override it per instance or per class; inside the
    #: method, ``_apt_cache_search`` resolves to the module-level function.
    _apt_cache_search = staticmethod(_apt_cache_search)

    def _find_cuda_root(self) -> Path | None:
        """First plausible CUDA toolkit root (contains bin/nvcc)."""
        candidates: list[str] = []
        for env in ("CUDA_PATH", "CUDAHOME", "CUDA_HOME"):
            v = os.environ.get(env)
            if v:
                candidates.append(v)
        candidates.append("/usr/local/cuda")
        try:
            base = Path("/usr/local")
            if base.is_dir():
                candidates += sorted((str(d) for d in base.glob("cuda-*")),
                                     reverse=True)   # newest first
        except OSError:
            pass
        for c in candidates:
            p = Path(c)
            if (p / "bin" / _exe("nvcc")).is_file():
                return p
        return None

    def _has_nvcc(self, cuda_root: Path | None = None) -> bool:
        """nvcc anywhere: on PATH *or* inside a discovered CUDA toolkit root.

        The previous implementation only checked ``shutil.which('nvcc')`` and
        a hard-coded ``/usr/local/cuda/bin/nvcc``, so toolkits installed at
        ``/usr/local/cuda-12.x`` or outside PATH were missed and the guided
        build silently fell back to CPU-only compilation.
        """
        if shutil.which("nvcc") is not None:
            return True
        if cuda_root is None:
            cuda_root = self._find_cuda_root()
        return cuda_root is not None

    def _has_nvidia_driver(self) -> bool:
        """True when an NVIDIA kernel driver is loaded (GPU actually usable)."""
        return self._has_nvidia_smi() or self._nvidia_proc_driver()

    @staticmethod
    def _has_nvidia_smi() -> bool:
        return shutil.which("nvidia-smi") is not None

    @staticmethod
    def _nvidia_proc_driver() -> bool:
        try:
            return Path("/proc/driver/nvidia/version").exists()
        except OSError:
            return False

    def _has_rocm(self) -> bool:
        """Legacy toolkit-presence probe ('/opt/rocm exists or hipcc on PATH').

        Kept for older call sites; :meth:`preflight` no longer trusts it for
        backend selection because a bare /opt/rocm directory without hipcc is
        frequently a leftover SDK install on NVIDIA-only machines.
        """
        return shutil.which("hipcc") is not None or self._find_rocm_root().is_dir()

    def _find_rocm_root(self) -> Path:
        """First plausible ROCm root (contains bin/hipcc), else /opt/rocm."""
        candidates: list[str] = []
        for env in ("ROCM_PATH", "ROCR_VISIBLE"):
            v = os.environ.get(env)
            if v:
                candidates.append(v)
        candidates.append("/opt/rocm")
        try:
            base = Path("/opt")
            if base.is_dir():
                # Newest rocm-X.Y release first, then the plain /opt/rocm
                # symlink target.
                releases = sorted((d for d in base.glob("rocm-*")
                                   if d.is_dir()), reverse=True)
                candidates += [str(d) for d in releases]
        except OSError:
            pass
        for c in candidates:
            p = Path(c)
            if (p / "bin" / _exe("hipcc")).is_file():
                return p
        return Path("/opt/rocm")

    @staticmethod
    def _amdgpu_driver_loaded() -> bool:
        """True when the amdgpu kernel driver owns at least one DRM card.

        Reads ``/sys/class/drm/card*/device/driver`` - works even without any
        ROCm userspace installed, which makes it the strongest *hardware*
        proof for the AMD family (mirroring what nvidia-smi//proc gives for
        NVIDIA).  Never *removes* a PCI-enumerated 'amd' family: an iGPU with
        only the minimal framebuffer driver still counts as AMD hardware.
        """
        try:
            cards = sorted(Path("/sys/class/drm").glob("card[0-9]*"))
        except OSError:
            return False
        for c in cards:
            try:
                drv = (c / "device" / "driver")
                if drv.is_symlink() and \
                        Path(os.readlink(str(drv))).name == "amdgpu":
                    return True
            except OSError:
                continue
        return False

    def _has_vulkan(self) -> bool:
        """True only when a Vulkan *development* build is actually possible.

        Checking for the runtime loader alone (libvulkan.so.1) was a bug:
        virtually every driver install ships it, but compiling ggml-vulkan
        additionally needs the SDK headers (vulkan/vulkan.h), the static/
        import library (libvulkan.so) and the shader compiler 'glslc' (or
        glslangValidator).  Picking vulkan without those made cmake's
        ``find_package(Vulkan)`` abort the whole guided install with
        "Could NOT find Vulkan (missing: Vulkan_LIBRARY Vulkan_INCLUDE_DIR
        glslc)" - even though the user merely had a runtime loader present.
        A machine can have a perfectly usable Vulkan *driver* yet no dev
        package; llama.cpp then has to fall back to CPU (or CUDA/ROCm)
        instead of failing the build outright.
        """
        return (self._vulkan_lib_dir() is not None
                and self._vulkan_header_dir() is not None
                and self._vulkan_shaderc() is not None)

    @staticmethod
    def _vulkan_lib_dir() -> Path | None:
        """Directory holding the *linkable* libvulkan.so (SDK/dev package).

        The bare runtime file is named ``libvulkan.so.1``; plain ``ld`` can
        only link against ``libvulkan.so``, which is created by the Vulkan
        development package.  CMake's FindVulkan reports exactly that as
        ``Vulkan_LIBRARY`` missing, so require the unversioned symlink.
        """
        dirs = []
        env = os.environ.get("VULKAN_SDK")
        if env:
            sdk = Path(env)
            dirs += [sdk / "lib", sdk]
        dirs += [Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/lib64"),
                 Path("/usr/lib"), Path("/usr/local/lib"),
                 Path("/lib/x86_64-linux-gnu")]
        for d in dirs:
            try:
                if (d / "libvulkan.so").is_file():
                    return d
            except OSError:
                continue
        return None

    @staticmethod
    def _vulkan_header_dir() -> Path | None:
        """Directory containing ``vulkan/vulkan.h`` (Vulkan headers package)."""
        dirs = []
        env = os.environ.get("VULKAN_SDK")
        if env:
            dirs.append(Path(env) / "include")
        prefix = os.environ.get("XDG_PREFIX") or "/usr"
        dirs += [Path(prefix) / "include", Path("/usr/local/include")]
        for d in dirs:
            try:
                if (d / "vulkan" / "vulkan.h").is_file():
                    return d
            except OSError:
                continue
        return None

    @staticmethod
    def _vulkan_shaderc() -> str | None:
        """Path to a SPIR-V shader compiler glslc-style (required by cmake)."""
        for name in ("glslc", "glslangValidator"):
            p = shutil.which(name)
            if p:
                return p
        sdk = os.environ.get("VULKAN_SDK")
        if sdk:
            for name in ("glslc", "glslangValidator"):
                cand = Path(sdk) / "bin" / name
                try:
                    if cand.is_file() and os.access(cand, os.X_OK):
                        return str(cand)
                except OSError:
                    continue
        return None

    def _run_capture(self, cmd: list[str], timeout: float = 8.0) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout)
            return (r.stdout or "") + (r.stderr or "")
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def _stream(self, cmd: list[str]):
        """Run a command, tailing combined output into the log signal."""
        self.log.emit("[install] $ " + " ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, encoding="utf-8",
                                    errors="replace")
        except OSError as exc:
            raise AppError(f"Could not execute '{cmd[0]}'",
                           detail=f"{type(exc).__name__}: {exc}",
                           suggestions=["Ensure the tool is installed and on "
                                        "PATH, then retry."]) from exc
        assert proc.stdout
        tail: list[str] = []
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                self.log.emit("[install] " + line)
                tail.append(line)
                del tail[:-12]          # keep the last few error lines
            if self._cancel.is_set():
                proc.kill()
                break
        rc = proc.wait()
        if rc != 0:
            blob = "\n".join(tail).lower()
            suggestions = ["Fix the reported problem (permissions, missing "
                           "SDK, network) and press the button again - the "
                           "installer resumes from the existing checkout."]
            # Translate common 'missing development package' cmake errors into
            # concrete privileged command lines instead of a raw stack trace.
            fixes = []
            if "could not find vulkan" in blob or "vulkan_library" in blob \
                    or "vulkan_include_dir" in blob:
                fixes.append(fix_command("vulkan"))
            # 'CUDA::cublas / curand / cudnn imported target not found' means
            # the toolkit root has nvcc but lacks the math dev libraries -
            # almost always a minimal 'cuda-compiler' install.  The exact
            # package names are resolved live from the apt cache (pinned to
            # the installed nvcc version) because they are version-suffixed
            # and change with every CUDA release; never suggest a hard-coded
            # name that may not exist in the user's repos.
            if ("cuda::cublas" in blob or "cuda::curand" in blob
                    or "cuda::cudnn" in blob
                    or ("cublas" in blob and "not found" in blob)):
                self.log.emit("[install] The CUDA toolkit is missing its "
                              "math libraries (cuBLAS dev files). nvcc alone "
                              "is not enough for llama.cpp.")
                v = (self._cuda_toolkit_version(self._cuda_root)
                     if self._cuda_root else "")
                mgr = detect_pkg_manager()
                # route *all* managers through the helper so the apt path
                # always uses this runner's (injectable) cache lookup.
                c, n = _cuda_libs_fix_command(mgr, v,
                                              search_fn=self._apt_cache_search)
                fixes += c
                for note in n:
                    self.log.emit("[install] NOTE: " + note)
            if "nvcc" in blob or "cuda compiler" in blob \
                    or "could not find cuda" in blob:
                fixes.append(fix_command("cuda"))
            if "hip" in blob and ("not found" in blob or "missing" in blob):
                fixes.append(fix_command("rocm"))
            if "no c/c++ compiler" in blob or "cmake c compiler" in blob \
                    or "g++ not found" in blob:
                fixes.append(fix_command("buildtools"))
            for f in dict.fromkeys(fixes):     # dedupe, keep order
                self.log.emit("[install] FIX: " + f)
            if fixes:
                suggestions.insert(
                    0, "A required development package is missing. Install "
                       "it (needs sudo/root), then press the button again: "
                       + ";  then ".join(dict.fromkeys(fixes)))
            raise AppError(
                f"Step failed with exit code {rc}: {' '.join(cmd[:3])} …",
                detail="See the '[install]' log lines above for the tool's own "
                       "error message."
                       + (" Fix command(s) provided below." if fixes else ""),
                suggestions=suggestions)


def suggested_install_root() -> str:
    """Default target shown in the Tab 2 folder picker."""
    return str(DEFAULT_INSTALL_ROOT)


def platform_hint() -> str:
    system = platform.system()
    if system == "Linux":
        return ("Guided install uses git + cmake + g++ "
                "(sudo apt install git cmake build-essential).")
    return f"Guided install support for {system} is best-effort."
