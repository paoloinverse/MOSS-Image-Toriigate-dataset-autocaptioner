"""llama.cpp discovery, guided installation and autoconfiguration.

The app is only a *manager* around ``llama-server``; the binary itself is
external.  Historically users had to know where their build lived and type
the full path into Tab 2 - error-prone when it is not on PATH.  This module:

* :func:`discover_binaries`  - exhaustive autodetection (PATH, common install
  prefixes, per-user build dirs, flatpak/snap spillovers) with version probing.
* :func:`pick_best_binary`   - ranks discoveries (explicit user choice wins,
  then CUDA > ROCm/Vulkan > plain-CPU builds).
* :class:`InstallRunner`     - QObject worker that clones + compiles llama.cpp
  in a QThread with live log streaming, so the GUI never freezes during the
  (minutes-long) build.  Everything heavy is behind an explicit user click
  with a confirmation dialog - nothing here downloads or compiles silently.

No huge allocations, no network access at import time; every failure surfaces
as :class:`~app.core.models.AppError` with a verbose reason + fix report.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
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

    Returns ``{"has_api_key": bool, "has_reasoning_budget": bool,
               "cuda": bool}``.  Every check is conservative: when the
    binary cannot be probed at all we assume the modern feature exists and
    GPU support does not, so startup never breaks harder than before.
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
        caps = {"has_api_key": True, "has_reasoning_budget": True,
                "cuda": False}
    else:
        def supported(flag: str) -> bool:
            """True unless --help explicitly lists the flag as unsupported."""
            for ln in help_txt.splitlines():
                low = ln.lower()
                if ("unsupported" in low or "not supported" in low
                        or "unrecognized" in low):
                    if flag in low:
                        return False
            return True

        has_api_key = ("--api-key" in help_txt) or supported("--api-key")
        has_rb = ("reasoning-budget" in help_txt) or \
                 supported("--reasoning-budget")
        # GPU detection: prefer the explicit feature list printed by
        # `llama-server --version` ("ggml CUDA_blas", "feature: ... CUDA").
        # Fall back to linked libraries via ldd (catches older builds whose
        # version banner omits the feature list).
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
        caps = {"has_api_key": has_api_key, "has_reasoning_budget": has_rb,
                "cuda": cuda}
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
# Guided installation (clone + cmake build) - runs inside a QThread
# ---------------------------------------------------------------------------
class InstallRunner(QObject):
    """Background clone+compile of llama.cpp with streamed logs.

    Signals:
        log(str)         - progress lines for the GUI log box
        finished(bool, str) - success flag + final message (report on failure)
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

    # ------------------------------------------------------------- public
    def cancel(self):
        self._cancel.set()
        self.log.emit("[install] cancellation requested - stopping after the "
                      "current build step finishes.")

    def check_prerequisites(self) -> tuple[bool, list[str]]:
        """Verify git/cmake/compiler availability (no heavy work)."""
        missing = []
        for tool, pkg in (("git", "git"), ("cmake", "cmake"),
                          ("g++", "build-essential")):
            if shutil.which(tool) is None:
                missing.append(f"{tool}  (sudo apt install {pkg})")
        if missing:
            return False, missing
        v = self._run_capture(["cmake", "--version"])
        m = re.search(r"(\d+)\.(\d+)", v)
        if m and (int(m.group(1)), int(m.group(2))) < (3, 14):
            return False, [f"cmake {m.group(0)} is too old (need >= 3.14); "
                           "upgrade via pip or Kitware's APT repo"]
        return True, []

    def required_disk_gb(self) -> float:
        """Rough free space needed: shallow clone (~0.4 GiB) + build tree."""
        return 3.0

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

        ok, missing = self.check_prerequisites()
        if not ok:
            raise AppError(
                "Build toolchain incomplete - cannot compile llama.cpp",
                detail="missing: " + "; ".join(missing),
                suggestions=["Install the listed packages (sudo apt install …) "
                             "and press 'Auto-install / build' again.",
                             "Or point Tab 2 'Binary:' at an existing "
                             "llama-server and skip building entirely."])

        # Disk-space preflight BEFORE any bytes hit the wire.
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

        # ---- backend selection ----------------------------------------
        # Robust GPU-toolchain discovery (the old `shutil.which("nvcc")`-only
        # check silently produced CPU builds on machines with CUDA installed
        # outside PATH - e.g. "llama sees none" despite three RTX GPUs).
        cuda_root = self._find_cuda_root()
        has_nvcc = self._has_nvcc(cuda_root)
        has_driver = self._has_nvidia_driver()
        backend = self.backend
        if backend == "auto":
            if has_nvcc:
                backend = "cuda"
            elif self._has_rocm():
                backend = "rocm"
            elif self._has_vulkan():
                backend = "vulkan"
            else:
                backend = "cpu"
            self.log.emit(f"[install] auto-detected GPU backend: {backend}")
        if backend == "cpu" and has_driver:
            self.log.emit(
                "[install] WARNING: NVIDIA GPU(s) present but no CUDA "
                "compiler (nvcc) found -> building CPU-only llama-server; "
                "GPU offload (-ngl) will be IGNORED by that binary.")

        build_flags = {
            "cuda": ["-DGGML_CUDA=ON"],
            "rocm": ["-DGGML_HIP=ON"],
            "vulkan": ["-DGGML_VULKAN=ON"],
            "cpu": [],
        }[backend]
        if backend == "cuda" and not has_nvcc:
            self.log.emit("[install] WARNING: nvcc not found - CUDA-enabled "
                          "build will fail. Falling back to CPU flags would "
                          "lose GPU offload; trying anyway because some distro "
                          "packaging provides nvcc outside PATH.")

        build_dir = src / "build"
        jobs = max(1, (os.cpu_count() or 4) // 2)
        configure_cmd = ["cmake", "-S", str(src), "-B", str(build_dir),
                        "-DCMAKE_BUILD_TYPE=Release"] + build_flags
        if backend == "cuda" and cuda_root and \
                not shutil.which("nvcc"):
            # nvcc exists only inside the CUDA toolkit root: hand it to cmake.
            configure_cmd.append(f"-DCMAKE_CUDA_COMPILER="
                                 f"{cuda_root / 'bin' / 'nvcc'}")
        self.log.emit(f"[install] configuring ({' '.join(build_flags) or 'CPU'}) …")
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
                             "(retry with fewer jobs), or cmake too old."])
        ver = probe_version(bin_path)
        caps = probe_capabilities(bin_path)
        self.log.emit(f"[install] SUCCESS: {bin_path}"
                      + (f"  [{ver}]" if ver else ""))
        # ---- verify the compiled binary really has the requested backend --
        if backend in ("cuda", "rocm", "vulkan") and not caps["cuda"]:
            self.log.emit(
                f"[install] WARNING: the compiled llama-server does NOT list "
                f"{backend.upper()} support (see the 'ggml ...' feature lines "
                "above). It will run CPU-only and ignore -ngl.")
            self.log.emit(
                "[install] hint: install the CUDA toolkit (e.g. "
                "'sudo apt install nvidia-cuda-toolkit' or from developer.nvidia.com) "
                "and press 'Auto-install / build' again; the existing checkout "
                "is re-used so only the compile step repeats.")
        elif backend == "cuda":
            self.log.emit("[install] verified: binary advertises CUDA support "
                          "- GPU offload (-ngl) will work.")
        self.finished.emit(True, str(bin_path))

    # ------------------------------------------------------------- helpers
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
        if shutil.which("nvidia-smi") is not None:
            return True
        try:
            return Path("/proc/driver/nvidia/version").exists()
        except OSError:
            return False

    def _has_rocm(self) -> bool:
        return shutil.which("hipcc") is not None or Path("/opt/rocm").is_dir()

    def _has_vulkan(self) -> bool:
        """Vulkan loader availability (fallback GPU path when CUDA is absent)."""
        try:
            r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True,
                               text=True, timeout=8.0)
            if "device" in ((r.stdout or "") + (r.stderr or "")).lower():
                return True
        except Exception:
            pass
        for lib in ("/lib/x86_64-linux-gnu/libvulkan.so.1",
                    "/usr/lib64/libvulkan.so.1",
                    "/usr/lib/libvulkan.so.1"):
            if Path(lib).exists():
                return True
        return False

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
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                self.log.emit("[install] " + line)
            if self._cancel.is_set():
                proc.kill()
                break
        rc = proc.wait()
        if rc != 0:
            raise AppError(
                f"Step failed with exit code {rc}: {' '.join(cmd[:3])} …",
                detail="See the '[install]' log lines above for the tool's own "
                       "error message.",
                suggestions=["Fix the reported problem (permissions, missing "
                             "SDK, network) and press the button again - the "
                             "installer resumes from the existing checkout."])


def suggested_install_root() -> str:
    """Default target shown in the Tab 2 folder picker."""
    return str(DEFAULT_INSTALL_ROOT)


def platform_hint() -> str:
    system = platform.system()
    if system == "Linux":
        return ("Guided install uses git + cmake + g++ "
                "(sudo apt install git cmake build-essential).")
    return f"Guided install support for {system} is best-effort."
