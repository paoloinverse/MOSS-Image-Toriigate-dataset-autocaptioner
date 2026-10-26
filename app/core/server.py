"""llama-server lifecycle manager.

Builds the command line from the resource plan + GUI settings, launches the
binary as a child process, tails stdout/stderr into the GUI log, watches for
OOM conditions and performs the automatic recovery loop required by the spec:

    OOM  ->  kill server  ->  -c reduced by 1024  ->  restart (+ warning log)
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from app.core import llamacpp
from app.core.hardware import HardwareManager
from app.core.models import ModelFile
from app.core.resources import ResourcePlan, MIN_CONTEXT

OOM_PATTERNS = (
    "out of memory", "oom", "cudaerror", "cuda error",
    "failed to allocate", "ggml_backend_alloc", "std::bad_alloc",
    "hipoutofmemory", "hip out of memory", "kv buffer",
)


@dataclass
class ServerConfig:
    model: ModelFile | None = None
    mmproj_path: str | None = None
    plan: ResourcePlan | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    # "layer" is the default split mode: it works on every ggml backend,
    # while "row"/"graph" need CUDA VMM/split-buffer support in the *build +
    # driver* stack (see llamacpp.probe_split_buffer_support - a missing one
    # aborts model load with "device CUDAxx does not support split buffers").
    split_mode: str = "layer"        # row | layer | graph
    tensor_split: str = ""           # e.g. "50,50"
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    reasoning_budget: int = 0        # --reasoning-budget (thinking level)
    extra_args: str = ""             # raw passthrough from Tab 2
    explicit_mode: bool = False      # disables safety-system prompts
    binary: str = "llama-server"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class ServerManager(QObject):
    """Owns exactly one llama-server child process."""

    log = Signal(str)
    state_changed = Signal(str)          # "stopped" | "starting" | "ready" | "crashed"
    oom_recovery = Signal(int)           # new context size after shrink

    def __init__(self, hw: HardwareManager, preferred_binary: str = "",
                 parent=None):
        super().__init__(parent)
        self.hw = hw
        # Live mirror of the Tab 2 'Binary:' field (kept in sync by
        # AppContext).  Used for discovery *before* the first start(), when
        # self.config is still None - otherwise a user-picked path shown as
        # OK in the Binary status would be ignored and find_binary() could
        # wrongly report "no runnable llama-server found".
        self.preferred_binary = preferred_binary
        self.proc: subprocess.Popen | None = None
        self.config: ServerConfig | None = None
        self._reader_threads: list[threading.Thread] = []
        self._stop_reader = threading.Event()
        self._lock = threading.RLock()
        self.ready = False
        self._shutdown_requested = False

    def _preferred_binary(self) -> str:
        """Current Tab 2 'Binary:' choice, with sane fallbacks.

        Priority: the last start() config (when present) > the live mirror
        of the GUI field maintained by AppContext > plain name lookup.
        """
        cfg = self.config
        if cfg is not None and getattr(cfg, "binary", ""):
            return cfg.binary
        pref = getattr(self, "preferred_binary", "") or ""
        return pref or "llama-server"

    # -- discovery ----------------------------------------------------------
    def find_binary(self) -> str | None:
        """Locate a runnable llama-server honouring the user's Tab 2 choice.

        Delegates to :mod:`app.core.llamacpp` so discovery, ranking and the
        "preferred path wins" logic live in one place (also used by the
        'Autodetect' button and startup autoconfiguration).  Honours the
        user's Binary: selection even *before* the first ``start()`` - when
        ``self.config`` is still ``None`` - by consulting the live
        ``preferred_binary`` mirror kept in sync by :class:`AppContext`.
        """
        from app.core import llamacpp
        best = llamacpp.pick_best(self._preferred_binary())
        return best["path"] if best else None

    def describe_binary_search(self) -> str:
        """Verbose explanation of where we looked - used in error reports."""
        from app.core import llamacpp
        cand = self._preferred_binary()
        return (f"preferred='{cand}' plus " +
                llamacpp.describe_search().replace("\n", "\n             "))

    # -- argv construction ---------------------------------------------------
    def build_args(self, cfg: ServerConfig) -> list[str]:
        binary = self.find_binary() or cfg.binary
        caps = llamacpp.probe_capabilities(binary)
        # Warn *before* spawning when the user selected GPUs but the binary
        # was compiled without any GPU backend (the silent CPU-only run that
        # produced "no usable GPU found, --gpu-layers option will be ignored"
        # in the field).  Guarded so duck-typed hw stubs keep working.
        try:
            want_gpu = bool(self.hw.selected_gpus)
        except Exception:
            want_gpu = False
        if want_gpu and not caps.get("gpu", False):
            self.log.emit(
                "[server] WARNING: this llama-server binary does NOT advertise "
                "CUDA/ROCm/Vulkan support - GPU offload (-ngl/-sm/-ts) will be "
                "ignored and inference runs CPU-only.\n"
                "    fix    : re-run 'Auto-install / build' in Tab 2 (the guided "
                "installer now detects nvcc outside PATH and verifies the "
                "compiled binary actually has GPU support), or point 'Binary:' "
                "at a GPU-enabled llama.cpp build.")
        args = [binary]
        if cfg.model:
            args += ["-m", str(cfg.model.local_path())]
        if cfg.mmproj_path:
            args += ["--mmproj", cfg.mmproj_path]
        plan = cfg.plan or ResourcePlan()
        ngl = plan.n_gpu_layers if plan.n_gpu_layers >= 0 else 0
        args += ["-ngl", str(ngl)]

        selected = self.hw.selected_gpus
        if len(selected) > 1:
            # Multi-GPU: pipeline/tensor parallelism per spec.
            args += ["-sm", cfg.split_mode]
            ts = cfg.tensor_split or self.hw.tensor_split_string()
            if ts:
                args += ["-ts", ts]
        elif len(selected) == 1:
            args += ["-sm", "none"]

        args += ["-c", str(plan.context_tokens)]
        if selected:
            args += ["--cache-type-k", cfg.cache_type_k,
                     "--cache-type-v", cfg.cache_type_v]
        # Thinking level -> server-side reasoning budget (Qwen3-family);
        # skipped when the binary predates the flag (older llama.cpp builds
        # reject it with 'invalid argument' and exit 1).
        if getattr(cfg, "reasoning_budget", 0) > 0:
            if caps["has_reasoning_budget"]:
                args += ["--reasoning-budget", str(cfg.reasoning_budget)]
            else:
                self.log.emit("[server] WARNING: this llama-server does not "
                              "support --reasoning-budget - thinking levels "
                              "are request-only (enable_thinking in payload).")
        if cfg.explicit_mode:
            # Disable safety-system prompts server-side; the explicit trigger
            # sentence itself is injected into the system prompt by presets.
            args += ["-sp", ""]      # empty built-in system prompt
        args += ["--host", cfg.host, "--port", str(cfg.port)]
        # Auth token for the API endpoints: the real flag is --api-key.
        # Bare `--api` was rejected by older builds ("error: invalid
        # argument: --api" -> exit code 1), so probe before passing it.
        if caps["has_api_key"]:
            args += ["--api-key", ""]
        else:
            self.log.emit("[server] NOTE: this llama-server has no --api-key "
                          "option (old build) - starting without an API "
                          "token; consider re-building via 'Auto-install / "
                          "build' for security updates.")
        args += ["-af"]      # autoplay flash attention
        if cfg.extra_args.strip():
            args += cfg.extra_args.split()
        return args

    # -- lifecycle ------------------------------------------------------------
    def _preflight(self, cfg: ServerConfig) -> list[str]:
        """Collect human-readable problems that would break the launch.

        Returns a list of blocking error strings (empty == safe to spawn).
        Verbose by design so the log tells the user exactly what to fix.
        """
        errors: list[str] = []
        binary = self.find_binary()
        if not binary:
            errors.append(
                "llama-server binary not found: " + self.describe_binary_search() + ".\n"
                "    fix    : press 'Autodetect' or 'Auto-install / build' in Tab 2\n"
                "             (guided installer clones + compiles llama.cpp for you),\n"
                "             or navigate to an existing build with 'Browse…',\n"
                "             or install manually: sudo apt install git cmake "
                "build-essential && cmake -B ~/llama.cpp/build -S ~/llama.cpp "
                "-DGGML_CUDA=ON && cmake --build ~/llama.cpp/build -j --target llama-server")
            return errors   # everything below is pointless without a binary

        if cfg.model is None:
            errors.append("No model selected - choose a model/quant in Tab 1 first.")
        else:
            mpath = Path(cfg.model.local_path())
            if not mpath.is_file():
                errors.append(
                    f"Model GGUF missing on disk: {mpath}\n"
                    "    fix    : press 'Download & Verify' in Tab 1 "
                    "(large files are only fetched on explicit click).")
            elif mpath.stat().st_size < 1024 * 1024:
                errors.append(
                    f"Model GGUF looks truncated ({mpath}, "
                    f"{mpath.stat().st_size} bytes).\n"
                    "    fix    : delete the file and re-run 'Download & Verify'.")
            mmproj_needed = cfg.mmproj_path or cfg.model.mmproj_filename
            if mmproj_needed and not (cfg.mmproj_path and Path(cfg.mmproj_path).is_file()):
                mp = cfg.model.mmproj_local_path()
                if mp is None or not Path(mp).is_file():
                    errors.append(
                        f"Multimodal projector (mmproj) missing: {mmproj_needed}\n"
                        "    fix    : image captioning requires the mmproj GGUF; run "
                        "'Download & Verify' in Tab 1.")

        # Port sanity / availability check (cheap socket bind probe).
        if not (1 <= cfg.port <= 65535):
            errors.append(f"Invalid port {cfg.port} (must be 1-65535).")
        else:
            try:
                import socket as _s
                probe = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
                probe.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
                probe.bind((cfg.host, cfg.port))
                probe.close()
            except OSError as exc:
                errors.append(
                    f"Cannot bind {cfg.host}:{cfg.port}: {exc}\n"
                    "    fix    : another llama-server or process already uses this "
                    "port; change the port in Tab 2 or stop the other instance.")

        plan = cfg.plan
        if plan is not None and plan.context_tokens < MIN_CONTEXT:
            errors.append(
                f"Planned context {plan.context_tokens} is below the minimum "
                f"{MIN_CONTEXT}; the resource plan is invalid - press "
                "'Refresh Plan' in Tab 2.")
        return errors

    def start(self, cfg: ServerConfig) -> bool:
        with self._lock:
            if self.proc and self.proc.poll() is None:
                self.log.emit("[server] already running")
                return True
            self.config = cfg

            problems = self._preflight(cfg)
            if problems:
                for p in problems:
                    self.log.emit("[server] PRE-FLIGHT ERROR: " + p)
                self.state_changed.emit("crashed")
                return False

            self._shutdown_requested = False
            args = self.build_args(cfg)
            self.log.emit("[server] exec: " + " ".join(args))
            binary = args[0]
            try:
                env = os.environ.copy()
                env.setdefault("LLAMA_LOG_PREFIX", "1")
                self.proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    start_new_session=True,
                )
            except PermissionError as exc:
                self.log.emit(
                    f"[server] ERROR: '{binary}' is not executable: {exc}\n"
                    "    fix    : chmod +x the binary, or point Tab 2 'Binary' at "
                    "the correct llama-server build.")
                self.state_changed.emit("crashed")
                self.proc = None
                return False
            except FileNotFoundError:
                self.log.emit(
                    "[server] ERROR: binary vanished between discovery and exec: "
                    f"{binary}\n    fix    : re-check the path in Tab 2.")
                self.state_changed.emit("crashed")
                self.proc = None
                return False
            except OSError as exc:
                self.log.emit(f"[server] failed to launch: {type(exc).__name__}: {exc}")
                self.state_changed.emit("crashed")
                self.proc = None
                return False
            self.state_changed.emit("starting")
            self._stop_reader.clear()
            t = threading.Thread(target=self._read_output, daemon=True)
            t.start()
            self._reader_threads.append(t)
            threading.Thread(target=self._watch_ready, daemon=True).start()
            return True

    def stop(self, graceful_ms: int = 3000):
        with self._lock:
            self._shutdown_requested = True
            self.ready = False
            if not self.proc:
                self.state_changed.emit("stopped")
                return
            try:
                self.proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=graceful_ms / 1000)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                except OSError:
                    pass
            self.log.emit("[server] stopped")
            self.state_changed.emit("stopped")
            self.proc = None

    def is_running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    # -- output tailing & OOM watchdog -----------------------------------------
    def _read_output(self):
        proc = self.proc
        if not proc or not proc.stdout:
            return
        for line in iter(proc.stdout.readline, ""):
            if self._stop_reader.is_set():
                break
            line = line.rstrip("\n")
            if line:
                self.log.emit(f"[llama] {line}")
                low = line.lower()
                if any(p in low for p in OOM_PATTERNS):
                    self._handle_oom(line)
        try:
            proc.stdout.close()
        except Exception:
            pass

    def _watch_ready(self):
        # Consider the server ready once llama.cpp prints its listening line
        # or /health responds.
        deadline = time.time() + 300
        while time.time() < deadline:
            if self._shutdown_requested or not self.proc:
                return
            if self.proc.poll() is not None:
                if not self._shutdown_requested:
                    self.state_changed.emit("crashed")
                    self.log.emit("[server] process exited unexpectedly "
                                  f"(code {self.proc.returncode})")
                return
            if self.ready:
                return
            if self._health_ok():
                self.ready = True
                self.state_changed.emit("ready")
                self.log.emit(f"[server] ready at {self.config.base_url}")
                return
            time.sleep(1.0)
        self.log.emit("[server] startup timed out after 300 s")

    def _health_ok(self) -> bool:
        try:
            import urllib.request
            with urllib.request.urlopen(
                    f"{self.config.base_url}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    # -- automatic OOM recovery -------------------------------------------------
    def _handle_oom(self, offending_line: str):
        with self._lock:
            if self._shutdown_requested or not self.config or not self.config.plan:
                return
            old_ctx = self.config.plan.context_tokens
            if old_ctx <= MIN_CONTEXT:
                self.log.emit("[server] OOM persists at minimum context - "
                              "cannot shrink further; giving up.")
                return
            self.log.emit("[server] WARNING: KV-cache OOM detected. Killing server "
                          f"and restarting with -c {old_ctx} -> {old_ctx - 1024}.")
            new_ctx = max(MIN_CONTEXT, old_ctx - 1024)
            self.config.plan.context_tokens = new_ctx
            self._shutdown_requested = True
            proc, self.proc = self.proc, None
            try:
                proc.kill()
            except OSError:
                pass
            self.ready = False
            self._stop_reader.set()
            self.state_changed.emit("starting")
            self._shutdown_requested = False
            cfg = self.config
            time.sleep(0.5)
        # Restart outside the lock so the reader thread can attach cleanly.
        self.start(cfg)
        self.oom_recovery.emit(new_ctx)
