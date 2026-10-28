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
    split_mode: str = "layer"          # row | layer | graph
    tensor_split: str = ""           # e.g. "50,50"
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    flash_attn: str = "auto"         # GUI switch: auto | on | off ("" = leave to server)
    reasoning_budget: int = 0        # --reasoning-budget (thinking level)
    extra_args: str = ""             # raw passthrough from Tab 2
    explicit_mode: bool = False      # disables safety-system prompts
    binary: str = "llama-server"
    api_key: str = ""                # Tab 2 'API Key:' (--api-key + client auth)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# --------------------------------------------------------------------------
# Host classification / liveness probes (shared by the preflight and Tab 2's
# on-commit reachability check - see app/tabs/inference_tab.py)
# --------------------------------------------------------------------------
LOCAL_HOSTNAMES = {"", "localhost", "localhost.localdomain", "ip6-localhost",
                   "127.0.0.1", "::1", "0.0.0.0", "::"}


def is_local_host(host: str) -> bool:
    """True for loopback/wildcard/this-machine names (bind locally)."""
    h = (host or "").strip().strip("[]").lower()
    if not h:
        return True
    if h in LOCAL_HOSTNAMES:
        return True
    if h.startswith("127."):                 # whole 127/8 loopback block
        return True
    try:
        import socket as _sk
        infos = _sk.getaddrinfo(h, None)
        addrs = {i[4][0] for i in infos}
        if any(a in ("127.0.0.1", "::1") for a in addrs):
            return True
        # A name that only resolves to this machine's own addresses
        # (e.g. the laptop's hostname or its LAN IP) still counts local.
        try:
            host_addrs = {i[4][0]
                          for i in _sk.getaddrinfo(_sk.gethostname(), None)}
            if addrs and addrs <= host_addrs:
                return True
        except OSError:
            pass
    except OSError:
        pass
    return False


def probe_server(base_url: str, api_key: str = "", timeout: float = 3.0,
                 connect_timeout: float = 3.0) -> tuple[str, str]:
    """Probe ``base_url/health``; returns one of:

    * ``("running", "")``            - HTTP 200: server up, no auth needed
    * ``("running-auth", "")``       - HTTP 401: server up, token accepted
                                       (or unauthenticated health endpoint)
    * ``("auth-required", msg)``     - HTTP 401/403 with NO token configured
    * ``("auth-failed", msg)``       - HTTP 401/403 WITH a token: it is wrong
    * ``("down", msg)``              - TCP reachable, nothing answering HTTP
    * ``("unreachable", msg)``       - DNS/TCP/connect failure

    ``("running", ...)`` and ``("running-auth", ...)`` both mean "the server
    answers requests" - the suffix only records whether a token was supplied.
    Never raises; every outcome carries a short human-readable detail.
    """
    url = base_url.rstrip("/") + "/health"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _ok(status: int) -> tuple[str, str]:
        label = "running-auth" if api_key else "running"
        return (label, "" if status == 200 else f"HTTP {status}")

    try:
        import urllib.request as _ur
        req = _ur.Request(url, headers=headers)
        with _ur.urlopen(req, timeout=timeout) as r:
            return _ok(r.status)
    except Exception as exc:
        import urllib.error as _ue
        status = None
        if isinstance(exc, _ue.HTTPError):
            status = exc.code
        if status in (401, 403):
            if not api_key:
                return ("auth-required",
                        f"HTTP {status}: server requires an API key")
            # A token WAS sent but /health refused it.  Some llama-server
            # builds gate /health inconsistently even for VALID keys, so
            # confirm against /v1/models before declaring the key wrong:
            # accepted there => server is fine for us (running-auth);
            # denied there too => the key really is bad (auth-failed).
            try:
                req2 = _ur.Request(base_url.rstrip("/") + "/v1/models",
                                   headers=headers)
                with _ur.urlopen(req2, timeout=timeout) as r2:
                    if r2.status < 400:
                        return ("running-auth", "")
            except _ue.HTTPError as e2:
                if e2.code in (401, 403):
                    return ("auth-failed",
                            f"HTTP {e2.code}: server rejected the API key")
            except Exception:
                pass
            return ("auth-failed",
                    f"HTTP {status}: server rejected the API key")
        # No HTTP answer at all - distinguish "port closed" from "no route".
        try:
            import socket as _sk
            from urllib.parse import urlparse as _up
            pr = _up(base_url)
            s = _sk.create_connection((pr.hostname or "127.0.0.1",
                                       pr.port or 8080),
                                       timeout=connect_timeout)
            s.close()
            return ("down", f"{type(exc).__name__}: {exc}")
        except OSError as exc2:
            return ("unreachable", f"{type(exc2).__name__}: {exc2}")


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
        # True when we ATTACHED to an already-running llama-server (found via
        # the host-aware preflight probe) instead of spawning our own child.
        # While attached there is no subprocess to supervise: "stop" merely
        # detaches and the remote/local server keeps running.
        self.attached = False

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
        # Older stubs / cached probes may lack newer keys - default them so
        # build_args never raises KeyError on a partial dict.
        def _cap(name: str, default: bool = False) -> bool:
            try:
                return bool(caps.get(name, default))
            except Exception:
                return default
        # Warn *before* spawning when the user selected GPUs but the binary
        # was compiled without any GPU backend (the silent CPU-only run that
        # produced "no usable GPU found, --gpu-layers option will be ignored"
        # in the field).  Guarded so duck-typed hw stubs keep working.
        try:
            want_gpu = bool(self.hw.selected_gpus)
        except Exception:
            want_gpu = False
        if want_gpu and not _cap("gpu", False):
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
            if _cap("has_reasoning_budget", True):
                args += ["--reasoning-budget", str(cfg.reasoning_budget)]
            else:
                self.log.emit("[server] WARNING: this llama-server does not "
                              "support --reasoning-budget - thinking levels "
                              "are request-only (enable_thinking in payload).")
        if cfg.explicit_mode:
            # Disable safety-system prompts server-side; the explicit trigger
            # sentence itself is injected into the system prompt by presets.
            # '-sp' takes an optional string, but builds without the flag
            # abort with "error: invalid argument:" (empty value included)
            # and exit 1 - so only pass it when the probe says it exists.
            if _cap("has_single_turn", False):
                args += ["-sp", ""]   # empty built-in system prompt
            else:
                self.log.emit("[server] NOTE: this llama-server does not "
                              "support '-sp' (--single-turn) - explicit mode "
                              "runs request-only (the trigger sentence is "
                              "still injected into the system prompt).")
        args += ["--host", cfg.host, "--port", str(cfg.port)]
        # Auth token for the API endpoints: the real flag is --api-key.
        # Bare `--api` was rejected by older builds ("error: invalid
        # argument: --api" -> exit code 1), so probe before passing it.
        # Tab 2 now has an explicit 'API Key:' field (cfg.api_key); when it
        # is empty we fall back to the MOSS_API_KEY / APPCAP_API_KEY env
        # vars.  An *empty* token was never passed any more - omitting the
        # flag entirely = no auth, which stays the default.
        _key = (getattr(cfg, "api_key", "") or
                os.environ.get("MOSS_API_KEY") or
                os.environ.get("APPCAP_API_KEY") or "")
        if _key:
            if _cap("has_api_key", False):
                args += ["--api-key", _key]
            else:
                self.log.emit("[server] NOTE: this llama-server has no "
                              "--api-key option (old build) - starting "
                              "without an API token; consider re-building "
                              "via 'Auto-install / build' in Tab 2.")
        # Flash attention: GUI switch (Tab 2 'Flash Attention': Auto/On/Off).
        # The old '-af' alias was deprecated and has been REMOVED from
        # llama.cpp - passing it makes current builds exit with
        # "error: invalid argument: -af", which blocked every launch.
        # Modern spelling is '--flash-attn auto|on|off'; slightly older
        # builds only understand the boolean '--flash-attn' / '-fa'.  We
        # probe the binary and translate accordingly; when the user left
        # the switch on "auto" and the server already defaults to flash
        # attention, passing nothing is the safest choice.
        fa = str(getattr(cfg, "flash_attn", "auto") or "auto").strip().lower()
        if fa not in ("auto", "on", "off"):
            fa = "auto"
        if _cap("flash_attn_auto", True):
            args += ["--flash-attn", fa]
        elif _cap("flash_attn_on_off"):
            # Boolean-era build: no per-value control; enable FA unless the
            # user explicitly turned it off.
            if fa != "off":
                args += ["-fa"]
            else:
                self.log.emit("[server] NOTE: this llama-server predates "
                              "'--flash-attn auto|on|off'; flash attention "
                              "will stay at the build default (off request "
                              "ignored).")
        else:
            self.log.emit("[server] NOTE: this llama-server has no flash-"
                          "attention option at all - skipping it; consider "
                          "re-building via 'Auto-install / build' in Tab 2.")
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
                        "    fix    : image captioning requires the mmproj GGUF; press\n"
                        "             'Fetch mmproj' next to 'Model to run' in Tab 2\n"
                        "             (or 'Download & Verify' in Tab 1) - both use the\n"
                        "             same live-Hub discovery pipeline.")
            # Wrong-projector guard: the auto-detection fallback used to grab
            # *any* mmproj from the model library, which paired ToriiGate's
            # projector with the Qwen3.8-27B main GGUF and made llama-server
            # die at load time with
            #   "mtmd_init_from_file: mismatch between text model (n_embd =
            #    5120) and mmproj (n_embd = 2560) / hint: wrong mmproj".
            # Catch it here - before spawning - with an actionable message.
            if cfg.mmproj_path and Path(cfg.mmproj_path).is_file():
                try:
                    bad = cfg.model.mmproj_companion_error(Path(cfg.mmproj_path))
                except Exception:
                    bad = None
                if bad:
                    errors.append(
                        f"Incompatible multimodal projector for "
                        f"'{cfg.model.filename}': {bad} "
                        f"(would pass: --mmproj {cfg.mmproj_path})\n"
                        "    fix    : download this model's own mmproj via "
                        "'Fetch mmproj' in Tab 2 (or 'Download & Verify' in "
                        "Tab 1 - both use the live repo listing, which "
                        "includes its projector), or clear 'Extra args' if "
                        "you set --mmproj manually there.")

        # Port sanity + host-aware availability check.
        #
        # LOCAL host (127.0.0.1 / localhost / 0.0.0.0 / this machine): we own
        # the bind, so a cheap socket-bind probe is the right test - if the
        # port is taken by another llama-server, reuse it transparently when
        # it is healthy (the Tab-2 commit-time check already told the user)
        # and block with an actionable error otherwise.
        #
        # REMOTE host: binding locally is meaningless; instead probe whether
        # a server is ALREADY answering on that address/port.  If one is, we
        # attach to it instead of spawning a duplicate; if not, verify the
        # address is network-reachable at all before trying to spawn a binary
        # on a remote box (which llama-server cannot do).
        if not (1 <= cfg.port <= 65535):
            errors.append(f"Invalid port {cfg.port} (must be 1-65535).")
        elif not str(cfg.host or "").strip():
            errors.append("Empty host - set an address in Tab 2 "
                          "(e.g. 127.0.0.1, 0.0.0.0 or a remote IP).")
        elif is_local_host(cfg.host):
            status, detail = probe_server(cfg.base_url,
                                          getattr(cfg, "api_key", ""))
            if status in ("running", "running-auth"):
                self._adopt_existing(cfg, status)
                return errors      # no further checks needed - we attached
            if status in ("auth-required", "auth-failed"):
                errors.append(
                    f"{cfg.base_url} answered but authentication failed "
                    f"({detail}).\n    fix    : put the server's token in "
                    "Tab 2 'API Key:' (or MOSS_API_KEY/APPCAP_API_KEY), "
                    "then press 'Check' again.")
            else:
                try:
                    import socket as _s
                    probe = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
                    probe.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
                    probe.bind((cfg.host if cfg.host not in ("", "::")
                                else "0.0.0.0", cfg.port))
                    probe.close()
                except OSError as exc:
                    if status == "down":
                        # Something holds the port but is not a healthy
                        # llama-server (or is still booting).
                        errors.append(
                            f"Cannot bind {cfg.host}:{cfg.port}: {exc}\n"
                            "    fix    : another process already uses this "
                            "port; change the port in Tab 2 or stop the "
                            "other instance.")
                    else:
                        errors.append(
                            f"Cannot bind {cfg.host}:{cfg.port}: {exc}\n"
                            "    fix    : another llama-server or process "
                            "already uses this port; change the port in "
                            "Tab 2 or stop the other instance.")
        else:
            status, detail = probe_server(cfg.base_url,
                                          getattr(cfg, "api_key", ""))
            if status in ("running", "running-auth"):
                self._adopt_existing(cfg, status)
                return errors
            if status in ("auth-required", "auth-failed"):
                # Reachable, but we cannot confirm health without credentials.
                self.log.emit(
                    f"[server] remote {cfg.base_url} reachable but "
                    f"{detail}; starting anyway - fix the 'API Key:' field "
                    "in Tab 2 if requests come back 401.")
            elif status == "unreachable":
                errors.append(
                    f"Remote host {cfg.base_url} is unreachable ({detail}).\n"
                    "    fix    : check the address/port, that the machine "
                    "is on the network/firewall-open, or start llama-server "
                    "there yourself ('--host 0.0.0.0') and use 'Check' in "
                    "Tab 2 to attach to it.")
            else:  # "down": TCP ok, no HTTP answer
                self.log.emit(
                    f"[server] NOTE: remote {cfg.base_url} is reachable but "
                    f"no llama-server answered /health ({detail}); note this "
                    "GUI cannot launch a process on a remote machine - "
                    "attempting anyway in case the service is still booting.")

        plan = cfg.plan
        if plan is not None and plan.context_tokens < MIN_CONTEXT:
            errors.append(
                f"Planned context {plan.context_tokens} is below the minimum "
                f"{MIN_CONTEXT}; the resource plan is invalid - press "
                "'Refresh Plan' in Tab 2.")
        return errors

    # -- attaching to an existing server --------------------------------------
    def _adopt_existing(self, cfg: ServerConfig, status: str) -> None:
        """Attach to a llama-server that is already answering on cfg.base_url.

        Called from :meth:`_preflight` when the host-aware probe finds a
        healthy server (HTTP 200, or 401-confirmed-with-token) at the
        configured address - either our own local bind being reused, or a
        remote instance the user started separately.  There is nothing to
        spawn in that case: we mirror the "ready" state exactly like
        ``_watch_ready`` does after a normal launch, so Start becomes a no-op
        and Stop later just detaches (see :meth:`stop`).
        """
        with self._lock:
            if self.proc and self.proc.poll() is None:
                return                      # we already own a child process
            self.config = cfg
            self.attached = True
            self.ready = True
            how = ("API-key authenticated" if status == "running-auth"
                   else "no authentication")
            self.log.emit(
                f"[server] NOTE: a llama-server is ALREADY running at "
                f"{cfg.base_url} ({how}) - attaching to it instead of "
                "starting a new one.\n"
                "    This GUI cannot control that external process: 'Stop' "
                "only detaches, and OOM auto-recovery is unavailable. To "
                "let this app manage the server, stop the other instance "
                "or pick a different port in Tab 2.")
            self.state_changed.emit("ready")

    def start(self, cfg: ServerConfig) -> bool:
        with self._lock:
            if self.proc and self.proc.poll() is None:
                self.log.emit("[server] already running")
                return True
            self.attached = False
            self.config = cfg

            problems = self._preflight(cfg)
            if problems:
                for p in problems:
                    self.log.emit("[server] PRE-FLIGHT ERROR: " + p)
                self.state_changed.emit("crashed")
                return False
            # _preflight may have ATTACHED to an already-running server
            # (healthy /health on the configured address) instead of leaving
            # room to spawn one - in that case there is nothing to launch.
            if self.ready:
                return True

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
            was_attached = self.attached
            self.ready = False
            self.attached = False
            if not self.proc:
                # Either we never started anything, or we had ATTACHED to an
                # external server (no child process of ours exists).  An
                # external llama-server must never be killed by this GUI.
                if was_attached:
                    self.log.emit("[server] detached from external server "
                                  f"({self.config.base_url if self.config else ''}"
                                  ") - that process keeps running; it is not "
                                  "managed by this GUI.")
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
        if self.proc and self.proc.poll() is None:
            return True
        # Attached mode: no child process, but the external server answers.
        return bool(self.attached and self.ready)

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
