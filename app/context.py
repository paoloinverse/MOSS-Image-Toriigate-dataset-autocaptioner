"""Shared application context passed to every tab / panel.

Holds the long-lived backend singletons plus cross-tab state so widgets stay
independent of each other (tabs talk through the context's signals).

Also provides :meth:`AppContext.report_error`, the single funnel every tab
uses for verbose, structured error reporting: the full multi-line message is
always written to the global log (Tab 5), and optionally shown in a modal
message box when ``dialog=True``.
"""
from __future__ import annotations

import os
import re
import threading
import traceback
import urllib.request

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QTimer, Signal

from app.core.hardware import HardwareManager
from app.core.models import (BUILTIN_SPEC_KEYS, ModelSpec, StorageManager,
                             build_registry)
from app.core.presets import DEFAULT_PRESETS, CaptionPreset, _clone
from app.core.server import ServerConfig, ServerManager


@dataclass
class InferenceSettings:
    """Live values from Tab 2 consumed when building server/payload config."""
    extra_args: str = ""
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    # llama.cpp sampler penalties sent per request (build_payload).  These were
    # never exposed before, so the server always used its own defaults; the
    # values here mirror those defaults (0.0 / 1.1) -> unchanged behaviour.
    presence_penalty: float = 0.0      # -1..2, default 0
    repetition_penalty: float = 1.1    # 0..2,  default 1.1
    # Qwen3.8-family reasoning controls (Tab 2 -> api_client.build_payload):
    # ``enable_thinking`` is the master switch mapped to
    # chat_template_kwargs.enable_thinking; ``preserve_thinking`` keeps
    # earlier  blocks verbatim in the history (safest when the effort
    # level changes mid-conversation); ``reasoning_budget`` is the token cap
    # (-1 = unlimited => no per-request override / no server-side cap).
    enable_thinking: bool = True
    preserve_thinking: bool = True
    reasoning_budget: int = -1
    thinking_level: str = "xhigh"    # reasoning_effort key ("" = off) - persisted
    # Response token cap sent as payload 'max_tokens' (Tab 2 spinbox).
    # Previously this value only lived inside each preset's GenerationParams
    # ('max_tokens' in captioner_config.json), so it was saved/exported but
    # had no GUI control and could not be adjusted from Tab 2 at all.
    max_tokens: int = 2048
    cache_type: str = "q8_0"
    # Multi-GPU tensor split mode passed to llama-server as '-sm <mode>'.
    # Default is 'layer' (llama.cpp's recommended split for pipeline-style
    # layer distribution across GPUs); 'row' / 'graph' remain selectable in
    # Tab 2.  Mirrors ServerConfig.split_mode in app/core/server.py.
    split_mode: str = "layer"          # row | layer | graph
    flash_attn: str = "auto"         # Tab 2 switch: auto | on | off (--flash-attn)
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""                # Tab 2 'API Key:' - server --api-key AND
                                     # the Bearer token used by this GUI's own
                                     # caption requests (Tab 4 / batch worker)
    binary: str = "llama-server"     # Tab 2 'Binary:' (autodetected / chosen)
    external_url: str = ""           # Full base URL for an external OpenAI-
                                     # compatible server (e.g. http://10.0.1.5:8080
                                     # or https://api.example.com/v1).  When set,
                                     # this takes precedence over host:port for
                                     # all requests; no local llama-server is needed.
    external_model: str = ""         # Tab 2 'External server model' - explicit
                                     # /v1/models id sent as 'model'; '' = auto
                                     # (the single loaded model, else 'default')
    run_model_key: str = ""          # Tab 2 explicit override ('' = follow Tab 1)
    run_quant: str = ""              # Tab 2 quant/filename override ('' = follow)
    verbose_logging: bool = False    # dump llama-server conversation detail
                                     # (prompts sent, answers + thinking) to
                                     # the Log Output panel
    # Companion caption file naming (Tab 2 'Companion caption files').
    # Each sidecar is stored as '<image_name><suffix>.<ext>' next to its
    # image.  Load and save may use DIFFERENT names so you can read one set of
    # files and write another.  An empty suffix means none; an empty extension
    # means "follow the active preset's output format" ('json' for MOSS, 'txt'
    # for tag presets) when saving, or ".json then .txt" (historical search)
    # when loading.  All-empty defaults reproduce the original naming exactly,
    # so existing datasets keep working unchanged.
    caption_load_suffix: str = ""
    caption_load_ext: str = ""
    caption_save_suffix: str = ""
    caption_save_ext: str = ""
    # When True the image's own extension is KEPT in the caption file name
    # (cat.png -> cat.png.md); when False (default) it is dropped (cat.md).
    caption_keep_image_ext: bool = False
    # ---- In-RAM upscaling before analysis (Tab 2 'Upscale Before Analysis') ---
    # When enabled, each image is resized in RAM right before its request is
    # built (Lanczos/bicubic; independent X/Y factors of 1/2/4) and the
    # re-encoded result is embedded in the payload instead of the raw file
    # bytes.  Source files are never modified and nothing hits the disk.
    # Off by default -> payloads stay byte-identical to the pre-feature app.
    upscale_enabled: bool = False
    upscale_method: str = "lanczos"     # key into app.core.upscale.METHODS
    upscale_scale_x: int = 2
    upscale_scale_y: int = 2
    # ---- Tab 4 bounding-box overlay ---------------------------------------------
    # 'overlay_box_layout' / 'overlay_coord_scale' are keys into
    # app.core.boxes.LAYOUTS / SCALES.  Their real home is the *preset*, so they
    # travel inside a preset/config export; the copy here is only a restart
    # mirror, because app_settings.json persists settings while presets are
    # rebuilt from defaults unless a captioner_config.json was imported.
    # Precedence: hardwired preset default -> this mirror -> imported config.
    overlay_enabled: bool = True
    overlay_box_layout: str = "yxyx"
    overlay_coord_scale: str = "0-1000"
    overlay_colour_mode: str = "vivid"   # 'vivid' (bright band) | 'raw' (md5 bytes)
    overlay_labels: str = "none"         # 'none' | 'number' | 'text' (always-on chips)
    #: power-user override: draw geometry even for tag/prose presets (Booru,
    #: Markdown), whose prompts ask for none but whose JSON may carry some.
    overlay_any_preset: bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def apply_dict(self, d: dict) -> None:
        """Copy recognised keys from *d* with type coercion; ignore the rest.

        Booleans are only accepted for bool fields — ``int(True) == 1`` used
        to let a stray JSON ``true`` clobber numeric settings (port, top_k,
        max_tokens...) exactly like the unified importer does now (see
        :func:`app.core.models._coerce`).  Numeric strings still coerce.
        """
        for k, v in (d or {}).items():
            if k not in self.__dataclass_fields__:
                continue
            try:
                f = self.__dataclass_fields__[k]
                if f.type in ("bool", bool):
                    if isinstance(v, bool):
                        setattr(self, k, v)
                    elif isinstance(v, str) \
                            and v.strip().lower() in ("true", "false"):
                        setattr(self, k, v.strip().lower() == "true")
                    # numbers never flip a bool field -> keep current value
                elif isinstance(v, bool):
                    pass                # bools never fill numeric/str fields
                elif f.type in ("float", float):
                    setattr(self, k, float(v))
                elif f.type in ("int", int):
                    setattr(self, k, int(v))
                elif f.type in ("str", str):
                    setattr(self, k, str(v))
                else:
                    setattr(self, k, v)
            except (TypeError, ValueError):
                pass    # keep the current value on garbage input


class AppContext(QObject):
    model_changed = Signal()            # model/quant selection changed
    preset_changed = Signal()           # active preset/explicit mode changed
    image_focused = Signal(str)         # path of focused image (Tab 4 -> panel)
    box_clicked = Signal(str, str)      # (image path, box text) — Tab 4 overlay
    server_state = Signal(str)          # mirrored from ServerManager.state_changed
    storage_changed = Signal()          # model storage location changed (Tab 1)
    log = Signal(str)                   # global log sink (Tab 5 text box)
    error_log = Signal(str, bool)       # (verbose report, show_modal_dialog)
    binary_changed = Signal()           # llama-server path autodetected/chosen
    registry_refreshed = Signal(str)    # live Hub listing updated a repo key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hw = HardwareManager(self)
        self.server = ServerManager(self.hw, parent=self)
        # Keep the server's discovery in sync with the Tab 2 'Binary:' field
        # *before* the first start() (when server.config is still None), so
        # find_binary() honours the user's choice instead of ignoring it.
        self.binary_changed.connect(self._sync_server_preferred_binary)
        self.registry: dict[str, ModelSpec] = build_registry()
        # Register this context as the process-wide one so module-level key
        # lookups (model_profile_for / payload gating) resolve repos added at
        # runtime through Tab 1 to their family profile instead of degrading
        # to the neutral one; headless/test processes never set it and keep
        # using the import-time REGISTRY.
        import app.core.models as _models_module
        _models_module._ACTIVE_CTX = self
        self.presets: dict[str, CaptionPreset] = {
            k: _clone(v) for k, v in DEFAULT_PRESETS.items()}
        self.active_preset_key: str = "moss"
        self.explicit_mode: bool = False
        self.settings = InferenceSettings()
        self.selected_model_key: str = "toriigate"
        self.selected_quant: str = "Q8_0"
        self.dataset_root: str | None = None
        self.recursive: bool = True
        self.focused_image: str | None = None
        self.batch_selection: list[str] = []      # checked images (Tab 5)
        # Raw JSON of the most recent analysis per image.  The Tab 4 overlay
        # prefers this over a sidecar on disk, so freshly analysed images show
        # their boxes immediately - and still show them when saving is off or the
        # save naming differs from the load naming.
        self.last_results: dict[str, str] = {}
        # When each of those results was produced.  A sidecar written *after*
        # the run (Caption Editor save, external tool, second batch) must win
        # again, so the cache is age-aware rather than absolute (see
        # app.core.boxes.resolve_box_source).
        self.last_results_ts: dict[str, float] = {}

        # Hydrate persisted settings (storage + inference incl. binary path).
        from app.core.models import load_app_settings
        self._persisted = load_app_settings()
        self.settings.apply_dict(self._persisted.get("settings") or {})

        # Restore the persisted Tab 1 model selection (app_settings.json).
        # A persisted key that no longer exists in the registry - e.g. a
        # custom repo removed on another machine, or hand-edited out of the
        # settings file - is logged and the built-in default stays selected:
        # start-up must never crash on stale selection state.
        _mdl = self._persisted.get("model") or {}
        if isinstance(_mdl, dict):
            _key = str(_mdl.get("key", "") or "")
            if _key:
                _spec = self.registry.get(_key)
                if _spec is None:
                    # print() as well: at this point no tab exists yet, so a
                    # signal-only message would be lost before the log panel
                    # connects (same convention as models.py's start-up output).
                    warn = (f"[models] persisted model key '{_key}' no longer "
                            f"exists in the registry (removed custom repo?) - "
                            f"falling back to '{self.selected_model_key}'.")
                    print(warn, flush=True)
                    self.log.emit(warn)
                else:
                    self.selected_model_key = _key
                    _q = str(_mdl.get("quant", "") or "")
                    if _q and _spec.file_for_quant(_q) is not None:
                        self.selected_quant = _q

        # Startup config override: an optional captioner_config.json sitting
        # next to the launcher (or pointed at by $APPCAP_CONFIG) is imported
        # automatically and OVERRIDES the hardwired preset defaults plus the
        # persisted Tab 1/Tab 2 values.  Applied before any tab exists, so
        # every widget hydrates from the imported state.
        self.startup_config_source: str | None = None
        self.startup_config_warnings: list[str] = []
        self._load_startup_config_override()

        # Wire internal mirrors ------------------------------------------------
        self.server.state_changed.connect(self.server_state.emit)
        self.server.log.connect(self.log.emit)
        self.hw.log_message.connect(self.log.emit)
        self.error_log.connect(self._on_error_log)

        # ---- Periodic server/memory monitor (60 s) ------------------------
        # Lightweight health + memory readout emitted to the GUI log every 60s
        # while the server is ready (local) or an external URL is configured.
        # Uses only /proc/meminfo and sysfs VRAM paths (no nvidia-smi, no
        # subprocess).  Zero disk writes — output goes exclusively through
        # self.log.emit() to the in-memory QPlainTextEdit widget.
        self._monitor_timer = QTimer(self)
        self._monitor_timer.setInterval(60_000)   # 60 s
        self._monitor_timer.timeout.connect(self._monitor_tick)
        self.server.state_changed.connect(self._on_monitor_server_state)

        # One-shot llama-server autodetection at startup (cheap --version
        # probes only; never downloads or compiles anything).
        self.autodetect_binary(silent=True)

    # ------------------------------------------------------- latest-run cache
    def remember_result(self, image_path: str, text: str) -> None:
        """Store the newest analysis JSON for *image_path* (+ its timestamp).

        The timestamp is what makes the Tab 4 overlay's in-session cache
        self-invalidating: a sidecar written after the run wins again.
        """
        import time
        self.last_results[image_path] = text
        self.last_results_ts[image_path] = time.time()

    def forget_results(self, image_path: str | None = None) -> None:
        """Drop one image's cached result, or every one of them."""
        if image_path is None:
            self.last_results.clear()
            self.last_results_ts.clear()
        else:
            self.last_results.pop(image_path, None)
            self.last_results_ts.pop(image_path, None)

    # ------------------------------------------------------------------ monitor
    @staticmethod
    def _read_sys_memory() -> str:
        """Read system RAM from /proc/meminfo (pure file read, no subprocess)."""
        try:
            with open("/proc/meminfo", "r") as f:
                info = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        info[parts[0].strip()] = int(parts[1].split()[0])  # kB
            total_gb = info.get("MemTotal", 0) / 1_048_576
            avail_gb = info.get("MemAvailable", 0) / 1_048_576
            used_gb = total_gb - avail_gb
            return f"RAM {used_gb:.1f}/{total_gb:.1f}G"
        except (OSError, ValueError, IndexError):
            return "RAM ?"

    @staticmethod
    def _read_gpu_vram() -> str:
        """Read GPU VRAM from sysfs (NVIDIA/AMD) — no nvidia-smi needed.

        NVIDIA: /sys/class/drm/card*/device/mem_info_vram_{used,total}
        AMD:    /sys/class/drm/card*/device/mem_info_vram_{used,total}  (amdgpu)
        Returns a short string or '' when no GPU sysfs is available.
        """
        import glob
        # Try all DRM card devices
        for pattern in ("/sys/class/drm/card[0-9]*/device/mem_info_vram_total",
                        "/sys/class/drm/card[0-9]*/device/vram_total"):
            for total_path in sorted(glob.glob(pattern)):
                used_path = total_path.replace("total", "used")
                try:
                    with open(total_path) as f:
                        total_b = int(f.read().strip())
                    with open(used_path) as f:
                        used_b = int(f.read().strip())
                    # Identify which card
                    card_id = os.path.basename(os.path.dirname(os.path.dirname(
                        total_path)))
                    return (f"VRAM {used_b / (1024**3):.1f}/"
                            f"{total_b / (1024**3):.1f}G ({card_id})")
                except (OSError, ValueError):
                    continue
        return ""

    def _monitor_tick(self):
        """60-second periodic: probe server + read memory stats -> log line.

        Runs in a short-lived daemon thread so the 3 s HTTP timeout never
        blocks the GUI event loop.  Zero disk writes — only emits to
        self.log (in-memory QPlainTextEdit).
        """
        def _do_probe():
            base = self.base_url()
            url = base.rstrip("/") + "/health"
            key = self.api_key()
            headers = {}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            # --- server status ---
            if self.using_external_server:
                # External servers (Unsloth studio et al.) may have NO /health
                # - use the models-first probe instead of logging HTTP 404s.
                from app.core.server import probe_server
                st, det = probe_server(base, key, timeout=3)
                srv_status = st + (f" ({det})" if det else "")
            else:
                srv_status = "?"
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=3) as r:
                        srv_status = f"OK ({r.status})"
                except Exception as e:
                    # Distinguish 'loading' from 'dead'
                    import urllib.error
                    if isinstance(e, urllib.error.HTTPError):
                        body = ""
                        try:
                            body = e.read().decode("utf-8", "replace")[:80]
                        except Exception:
                            pass
                        srv_status = f"HTTP {e.code}"
                        if "Loading model" in body:
                            srv_status += " (loading)"
                    else:
                        srv_status = "down"
            # --- memory ---
            mem_parts = [self._read_sys_memory()]
            vram = self._read_gpu_vram()
            if vram:
                mem_parts.append(vram)
            line = f"[monitor] server={srv_status} | {'  |  '.join(mem_parts)}"
            self.log.emit(line)

        t = threading.Thread(target=_do_probe, daemon=True,
                             name="server-monitor")
        t.start()

    def _on_monitor_server_state(self, state: str):
        """Start/stop the monitor timer based on server readiness."""
        active = (state in ("starting", "ready") or
                  self.using_external_server)
        if active and not self._monitor_timer.isActive():
            self._monitor_timer.start()
            # Fire once immediately so the first readout appears promptly.
            self._monitor_tick()
        elif not active and self._monitor_timer.isActive():
            self._monitor_timer.stop()

    # ------------------------------------------------------------------ binary
    def _sync_server_preferred_binary(self):
        """Mirror the current Tab 2 'Binary:' value into ServerManager.

        Called whenever the binary path is autodetected/user-chosen and at
        startup after persisted settings are hydrated; makes find_binary()
        honour the field even before the first start() (server.config None).
        """
        self.server.preferred_binary = self.settings.binary

    def autodetect_binary(self, silent: bool = False) -> str | None:
        """Find a usable llama-server and adopt it into the settings.

        Honours a previously user-chosen/autodetected path when still valid;
        otherwise picks the best candidate (CUDA > ROCm > Vulkan > CPU).
        Returns the adopted path or ``None`` when nothing was found.
        """
        from app.core import llamacpp
        chosen = llamacpp.pick_best(self.settings.binary)
        if chosen is not None:
            self.settings.binary = chosen["path"]
            self._sync_server_preferred_binary()
            if not silent:
                self.log.emit(
                    f"[llama.cpp] using '{chosen['path']}' "
                    f"(backend: {chosen['backend']})"
                    + (f"  [{chosen['version']}]" if chosen.get("version") else ""))
            self.binary_changed.emit()
            return chosen["path"]
        if not silent:
            self.log.emit("[llama.cpp] autodetect found no llama-server.\n"
                          + llamacpp.describe_search())
        return None

    # ------------------------------------------------- startup config override
    @staticmethod
    def startup_config_path() -> str | None:
        """Locate an optional ``captioner_config.json`` for this launch.

        Thin wrapper around :func:`app.core.models.find_startup_config`
        ($APPCAP_CONFIG → CWD → project root); see there for the details.
        """
        from app.core.models import find_startup_config
        return find_startup_config()

    def _load_startup_config_override(self):
        """Import :func:`startup_config_path` if present, overriding defaults.

        Uses the same importer as Tab 3's Import button, so both the unified
        bundle (presets + settings + storage + model) and legacy preset-only
        files work.  A broken file never blocks startup: the problem is logged
        and the hardwired/persisted configuration stays in effect.
        """
        path = self.startup_config_path()
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            self.log.emit(
                f"[config] captioner_config.json found at {path} but could "
                f"not be read ({type(exc).__name__}: {exc}); using built-in "
                "defaults instead.")
            return
        try:
            from app.core.models import import_full_config
            warnings = import_full_config(self, raw)
        except Exception as exc:
            self.log.emit(
                f"[config] startup import of {path} FAILED: "
                f"{type(exc).__name__}: {exc}; using built-in defaults.")
            return
        self.startup_config_source = path
        self.startup_config_warnings = list(warnings or [])
        n_presets = len(self.presets)
        self.log.emit(
            f"[config] startup: imported {path} — overrode hardwired presets "
            f"({n_presets} preset(s), active '{self.active_preset_key}', "
            f"explicit mode {'on' if self.explicit_mode else 'off'}), plus "
            "Tab 1/Tab 2 settings/storage/model selection.")
        for w in self.startup_config_warnings:
            self.log.emit(f"[config] startup import warning: {w}")

    def save_settings(self):
        """Persist Tab-1/Tab-2 configuration to app_settings.json."""
        from app.core.models import save_app_settings
        data = dict(self._persisted)
        data["storage"] = self.storage.to_dict()
        data["settings"] = self.settings.to_dict()
        data["model"] = {"key": self.selected_model_key,
                         "quant": self.selected_quant,
                         "repo": self.selected_repo_id}
        # User-added HF repos survive restarts (labels round-trip through the
        # specs' display names; built-ins are never persisted here - they are
        # hardwired in build_registry()).
        data["custom_repos"] = [{"repo": s.source_repo, "label": s.display_name}
                                for k, s in self.registry.items()
                                if k not in BUILTIN_SPEC_KEYS]
        try:
            save_app_settings(data)
        except Exception as exc:
            self.report_error(exc, context="saving application settings",
                              dialog=False)
        self._persisted = data

    # ------------------------------------------------------------------ errors
    def report_error(self, exc: Exception | str, context: str = "",
                     dialog: bool = False):
        """Centralized, verbose error reporting.

        Structured errors (:class:`AppError` and subclasses such as
        :class:`StorageLocationError`, plus
        :class:`ResourcePlanError`) render their own reason + fix-suggestion
        reports; anything else is wrapped with a short traceback so live
        testing on the user's machine produces actionable bug reports
        instead of silent failures or crash dialogs.
        """
        lines: list[str] = []
        if context:
            lines.append(f"[app] while: {context}")
        if isinstance(exc, str):
            lines.append(exc)
        elif hasattr(exc, "report") and callable(exc.report):
            # Typed, verbose error - use its structured multi-line rendering.
            lines.extend(exc.report().splitlines())
        else:
            tb = "".join(traceback.format_exception(type(exc), exc,
                                                    exc.__traceback__, limit=8))
            lines.append(f"[app] ERROR: {type(exc).__name__}: {exc}")
            lines.append("    trace  : " + tb.replace("\n", "\n    ").rstrip())
            lines.append("    fix    : copy this whole block into the bug report.")
        report = "\n".join(lines)
        self.error_log.emit(report, bool(dialog))

    def _on_error_log(self, report: str, dialog: bool):
        # Always land in the global log first (works even before tabs exist).
        self.log.emit(report)
        if dialog:
            try:
                from PySide6.QtWidgets import QApplication, QMessageBox
                top = QApplication.topLevelWidgets()
                parent = next((w for w in top if w.isVisible()), None)
                QMessageBox.warning(parent, "Auto-Captioner - Error", report)
            except Exception:
                # No display / headless: the log entry above is sufficient.
                pass

    # ------------------------------------------------------------------ helpers
    @property
    def active_model_spec(self) -> ModelSpec:
        return self.registry[self.selected_model_key]

    # ------------------------------------------------- tracked repo ids (§3.1)
    @property
    def selected_repo_id(self) -> str:
        """Hugging Face repo id of the Tab 1 model selection."""
        spec = self.registry.get(self.selected_model_key)
        return spec.source_repo if spec is not None else ""

    @property
    def run_repo_id(self) -> str:
        """Hugging Face repo id of the Tab 2 run target (follow-mode aware)."""
        spec = self.registry.get(self.run_model_key)
        return spec.source_repo if spec is not None else self.selected_repo_id

    # ------------------------------------------------------ run-target model
    @property
    def run_model_key(self) -> str:
        """Model key llama-server should load (Tab 2 override, else Tab 1)."""
        key = getattr(self.settings, "run_model_key", "") or ""
        if key in self.registry:
            return key
        return self.selected_model_key

    @property
    def run_quant(self) -> str:
        """Quant tag of the run target (falls back to the Tab 1 selection)."""
        spec = self.registry.get(self.run_model_key)
        if spec is None:
            return self.selected_quant
        q = getattr(self.settings, "run_quant", "") or ""
        if q and spec.file_for_quant(q) is not None:
            return q
        if self.run_model_key == self.selected_model_key \
                and spec.file_for_quant(self.selected_quant) is not None:
            return self.selected_quant
        for cand in ("Q8_0", "Q8-0"):
            f = spec.file_for_quant(cand)
            if f is not None:
                return f.quant
        return spec.quants()[0] if spec.quants() else ""

    @property
    def run_model_file(self):
        """The concrete :class:`ModelFile` llama-server must launch with."""
        spec = self.registry[self.run_model_key]
        return spec.file_for_quant(self.run_quant)

    @property
    def storage(self) -> StorageManager:
        """Active model-storage configuration (Tab 1 -> Storage Location)."""
        import app.core.models as _models
        return _models.STORAGE

    def set_storage_location(self, mode: str, unsloth_root: str = "") -> None:
        """Persist a new storage location and notify every dependent tab.

        Any failure is re-raised as a verbose AppError subclass so the caller
        can funnel it through :meth:`report_error`.
        """
        from app.core.models import set_storage
        set_storage(mode, unsloth_root)
        self.log.emit(f"[storage] {self.storage.describe()}")
        self.storage_changed.emit()
        self.model_changed.emit()      # cached-file badges must refresh

    # ------------------------------------------------- custom HF repositories
    def add_custom_repo(self, repo_id: str,
                        label: str = "") -> ModelSpec | None:
        """Register a user-added Hugging Face repository in the live registry.

        Synchronous and offline-safe by contract: the caller (Tab 1's Add
        flow) has ALREADY verified on its worker thread that the repo lists
        at least one main GGUF, so :func:`build_spec_from_repo` here works
        off the TTL-cached Hub listing instead of blocking the GUI.  Dedupes
        against every known spec (built-ins included - in which case the
        existing entry is simply returned); persists the 'custom_repos'
        section and emits ``registry_refreshed`` so Tab 2 picks up the new
        model immediately.

        Returns the registered :class:`ModelSpec`, or ``None`` when validation
        failed (the verbose AppError report is surfaced via
        :meth:`report_error` with a modal dialog).
        """
        from app.core.models import build_spec_from_repo, validate_repo_id
        try:
            norm = validate_repo_id(repo_id)
        except Exception as exc:      # AppError (invalid id) or surprise
            self.report_error(exc, context=f"adding custom repo '{repo_id}'",
                              dialog=True)
            return None
        for s in self.registry.values():
            if s.source_repo.lower() == norm.lower():
                # Already known (built-in or previously added): select it
                # instead of duplicating (plan §6, case 5).
                self.log.emit(
                    f"[models] '{norm}' is already available as "
                    f"'{s.display_name}' - selecting the existing entry.")
                return s
        try:
            spec = build_spec_from_repo(norm, label)
        except Exception as exc:      # listing failed / no GGUFs in between
            self.report_error(exc, context=f"adding custom repo '{norm}'",
                              dialog=True)
            return None
        self.registry[spec.key] = spec
        self.log.emit(
            f"[models] added custom repository '{norm}' "
            f"(family: {spec.family}, quants: {len(spec.files)}) - key "
            f"'{spec.key}'.")
        try:
            self.save_settings()
        except Exception as exc:
            self.report_error(exc, context="persisting custom repositories",
                              dialog=False)
        self.registry_refreshed.emit(spec.key)
        self.model_changed.emit()
        return spec

    def remove_custom_repo(self, key: str) -> bool:
        """Remove a user-added repository from the live registry (Tab 1).

        Refused while that spec is the current Tab 1 selection or an active
        Tab 2 explicit run target; otherwise removes it, persists, emits
        ``registry_refreshed`` and lets dependent widgets fall back.  The
        built-in models can never be removed.  Downloaded files are
        intentionally NOT touched - they remain on disk in the storage tree.
        """
        if key in BUILTIN_SPEC_KEYS:
            self.log.emit(f"[models] '{key}' is a built-in model and cannot "
                          "be removed.")
            return False
        spec = self.registry.get(key)
        if spec is None:
            return True               # already gone: nothing to do
        if key == self.selected_model_key:
            self.log.emit(
                f"[models] cannot remove '{spec.display_name}': it is the "
                "current selection - switch models first.")
            return False
        if key == (self.settings.run_model_key or ""):
            self.log.emit(
                f"[models] cannot remove '{spec.display_name}': Tab 2 has it "
                "as the explicit run target - clear that override first.")
            return False
        del self.registry[key]
        # Fall the persisted selection back to a built-in with a warning so
        # start-up validation on the next launch stays consistent (plan §6,
        # case 6).
        if self.selected_model_key == key:
            self.log.emit(f"[models] '{spec.display_name}' removed - model "
                          f"selection falls back to 'toriigate'.")
            self.selected_model_key = "toriigate"
            tq = ("Q8_0" if "Q8_0" in self.registry["toriigate"].quants()
                  else None) or next(iter(self.registry["toriigate"].quants()),
                                     "")
            self.selected_quant = tq
        elif (self.settings.run_model_key or "") == key:
            self.log.emit(f"[models] '{spec.display_name}' removed - Tab 2 "
                          "run target cleared (follow mode).")
            self.settings.run_model_key = ""
            self.settings.run_quant = ""
        self.log.emit(f"[models] removed custom repository "
                      f"'{spec.source_repo}' (key '{key}'). Downloaded files "
                      "were NOT deleted.")
        try:
            self.save_settings()
        except Exception as exc:
            self.report_error(exc, context="persisting custom repositories",
                              dialog=False)
        self.registry_refreshed.emit("")
        self.model_changed.emit()
        return True

    @property
    def active_preset(self) -> CaptionPreset:
        return self.presets[self.active_preset_key]

    def effective_system_prompt(self) -> str:
        spec = self.active_model_spec
        return self.active_preset.effective_system_prompt(
            explicit=self.explicit_mode, abliterated_model=spec.abliterated)

    # ------------------------------------------------------------- server URL
    @staticmethod
    def _client_host(host: str) -> str:
        """Map a *bind* address to an address this GUI can CONNECT to.

        Wildcard binds (0.0.0.0 / ::) mean "all local interfaces"; the
        matching client target is loopback.  Anything else - including real
        LAN IPs and remote hostnames - is used as-is.
        """
        h = (host or "").strip().strip("[]").lower()
        if h in ("", "0.0.0.0", "::"):
            return "127.0.0.1"
        return h

    @property
    def using_external_server(self) -> bool:
        """True when the user has configured an external server URL."""
        return bool((self.settings.external_url or "").strip())

    def base_url(self) -> str:
        """Base URL for caption requests against the configured endpoint.

        Priority:
          1. Tab 2 'External Server URL' (full URL like http://host:port)
          2. Live ServerConfig of the running/attached local server
          3. Tab 2 host/port fields (before the first start())

        Local wildcard hosts collapse to 127.0.0.1 so '0.0.0.0' still works.
        """
        ext = (self.settings.external_url or "").strip()
        if ext:
            # Bare origin, '.../v1' or '.../v1/chat/completions' all mean the
            # same server - normalise to the bare base; CaptionClient appends
            # its own /v1/... paths.
            from app.core.api_client import normalize_base_url
            return normalize_base_url(ext)
        cfg = self.server.config
        if cfg is not None:
            return f"http://{self._client_host(cfg.host)}:{cfg.port}"
        s = self.settings
        return f"http://{self._client_host(s.host)}:{int(s.port)}"

    def api_key(self) -> str:
        """Bearer token for our own requests (Tab 2 'API Key:' field).

        Empty when no key was set; falls back to MOSS_API_KEY /
        APPCAP_API_KEY so env-var setups keep working unchanged.
        """
        key = (self.settings.api_key or "").strip()
        if not key:
            import os
            key = (os.environ.get("MOSS_API_KEY")
                   or os.environ.get("APPCAP_API_KEY") or "")
        return key

    def resolve_external_model_id(self, timeout: float = 5.0) -> str:
        """Model id for EXTERNAL chat requests (see api_client docstring)."""
        from app.core.api_client import resolve_external_model
        return resolve_external_model(
            self.base_url(), self.api_key(),
            explicit=(self.settings.external_model or "").strip(),
            log_cb=self.log.emit, timeout=timeout)

    def make_client(self, timeout: float = 600.0):
        """CaptionClient wired to the current host/port/API key."""
        from app.core.api_client import CaptionClient
        return CaptionClient(self.base_url(), timeout=timeout,
                             api_key=self.api_key())
