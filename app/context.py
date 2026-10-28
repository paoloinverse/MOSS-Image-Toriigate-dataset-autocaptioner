"""Shared application context passed to every tab / panel.

Holds the long-lived backend singletons plus cross-tab state so widgets stay
independent of each other (tabs talk through the context's signals).

Also provides :meth:`AppContext.report_error`, the single funnel every tab
uses for verbose, structured error reporting: the full multi-line message is
always written to the global log (Tab 5), and optionally shown in a modal
message box when ``dialog=True``.
"""
from __future__ import annotations

import traceback

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from app.core.hardware import HardwareManager
from app.core.models import ModelSpec, StorageManager, build_registry
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
    reasoning_budget: int = 0
    thinking_level: str = ""         # Tab 2 combo key ("" = off) - persisted
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
    run_model_key: str = ""          # Tab 2 explicit override ('' = follow Tab 1)
    run_quant: str = ""              # Tab 2 quant/filename override ('' = follow)
    verbose_logging: bool = False    # dump llama-server conversation detail
                                     # (prompts sent, answers + thinking) to
                                     # the Log Output panel

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def apply_dict(self, d: dict) -> None:
        """Copy recognised keys from *d* with type coercion; ignore the rest."""
        for k, v in (d or {}).items():
            if k not in self.__dataclass_fields__:
                continue
            try:
                f = self.__dataclass_fields__[k]
                if f.type in ("float", float):
                    setattr(self, k, float(v))
                elif f.type in ("int", int):
                    setattr(self, k, int(v))
                elif f.type in ("str", str):
                    setattr(self, k, str(v))
                elif f.type in ("bool", bool):
                    setattr(self, k, bool(v))
                else:
                    setattr(self, k, v)
            except (TypeError, ValueError):
                pass    # keep the current value on garbage input


class AppContext(QObject):
    model_changed = Signal()            # model/quant selection changed
    preset_changed = Signal()           # active preset/explicit mode changed
    image_focused = Signal(str)         # path of focused image (Tab 4 -> panel)
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

        # Hydrate persisted settings (storage + inference incl. binary path).
        from app.core.models import load_app_settings
        self._persisted = load_app_settings()
        self.settings.apply_dict(self._persisted.get("settings") or {})

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

        # One-shot llama-server autodetection at startup (cheap --version
        # probes only; never downloads or compiles anything).
        self.autodetect_binary(silent=True)

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
                        "quant": self.selected_quant}
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

    def base_url(self) -> str:
        """Base URL for caption requests against the configured endpoint.

        Prefers the live ServerConfig of the running/attached server; falls
        back to the Tab 2 host/port fields before the first start().  Local
        wildcard hosts collapse to 127.0.0.1 so '0.0.0.0' still works.
        """
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

    def make_client(self, timeout: float = 600.0):
        """CaptionClient wired to the current host/port/API key."""
        from app.core.api_client import CaptionClient
        return CaptionClient(self.base_url(), timeout=timeout,
                             api_key=self.api_key())
