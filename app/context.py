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
from app.core.server import ServerManager


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
    # Verbose llama-server traffic logging: when True the batch client logs
    # the exact request payload (prompts + redacted image) and the full raw
    # response incl. streamed thinking to the global log.  None = auto
    # (honours the MOSS_VERBOSE_LOG env var), True/False = explicit override.
    verbose_logging: bool | None = None
    cache_type: str = "q8_0"
    split_mode: str = "layer"        # row | layer | graph (see ServerConfig)
    host: str = "127.0.0.1"
    port: int = 8080
    binary: str = "llama-server"     # Tab 2 'Binary:' (autodetected / chosen)

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
