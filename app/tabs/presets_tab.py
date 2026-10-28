"""Tab 3 - Captioning Presets & Prompts.

* Hardwired preset selector (MOSS-Image / Booru-SDXL / Custom).
* Explicit Mode checkbox -> injects NSFW triggers into the system prompt and
  tells the server manager to disable safety-system prompts.
* System/User prompt editors (live two-way bound to the active preset).
* Export/Import Config as JSON - the **unified** bundle covering every
  configurable piece: captioning presets (this tab), inference settings +
  llama-server binary (Tab 2) and storage/model selection (Tab 1).
"""
from __future__ import annotations

import json

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QTextEdit, QVBoxLayout, QWidget)

from app.context import AppContext
from app.core.models import export_full_config, import_full_config
from app.core.presets import EXPLICIT_TRIGGER, _clone


class PresetsTab(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._loading = False

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for p in ctx.presets.values():
            self.preset_combo.addItem(p.display_name, p.key)
        idx = self.preset_combo.findData(ctx.active_preset_key)
        self.preset_combo.setCurrentIndex(max(0, idx))
        top.addWidget(self.preset_combo, stretch=1)

        self.explicit_check = QCheckBox("Explicit Mode (NSFW)")
        self.explicit_check.setChecked(ctx.explicit_mode)
        top.addWidget(self.explicit_check)
        root.addLayout(top)

        self.explicit_hint = QLabel(
            "When enabled and an abliterated model is selected, the trigger\n"
            f"\u201c{EXPLICIT_TRIGGER}\u201d\n"
            "is appended to the system prompt and server safety prompts are disabled.")
        self.explicit_hint.setStyleSheet("color: #a66;")
        self.explicit_hint.setWordWrap(True)
        root.addWidget(self.explicit_hint)

        sp_box = QGroupBox("System Prompt")
        sp_l = QVBoxLayout(sp_box)
        self.system_edit = QTextEdit()
        self.system_edit.setAcceptRichText(False)
        sp_l.addWidget(self.system_edit)
        root.addWidget(sp_box, stretch=2)

        up_box = QGroupBox("User Prompt")
        up_l = QVBoxLayout(up_box)
        self.user_edit = QTextEdit()
        self.user_edit.setAcceptRichText(False)
        self.user_edit.setMaximumHeight(90)
        up_l.addWidget(self.user_edit)
        root.addWidget(up_box, stretch=1)

        io_row = QHBoxLayout()
        self.export_btn = QPushButton("Export Config")
        self.import_btn = QPushButton("Import Config")
        io_row.addWidget(self.export_btn)
        io_row.addWidget(self.import_btn)
        io_row.addStretch(1)
        root.addLayout(io_row)

        # ---- wiring ------------------------------------------------------------
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        self.explicit_check.toggled.connect(self.on_explicit_toggled)
        self.system_edit.textChanged.connect(self.save_edits)
        self.user_edit.textChanged.connect(self.save_edits)
        self.export_btn.clicked.connect(self.export_cfg)
        self.import_btn.clicked.connect(self.import_cfg)

        self.load_preset()

    # ------------------------------------------------------------------ slots
    def load_preset(self):
        key = self.preset_combo.currentData() or self.ctx.active_preset_key
        preset = self.ctx.presets.get(key) or self.ctx.active_preset
        self.ctx.active_preset_key = key
        self._loading = True
        self.system_edit.setPlainText(preset.system_prompt)
        self.user_edit.setPlainText(preset.user_prompt)
        self._loading = False
        self.ctx.preset_changed.emit()

    def save_edits(self):
        if self._loading:
            return
        key = self.ctx.active_preset_key
        preset = self.ctx.presets[key]
        preset.system_prompt = self.system_edit.toPlainText()
        preset.user_prompt = self.user_edit.toPlainText()

    def on_explicit_toggled(self, checked: bool):
        self.ctx.explicit_mode = checked
        spec = self.ctx.active_model_spec
        if checked and not spec.abliterated:
            self.ctx.log.emit(
                f"[presets] Explicit mode ON but '{spec.display_name}' is not an "
                "abliterated build - trigger injection skipped; switch to "
                "Qwen3.8-27B-abliterated for uncensored captions.")
        else:
            self.ctx.log.emit(f"[presets] Explicit mode {'enabled' if checked else 'disabled'}")
        self.ctx.preset_changed.emit()

    # ------------------------------------------------------------------ IO
    def export_cfg(self):
        """Export the UNIFIED config: presets + Tab 2 settings + storage."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export full application config", "captioner_config.json",
            "JSON (*.json)")
        if not path:
            return
        try:
            payload = export_full_config(self.ctx)
            raw = json.dumps(payload, indent=2, ensure_ascii=False)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(raw)
        except OSError as exc:
            self.ctx.report_error(
                f"[presets] EXPORT FAILED: {type(exc).__name__}: {exc}\n"
                f"    path   : {path}\n"
                "    fix    : check that the target directory exists and is writable.",
                context="exporting config", dialog=True)
            return
        except Exception as exc:
            self.ctx.report_error(exc, context="building config export", dialog=True)
            return
        self.ctx.save_settings()
        self.ctx.log.emit(
            f"[config] unified config exported -> {path} "
            "(presets + inference/binary + storage + model selection)")

    def import_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import application config (unified or legacy preset-only)",
            "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except FileNotFoundError:
            self.ctx.report_error(
                f"[presets] IMPORT FAILED: file does not exist:\n    {path}\n"
                "    fix    : re-pick the file in the dialog.",
                context="importing config", dialog=True)
            return
        except OSError as exc:
            self.ctx.report_error(
                f"[presets] IMPORT FAILED: cannot read file: "
                f"{type(exc).__name__}: {exc}",
                context=f"importing config from {path}", dialog=True)
            return
        try:
            warnings = import_full_config(self.ctx, raw)
        except FileNotFoundError:
            self.ctx.report_error(
                f"[presets] IMPORT FAILED: file does not exist:\n    {path}",
                context="importing config", dialog=True)
            return
        except (ValueError, json.JSONDecodeError) as exc:
            self.ctx.report_error(
                f"[presets] IMPORT FAILED: the file is not a valid captioner "
                f"config.\n    reason : {type(exc).__name__}: {exc}\n"
                "    fix    : export a fresh config and edit that, or check "
                "the JSON syntax (keys: version/presets/settings/storage/model).",
                context="importing config", dialog=True)
            return
        except Exception as exc:
            self.ctx.report_error(exc, context=f"importing config from {path}",
                                  dialog=True)
            return
        for w in warnings:
            self.ctx.log.emit(f"[config] import warning: {w}")
        self.ctx.save_settings()
        self.ctx.log.emit(f"[config] unified config imported from {path} "
                          f"({len(self.ctx.presets)} preset(s); all tabs "
                          "refreshed)")
        self.reload_combos()
        # Push the new values into the other tabs' widgets: Tab 2 (binary +
        # sliders/host/port) and Tab 1 (storage + model/quant selectors).
        # Use duck-typing instead of an isinstance check against MainWindow:
        # importing app.main_window here would be a circular import (it builds
        # the tabs), and hard-coupling the tab to the window class previously
        # raised NameError at runtime.
        mw = self.window()
        for attr in ("tab_inference", "tab_hardware"):
            tab = getattr(mw, attr, None)
            refresher = getattr(tab, "refresh_from_context", None)
            if callable(refresher):
                try:
                    refresher()
                except Exception as exc:  # never break the import flow
                    self.ctx.report_error(exc, context=f"refreshing {attr}")

    def reload_combos(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for p in self.ctx.presets.values():
            self.preset_combo.addItem(p.display_name, p.key)
        idx = self.preset_combo.findData(self.ctx.active_preset_key)
        self.preset_combo.setCurrentIndex(max(0, idx))
        self.preset_combo.blockSignals(False)
        self.explicit_check.blockSignals(True)
        self.explicit_check.setChecked(self.ctx.explicit_mode)
        self.explicit_check.blockSignals(False)
        self.load_preset()
