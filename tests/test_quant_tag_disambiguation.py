"""Regression tests: ambiguous quant tags must never resolve to the wrong
file, and Tab 1 must keep a valid quant selection across dropdown
re-population.

Background (the fixed bugs):
* ``_quant_tag_from_name`` used to parse every '...-Q*_K_P.gguf' file of the
  HauhauCS Qwen3.6 repos down to the bare tag 'P' (likewise six UD-*_XL files
  -> 'XL', all -L variants -> 'L').  Selection, persistence and download
  planning key on that tag, so picking any one of the colliding entries
  silently downloaded the FIRST file carrying the tag.
* ``HardwareModelsTab.refresh_quants`` restored the previous selection from
  the combo's *display text* (which can never match a tag lookup), so every
  re-population reset the dropdown to the topmost quant (IQ2_M for repos
  without a Q8_0).

All tests are headless and offline: Hugging Face access is faked through a
monkeypatched ``_lookup_entry``/``_fetch_hf_config`` (same convention as
tests/test_custom_repo.py); the Qt test runs on the offscreen platform.

    python -m pytest tests/test_quant_tag_disambiguation.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.core import models as M                                        # noqa: E402
from app.core.models import (_disambiguate_quant_tags,                     # noqa: E402
                             _quant_tag_from_name)

# The real HauhauCS Qwen3.6-27B-Uncensored-HauhauCS-Aggressive listing
# (filenames from the Hub; mmproj + README included on purpose).
HAU = "HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive"
P_QUANTS = ("Q2_K_P", "Q3_K_P", "Q4_K_P", "Q5_K_P", "Q6_K_P", "Q8_K_P")
HAU_FILES = ([f"Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-{q}.gguf"
              for q in ("IQ2_M", "IQ3_M", "IQ3_XS", "IQ4_XS", *P_QUANTS)]
             + ["mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-f16.gguf",
                "README.md"])

HUIHUI = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
HUIHUI_FILES = [
    "Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf",
    "Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S.gguf",
    "Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf",
    "Huihui-Qwen3.8-27B-abliterated-Q4_K_L.gguf",
    "Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf",
    "Huihui-Qwen3.8-27B-abliterated-Q8_0_L.gguf",
    "Huihui-Qwen3.8-27B-abliterated-UD-DW-IQ3_S.gguf",
    "Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf",
    "Huihui-Qwen3.8-27B-abliterated-UD-IQ2_S-MTP.gguf",
    "Huihui-Qwen3.8-27B-abliterated-UD-IQ2_S.gguf",
    "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf",
    "Huihui-Qwen3.8-27B-abliterated-UD-Q2_K_XL.gguf",
    "Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf",
    "Huihui-Qwen3.8-27B-abliterated-UD-Q8_K_XL.gguf",
    "Huihui-Qwen3.8-27B-abliterated-bf16.gguf",
    "mmproj-model-bf16.gguf",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _serve(monkeypatch, table):
    """Fake every Hub-backed lookup (listing + config) so tests stay offline."""
    def _fake(repo_id, force=False):
        files = table.get(repo_id, [])
        return {"files": list(files), "sizes": {f: 1 << 30 for f in files},
                "ts": time.monotonic()}
    monkeypatch.setattr(M, "_lookup_entry", _fake)
    monkeypatch.setattr(M, "_fetch_hf_config", lambda repo_id: None)


# --------------------------------------------------------------------------
# 1. parser: full quant spans, not bare suffixes
# --------------------------------------------------------------------------
def test_parser_full_spans():
    cases = {
        # the reported P-collision (old parser returned 'P' for all six)
        "Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf": "Q2_K_P",
        "Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf": "Q8_K_P",
        # the reported XL-collision (old parser returned 'XL' for all six)
        "Huihui-Qwen3.8-27B-abliterated-UD-Q2_K_XL.gguf": "UD-Q2_K_XL",
        "Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf": "UD-Q4_K_XL",
        # -L variants must not all collapse to 'L'; Q8_0_L stays intact
        "Huihui-Qwen3.8-27B-abliterated-Q4_K_L.gguf": "Q4_K_L",
        "Huihui-Qwen3.8-27B-abliterated-Q8_0_L.gguf": "Q8_0_L",
        # family prefixes + MTP suffixes stay part of the identity
        "Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S-mtp.gguf":
            "GSQ-RCO-IQ3_S-MTP",
        "Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_S.gguf": "GSQ-RCO-IQ3_S",
        "Huihui-Qwen3.8-27B-abliterated-UD-DW-IQ3_S.gguf": "UD-DW-IQ3_S",
        "Huihui-Qwen3.8-27B-abliterated-UD-IQ2_S-MTP.gguf": "UD-IQ2_S-MTP",
        # K-size classes kept (old parser dropped the trailing _M/_S/L)
        "Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf": "UD-DW-Q4_K_M",
        # lowercase + bare quants unchanged
        "Huihui-Qwen3.8-27B-abliterated-bf16.gguf": "BF16",
        "ToriiGate-0.5-BF16.gguf": "BF16",
        "ToriiGate-0.5-Q8_0.gguf": "Q8_0",
        "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf": "UD-IQ4_XS",
        # quant mid-name still found
        "model-Q4_K-ctx128k.gguf": "Q4_K",
    }
    for name, want in cases.items():
        assert _quant_tag_from_name(name) == want, name


# --------------------------------------------------------------------------
# 2. disambiguation: uniqueness guarantee for arbitrary future naming
# --------------------------------------------------------------------------
def test_disambiguate_unique_and_deterministic():
    ents = [("model-A-Q4_K.gguf", "Q4_K", 0),
            ("model-B-Q4_K.gguf", "Q4_K", 0),
            ("model-C-Q4_K.gguf", "Q4_K", 0),
            ("plain-IQ2_M.gguf", "IQ2_M", 0)]
    out = _disambiguate_quant_tags(ents)
    tags = [t for _f, t, _s in out]
    assert len({t.upper() for t in tags}) == len(tags), tags
    # untouched files keep their tag; first of a colliding group keeps it too
    assert tags[3] == "IQ2_M"
    assert tags[0] == "Q4_K"
    # deterministic: same input -> same output
    assert _disambiguate_quant_tags(ents) == out
    # filenames pass through untouched
    assert [f for f, _t, _s in out] == [e[0] for e in ents]
    # single-entry listings are a no-op
    assert _disambiguate_quant_tags([("a.gguf", "Q8_0", 1)]) == [
        ("a.gguf", "Q8_0", 1)]


# --------------------------------------------------------------------------
# 3. end-to-end: discovery -> unique tags -> select-by-tag roundtrip
# --------------------------------------------------------------------------
def _roundtrip(monkeypatch, repo_id, files):
    _serve(monkeypatch, {repo_id: files})
    spec = M.build_spec_from_repo(repo_id)
    tags = [f.quant for f in spec.files]
    assert len(set(tags)) == len(tags), f"colliding tags: {tags}"
    # THE invariant the bug broke: selecting any entry resolves to exactly
    # that file - never the first one sharing a tag.
    for f in spec.files:
        resolved = spec.file_for_quant(f.quant)
        assert resolved is not None and resolved.filename == f.filename, (
            f.quant, f.filename)
    return spec


def test_discover_roundtrip_hauhaucs(monkeypatch):
    spec = _roundtrip(monkeypatch, HAU, HAU_FILES)
    assert [f.quant for f in spec.files[4:]] == list(P_QUANTS)
    # the exact symptom from the report: picking Q4_K_P must NOT land on
    # the topmost IQ2_M (nor on the first 'P' file, Q2_K_P).
    f = spec.file_for_quant("Q4_K_P")
    assert f is not None and f.filename.endswith("-Q4_K_P.gguf")
    # mmproj excluded from the base-model list
    assert all("mmproj" not in f.filename.lower() for f in spec.files)


def test_discover_roundtrip_huihui(monkeypatch):
    spec = _roundtrip(monkeypatch, HUIHUI, HUIHUI_FILES)
    tags = set(spec.quants())
    # the old parser emitted bare 'XL'/'L' here - they must not reappear
    assert "XL" not in tags and "P" not in tags and "L" not in tags
    assert "UD-Q2_K_XL" in tags and "Q4_K_L" in tags
    assert spec.file_for_quant("UD-Q4_K_XL").filename.endswith(
        "-UD-Q4_K_XL.gguf")


def test_toriigate_unaffected(monkeypatch):
    torii = "DraconicDragon/ToriiGate-0.5-GGUF"
    files = ["ToriiGate-0.5-BF16.gguf", "ToriiGate-0.5-Q8_0.gguf"]
    _serve(monkeypatch, {torii: files})
    out = M.discover_repo_ggufs(torii)
    assert [t for _f, t, _s in out] == ["BF16", "Q8_0"]
    # ...and if the repo ever gained a case-twin file, disambiguation still
    # keeps every tag unique (selection stays exact).
    files += ["ToriiGate-0.5-Q8_0.gguf.bak",   # not a .gguf: ignored
              "torriigate-0.5-bf16.gguf"]      # case-twin of the BF16 name
    out = M.discover_repo_ggufs(torii, force_refresh=True)
    tags = [t for _f, t, _s in out]
    assert len({t.upper() for t in tags}) == len(tags), tags


def test_bits_suffix_normalization():
    # K-P quants estimate at their base quant, not the 8.5 unknown default
    assert M._quant_bits_for("Q4_K_P", "x-Q4_K_P.gguf") \
        == M.QUANT_BITS["Q4_K"]
    assert M._quant_bits_for("Q8_0_L", "x-Q8_0_L.gguf") \
        == M.QUANT_BITS["Q8_0"]
    assert M._quant_bits_for("IQ4_XS", "x-IQ4_XS.gguf") \
        == M.QUANT_BITS["IQ4_XS"]
    assert M._quant_bits_for("UNKNOWN9", "x-UNKNOWN9.gguf") == 8.5


# --------------------------------------------------------------------------
# 4. GUI regression (offscreen Qt): selection survives re-population
# --------------------------------------------------------------------------
def test_tab1_refresh_keeps_selected_quant(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    monkeypatch.setenv("APPCAP_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(M, "SETTINGS_PATH", tmp_path / "app_settings.json")
    # no startup-config import (the project root ships one next to the app)
    monkeypatch.setattr(M, "find_startup_config", lambda name=None: None)
    _serve(monkeypatch, {HAU: HAU_FILES})

    from app.context import AppContext
    from app.tabs.hardware_tab import HardwareModelsTab

    ctx = AppContext()
    spec = M.build_spec_from_repo(HAU)
    ctx.registry[spec.key] = spec          # before tab build -> in combo
    tab = HardwareModelsTab(ctx)
    try:
        mi = tab.model_combo.findData(spec.key)
        assert mi >= 0, "HauhauCS spec missing from the Model dropdown"
        tab.model_combo.setCurrentIndex(mi)      # fires refresh_quants
        # repo has no Q8_0 -> initial default is the topmost quant (IQ2_M);
        # that is intended FIRST-LOOK behaviour, not a reset of a choice.
        assert tab.quant_combo.currentData() == "IQ2_M"

        qi = next(i for i in range(tab.quant_combo.count())
                  if tab.quant_combo.itemData(i) == "Q4_K_P")
        tab.quant_combo.setCurrentIndex(qi)      # user picks a P-type quant
        assert tab.current_file() is not None
        assert tab.current_file().filename.endswith("-Q4_K_P.gguf")

        # THE reported bug: any re-population (live refresh landing, model
        # combo re-render) used to reset this to IQ2_M / the first 'P' file.
        tab.refresh_quants()
        assert tab.quant_combo.currentData() == "Q4_K_P"
        assert ctx.selected_quant == "Q4_K_P"
        assert tab.current_file() is not None
        assert tab.current_file().filename.endswith("-Q4_K_P.gguf")
    finally:
        if tab._repo_worker is not None and tab._repo_worker.isRunning():
            tab._repo_worker.wait(2000)
        tab.deleteLater()
