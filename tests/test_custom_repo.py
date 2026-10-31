"""Generalized-model-support tests (plan: PLAN_generalized_model_support.md, M4).

All headless, no network: Hugging Face access is faked through the module's
own listing cache (_REPO_LOOKUP_CACHE) and targeted monkeypatches of
_lookup_entry / _fetch_hf_config.  Run with pytest:

    python -m pytest tests/test_custom_repo.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import models as M                       # noqa: E402
from app.context import InferenceSettings              # noqa: E402
from app.core.api_client import build_payload          # noqa: E402
from app.core.presets import (DEFAULT_PRESETS, _clone,  # noqa: E402
                              export_config)

GLM_REPO = "Acme/GLM-5.3-Vision-GGUF"
QWEN8_REPO = "Stable/Qwen3.8-Captioner-GGUF"
ABLIT_REPO = "Huihui-AI/Some-Qwen3.6-abliterated-GGUF"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fake_listing(repo_id: str, files) -> None:
    """Seed the module listing cache so every Hub-backed path becomes a hit."""
    if isinstance(files, list):
        sizes = {f: 1 << 30 for f in files}
    else:
        sizes = dict(files)
    M._REPO_LOOKUP_CACHE[repo_id] = {"files": list(sizes), "sizes": sizes,
                                     "ts": time.monotonic()}


def _serve_lookup(monkeypatch, table: "dict[str, list[str]]") -> None:
    """Replace _lookup_entry wholesale (honours force=True refreshes) so a
    build_registry()/spec.refresh() cycle stays fully offline and fast."""
    def _fake(repo_id, force=False):
        files = table.get(repo_id, [])
        return {"files": list(files), "sizes": {f: 1 << 30 for f in files},
                "ts": time.monotonic()}
    monkeypatch.setattr(M, "_lookup_entry", _fake)


def _family_stub(family: str):
    """Minimal spec stand-in for the family-first profile lookup (duck-typed)."""
    return SimpleNamespace(family=family)


# --------------------------------------------------------------------------
# 1. repo-id validation (plan section 7, edge case 3)
# --------------------------------------------------------------------------
def test_repo_id_validation_table():
    valid = {
        "Acme/GLM-5.3-Vision-GGUF": "Acme/GLM-5.3-Vision-GGUF",
        "a/b": "a/b",
        "Org.X_1/Model-2.gguf-v3": "Org.X_1/Model-2.gguf-v3",
    }
    for raw, want in valid.items():
        assert M.validate_repo_id(raw) == want

    invalid = ["Acme/GLM-5.3-Vision-GGUF/tree/main",   # extra segments
               "no-slash-here",                        # missing slash
               "/name", "org/", "a//b",                # empty parts / double slash
               "org name/model",                       # illegal characters (space)
            ""]                                   # nothing at all
    for raw in invalid:
        try:
            M.validate_repo_id(raw)
        except M.AppError as exc:
            text = (exc.message + " " + exc.detail).lower()
            assert ("repository id" in text or "segments" in text
                    or "slash" in text or "character" in text), exc.report()
        else:
            raise AssertionError(f"{raw!r} must be rejected")

    # Pasted BARE Hub URLs are normalized to org/name and ACCEPTED...
    assert M.validate_repo_id(
        "https://huggingface.co/Acme/GLM-5.3-Vision-GGUF") == \
        "Acme/GLM-5.3-Vision-GGUF"
    # ...while URLs carrying extra path segments are rejected, with a hint:
    try:
        M.validate_repo_id(
            "https://huggingface.co/Acme/GLM-5.3-Vision-GGUF/tree/main")
    except M.AppError as exc:
        joined = " ".join(exc.suggestions) + " " + exc.detail
        assert "Did you mean 'Acme/GLM-5.3-Vision-GGUF'" in joined, exc.report()
    else:
        raise AssertionError("URL with path segments must be rejected")

    # normalize_repo_id keeps trailing path segments for the validator to hint on.
    assert M.normalize_repo_id(
        "  https://huggingface.co/Org/Model/tree/main  ") == "Org/Model/tree/main"


# --------------------------------------------------------------------------
# 2. slug/key derivation + storage-path consistency (/ -> --)
# --------------------------------------------------------------------------
def test_slug_key_and_storage_path_consistency(tmp_path, monkeypatch):
    mixed = GLM_REPO[:3].upper() + GLM_REPO[3:]        # "ACMe/..." - still valid chars
    _fake_listing(mixed, ["GLM-5.3-Q8_0.gguf"])
    _fake_listing(GLM_REPO.lower(), ["GLM-5.3-Q8_0.gguf"])
    monkeypatch.setattr(M, "_fetch_hf_config", lambda rid: None)

    spec = M.build_spec_from_repo(mixed)               # case as pasted
    expected_slug = mixed.lower().replace("/", "--")   # ...slug is canonical lowercase
    assert spec.key == expected_slug, f"key {spec.key!r} != slug {expected_slug!r}"
    assert spec.source_repo == mixed                   # id itself keeps user's case

    # Determinism: the same repo in another case form yields the identical key.
    spec2 = M.build_spec_from_repo(GLM_REPO.lower())
    assert spec.key == spec2.key and spec2.files[0].quant == "Q8_0"

    # Storage layout: on-disk folders are named from the repo id AS PASTED
    # (case preserved) while registry keys/configs use the lowercase slug.
    # Both schemes are deterministic, and every runtime lookup goes through
    # the same source_repo string, so they never diverge in practice - lock
    # both halves of that contract here.
    monkeypatch.setenv("APPCAP_CACHE_DIR", str(tmp_path / "cache"))
    mf = M.ModelFile(filename="GLM-5.3-Q8_0.gguf", quant="Q8_0", repo_id=mixed)
    assert mf.local_path().parent.name == mixed.replace("/", "--"), \
        f"storage folder {mf.local_path()!r} != pasted-case slug dir"


# --------------------------------------------------------------------------
# 3. build_spec_from_repo: fake listing -> files/quants; mmproj companion;
#    abliterated heuristic on/off (plan section 7)
# --------------------------------------------------------------------------
def test_build_spec_from_fake_listing(monkeypatch):
    _fake_listing(GLM_REPO, ["GLM-5.3-Q8_0.gguf", "GLM-5.3-BF16.gguf",
                             "GLM-5.3-Q8_0.mmproj.gguf", "README.md"])
    monkeypatch.setattr(M, "_fetch_hf_config", lambda rid: None)
    spec = M.build_spec_from_repo(GLM_REPO)

    assert spec.key == GLM_REPO.lower().replace("/", "--")
    # mmproj + non-GGUF files are excluded from the quant list (verbatim Hub names):
    assert [f.filename for f in spec.files] == ["GLM-5.3-Q8_0.gguf", "GLM-5.3-BF16.gguf"]
    assert [f.quant for f in spec.files] == ["Q8_0", "BF16"]
    assert spec.live is True and not spec.abliterated

    # Live-first companion discovery off the same (cached) listing:
    name = M.resolve_mmproj_live(GLM_REPO, "Q8_0")
    assert name == "GLM-5.3-Q8_0.mmproj.gguf"


def test_build_spec_abliterated_heuristic(monkeypatch):
    monkeypatch.setattr(M, "_fetch_hf_config", lambda rid: None)

    # via the repo id (no filename markers needed)...
    _fake_listing(ABLIT_REPO, ["SomeQwen-Q8_0.gguf"])
    assert M.build_spec_from_repo(ABLIT_REPO).abliterated is True

    # ...and via a filename even when the id says nothing:
    repo2 = "Acme/PlainCaptioner-GGUF"
    _fake_listing(repo2, ["Plain-UNCENSORED-Q8_0.gguf"])
    assert M.build_spec_from_repo(repo2).abliterated is True

    # and off by default:
    repo3 = "Acme/CleanCaptioner-GGUF"
    _fake_listing(repo3, ["Clean-Q8_0.gguf"])
    assert M.build_spec_from_repo(repo3).abliterated is False


def test_build_spec_refuses_no_gguf_repos():
    repo = "Acme/SafetensorsOnly"
    _fake_listing(repo, ["model-00001-of-00002.safetensors",
                         "model-00002-of-00002.safetensors", "README.md"])
    try:
        M.build_spec_from_repo(repo)
    except M.AppError as exc:
        assert "no *.gguf files" in (exc.message + " " + exc.detail), exc.report()
    else:
        raise AssertionError("safetensors-only repo must be refused")


def test_build_spec_classified_gated_repo_error(monkeypatch):
    def _deny(repo_id, force=False):
        raise M.AppError.classify(RuntimeError("403 Forbidden"),
                                  context=f"listing '{repo_id}'")
    monkeypatch.setattr(M, "_lookup_entry", _deny)
    try:
        M.build_spec_from_repo(GLM_REPO)
    except M.AppError as exc:
        assert "gated/private repo" in exc.message, exc.report()
        joined = "\n".join(exc.suggestions).lower()
        assert "hf auth login" in joined and "hf_token" in joined, \
            f"gated remediation hint missing: {exc.suggestions}"
    else:
        raise AssertionError("gated repo must surface a classified AppError")


# --------------------------------------------------------------------------
# 4. detect_family table incl. flash-next precedence + config model_type hint
# --------------------------------------------------------------------------
def test_detect_family_table():
    assert M.detect_family(GLM_REPO) == "glm5.3"
    assert M.detect_family(QWEN8_REPO) == "qwen3.8"          # plain 3.8
    assert M.detect_family("Acme/qwen3_5-mini-gguf") == "qwen3.5"
    assert M.detect_family("Org/Qwen-3.6-VL-Instruct-GGUF") == "qwen3.6"
    # flash/next variant must WIN over plain 3.8 (ordered heuristics):
    assert M.detect_family("Acme/Qwen3.8-VL-Flash-Next-GGUF") == "qwen3.8-flash-next"
    assert M.detect_family("Org/qwen3_8_next_air") == "qwen3.8-flash-next"
    # built-in torii detection is unchanged...
    assert M.detect_family("DraconicDragon/ToriiGate-0.5-GGUF") == "toriigate"
    # ...and uninformative names fall back to generic (or a config hint):
    assert M.detect_family("Acme/WeirdModel-GGUF") == "generic"
    hinted = M.detect_family("Acme/WeirdModel-GGUF", {"model_type": "glm5_3_air"})
    assert hinted == "glm5.3", f"config model_type hint ignored: {hinted}"


# --------------------------------------------------------------------------
# 5. family profiles incl. the plain qwen3.8 entry (acceptance criterion 3)
# --------------------------------------------------------------------------
def test_family_profile_entries():
    defaults = M._MODEL_FAMILY_DEFAULTS
    for fam in ("qwen3.5", "qwen3.6", "qwen3.8", "qwen3.8-flash-next"):
        prof = defaults.get(fam)
        assert prof is not None, f"family table missing {fam!r}"
        assert [l.key for l in prof.thinking_levels] == ["xhigh", "medium", "low", ""], fam
        assert prof.temperature == 0.6 and prof.supports_reasoning_kwargs is True, fam
    glm = defaults.get("glm5.3")
    assert glm is not None and not list(glm.thinking_levels) \
        and glm.supports_reasoning_kwargs is False and glm.temperature == 0.7
    gen = defaults.get("generic")
    assert gen is not None and gen.supports_reasoning_kwargs is False


def test_profile_lookup_via_live_ctx_registry(monkeypatch):
    """Regression: module-level key lookups must consult the LIVE AppContext
    registry (app/context.py wires _ACTIVE_CTX), else repos added at runtime
    through Tab 1 degrade to the neutral profile and leak Qwen kwargs."""
    live_ctx = SimpleNamespace(registry={
        "acme--glm-5.3-vision-gguf": _family_stub("glm5.3"),
        "stable--qwen3-8-captioner-gguf": _family_stub("qwen3.8"),
    })

    # Hermeticity: on machines where an earlier GUI session persisted this
    # custom repo (real app_settings.json), the import-time REGISTRY may
    # already contain it - drop it for the duration of the neutral claim.
    _clean_registry = {k: v for k, v in M.REGISTRY.items()
                       if k != "acme--glm-5.3-vision-gguf"}
    monkeypatch.setattr(M, "REGISTRY", _clean_registry)

    # Headless baseline: without a context, runtime-only keys resolve neutral.
    monkeypatch.setattr(M, "_ACTIVE_CTX", None)
    pre = M.get_model_profile("acme--glm-5.3-vision-gguf")
    assert pre.supports_reasoning_kwargs is True and not list(pre.thinking_levels)

    # ...with the live ctx wired (as AppContext.__init__ does): family profile.
    monkeypatch.setattr(M, "_ACTIVE_CTX", live_ctx)
    p_glm = M.model_profile_for("acme--glm-5.3-vision-gguf")
    assert p_glm.supports_reasoning_kwargs is False     # payload gate active
    assert not list(p_glm.thinking_levels)               # Tab 2 combo disabled
    assert p_glm.temperature == 0.7                     # GLM defaults, not neutral

    p_q8 = M.model_profile_for("stable--qwen3-8-captioner-gguf")
    assert [l.key for l in p_q8.thinking_levels] == ["xhigh", "medium", "low", ""], \
        "plain qwen3.8 custom repo must get family thinking levels"
    assert p_q8.temperature == 0.6

    # Built-ins and truly unknown keys are unaffected by the ctx wiring:
    assert M.get_model_profile("qwen3.8-27b").supports_reasoning_kwargs is True
    unk = M.model_profile_for("org--totally-unknown-gguf")   # in neither registry nor table
    assert not list(unk.thinking_levels)


# --------------------------------------------------------------------------
# 6. custom_repos persistence round-trip (incl. corrupt-file safety + merge)
# --------------------------------------------------------------------------
def test_custom_repos_roundtrip(tmp_path, monkeypatch):
    settings = tmp_path / "app_settings.json"
    monkeypatch.setattr(M, "SETTINGS_PATH", settings)

    payload = {"custom_repos": [{"repo": GLM_REPO, "label": "GLM custom"},
                                {"repo": ABLIT_REPO, "label": ""}]}
    M.save_app_settings(payload)
    assert M.load_app_settings().get("custom_repos") == payload["custom_repos"]

    settings.write_text("{ this is not json", encoding="utf-8")
    assert M.load_app_settings() == {}, "corrupt settings file must degrade to {}"


def test_build_registry_merges_persisted_customs(tmp_path, monkeypatch):
    """Offline-safe merge: no real network (lookup faked incl. force refresh);
    built-in source repos listed in custom_repos are deduped, not duplicated."""
    settings = tmp_path / "app_settings.json"
    monkeypatch.setattr(M, "SETTINGS_PATH", settings)
    _serve_lookup(monkeypatch, {
        "DraconicDragon/ToriiGate-0.5-GGUF": ["ToriiGate-0.5-Q8_0.gguf"],
        "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF":
            ["Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf"],
        GLM_REPO: ["GLM-5.3-Q8_0.gguf"],
    })

    M.save_app_settings({"custom_repos": [
        {"repo": GLM_REPO, "label": "My GLM"},
        # duplicates a BUILT-IN source repo: must not create a third entry
        {"repo": "DraconicDragon/ToriiGate-0.5-GGUF", "label": "dup?"},
    ]})

    reg = M.build_registry()
    expected_custom_key = GLM_REPO.lower().replace("/", "--")
    assert set(reg) == {"toriigate", "qwen3.8-27b", expected_custom_key}
    custom = reg[expected_custom_key]
    assert custom.source_repo == GLM_REPO and custom.display_name == "My GLM"


# --------------------------------------------------------------------------
# 7. unified config bundle v3: export shape + import behaviour (never aborts)
# --------------------------------------------------------------------------
class _StubCtx:
    """Just enough AppContext surface for export_full_config/import_full_config."""

    def __init__(self, monkeypatch):
        settings_file = Path("/nonexistent-dir/app_settings.json")   # empty state
        monkeypatch.setattr(M, "SETTINGS_PATH", settings_file)
        self.registry = M.build_registry(refresh=False)      # seeds only, no Hub

        from app.core.presets import DEFAULT_PRESETS as _DP, _clone as _cl
        self.presets = {k: _cl(v) for k, v in _DP.items()}
        self.active_preset_key = "moss"
        self.explicit_mode = False
        self.settings = InferenceSettings()
        self.selected_model_key = "toriigate"
        self.selected_quant = "Q8_0"
        self.selected_repo_id = "DraconicDragon/ToriiGate-0.5-GGUF"

        calls: list[tuple] = []
        monkeypatch.setattr(M, "set_storage",
                            lambda mode, root="": calls.append((mode, root)))
        self._storage_calls = calls
        self.storage = SimpleNamespace(to_dict=lambda: {"mode": M.STORAGE_DEFAULT})


def _presets_bundle(ctx) -> dict:
    return json.loads(export_config(ctx.presets, ctx.active_preset_key,
                                    ctx.explicit_mode))


def test_v3_export_shape(monkeypatch):
    ctx = _StubCtx(monkeypatch)
    monkeypatch.setattr(M, "_fetch_hf_config", lambda rid: None)
    _fake_listing(GLM_REPO, ["GLM-5.3-Q8_0.gguf"])
    spec = M.build_spec_from_repo(GLM_REPO)
    ctx.registry[spec.key] = spec

    data = M.export_full_config(ctx)
    assert data["version"] == M.CONFIG_VERSION == 3
    assert data["model"]["repo"] == "DraconicDragon/ToriiGate-0.5-GGUF"
    # custom repos round-trip as repo+label pairs; built-ins are never persisted:
    assert {"repo": GLM_REPO, "label": spec.display_name} in data["custom_repos"]
    assert all(e["repo"] != "DraconicDragon/ToriiGate-0.5-GGUF"
               for e in data["custom_repos"])


def test_v3_import_unknown_repo_creates_spec_or_warns(monkeypatch):
    ctx = _StubCtx(monkeypatch)

    # (a) reachable unknown repo: spec created AND selection follows it...
    monkeypatch.setattr(M, "_fetch_hf_config", lambda rid: None)
    new_key = QWEN8_REPO.lower().replace("/", "--")
    _fake_listing(QWEN8_REPO, ["Qwen3.8-Captioner-Q8_0.gguf"])
    bundle = {"version": 3, "presets": _presets_bundle(ctx),
              "settings": ctx.settings.to_dict(),
              "model": {"key": new_key, "quant": "Q8_0", "repo": QWEN8_REPO}}
    warns = M.import_full_config(ctx, json.dumps(bundle))
    assert new_key in ctx.registry, f"v3 import must (re)create the spec; {warns}"

    # (b) unreachable / no-GGUF repo: warned about, selection intact, no abort.
    bad = "Acme/Gone-Missing"
    _fake_listing(bad, ["model.safetensors"])            # exists but no GGUFs
    bundle["model"] = {"key": bad.lower().replace("/", "--"), "quant": "",
                       "repo": bad}
    warns2 = M.import_full_config(ctx, json.dumps(bundle))
    joined = "\n".join(warns2)
    assert ("could not be verified" in joined or "no longer exists" in joined), \
        f"expected a no-abort warning for the dead repo, got: {warns2}"
    assert ctx.selected_model_key == new_key             # unchanged by (b)


def test_v3_import_legacy_sections_still_work(monkeypatch):
    """A v2-style bundle (no model/custom_repos sections) imports unchanged."""
    ctx = _StubCtx(monkeypatch)
    raw = {"version": 2, "presets": _presets_bundle(ctx),
           "settings": dict(ctx.settings.to_dict(), temperature=0.42)}
    warns = M.import_full_config(ctx, json.dumps(raw))
    assert ctx.settings.temperature == 0.42
    assert all("legacy" not in w for w in warns)


# --------------------------------------------------------------------------
# 8. build_payload family gate (plan section 7 / edge case 9)
# --------------------------------------------------------------------------
def test_build_payload_family_gate(tmp_path, monkeypatch):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

    settings = InferenceSettings(enable_thinking=True, thinking_level="medium",
                                 reasoning_budget=4096)
    preset = DEFAULT_PRESETS["moss"]
    live_ctx = SimpleNamespace(registry={
        "acme--glm-5.3-vision-gguf": _family_stub("glm5.3"),
        "stable--qwen3-8-captioner-gguf": _family_stub("qwen3.8"),
    })
    monkeypatch.setattr(M, "_ACTIVE_CTX", live_ctx)

    p_glm = build_payload(str(img), preset, "SYS", settings=settings,
                          model_key="acme--glm-5.3-vision-gguf")
    assert "chat_template_kwargs" not in p_glm, \
        f"GLM payload must be sampling-only: {sorted(p_glm)}"
    assert "reasoning_budget" not in p_glm               # no Qwen extras at all
    # ...yet it is still a fully usable caption request:
    assert p_glm["messages"] and p_glm["stream"] is True

    p_q8 = build_payload(str(img), preset, "SYS", settings=settings,
                         model_key="stable--qwen3-8-captioner-gguf")
    ctk = p_q8.get("chat_template_kwargs") or {}
    assert ctk.get("reasoning_effort") == "medium" \
        and p_q8.get("reasoning_budget") == 4096, \
        f"qwen3.8 custom repo lost its family kwargs: {p_q8}"

    # Empty/legacy key keeps the historical full-Qwen contract (headless callers).
    p_legacy = build_payload(str(img), preset, "SYS", settings=settings)
    assert (p_legacy.get("chat_template_kwargs") or {}).get("reasoning_effort") == "medium"


if __name__ == "__main__":
    sys.exit("run with: python -m pytest tests/test_custom_repo.py")
