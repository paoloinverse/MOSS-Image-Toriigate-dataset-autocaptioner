"""Regression tests for the mmproj scoping fix (commit "Add explicit
run-model dropdown in Tab 2 and fix mmproj compatibility checks").

Field failure being guarded against: after switching the run model from
ToriiGate-0.5 to Qwen3.8-27B, llama-server was launched with

    -m .../Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf
    --mmproj .../DraconicDragon--ToriiGate-0.5-GGUF/ToriiGate-0.5-Q8_0.mmproj.gguf

and died with
    E mtmd_init_from_file: mismatch between text model (n_embd = 5120)
      and mmproj (n_embd = 2560)
    hint: you may be using wrong mmproj

Root cause: an *unscoped* recursive scan of the whole Unsloth tree handed
back any projector it found - including another repository's one.  The fix
scopes the scan to the file's own repo folder and adds a pre-launch
compatibility gate (``mmproj_companion_error``) inside ``ServerManager._preflight``.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import models as M
from app.core.server import ServerConfig, ServerManager


# --------------------------------------------------------------------------
# helpers: fake GGUF files with controllable architecture metadata
# --------------------------------------------------------------------------
def _write_gguf(path: Path, arch: str) -> Path:
    """Write a minimal-but-valid GGUF v3 header carrying general.architecture.

    Also pads the file to >1 MiB so ``ServerManager._preflight``'s truncation
    heuristic (size < 1 MiB) doesn't flag these synthetic fixtures.
    """
    def s(v: str) -> bytes:
        b = v.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    kv = s("general.architecture") + struct.pack("<I", 9) + s(arch)
    data = (b"GGUF" + struct.pack("<I", 3)
            + struct.pack("<Q", 0)          # n_tensors
            + struct.pack("<Q", 1)          # n_kv
            + kv)
    data += b"\x00" * (1024 * 1024 - len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


TORII_REPO = "DraconicDragon/ToriiGate-0.5-GGUF"
QWEN_REPO = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"


class _FakeSpec:
    """Enough ModelSpec surface for _all_specs()-driven name sniffing and
    the unified mmproj planner (spec_for_repo / plan_mmproj_download)."""
    def __init__(self, source_repo):
        self.source_repo = source_repo
        self.key = source_repo              # spec_for_repo() may probe .key
        self.mmproj_candidates: list[str] = []

    def file_for_quant(self, quant):        # used by resolve_mmproj_companion
        return None


def _install_storage(tmp_path: Path, monkeypatch):
    """Unsloth storage rooted at tmp_path with the field-reported layout:
    ToriiGate repo has its own projector; the Qwen repo has NO projector."""
    root = tmp_path / "big_models_cache_huggingface" / "unsloth"
    torii_dir = root / TORII_REPO.replace("/", "--")
    qwen_dir = root / QWEN_REPO.replace("/", "--")
    _write_gguf(torii_dir / "ToriiGate-0.5-Q8_0.gguf", "qwen2vl")
    _write_gguf(torii_dir / "ToriiGate-0.5-Q8_0.mmproj.gguf", "qwen2vl")
    _write_gguf(qwen_dir / "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf",
                "qwen3")
    monkeypatch.setattr(M, "STORAGE",
                        M.StorageManager(mode=M.STORAGE_UNSLOTH,
                                         unsloth_root=str(root)))
    monkeypatch.setattr(M, "_all_specs", lambda: [
        _FakeSpec(TORII_REPO), _FakeSpec(QWEN_REPO)])
    return root


def _torii_file():
    return M.ModelFile(filename="ToriiGate-0.5-Q8_0.gguf", quant="Q8_0",
                       repo_id=TORII_REPO, params_billions=0.5)


def _qwen_file():
    return M.ModelFile(
        filename="Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf",
        quant="UD-IQ4_XS", repo_id=QWEN_REPO, params_billions=27.0)


# --------------------------------------------------------------------------
# 1. scoping regression: no cross-repo projector pickup
# --------------------------------------------------------------------------
def test_qwen_without_own_mmproj_never_gets_toriigate_projector(tmp_path,
                                                                monkeypatch):
    """The exact field bug: Qwen (no local projector) must NOT resolve to
    ToriiGate's mmproj even though it sits in the same Unsloth tree."""
    _install_storage(tmp_path, monkeypatch)
    path, name = _qwen_file().resolve_mmproj_companion()
    assert path is None, f"cross-repo projector leaked: {path}"
    assert name is None or TORII_REPO.split("/")[-1].lower() not in name.lower()


def test_toriigate_still_finds_its_own_projector(tmp_path, monkeypatch):
    """Positive control: scoping must not break same-repo resolution."""
    root = _install_storage(tmp_path, monkeypatch)
    path, name = _torii_file().resolve_mmproj_companion()
    expected = (root / TORII_REPO.replace("/", "--")
                / "ToriiGate-0.5-Q8_0.mmproj.gguf")
    assert path == expected
    assert name == "ToriiGate-0.5-Q8_0.mmproj.gguf"


# --------------------------------------------------------------------------
# 2. mmproj_companion_error detection
# --------------------------------------------------------------------------
def test_companion_error_flags_cross_repo_name(tmp_path, monkeypatch):
    _install_storage(tmp_path, monkeypatch)
    root = tmp_path / "big_models_cache_huggingface" / "unsloth"
    bad = (root / TORII_REPO.replace("/", "--")
           / "ToriiGate-0.5-Q8_0.mmproj.gguf")
    err = _qwen_file().mmproj_companion_error(bad)
    # either detection path is acceptable - name sniffing or header arch
    # mismatch; both must fire for the field-reported pairing.
    assert err is not None and ("ToriiGate" in err or "architect" in err), err


def test_companion_error_name_sniffing_without_headers(tmp_path, monkeypatch):
    """Header unreadable -> cross-repo *name* sniffing alone must still flag."""
    _install_storage(tmp_path, monkeypatch)
    root = tmp_path / "big_models_cache_huggingface" / "unsloth"
    bogus = (root / TORII_REPO.replace("/", "--")
             / "ToriiGate-0.5-Q8_0.mmproj.gguf")
    bogus.write_bytes(b"not a gguf at all")
    qwen_main = (root / QWEN_REPO.replace("/", "--")
                 / "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf")
    qwen_main.write_bytes(b"also unreadable")   # force main_arch=None too
    err = _qwen_file().mmproj_companion_error(bogus)
    assert err is not None and "ToriiGate" in err, err


def test_companion_error_accepts_own_projector(tmp_path, monkeypatch):
    root = _install_storage(tmp_path, monkeypatch)
    good = (root / TORII_REPO.replace("/", "--")
            / "ToriiGate-0.5-Q8_0.mmproj.gguf")
    assert _torii_file().mmproj_companion_error(good) is None


# --------------------------------------------------------------------------
# 3. preflight gate blocks the bad launch end-to-end
# --------------------------------------------------------------------------
class _HW:
    selected_gpus = []

    def tensor_split_string(self):
        return ""


def _make_manager(monkeypatch):
    sm = ServerManager(hw=_HW())
    monkeypatch.setattr(sm, "find_binary",
                        lambda: "/usr/local/bin/llama-server")
    # port probe would bind a real socket; keep the test hermetic by using
    # an ephemeral high port that is essentially never occupied.
    return sm


def test_preflight_blocks_wrong_mmproj(tmp_path, monkeypatch):
    root = _install_storage(tmp_path, monkeypatch)
    qwen = _qwen_file()
    bad_mp = (root / TORII_REPO.replace("/", "--")
              / "ToriiGate-0.5-Q8_0.mmproj.gguf")
    sm = _make_manager(monkeypatch)
    cfg = ServerConfig(model=qwen, mmproj_path=str(bad_mp), port=48317)
    errors = sm._preflight(cfg)
    joined = "\n".join(errors).lower()
    assert any("incompatible multimodal projector" in e for e in errors), \
        f"wrong-mmproj gate did not fire: {errors}"
    assert "mmproj" in joined


def test_preflight_allows_matching_pair(tmp_path, monkeypatch):
    root = _install_storage(tmp_path, monkeypatch)
    torii = _torii_file()
    good_mp = (root / TORII_REPO.replace("/", "--")
               / "ToriiGate-0.5-Q8_0.mmproj.gguf")
    sm = _make_manager(monkeypatch)
    cfg = ServerConfig(model=torii, mmproj_path=str(good_mp), port=48318)
    errors = sm._preflight(cfg)
    assert errors == [], errors


# --------------------------------------------------------------------------
# 4. build_args uses the run-target paths (GUI smoke-level check)
# --------------------------------------------------------------------------
def test_build_args_pairs_qwen_main_without_foreign_mmproj(tmp_path,
                                                           monkeypatch):
    root = _install_storage(tmp_path, monkeypatch)
    qwen = _qwen_file()
    torii_mp = (root / TORII_REPO.replace("/", "--")
                / "ToriiGate-0.5-Q8_0.mmproj.gguf")

    class _HW:
        selected_gpus = []

        def tensor_split_string(self):
            return ""

    sm = ServerManager(hw=_HW())
    monkeypatch.setattr(sm, "find_binary",
                        lambda: "/usr/local/bin/llama-server")
    from app.core import llamacpp
    monkeypatch.setattr(llamacpp, "probe_capabilities",
                        lambda b: {"gpu": True})
    # correct wiring: Qwen main GGUF, no mmproj (Tab 2 resolves per run target)
    args = sm.build_args(ServerConfig(model=qwen, mmproj_path=None))
    assert args[args.index("-m") + 1] == str(
        qwen.local_path())
    assert "--mmproj" not in args
    # and had the GUI wired the stale ToriiGate projector, the gate above
    # (_preflight) would already have blocked it.
    cfg_bad = ServerConfig(model=qwen, mmproj_path=str(torii_mp), port=48319)
    errors = sm._preflight(cfg_bad)
    assert any("incompatible multimodal projector" in e for e in errors) \
        and any("ToriiGate" in e for e in errors), errors


# --------------------------------------------------------------------------
# 5. plan_mmproj_download - unified discovery pipeline (both tabs)
#    local cache -> live main-repo listing -> companion repo -> offline
# --------------------------------------------------------------------------
def test_plan_reuses_cached_local_projector(tmp_path, monkeypatch):
    """ToriiGate-style: cached same-repo mmproj -> no download needed."""
    _install_storage(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "resolve_mmproj_live",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not hit the network")))
    needed, msg, path = M.plan_mmproj_download(_torii_file())
    assert needed is False
    assert path is not None and path.name == "ToriiGate-0.5-Q8_0.mmproj.gguf"
    assert "re-using cached projector" in msg


def test_plan_discovers_qwen_style_name_from_live_listing(tmp_path, monkeypatch):
    """Qwen-style 'mmproj-*.gguf' names come verbatim from the live Hub
    listing - never composed locally - and are adopted onto the ModelFile."""
    _install_storage(tmp_path, monkeypatch)
    qwen = _qwen_file()
    seen = []

    def fake_live(repo, quant, force_refresh=False):
        seen.append(repo)
        return "mmproj-Huihui-Qwen3.8-27B-abliterated-F16.gguf"

    monkeypatch.setattr(M, "resolve_mmproj_live", fake_live)
    needed, msg, path = M.plan_mmproj_download(qwen)
    assert needed is True and path is None
    assert msg == "mmproj-Huihui-Qwen3.8-27B-abliterated-F16.gguf"
    assert qwen.mmproj_filename == msg          # adopted for verify/preflight
    assert seen == [QWEN_REPO]                  # main repo tried first


def test_plan_prefers_companion_repo_override(tmp_path, monkeypatch):
    """MMPROJ_REPO_OVERRIDE puts the companion repo FIRST in the lookup
    order; the main repo remains as fallback candidate."""
    _install_storage(tmp_path, monkeypatch)
    companion = "unsloth/Huihui-Qwen3.8-27B-Vision"
    # _FakeSpec.key == source_repo, so key the override by repo id
    monkeypatch.setattr(M, "MMPROJ_REPO_OVERRIDE", {QWEN_REPO: companion})
    seen = []

    def fake_live(repo, quant, force_refresh=False):
        seen.append(repo)
        return f"mmproj-{repo.split('/')[-1]}-F16.gguf"

    monkeypatch.setattr(M, "resolve_mmproj_live", fake_live)
    needed, msg, _p = M.plan_mmproj_download(_qwen_file())
    assert needed is True
    assert seen[0] == companion                 # companion queried first
    assert companion in msg                     # exact filename from it


def test_plan_offline_falls_back_to_remembered_name(tmp_path, monkeypatch):
    """Hub unreachable (resolve_mmproj_live -> None): a previously adopted
    name may still be used to queue the download."""
    _install_storage(tmp_path, monkeypatch)
    qwen = _qwen_file()
    qwen.mmproj_filename = "mmproj-Huihui-Qwen3.8-27B-abliterated-F16.gguf"
    monkeypatch.setattr(M, "resolve_mmproj_live",
                        lambda *a, **k: None)
    needed, msg, _p = M.plan_mmproj_download(qwen)
    assert needed is True
    assert msg == qwen.mmproj_filename


def test_plan_reports_unavailability_when_nothing_found(tmp_path, monkeypatch):
    """No local file, no live listing, no remembered name, no bundled
    candidates -> honest report instead of a composed guess."""
    _install_storage(tmp_path, monkeypatch)
    qwen = _qwen_file()
    monkeypatch.setattr(M, "resolve_mmproj_live",
                        lambda *a, **k: None)
    needed, msg, path = M.plan_mmproj_download(qwen)
    assert needed is False and path is None
    assert "unavailable" in msg and QWEN_REPO in msg


def test_ensure_mmproj_downloaded_success(tmp_path, monkeypatch):
    _install_storage(tmp_path, monkeypatch)
    qwen = _qwen_file()
    monkeypatch.setattr(M, "resolve_mmproj_live",
                        lambda *a, **k: "mmproj-Huihui-Qwen3.8-27B-abliterated-F16.gguf")
    got_dir = (tmp_path / "big_models_cache_huggingface" / "unsloth"
               / QWEN_REPO.replace("/", "--"))
    made = got_dir / "mmproj-Huihui-Qwen3.8-27B-abliterated-F16.gguf"

    def fake_dl(repo, files):
        assert repo == QWEN_REPO and files == \
            ["mmproj-Huihui-Qwen3.8-27B-abliterated-F16.gguf"]
        made.write_bytes(b"x" * 2048)
        return [made]

    monkeypatch.setattr(M, "download_model_files", fake_dl)
    logs = []
    ok, msg = M.ensure_mmproj_downloaded(qwen, log_cb=logs.append)
    assert ok is True and "projector ready" in msg
    assert any("[mmproj]" in l for l in logs)


def test_ensure_mmproj_downloaded_failure_report(tmp_path, monkeypatch):
    _install_storage(tmp_path, monkeypatch)
    qwen = _qwen_file()
    monkeypatch.setattr(M, "resolve_mmproj_live",
                        lambda *a, **k: "mmproj-Qwen-F16.gguf")

    def boom(repo, files):
        raise M.AppError("File or repository not found on the Hub",
                         detail="404 Client Error",
                         suggestions=["verify filenames in the registry"])

    monkeypatch.setattr(M, "download_model_files", boom)
    ok, msg = M.ensure_mmproj_downloaded(qwen)
    assert ok is False
    assert "not found" in msg.lower() and "404" in msg


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
