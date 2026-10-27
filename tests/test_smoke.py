"""Headless smoke tests for the captioner's non-GUI logic.

Run:  python3 tests/test_smoke.py            (plain, no pytest needed)
      xvfb-run -a python3 tests/test_gui.py  (GUI construction under Xvfb)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- hardware stub
class FakeHW:
    def __init__(self, gpus, ram_avail_mb=32768):
        self.gpus = gpus
        self.ram_available_bytes = ram_avail_mb * 1024 * 1024

    @property
    def selected_gpus(self):
        return [g for g in self.gpus if g.selected]


from app.core.hardware import GPUInfo                      # noqa: E402
from app.core.models import ArchInfo, ModelFile, build_registry, _arch_from_config  # noqa: E402
from app.core.resources import (plan_resources, shrink_context, MIN_CONTEXT,   # noqa: E402
                                MAX_CONTEXT_CAP)
from app.core.presets import (DEFAULT_PRESETS, parse_moss_json, clean_booru_tags,   # noqa: E402
                              save_caption, load_existing_caption, export_config,
                              import_config, EXPLICIT_TRIGGER, CaptionPreset,
                              GenerationParams, _clone)


def gpu(i, name, total_mb, used_mb=0, sel=True, w=50):
    return GPUInfo(index=i, name=name, total_vram_mb=total_mb,
                   used_vram_mb=used_mb, selected=sel, split_weight=w)


# ------------------------------------------------------------------ KV/context
def test_kv_context_sizing():
    arch = ArchInfo(hidden_size=4096, num_layers=32, num_kv_heads=8, head_dim=128)
    # f16 KV/token = 2*32*8*128*2 = 131072 B ; q8_0 ≈ 0.5625x -> 73728 B
    check("kv f16 bytes/token", arch.kv_bytes_per_token("f16") == 131072,
          str(arch.kv_bytes_per_token("f16")))
    check("kv q8_0 smaller than f16",
          arch.kv_bytes_per_token("q8_0") < arch.kv_bytes_per_token("f16"))

    mf = ModelFile("m.gguf", "Q8_0", "repo/x", 0.5)
    with tempfile.TemporaryDirectory() as td:
        real = Path(td) / "m.gguf"
        real.write_bytes(b"\0" * (1024 * 1024 * 600))     # 600 MiB model
        mf.local_path = lambda: real                       # type: ignore
        mm = Path(td) / "mm.gguf"
        mm.write_bytes(b"\0" * (1024 * 1024 * 100))        # 100 MiB mmproj

        hw = FakeHW([gpu(0, "RTX 4090", 24576)])
        plan = plan_resources(hw, mf, arch, "q8_0", mmproj_path=mm)
        # remaining = 24576-600-100-500 = 23376 MiB -> tokens floor/1024,
        # then clamped to the global MAX_CONTEXT_CAP ceiling.
        expect = min(int(23376 * 1024 * 1024 // arch.kv_bytes_per_token("q8_0"))
                     // 1024 * 1024, MAX_CONTEXT_CAP)
        check("dynamic -c is multiple of 1024", plan.context_tokens % 1024 == 0,
              str(plan.context_tokens))
        check("dynamic -c matches formula", plan.context_tokens == expect,
              f"{plan.context_tokens} vs {expect}")
        check("full offload when fits", plan.n_gpu_layers == 99 and plan.full_offload)

        # q4_0 KV should allow a longer context than q8_0 (or both sit at the
        # shared cap, which is fine - never shorter).
        plan_q4 = plan_resources(hw, mf, arch, "q4_0", mmproj_path=mm)
        check("KV quant extends context",
              plan_q4.context_tokens >= plan.context_tokens,
              f"q4_0={plan_q4.context_tokens} q8_0={plan.context_tokens}")

        # ---- spill to RAM: big Q8_0 on one 24GB GPU --------------------------
        big = ModelFile("big.gguf", "Q8_0", "repo/y", 27)
        bigpath = Path(td) / "big.gguf"
        # Sparse file: logical size 26 GiB without consuming real disk/RAM.
        with open(bigpath, "wb") as fh:
            fh.truncate(1024 * 1024 * 26000)
        big.local_path = lambda: bigpath                     # type: ignore
        arch40 = ArchInfo(hidden_size=5120, num_layers=40, num_kv_heads=8, head_dim=128)
        sp = plan_resources(FakeHW([gpu(0, "RTX 3090", 24576)]), big, arch40, "q8_0")
        check("spill triggers", sp.spilled_to_ram and 0 <= sp.n_gpu_layers < 40,
              f"ngl={sp.n_gpu_layers}")
        check("spill keeps min context", sp.context_tokens >= MIN_CONTEXT)

    # no GPUs selected -> CPU-only
    p = plan_resources(FakeHW([gpu(0, "x", 1000, sel=False)]), mf, arch, "q8_0")
    check("cpu-only plan", p.n_gpu_layers == 0 and p.context_tokens == MIN_CONTEXT)

    check("oom shrink step", shrink_context(16384) == 15360)
    check("oom shrink floor", shrink_context(MIN_CONTEXT) == MIN_CONTEXT)


def test_arch_from_config():
    cfg = {"model_type": "qwen3",
           "text_config": {"hidden_size": 5120, "num_hidden_layers": 64,
                           "num_attention_heads": 40, "num_key_value_heads": 8,
                           "head_dim": 128}}
    a = _arch_from_config(cfg)
    check("GQA heads parsed", a.num_kv_heads == 8 and a.num_layers == 64)
    check("kv per token math", a.kv_bytes_per_token("f16") == 2 * 64 * 8 * 128 * 2)


# ------------------------------------------------------------------ presets
def test_presets_and_postproc():
    moss = DEFAULT_PRESETS["moss"]
    booru = DEFAULT_PRESETS["booru"]
    check("MOSS temp/topp", moss.params.temperature == 1.0 and moss.params.top_p == 0.95)
    check("MOSS thinking mode", moss.params.enable_thinking is True)
    check("Booru temp 0.7 low budget",
          booru.params.temperature == 0.7 and booru.params.reasoning_budget == 0)
    check("MOSS schema in prompt", '"box_2d"' in moss.system_prompt
          and "normalized 0-1000" in moss.system_prompt)

    raw = ('Here you go:\n```json\n{"caption": {"long": "L", "medium": "M",'
           ' "short": "S"}, "watermark": [{"box_2d": [10, 20, 9999, 30],'
           ' "label": "(C) artist"}], "borders": ["none"]}\n```')
    obj = parse_moss_json(raw)
    check("moss json parse", obj["caption"]["long"] == "L")
    check("watermark box clamped 0-1000",
          obj["watermark"][0]["box_2d"] == [10, 20, 1000, 30])

    # Regression: models often emit a single watermark as a bare object
    # instead of a list -- must not crash with "'str' object has no
    # attribute 'get'" during batch save.
    bare = ('{"caption": {"long": "L", "medium": "M", "short": "S"},'
            ' "watermark": {"box_2d": [767, 695, 852, 765],'
            ' "label": "Signature and date"}, "borders": ["none"]}')
    try:
        obj2 = parse_moss_json(bare)
        check("bare-dict watermark coerced to list",
              obj2["watermark"] == [{"box_2d": [767, 695, 852, 765],
                                     "label": "Signature and date"}])
    except Exception as exc:      # noqa: BLE001
        check("bare-dict watermark coerced to list", False, str(exc))
    junk = ('{"caption": "stringified", "watermark": ["nope",'
            ' {"box_2d": [1, 2, 3]}], "borders": "none"}')
    try:
        obj3 = parse_moss_json(junk)
        # fuzzy parsing: the bare string is captured as text-only data and
        # the malformed 3-value box is skipped -- nothing crashes.
        check("junk watermark entries handled",
              obj3["watermark"] == [{"label": "nope"}]
              and obj3["borders"] == ["none"])
    except Exception as exc:      # noqa: BLE001
        check("junk watermark entries handled", False, str(exc))

    # Regression (ToriiGate-0.5): fuzzy schema -- watermark is a nested dict
    # of per-panel text entries AND the model adds an out-of-spec
    # bounding_boxes section with 0-1 coords and null panels.  None of that
    # data may be silently dropped.
    fuzzy = json.dumps({
        "caption": {"long": "L", "medium": "M", "short": "S"},
        "watermark": {
            "panel_6": {"text": "01.26"},
            "panel_6_signature": {"text": "2023"},
            "panel_6_additional_signature": {"text": "SARA"},
            "page_title": "Brooke draws a manga page every week challenge 2023 #1",
        },
        "bounding_boxes": {
            "panel_1": {"character": [0.2, 0.7, 0.3, 0.8],
                        "face": [0.25, 0.75, 0.28, 0.78]},
            "panel_2": None,
            "panel_4": {"character": [0.5, 0.75, 0.6, 0.8], "face": None},
            "panel_6": {"character": [0.55, 0.7, 0.85, 0.85],
                        "face": [0.75, 0.75, 0.78, 0.78]},
        },
        "borders": ["none"],
    })
    try:
        obj4 = parse_moss_json(fuzzy)
        wm_labels = [w["label"] for w in obj4["watermark"]]
        check("fuzzy nested watermark captured",
              wm_labels == ["01.26", "2023", "SARA",
                            "Brooke draws a manga page every week challenge 2023 #1"],
              str(wm_labels))
        bb_ids = [r["id"] for r in obj4.get("bounding_boxes", [])]
        check("fuzzy bounding_boxes captured",
              bb_ids == ["bounding_boxes.panel_1.character",
                         "bounding_boxes.panel_1.face",
                         "bounding_boxes.panel_4.character",
                         "bounding_boxes.panel_6.character",
                         "bounding_boxes.panel_6.face"],
              str(bb_ids))
        first = obj4["bounding_boxes"][0]
        check("0-1 coords scaled to 0-1000",
              first["box_2d"] == [200, 700, 300, 800], str(first))
        check("null panels skipped", all("box_2d" in r for r in obj4["bounding_boxes"]))
    except Exception as exc:      # noqa: BLE001
        check("fuzzy nested watermark captured", False, str(exc))

    tags = clean_booru_tags("1girl, smile, 1girl, looking at viewer\nbrown hair")
    check("booru dedupe+join", tags == "1girl, smile, looking at viewer, brown hair",
          tags)

    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "a.png")
        open(img, "wb").close()
        j = save_caption(img, raw, "json")
        check("json sidecar written", j.endswith(".json") and os.path.isfile(j))
        kind, text = load_existing_caption(img)
        check("json sidecar readback", kind == "json" and '"long"' in text)
        open(os.path.join(td, "b.png"), "wb").close()
        t2 = save_caption(os.path.join(td, "b.png"), "1girl, solo", "txt")
        check("txt sidecar written", t2.endswith(".txt") and "solo" in open(t2).read())

    # explicit-mode injection only for abliterated models
    p = _clone(moss)
    check("explicit injects", EXPLICIT_TRIGGER in
          p.effective_system_prompt(True, True))
    check("explicit skipped non-abliterated", EXPLICIT_TRIGGER not in
          p.effective_system_prompt(True, False))

    # export/import round-trip
    presets = {"moss": _clone(DEFAULT_PRESETS["moss"])}
    presets["moss"].user_prompt = "custom prompt!"
    raw_cfg = export_config(presets, "moss", True)
    got, active, expl = import_config(raw_cfg)
    check("config roundtrip", active == "moss" and expl is True
          and got["moss"].user_prompt == "custom prompt!")


# ------------------------------------------------------------------ server args
def test_server_args():
    from app.core.server import ServerManager, ServerConfig
    from app.core.resources import ResourcePlan

    class HWStub:
        def __init__(self):
            self.gpus = [gpu(0, "A", 24576, w=50), gpu(1, "B", 24576, w=50)]

        @property
        def selected_gpus(self):
            return [g for g in self.gpus if g.selected]

        def tensor_split_string(self):
            return ",".join(str(g.split_weight) if g.selected else 0
                            for g in self.gpus)

    sm = ServerManager.__new__(ServerManager)   # skip QObject init
    sm.hw = HWStub()
    # build_args logs capability warnings through the Qt signal; with the
    # QObject never properly initialised (and no event loop) emitting would
    # raise "Signal source has been deleted", so capture notes instead.
    sm.log = type("LogCapture", (), {"emit": staticmethod(lambda *a, **k: None)})()
    cfg = ServerConfig(
        model=ModelFile("q.gguf", "Q8_0", "r/s", 27),
        mmproj_path="/cache/mm.gguf",
        plan=ResourcePlan(context_tokens=16384, n_gpu_layers=99),
        split_mode="row", cache_type_k="q8_0", cache_type_v="q8_0",
        extra_args="--flash-attn", binary="/usr/bin/llama-server",
    )
    sm.find_binary = lambda: "/usr/bin/llama-server"
    args = sm.build_args(cfg)
    joined = " ".join(args)
    check("-m passed", "-m" in args and "q.gguf" in joined)
    check("--mmproj passed", "--mmproj" in args)
    check("-ngl 99", args[args.index("-ngl") + 1] == "99")
    check("split mode row", args[args.index("-sm") + 1] == "row")
    check("tensor split 50,50", args[args.index("-ts") + 1] == "50,50")
    check("-c from plan", args[args.index("-c") + 1] == "16384")
    check("kv cache types", "--cache-type-k" in args and "q8_0" in args)
    check("host/port", "--host" in args and "--port" in args)
    check("extra passthrough", "--flash-attn" in args)
    # Regression: the old code appended a bare '--api' which llama.cpp never
    # had ('error: invalid argument: --api' -> exit 1).  The flag must now be
    # the real --api-key (probed).
    check("no bogus --api flag", "--api" not in args)
    check("--api-key present", "--api-key" in args)
    # '-af' was deprecated and removed from llama.cpp - passing it made every
    # launch die with "invalid argument".  It must never appear again; flash
    # attention is a GUI switch mapped to the modern spelling instead.
    check("no removed -af flag", "-af" not in args)
    # Modern spelling: the probed test binary (nonexistent path -> assumed
    # up-to-date) gets '--flash-attn <value>' from the GUI switch, defaulting
    # to 'auto'.
    fa_i = args.index("--flash-attn")
    check("--flash-attn value passed",
          args[fa_i + 1] == "auto")
    cfg_fa = ServerConfig(model=cfg.model, plan=cfg.plan, flash_attn="on",
                          binary="/usr/bin/llama-server")
    args_fa = sm.build_args(cfg_fa)
    check("GUI switch 'on' honoured",
          args_fa[args_fa.index("--flash-attn") + 1] == "on")
    cfg_fa_off = ServerConfig(model=cfg.model, plan=cfg.plan, flash_attn="off",
                              binary="/usr/bin/llama-server")
    args_fa_off = sm.build_args(cfg_fa_off)
    check("GUI switch 'off' honoured",
          args_fa_off[args_fa_off.index("--flash-attn") + 1] == "off")

    # Older boolean-only build (-fa supported, no auto|on|off): the switch
    # must degrade to '-fa' when FA is wanted, and add nothing when off.
    # The nonexistent test path cannot be stat()'ed, so probe_capabilities()
    # never caches under a key - monkeypatch it for these two cases instead.
    import app.core.llamacpp as _lc
    legacy_caps = {"has_api_key": True, "has_reasoning_budget": True,
                   "flash_attn_auto": False, "flash_attn_on_off": True,
                   "cuda": False, "gpu": False}
    none_caps = {"has_api_key": True, "has_reasoning_budget": True,
                 "flash_attn_auto": False, "flash_attn_on_off": False,
                 "cuda": False, "gpu": False}
    orig_probe = _lc.probe_capabilities
    try:
        _lc.probe_capabilities = lambda *a, **k: dict(legacy_caps)
        cfg_fa_old = ServerConfig(model=cfg.model, plan=cfg.plan,
                                  flash_attn="on",
                                  binary="/usr/bin/llama-server")
        args_fa_old = sm.build_args(cfg_fa_old)
        check("legacy build uses -fa", "-fa" in args_fa_old
              and "--flash-attn" not in args_fa_old
              and "-af" not in args_fa_old)
        cfg_fa_old_off = ServerConfig(model=cfg.model, plan=cfg.plan,
                                      flash_attn="off",
                                      binary="/usr/bin/llama-server")
        args_fa_old_off = sm.build_args(cfg_fa_old_off)
        check("legacy build omits fa when off",
              "-fa" not in args_fa_old_off and "-af" not in args_fa_old_off
              and "--flash-attn" not in args_fa_old_off)
        # Ancient build without any flash-attn flag: launch must still work,
        # just without FA arguments (never fall back to the removed '-af').
        _lc.probe_capabilities = lambda *a, **k: dict(none_caps)
        args_none = sm.build_args(ServerConfig(
            model=cfg.model, plan=cfg.plan, flash_attn="auto",
            binary="/usr/bin/llama-server"))
        check("no-fa build skips fa entirely",
              "-af" not in args_none and "-fa" not in args_none
              and "--flash-attn" not in args_none)
    finally:
        _lc.probe_capabilities = orig_probe
    # Reasoning budget flows through only when the binary supports it.
    cfg2 = ServerConfig(model=cfg.model, plan=cfg.plan,
                        reasoning_budget=4096, binary="/usr/bin/llama-server")
    args2 = sm.build_args(cfg2)
    check("reasoning budget passed",
          "--reasoning-budget" in args2
          and args2[args2.index("--reasoning-budget") + 1] == "4096")


# ------------------------------------------------------------------ profiles
def test_model_profiles():
    from app.core.models import (KV_CACHE_TYPES, MODEL_PROFILES,
                                 get_model_profile)

    check("bf16 in KV types", "bf16" in KV_CACHE_TYPES)
    tg = get_model_profile("toriigate")
    qw = get_model_profile("qwen3.8-27b")
    check("toriigate Q8_0 context cap",
          tg.max_context_by_quant.get("Q8_0") == 32768)
    check("qwen3.8 IQ4_XS context cap",
          qw.max_context_by_quant.get("UD-IQ4_XS") == 131072)
    check("thinking levels present",
          len(tg.thinking_levels) >= 4 and len(qw.thinking_levels) >= 4)
    budgets = [l.budget for l in qw.thinking_levels]
    check("thinking budgets ascend", budgets == sorted(budgets))
    unknown = get_model_profile("does-not-exist")
    check("unknown model neutral profile",
          unknown.max_context_by_quant == {}
          and unknown.thinking_levels == ())
    check("profiles registered", set(MODEL_PROFILES) >= {"toriigate",
                                                         "qwen3.8-27b"})


# ------------------------------------------------------------------ payload
def test_payload():
    from app.core.api_client import build_payload
    with tempfile.TemporaryDirectory() as td:
        img = os.path.join(td, "x.png")
        Path(img).write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        pl = build_payload(img, DEFAULT_PRESETS["moss"], "SYS PROMPT")
        check("system msg", pl["messages"][0]["content"] == "SYS PROMPT")
        content = pl["messages"][1]["content"]
        kinds = {c["type"] for c in content}
        check("multimodal parts", kinds == {"text", "image_url"})
        url = next(c["image_url"]["url"] for c in content if c["type"] == "image_url")
        check("base64 data url", url.startswith("data:image/png;base64,"))
        check("thinking flag", pl.get("enable_thinking") is True)
        check("reasoning budget", pl.get("reasoning_budget_tokens") == 512)
        check("sampling params", pl["temperature"] == 1.0 and pl["top_p"] == 0.95
              and pl["top_k"] == 40 and "min_p" in pl)
        check("streaming", pl["stream"] is True)

        # Live Tab 2 settings override the preset values (thinking levels).
        from app.context import InferenceSettings
        s = InferenceSettings(temperature=0.6, top_p=0.9, top_k=20, min_p=0.0,
                              reasoning_budget=4096)
        pl2 = build_payload(img, DEFAULT_PRESETS["moss"], "SYS", settings=s)
        check("settings override sampling",
              pl2["temperature"] == 0.6 and pl2["top_p"] == 0.9
              and pl2["top_k"] == 20 and pl2["min_p"] == 0.0)
        check("settings drive thinking",
              pl2.get("enable_thinking") is True
              and pl2.get("reasoning_budget_tokens") == 4096)
        s_off = InferenceSettings(reasoning_budget=0)
        # Preset enable_thinking stays on unless overridden by a preset that
        # disables it; with budget 0 no reasoning tokens are injected.
        pl3 = build_payload(img, DEFAULT_PRESETS["moss"], "SYS", settings=s_off)
        check("zero budget -> no reasoning tokens",
              "reasoning_budget_tokens" not in pl3)


# ------------------------------------------------------------------ batch collect
def test_collect_images():
    from app.core.batch import collect_images
    with tempfile.TemporaryDirectory() as td:
        sub = os.path.join(td, "sub")
        os.makedirs(sub)
        for p in ("a.png", "b.txt", os.path.join("sub", "c.jpg")):
            Path(os.path.join(td, p)).write_bytes(b"x")
        rec = collect_images(td, True)
        nonrec = collect_images(td, False)
        check("recursive finds both", len(rec) == 2, str(rec))
        check("non-recursive finds one", len(nonrec) == 1, str(nonrec))


if __name__ == "__main__":
    test_kv_context_sizing()
    test_arch_from_config()
    test_presets_and_postproc()
    test_server_args()
    test_model_profiles()
    test_payload()
    test_collect_images()
    print("\n%d failure(s): %s" % (len(FAILURES), FAILURES or "none"))
    sys.exit(1 if FAILURES else 0)
