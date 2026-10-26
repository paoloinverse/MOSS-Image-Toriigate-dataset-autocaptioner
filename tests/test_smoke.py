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

    # robustness: model variants that must not crash the batch worker
    obj = parse_moss_json('{"caption": "plain string caption", "borders": ["none"]}')
    check("moss json caption-as-string normalized",
          obj["caption"]["long"] == "plain string caption"
          and obj["caption"]["short"] == "")
    obj = parse_moss_json('{"caption": {"long": "L"}, "watermark": ["junk", '
                          '{"box_2d": [1,2,3,4], "label": "x"}]}')
    check("moss json malformed watermark tolerated",
          len(obj["watermark"]) == 1 and obj["watermark"][0]["box_2d"] == [1, 2, 3, 4])

    tags = clean_booru_tags("1girl, smile, 1girl, looking at viewer\nbrown hair")
    check("booru dedupe+join", tags == "1girl, smile, looking at viewer, brown hair",
          tags)

    # normalize_moss_obj: never raises, keeps extra keys, str-caption safe
    from app.core.presets import normalize_moss_obj, load_caption_obj
    n = normalize_moss_obj({"caption": "just a string", "extra": 1})
    check("normalize str caption -> long",
          n["caption"]["long"] == "just a string" and n["extra"] == 1)
    n = normalize_moss_obj({"caption": {"long": " L ", "medium": None}})
    check("normalize strips/defaults levels",
          n["caption"]["long"] == "L" and n["caption"]["medium"] == "")
    n = normalize_moss_obj({"caption": ["weird"], "watermark": "junk",
                            "borders": "none"})
    check("normalize garbage shapes fall back",
          isinstance(n["caption"], dict) and n["watermark"] == []
          and n["borders"] == ["none"])
    check("normalize non-dict -> {}", normalize_moss_obj(["x"]) == {})

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

        # load_caption_obj must survive a hand-made string-caption .json -
        # this was the exact shape that crashed re-reads with
        # "'str' object has no attribute 'get'".
        cimg = os.path.join(td, "c.png")
        open(cimg, "wb").close()
        with open(os.path.join(td, "c.json"), "w") as fh:
            json.dump({"caption": "string caption", "watermark": [],
                       "borders": ["none"]}, fh)
        kind, obj = load_caption_obj(cimg)
        check("load_caption_obj normalizes string caption",
              kind == "json" and obj["caption"]["long"] == "string caption"
              and obj["caption"].get("short") == "")
        kind, obj = load_caption_obj(os.path.join(td, "b.png"))
        check("load_caption_obj txt fallback",
              kind == "txt" and obj["caption"]["long"] == "1girl, solo")
        kind, obj = load_caption_obj(os.path.join(td, "missing.png"))
        check("load_caption_obj missing -> empty", kind == "" and obj == {})

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
    captured: list[str] = []
    sm.log = type("LogCapture", (), {"emit": staticmethod(
        lambda msg, _c=captured: _c.append(str(msg)))})()
    # The split-buffer probe must never spawn a real llama-server in tests;
    # stub it at module level (build_args calls it via `llamacpp.` attribute
    # lookup, so monkeypatching the module is enough).  None == inconclusive,
    # i.e. exactly what an unknown binary yields - behaviour stays unchanged.
    from app.core import llamacpp as L
    orig_probe = L.probe_split_buffer_support
    L.probe_split_buffer_support = lambda *a, **k: None
    # The GPU-backend warning in build_args probes --help/--version of the
    # binary; with a fake stub here that would flood the log capture, so pin
    # "GPU-capable" for the whole test (the real detection is covered by the
    # preflight/installer suites).
    orig_caps = L.probe_capabilities
    L.probe_capabilities = lambda *a, **k: {"has_api_key": True,
                                            "has_reasoning_budget": True,
                                            "cuda": True, "gpu": True}
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
    # the real --api-key (probed), and '-af' must survive.
    check("no bogus --api flag", "--api" not in args)
    check("--api-key present", "--api-key" in args)
    check("-af flash attn", "-af" in args)
    # Reasoning budget flows through only when the binary supports it.
    cfg2 = ServerConfig(model=cfg.model, plan=cfg.plan,
                        reasoning_budget=4096, binary="/usr/bin/llama-server")
    args2 = sm.build_args(cfg2)
    check("reasoning budget passed",
          "--reasoning-budget" in args2
          and args2[args2.index("--reasoning-budget") + 1] == "4096")

    # ---- split-buffer probe integration (ServerManager._check_split_support)
    # Default preset must be the universally supported 'layer' mode: builds
    # or driver stacks without CUDA VMM die on '-sm row' with
    # "device CUDAxx does not support split buffers".
    from app.core.server import ServerConfig as SC
    check("default split mode is layer", SC().split_mode == "layer")
    try:
        from app.context import InferenceSettings
        check("GUI default split mode is layer",
              InferenceSettings().split_mode == "layer")
    except Exception as exc:                       # GUI import must not break
        check("GUI default split mode is layer", False, str(exc))

    with tempfile.TemporaryDirectory() as td:
        fake_bin = Path(td) / "llama-server"
        fake_bin.write_bytes(b"#!/bin/sh\nexit 0\n")
        fake_bin.chmod(0o755)
        calls: list[tuple] = []
        L.probe_split_buffer_support = \
            lambda b, *a, **k: (calls.append((str(b),)), False)[1]
        sm.find_binary = lambda: str(fake_bin)
        sm._SPLIT_CACHE = {}
        captured.clear()
        cfg_row = ServerConfig(model=cfg.model, plan=cfg.plan,
                               split_mode="row", binary=str(fake_bin))
        a = sm.build_args(cfg_row)
        check("probe fires for multi-GPU row mode",
              calls and calls[0][0] == str(fake_bin)
              and a[a.index("-sm") + 1] == "row")
        check("unsupported split warns with fix hint",
              any("split-buffer" in m and "layer" in m for m in captured),
              str(captured))
        # Second call for the same (path, mtime) must hit the cache.
        n_before = len(calls)
        sm.build_args(cfg_row)
        check("probe result cached per binary", len(calls) == n_before)
        # 'layer' never triggers the probe; single-GPU runs force -sm none.
        calls.clear(); captured.clear()
        sm.build_args(ServerConfig(model=cfg.model, plan=cfg.plan,
                                   split_mode="layer", binary=str(fake_bin)))
        check("layer mode skips probe", not calls and not captured)
        one = HWStub(); one.gpus = [gpu(0, "A", 24576, w=50)]
        sm.hw = one
        a = sm.build_args(cfg_row)
        check("single GPU forces -sm none, no probe",
              a[a.index("-sm") + 1] == "none" and not calls)
        sm.hw = HWStub()
        # Inconclusive verdict (None) must NOT warn (never block the user).
        L.probe_split_buffer_support = lambda b, *a, **k: None
        sm._SPLIT_CACHE = {}
        captured.clear()
        sm.build_args(cfg_row)
        check("inconclusive probe stays silent", not captured, str(captured))
        # Live-log hint: the reader thread must translate llama.cpp's own
        # complaint into exactly one actionable '[server] HINT' line.
        _lines = iter([
            "0.01 E llama_model_load: error loading model: device CUDA0 "
            "does not support split buffers\n",
            "0.01 E llama_model_load: error loading model: device CUDA0 "
            "does not support split buffers\n"])

        class _P:
            stdout = type("S", (), {"readline": staticmethod(
                lambda: next(_lines, ""))})()
        sm.proc = _P(); sm._stop_reader = threading.Event()
        sm._shutdown_requested = False
        sm._split_hint_shown = False
        captured.clear()
        sm._read_output()
        hints = [m for m in captured if "[server] HINT" in m]
        check("live log split hint fires once", len(hints) == 1,
              str(captured))
        check("hint carries the fix text", "Split mode" in (hints[0] if hints else ""))
    L.probe_split_buffer_support = orig_probe
    L.probe_capabilities = orig_caps


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


# ------------------------------------------------- streaming response handling
def test_stream_delta_parsing():
    """CaptionClient must survive every llama-server delta shape.

    Regression for "'str' object has no attribute 'get'": older code did
    (obj.get('choices') or [{}])[0] and crashed when the server emitted
    choices: [] / null, or a non-dict choice.  Also verifies thinking
    deltas (delta.reasoning_content) are captured separately from content.
    """
    import asyncio
    from app.core.api_client import CaptionClient

    class _Resp:
        status_code = 200

        async def aiter_lines(self):
            for s in self.lines:
                yield s

    class _Ctx:
        def __init__(self, lines): self.lines = lines
        async def __aenter__(self): return _Resp_ctx(self.lines)
        async def __aexit__(self, *a): return False

    class _Resp_ctx(_Resp):
        def __init__(self, lines): self.lines = lines

    class _Client:
        def __init__(self, lines): self.lines = lines
        def stream(self, *a, **k): return _Ctx(self.lines)

    def make_lines(events):
        out = []
        for e in events:
            if isinstance(e, str):
                out.append("data: " + e)
            else:
                out.append("data: " + json.dumps(e))
        out.append("data: [DONE]")
        return out

    img_dir = tempfile.mkdtemp()
    img = os.path.join(img_dir, "x.png")
    Path(img).write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

    # 1) empty choices list + null choices -> tolerated, content still parsed
    lines = make_lines([
        {"choices": []},
        {"choices": None},
        {"choices": ["not-a-dict"]},
        {"choices": [{"delta": {"content": "AB"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "text": "CD", "finish_reason": "stop"}]},
    ])
    c = CaptionClient("http://127.0.0.1:1")
    c._test_lines = lines
    orig_async = __import__("httpx").AsyncClient
    __import__("httpx").AsyncClient = lambda *a, **k: _Client(lines)
    try:
        got = asyncio.run(c.caption(img, DEFAULT_PRESETS["moss"], "SYS"))
    finally:
        __import__("httpx").AsyncClient = orig_async
    check("empty/null/non-dict choices tolerated", got == "ABCD", got)
    check("finish_reason captured", c.last_finish_reason == "stop")

    # 2) reasoning_content is NOT part of the caption but lands in last_reasoning
    lines2 = make_lines([
        {"choices": [{"delta": {"reasoning_content": "think1"}}]},
        {"choices": [{"delta": {"reasoning_content": "think2"}}]},
        {"choices": [{"delta": {"content": "{\"caption\":"}}]},
        {"choices": [{"delta": {"content": "\"ok\"}"}}]},
    ])
    c2 = CaptionClient("http://127.0.0.1:1")
    __import__("httpx").AsyncClient = lambda *a, **k: _Client(lines2)
    try:
        got2 = asyncio.run(c2.caption(img, DEFAULT_PRESETS["moss"], "SYS"))
    finally:
        __import__("httpx").AsyncClient = orig_async
    check("reasoning kept out of answer", got2 == '{"caption":"ok"}', got2)
    check("reasoning captured separately", c2.last_reasoning == "think1think2")

    # 3) verbose logging emits request payload + raw response to the log sink
    logged = []
    c3 = CaptionClient("http://127.0.0.1:1", log_cb=logged.append)
    __import__("httpx").AsyncClient = lambda *a, **k: _Client(lines2)
    try:
        from app.context import InferenceSettings
        asyncio.run(c3.caption(img, DEFAULT_PRESETS["moss"], "SYS PROMPT TEXT",
                               settings=InferenceSettings(verbose_logging=True)))
    finally:
        __import__("httpx").AsyncClient = orig_async
    joined = "\n".join(logged)
    check("verbose logs request", "[request]" in joined
          and "SYS PROMPT TEXT" in joined)
    check("verbose redacts image blob", "base64," not in joined.replace(
        "data:image/png;base64,", ""))
    check("verbose logs thinking + answer",
          "[response] thinking" in joined and "think1think2" in joined
          and "[response] answer" in joined)
    # and stays silent when disabled
    logged.clear()
    c4 = CaptionClient("http://127.0.0.1:1", log_cb=logged.append)
    __import__("httpx").AsyncClient = lambda *a, **k: _Client(lines2)
    try:
        asyncio.run(c4.caption(img, DEFAULT_PRESETS["moss"], "SYS",
                               settings=InferenceSettings(verbose_logging=False)))
    finally:
        __import__("httpx").AsyncClient = orig_async
    check("quiet mode logs nothing", logged == [], str(logged))


def test_batch_tab_cursor_streaming():
    """Regression: on_chunk used cursor.End (AttributeError spam in PySide6).

    Verified statically because the sandbox may lack Qt libs; additionally
    executed against real QTextCursor when PySide6 imports succeed.
    """
    import re as _re
    src = open(os.path.join(os.path.dirname(__file__), "..",
                            "app", "tabs", "batch_tab.py"), encoding="utf-8").read()
    check("on_chunk avoids instance cursor.End",
          "cursor.movePosition(cursor.End)" not in src)
    check("on_chunk uses QTextCursor.MoveOperation.End",
          "QTextCursor.MoveOperation.End" in src)
    check("batch_tab imports QTextCursor",
          _re.search(r"from PySide6\.QtGui import .*QTextCursor", src) is not None)
    try:
        from PySide6.QtGui import QTextCursor  # noqa: F401
        check("QTextCursor.MoveOperation.End exists at runtime",
              hasattr(QTextCursor.MoveOperation, "End"))
    except Exception:
        pass  # Qt not importable in this environment; static checks suffice


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
    test_stream_delta_parsing()
    test_batch_tab_cursor_streaming()
    test_collect_images()
    print("\n%d failure(s): %s" % (len(FAILURES), FAILURES or "none"))
    sys.exit(1 if FAILURES else 0)
