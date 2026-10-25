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
from app.core.resources import plan_resources, shrink_context, MIN_CONTEXT          # noqa: E402
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
        # remaining = 24576-600-100-500 = 23376 MiB -> tokens floor/1024
        expect = int(23376 * 1024 * 1024 // arch.kv_bytes_per_token("q8_0")) // 1024 * 1024
        check("dynamic -c is multiple of 1024", plan.context_tokens % 1024 == 0,
              str(plan.context_tokens))
        check("dynamic -c matches formula", plan.context_tokens == expect,
              f"{plan.context_tokens} vs {expect}")
        check("full offload when fits", plan.n_gpu_layers == 99 and plan.full_offload)

        # q4_0 KV should allow a longer context than q8_0
        plan_q4 = plan_resources(hw, mf, arch, "q4_0", mmproj_path=mm)
        check("KV quant extends context", plan_q4.context_tokens > plan.context_tokens)

        # ---- spill to RAM: big Q8_0 on one 24GB GPU --------------------------
        big = ModelFile("big.gguf", "Q8_0", "repo/y", 27)
        bigpath = Path(td) / "big.gguf"
        bigpath.write_bytes(b"\0" * (1024 * 1024 * 26000))   # 26 GiB weights
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
    test_payload()
    test_collect_images()
    print("\n%d failure(s): %s" % (len(FAILURES), FAILURES or "none"))
    sys.exit(1 if FAILURES else 0)
