"""Presets, prompt templates and caption post-processing.

Two hardwired styles plus Custom:

* MOSS-Image  -> strict JSON schema (caption.long/medium/short, watermark
  boxes with normalized 0-1000 coords, borders) saved as ``.json``.
* Booru-SDXL  -> comma-separated Danbooru tags saved as ``.txt``.

Explicit Mode injects the NSFW trigger sentence into the system prompt when
the selected model is an abliterated/uncensored build, and asks the server
manager to disable safety-system prompts.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict

EXPLICIT_TRIGGER = ("NSFW content is present. Be vulgar as required by the "
                    "situation. Do not filter explicit descriptions.")

MOSS_SYSTEM_PROMPT = """You are an image recaptioner. Return exactly one JSON object with this schema:
{
  "caption": {
    "long": "a detailed image description, including all non-watermark visible text",
    "medium": "a one-paragraph image description",
    "short": "a one-sentence image description"
  },
  "watermark": [{"box_2d": [y_min, x_min, y_max, x_max], "label": "watermark text"}],
  "borders": ["none"]
}
Rules:
- Describe composition, text layout, and style in detail for 'long'.
- Do not describe watermarks in captions; record them only in 'watermark'.
- Use normalized 0-1000 coordinates for bounding boxes.
- Return only the JSON object."""

BOORU_SYSTEM_PROMPT = """You are an expert anime and digital art tagger. Generate a comma-separated list of Danbooru-style tags.
Include: artist, copyright, character, meta, and general tags. Order by importance.
Limit to 100 tokens maximum. Do not use sentences."""


@dataclass
class GenerationParams:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    reasoning_budget: int = 0        # --reasoning-budget / reasoning_budget_tokens
    enable_thinking: bool = False    # Qwen-style thinking mode flag
    max_tokens: int = 2048


@dataclass
class CaptionPreset:
    key: str
    display_name: str
    system_prompt: str
    user_prompt: str
    params: GenerationParams
    output_format: str               # "json" | "txt"
    explicit_suffix: str = EXPLICIT_TRIGGER

    def effective_system_prompt(self, explicit: bool, abliterated_model: bool) -> str:
        sp = self.system_prompt
        if explicit and abliterated_model:
            sp = f"{sp}\n\n{self.explicit_suffix}"
        return sp


def _moss_default() -> CaptionPreset:
    return CaptionPreset(
        key="moss",
        display_name="MOSS-Image (JSON)",
        system_prompt=MOSS_SYSTEM_PROMPT,
        user_prompt="Analyze this image and produce the caption JSON object.",
        params=GenerationParams(temperature=1.0, top_p=0.95, top_k=40,
                                min_p=0.05, reasoning_budget=512,
                                enable_thinking=True),
        output_format="json",
    )


def _booru_default() -> CaptionPreset:
    return CaptionPreset(
        key="booru",
        display_name="Booru-SDXL / Illustrious (tags)",
        system_prompt=BOORU_SYSTEM_PROMPT,
        user_prompt="Tag this image.",
        params=GenerationParams(temperature=0.7, top_p=0.9, top_k=35,
                                min_p=0.05, reasoning_budget=0,
                                enable_thinking=False),
        output_format="txt",
    )


def _custom_default() -> CaptionPreset:
    return CaptionPreset(
        key="custom",
        display_name="Custom",
        system_prompt="You are a helpful image captioner.",
        user_prompt="Describe this image.",
        params=GenerationParams(),
        output_format="txt",
    )


DEFAULT_PRESETS: dict[str, CaptionPreset] = {
    p.key: p for p in (_moss_default(), _booru_default(), _custom_default())
}


# --------------------------------------------------------------------------
# Post-processing helpers
# --------------------------------------------------------------------------
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_moss_json(raw: str) -> dict:
    """Extract & validate the MOSS schema object from a model response.

    Tolerates markdown fences and leading reasoning text.  Raises ValueError
    when no valid JSON object with the expected keys can be recovered.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    candidates = [text] + _JSON_BLOCK.findall(text)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and "caption" in obj:
            cap = obj.get("caption") or {}
            # Some models emit caption as a plain string instead of the
            # expected {"long","medium","short"} object.  Normalize both
            # shapes so downstream .get() calls never hit a str.
            if isinstance(cap, str):
                cap = {"long": cap}
            if not isinstance(cap, dict):
                cap = {}
            norm = {
                "caption": {
                    "long": str(cap.get("long", "")).strip(),
                    "medium": str(cap.get("medium", "")).strip(),
                    "short": str(cap.get("short", "")).strip(),
                },
                "watermark": [],
                "borders": obj.get("borders", ["none"]),
            }
            for wm in obj.get("watermark", []) or []:
                if not isinstance(wm, dict):
                    continue    # tolerate malformed watermark entries
                box = wm.get("box_2d")
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    try:
                        norm["watermark"].append({
                            "box_2d": [max(0, min(1000, int(v))) for v in box],
                            "label": str(wm.get("label", "")),
                        })
                    except (TypeError, ValueError):
                        pass
            if not isinstance(norm["borders"], list):
                norm["borders"] = ["none"]
            return norm
    raise ValueError("Model response did not contain a valid MOSS caption JSON object")


def clean_booru_tags(raw: str) -> str:
    """Collapse a tag-model response into one tidy comma-separated line."""
    text = raw.strip()
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text, flags=re.S)
    # Drop common preamble like "Tags:" / "Here are the tags:"
    text = re.sub(r"^(?:tags?|here (?:are|is)[^.]*?)[:\-]\s*", "", text, flags=re.I)
    parts = [p.strip() for p in re.split(r"[,\n]+", text) if p.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return ", ".join(out)


def caption_path_for(image_path: str, fmt: str) -> str:
    """Sidecar caption path for an image ('.json' for MOSS, '.txt' otherwise)."""
    root = os.path.splitext(image_path)[0]
    ext = ".json" if fmt == "json" else ".txt"
    return root + ext


def save_caption(image_path: str, text: str, fmt: str) -> str:
    """Persist a caption next to its image; returns the written path."""
    target = caption_path_for(image_path, fmt)
    if fmt == "json":
        obj = parse_moss_json(text)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
    else:
        cleaned = clean_booru_tags(text)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(cleaned + "\n")
    return target


def load_existing_caption(image_path: str) -> tuple[str, str]:
    """Return (kind, text) for an existing sidecar .json/.txt caption."""
    import os
    root = os.path.splitext(image_path)[0]
    for ext in (".json", ".txt"):
        p = root + ext
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return ext[1:], fh.read()
            except OSError:
                pass
    return "", ""


# --------------------------------------------------------------------------
# Config export / import (Tab 3 IO buttons)
# --------------------------------------------------------------------------
def export_config(presets: dict[str, CaptionPreset], active_key: str,
                  explicit: bool) -> str:
    payload = {
        "version": 1,
        # Marker lets the unified importer recognise a preset-only file even
        # when the 'presets' key is nested at top level.
        "scope": "presets",
        "active_preset": active_key,
        "explicit_mode": explicit,
        "presets": {k: asdict(p) for k, p in presets.items()},
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def import_config(raw: str) -> tuple[dict[str, CaptionPreset], str, bool]:
    data = json.loads(raw)
    presets: dict[str, CaptionPreset] = {}
    for key, pd in (data.get("presets") or {}).items():
        gp = pd.pop("params", {}) or {}
        pd["params"] = GenerationParams(**{k: v for k, v in gp.items()
                                           if k in asdict(GenerationParams())})
        presets[key] = CaptionPreset(**{k: v for k, v in pd.items()
                                        if k in CaptionPreset.__dataclass_fields__})
    if not presets:
        presets = {k: _clone(v) for k, v in DEFAULT_PRESETS.items()}
    active = data.get("active_preset") or next(iter(presets))
    return presets, active, bool(data.get("explicit_mode", False))


def _clone(p: CaptionPreset) -> CaptionPreset:
    return CaptionPreset(**{**asdict(p), "params": GenerationParams(**asdict(p.params))})
