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


def _coerce_dict(value) -> dict:
    """Return ``value`` as a dict.

    Models sometimes emit a single object where the schema expects a list
    (e.g. ``"watermark": {...}``); iterating such a value would yield its
    string *keys*, so we wrap lone mappings into a one-element list instead.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return {}


# Keys that identify a dict as a concrete region/entity rather than a
# container whose values should be walked further (fuzzy-schema detection).
_REGION_KEYS = {"box_2d", "bbox", "bounding_box", "bounding_boxes",
                "coordinates", "coords", "points"}
_TEXT_KEYS = ("label", "text", "description", "name")

# Box keys may also appear as suffixes/qualifiers, e.g. ToriiGate emits
# ``face_bounding_box`` next to ``bounding_box``, and panel sub-keys such as
# ``panel_1_character`` carry the entity name after the last dot/underscore.
_BOX_KEY_WORDS = ("box", "bbox", "boxes", "coord", "coords", "coordinates",
                  "region", "rect")


def _looks_like_region(d: dict) -> bool:
    """True if ``d`` carries a bounding box or descriptive text payload."""
    if any(k in d for k in _REGION_KEYS):
        return True
    return any(isinstance(d.get(k), str) and d[k].strip() for k in _TEXT_KEYS)


def _bare_box(node) -> bool:
    """True if ``node`` is itself a raw 4-number box (e.g. ``[0.2, 0.7, ...]``)."""
    return (isinstance(node, list) and len(node) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in node))


def _xywh_box(d: dict) -> list | None:
    """Return a 4-value box assembled from corner keys, or None.

    llama.cpp's structured-output grammar forces models to emit boxes as
    objects like ``{"x_min": .., "y_min": .., "x_max": .., "y_max": ..}``
    (also ``xmin/xmax/ymin/ymax`` or ``left/top/right/bottom`` variants);
    these count as valid boxes just like the ``[.., .., .., ..]`` lists.
    """
    def num(*names):
        if not isinstance(d, dict):
            return None
        for n in names:
            v = d.get(n)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return None

    x1 = num("x_min", "xmin", "x1", "left")
    y1 = num("y_min", "ymin", "y1", "top")
    x2 = num("x_max", "xmax", "x2", "right")
    y2 = num("y_max", "ymax", "y2", "bottom")
    if None in (x1, y1, x2, y2):
        return None
    return [x1, y1, x2, y2]


def _is_box_key(key: str) -> bool:
    """True if a JSON key denotes a bounding box (incl. qualified names)."""
    k = str(key).lower()
    if k in _REGION_KEYS:
        return True
    words = re.split(r"[^a-z0-9]+", k)
    return any(w in _BOX_KEY_WORDS or w.endswith(("box", "bbox"))
               for w in words)


def _collect_boxes(node, path: list, out: list) -> None:
    """Recursively harvest every box under ``node`` into ``out``.

    Each hit is ``(suffix_id, box_list)`` where ``suffix_id`` keeps the
    qualifier of the box key (``face_bounding_box`` -> ``"face"``,
    ``panel_1.character`` -> ``"character"``) so nested boxes never collide.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            sub = path + [str(k)]
            box = None
            if _is_box_key(k):
                if _bare_box(v):
                    box = list(v)
                else:
                    box = _xywh_box(v)      # {x_min,y_min,x_max,y_max} object
            if box is not None:
                out.append((".".join(sub), box))
            else:
                _collect_boxes(v, sub, out)
    elif isinstance(node, (list, tuple)):
        if _bare_box(node):
            out.append((".".join(path), list(node)))
            return
        for i, v in enumerate(node):
            _collect_boxes(v, path + [str(i)], out)


def _box_suffix(key: str) -> str:
    """Qualifier of a box key: 'face_bounding_box' -> 'face', 'box_2d' -> ''.

    Strips generic box vocabulary (``bounding``, ``box``, ``bbox``, ``rect``,
    ``region``, coordinates, corners...) from the key's words and keeps the
    remainder; digits-only fragments (panel ids like ``2023``) survive.
    """
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", str(key)) if p]
    noise = set(_BOX_KEY_WORDS) | {"box_2d", "min", "max", "2d", "bboxes"}
    keep = []
    for p in parts:
        q = p.lower()
        if q in noise or q.rstrip("s") in noise or q == "bounding":
            continue
        # composite box words ('bbox', 'boundingbox') — but never a bare
        # numeric fragment like the '2' inside 'box_2d'.
        if len(p) >= 3 and not p.isdigit() and _is_box_key(q):
            continue
        keep.append(p)
    return "_".join(keep).lower()


def _clean_box(box) -> list | None:
    """Clamp a 4-value box to integers; floats in 0-1 are scaled to 0-1000."""
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if all(0.0 <= v <= 1.0 for v in vals):        # e.g. ToriiGate 0-1 coords
        vals = [round(v * 1000) for v in vals]
    return [max(0, min(1000, int(v))) for v in vals]


def _extract_text(d: dict) -> str:
    """Pull a human-readable text out of a region dict ('' when absent)."""
    for key in _TEXT_KEYS:
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _balanced_span(text: str):
    """Return the first balanced ``{...}`` span, or None.

    Unlike a greedy ``\\{.*\\}`` regex this cannot truncate the object when
    reasoning text or a second brace follows it on the wire.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _walk_regions(node, path: list, out: list) -> None:
    """Recursively harvest every region-like dict from a fuzzy structure.

    Handles all the shapes real models emit: flat lists of boxes, a single
    bare object, nested per-panel objects, null panels, mixed-depth trees,
    ``{description, bounding_box}`` entity records, and corner-key boxes
    (``{"x_min": .., "y_min": .., ...}``) forced by llama.cpp's JSON-grammar
    sampler.  Each hit becomes ``{id, box_2d?, boxes?, text?}`` with ``id``
    recording the JSON path (e.g. ``watermark.panel_6_signature``).  Bare
    strings in lists/dicts are captured as text-only regions so nothing is
    dropped.
    """
    if isinstance(node, dict):
        text = _extract_text(node)
        own_box = None                  # box carried by this dict itself
        for bk in ("box_2d", "bbox", "bounding_box", "coordinates", "coords"):
            v = node.get(bk)
            if _bare_box(v):
                own_box = list(v)
                break
            ob = _xywh_box(v)           # corner-key object forced by grammar
            if ob is not None:
                own_box = ob
                break
        if own_box is None and not text and any(_is_box_key(k) for k in node):
            # Bare corner-key box object ({x_min..} etc.) — the *dict itself*
            # is the region; never treat its coordinate fields as children.
            own_box = _xywh_box(node)
        if own_box is not None or text:
            # Concrete region record.  Harvest any *additional* boxes below it
            # (face_bounding_box siblings, nested panel sub-keys) so nothing
            # is silently dropped.
            found: list = []
            _collect_boxes(node, [], found)
            prefix = ".".join(path)
            extra = [(((prefix + "." + bp) if bp else prefix), b)
                     for bp, b in found
                     if list(b) != (own_box or [])]
            entry: dict = {"id": prefix or "root"}
            if own_box is not None and not extra:
                cb = _clean_box(own_box)
                if cb is not None:
                    entry["box_2d"] = cb
            elif own_box is not None or extra:
                # Multiple boxes (body + face): keep them all, keyed by the
                # qualifier of their JSON key.
                boxes: dict = {}
                if own_box is not None:
                    cb = _clean_box(own_box)
                    if cb is not None:
                        boxes[""] = cb
                for bp, b in extra:
                    suffix = _box_suffix(bp.rsplit(".", 1)[-1])
                    if not suffix and boxes:      # avoid a literal "" key
                        suffix = str(len(boxes) + 1)
                    cb = _clean_box(b)
                    if cb is not None:
                        boxes[suffix] = cb
                if len(boxes) == 1:
                    only = next(iter(boxes.values()))
                    boxes.clear()
                    entry["box_2d"] = only
                else:
                    entry["boxes"] = boxes
            if text:
                entry["text"] = text
            # Preserve any extra scalar fields the model invented.
            known = set(_REGION_KEYS) | set(_TEXT_KEYS) | {
                k for k in node if _is_box_key(k)}
            for k, v in node.items():
                if (k not in known and k not in entry
                        and isinstance(v, (str, int, float, bool))):
                    entry[k] = v
            out.append(entry)
            return                      # treated as leaf region
        for k, v in node.items():       # container: recurse into values
            if isinstance(v, str) and v.strip():
                # plain string value == text-only region (e.g. page_title)
                out.append({"id": ".".join(path + [str(k)]) or "root",
                            "text": v.strip()})
            elif _bare_box(v):
                # raw 4-number list == box-only region (ToriiGate panels)
                cb = _clean_box(v)
                if cb is not None:
                    out.append({"id": ".".join(path + [str(k)]) or "root",
                                "box_2d": cb})
            else:
                _walk_regions(v, path + [str(k)], out)
    elif isinstance(node, (list, tuple)):
        if _bare_box(node):             # the section itself is one bare box
            cb = _clean_box(node)
            if cb is not None:
                out.append({"id": ".".join(path) or "root", "box_2d": cb})
            return
        for i, v in enumerate(node):
            if isinstance(v, str) and v.strip():
                out.append({"id": ".".join(path + [str(i)]) or "root",
                            "text": v.strip()})
            elif _bare_box(v):
                cb = _clean_box(v)
                if cb is not None:
                    out.append({"id": ".".join(path + [str(i)]) or "root",
                                "box_2d": cb})
            else:
                _walk_regions(v, path + [str(i)], out)


def _collect_regions(value, section: str) -> list:
    """Normalise a whole schema section into a captured list of regions."""
    out: list = []
    _walk_regions(value, [section], out)
    return out


def _regions_to_watermarks(regions: list) -> list:
    """Render harvested watermark regions in the classic MOSS wire format.

    Entries keep ``box_2d``/``label`` when the model supplied them; box-less
    textual watermarks get their panel id as ``label`` so nothing is lost.
    Extra boxes (e.g. a secondary crop) ride along under ``boxes``.
    """
    wm: list = []
    for r in regions:
        last = str(r["id"]).rsplit(".", 1)[-1]
        entry: dict = {}
        if "box_2d" in r:
            entry["box_2d"] = r["box_2d"]
        entry["label"] = r.get("text") or last
        for k, v in r.items():
            if k not in ("id", "box_2d", "text"):
                entry[k] = v
        wm.append(entry)
    return wm


# Top-level keys handled explicitly by parse_moss_json; everything else the
# model volunteers (characters, background, texts, tags, ...) is preserved
# losslessly via _collect_regions().
_CORE_KEYS = {"caption", "watermark", "borders"}


def _harvest_scalar_section(value, section: str) -> list:
    """Capture a non-container top-level section without dropping data.

    Plain strings/numbers become ``[{id, text}]`` records and lists of
    scalars are walked element-wise — ``_walk_regions`` only understands
    containers, so e.g. ToriiGate's ``"overall_atmosphere": "..."`` used to
    vanish entirely.
    """
    if isinstance(value, str):
        v = value.strip()
        return [{"id": section, "text": v}] if v else []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [{"id": section, "text": str(value)}]
    out: list = []
    _walk_regions(value, [section], out)
    return out


def parse_moss_json(raw: str) -> dict:
    """Extract & validate the MOSS schema object from a model response.

    Tolerates markdown fences, leading reasoning text, and *fuzzy* schemas:
    models (e.g. ToriiGate-0.5) often deviate from the strict prompt spec by
    emitting ``watermark`` as a nested per-panel dict or adding a separate
    ``bounding_boxes`` section, and — when llama-server runs with a
    JSON-schema grammar — boxes appear as ``{"x_min":..,"y_min":..,
    "x_max":..,"y_max":..}`` corner objects next to ``face_bounding_box``,
    ``characters``/``background``/``texts``/``overall_atmosphere`` sections.
    Instead of dropping that data, every region-like object is harvested
    into normalised lists.  Raises ValueError when no valid JSON object with
    the expected keys can be recovered.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    candidates = [text]
    span = _balanced_span(text)
    if span is not None and span != text:
        candidates.append(span)
    candidates += _JSON_BLOCK.findall(text)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and "caption" in obj:
            cap = _coerce_dict(obj.get("caption"))
            norm = {
                "caption": {
                    "long": str(cap.get("long", "")).strip(),
                    "medium": str(cap.get("medium", "")).strip(),
                    "short": str(cap.get("short", "")).strip(),
                },
                "watermark": _regions_to_watermarks(
                    _collect_regions(obj.get("watermark"), "watermark")),
                "borders": obj.get("borders", ["none"]),
            }
            if not isinstance(norm["borders"], list):
                norm["borders"] = ["none"]
            # Every other top-level section the model volunteered beyond
            # the strict schema (bounding_boxes, characters, background,
            # texts, overall_atmosphere, extras, ...) is harvested into
            # its own list so no data is silently dropped.
            for key, value in obj.items():
                if key in _CORE_KEYS:
                    continue
                regions = _harvest_scalar_section(value, str(key))
                if regions:
                    norm.setdefault(key, [])
                    norm[key].extend(regions)
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


def _coerce_param(name: str, value):
    """Type-safe coercion of one imported generation parameter.

    Mirrors the field types of :class:`GenerationParams` and — unlike a bare
    ``int(value)``/``float(value)`` — refuses booleans for numeric fields:
    JSON files hand-edited from thinking flags can otherwise smuggle
    ``true``/``false`` into ``max_tokens`` / ``top_k`` (becoming 1/0), which
    silently mangles every generation setting on import.  Garbage input
    falls back to the dataclass default instead of corrupting the preset.
    """
    f = GenerationParams.__dataclass_fields__[name]
    target = f.type
    default = f.default
    if isinstance(value, bool):
        if target is bool:
            return value
        return default
    try:
        if target is int:
            return int(value)
        if target is float:
            return float(value)
        if target is bool:
            return bool(value)
    except (TypeError, ValueError):
        return default
    return value


def import_config(raw: str) -> tuple[dict[str, CaptionPreset], str, bool]:
    """Parse a preset config payload into fresh :class:`CaptionPreset` objects.

    Non-destructive: the parsed mapping is never mutated (the old in-place
    ``pop("params")`` corrupted any dict shared with the caller, so a second
    import of the same object silently lost *all* generation parameters).
    Unknown keys are ignored; known keys are coerced type-safely so numeric
    settings can never be clobbered by boolean JSON values.
    """
    data = json.loads(raw)
    param_names = set(asdict(GenerationParams()))
    preset_names = set(CaptionPreset.__dataclass_fields__)
    presets: dict[str, CaptionPreset] = {}
    for key, pd in (data.get("presets") or {}).items():
        if not isinstance(pd, dict):
            continue
        pd = dict(pd)                                # never mutate the input
        gp = pd.pop("params", None)
        gp = gp if isinstance(gp, dict) else {}
        pd["params"] = GenerationParams(**{
            k: _coerce_param(k, v) for k, v in gp.items() if k in param_names})
        presets[key] = CaptionPreset(**{k: v for k, v in pd.items()
                                       if k in preset_names})
    if not presets:
        presets = {k: _clone(v) for k, v in DEFAULT_PRESETS.items()}
    active = data.get("active_preset") or next(iter(presets))
    return presets, active, bool(data.get("explicit_mode", False))


def _clone(p: CaptionPreset) -> CaptionPreset:
    return CaptionPreset(**{**asdict(p), "params": GenerationParams(**asdict(p.params))})
