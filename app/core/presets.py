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

# Markdown mode: the model is asked to emit raw Markdown.  The *user* decides
# what belongs in the document by editing this prompt (it is a preset like the
# others, so custom instructions are saved into captioner_config.json and can
# be recycled).  Kept deliberately generic; sidecars save as '.md'.
MARKDOWN_SYSTEM_PROMPT = """You are an image captioner that responds ONLY in Markdown.
Output raw Markdown text: no code fences, no preamble, no explanation.
Use a short title plus structured sections or bullet lists as appropriate.
Follow the user's request for exactly what to include.  This default is
intentionally generic - replace it with your own Markdown instructions."""

OCR_TRANSCRIPT_SYSTEM_PROMPT = """You are an image recaptioner working for a Japanese ero-manga company: your job is to extract the Japanese transcript from the scan. Return exactly one JSON object with this schema:
{
  "information": {
    "page_reading_order": "RTL-TB",
    "text_writing": "vertical, top-to-down right-to-left",
    "number_of_bounding_boxes": "NNN",
    "image_filename": "image_filename.ext"
  },
  "bbox": [{"box_2d": [y_min0, x_min0, y_max0, x_max0], "label": "text_transcript", "chronological_number": "0"}],
  "bbox": [{"box_2d": [y_min1, x_min1, y_max1, x_max1], "label": "text_transcript", "chronological_number": "1"}]
}
Rules:
- Preserve the original Japanese text as it is: kana, kanji, without romanization and/or translation.
- Use the vision's native normalized 0-1000 coordinates for bounding boxes.
- Return only the JSON object."""


@dataclass
class GenerationParams:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    # llama.cpp sampler penalties (previously never sent -> server defaults).
    presence_penalty: float = 0.0     # -1..2, default 0; nudge away from seen tokens
    repetition_penalty: float = 1.1   # 0..2,  default 1.1; >1 discourages repeats
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
    output_format: str               # "json" | "txt" | "md" | "ocr_json"
    explicit_suffix: str = EXPLICIT_TRIGGER
    include_image_filename: bool = False  # inject image basename into system prompt
    # Bounding-box overlay coordinate standard for THIS preset (Tab 4 dropdowns).
    # MOSS-Image and OCR Transcript both ask for Y-X corners in native 0-1000
    # units; Custom is included because only the user knows what a custom prompt
    # emits.  They travel with the preset: ``export_config`` serialises presets via
    # ``asdict`` and ``import_config`` filters by these field names, so both keys
    # round-trip through Tab 3 export/import, the unified bundle and a startup
    # captioner_config.json - and a legacy file simply gets the defaults.
    overlay_box_layout: str = "yxyx"        # key into app.core.boxes.LAYOUTS
    overlay_coord_scale: str = "0-1000"     # key into app.core.boxes.SCALES
    # Presentation travels with the preset too, so a "this dataset is reviewed
    # with numbered chips, bright colours" preset exports and imports whole.
    overlay_labels: str = "none"            # none | number | text
    overlay_colour_mode: str = "vivid"      # vivid | raw
    overlay_any_preset: bool = False        # gate override, per preset

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


def _markdown_default() -> CaptionPreset:
    return CaptionPreset(
        key="markdown",
        display_name="Markdown",
        system_prompt=MARKDOWN_SYSTEM_PROMPT,
        user_prompt="Describe this image as Markdown.",
        params=GenerationParams(temperature=0.7, top_p=0.9, top_k=40,
                                min_p=0.05, reasoning_budget=0,
                                enable_thinking=False),
        output_format="md",
    )


def _ocr_transcript_default() -> CaptionPreset:
    return CaptionPreset(
        key="ocr_transcript",
        display_name="OCR Transcript (Japanese)",
        system_prompt=OCR_TRANSCRIPT_SYSTEM_PROMPT,
        user_prompt="Extract the Japanese transcript from this scan.",
        params=GenerationParams(temperature=0.7, top_p=0.9, top_k=40,
                                min_p=0.05, reasoning_budget=0,
                                enable_thinking=False, max_tokens=8192),
        output_format="ocr_json",
        include_image_filename=True,
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
    p.key: p for p in (_moss_default(), _booru_default(), _markdown_default(),
                       _ocr_transcript_default(), _custom_default())
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


# --------------------------------------------------------------------------
# Robust (fuzzy) JSON recovery
# --------------------------------------------------------------------------
#: The model is unreliable about bracket discipline - a dropped closing brace,
#: a trailing comma or an unterminated string all kill a stock json.loads while
#: the data is still perfectly recoverable.  These helpers make BOTH the save
#: path (parse_moss_json / parse_fuzzy_json) and the overlay path
#: (boxes.extract_boxes) as forgiving as each other - in fact they now share
#: exactly this recovery code, so a bracket-dropped response is saved repaired
#: instead of failing the image.  The object is located even when it is wrapped
#: in prose or code fences, and missing brackets are balanced so a truncated
#: response still parses.  This mirrors the strategy of a fuzzy-JSON parser
#: (e.g. github.com/livingbio/fuzzy-json) without adding a dependency to an
#: otherwise self-contained app.
def _strip_fences(text: str) -> str:
    """Drop markdown code-fence lines at the very start / end of *text*."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[A-Za-z0-9_]*[ \t]*\r?\n?", "", t)
        t = re.sub(r"\r?\n?[ \t]*```+[ \t]*$", "", t)
    return t.strip()


def _balance(text: str) -> tuple[str, int, int]:
    """Close unterminated strings / containers in *text* (best-effort).

    Walks the string honouring JSON string literals and escapes.  A closer that
    does not match the innermost open container is resolved by auto-closing
    whatever sits between it and the matching opener, so a dropped ``}`` before a
    ``]`` is reconstructed in the right order; a closer with no possible opener is
    dropped.  At the end an unterminated string is closed and every still-open
    container is closed (in reverse).  Trailing commas directly before a closer
    are removed, since json.loads rejects them.  Returns
    ``(repaired_text, n_brackets, n_commas)``.
    """
    out: list[str] = []
    stack: list[str] = []
    in_str = False
    esc = False
    fixes = 0
    for ch in text:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            want = "{" if ch == "}" else "["
            if not stack:
                continue                                # spurious closer
            if stack[-1] == want:
                stack.pop()
            else:
                for idx in range(len(stack) - 1, -1, -1):
                    if stack[idx] == want:
                        for j in range(len(stack) - 1, idx, -1):
                            out.append("}" if stack[j] == "{" else "]")
                            fixes += 1
                        del stack[idx:]                 # consume opener + above
                        break
                else:
                    continue                            # no opener -> drop
            out.append(ch)
        else:
            out.append(ch)                              # :, digits, commas, spaces...
    s = "".join(out)
    if in_str:
        s += '"'
        fixes += 1
    for opener in reversed(stack):                      # close what is still open
        s += "}" if opener == "{" else "]"
        fixes += 1
    commas = 0

    def _sub(m):
        nonlocal commas
        commas += 1
        return m.group(1)

    s = re.sub(r",(\s*[}\]])", _sub, s)                 # drop trailing commas
    return s, fixes, commas


def _robust_candidates(text: str) -> list[tuple[str, str]]:
    """(candidate_string, note) pairs for *text*, least-intervention first.

    Shared by the save path (parse_moss_json / parse_fuzzy_json) and the overlay
    path (boxes.extract_boxes) so both tolerate identical LLM noise - a markdown
    code fence, prose around an embedded object, dropped brackets or trailing
    commas.  Candidates are deduplicated; callers run ``json.loads`` on them in
    turn and apply their own validation / best-pick rule.  *note* says what was
    done to the string ('' for the untouched input).
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(s, note):
        if s and s not in seen:
            seen.add(s)
            out.append((s, note))

    t = (text or "").strip()
    if not t:
        return out
    add(t, "")                                    # raw, untouched
    sf = _strip_fences(t)
    if sf != t:
        add(sf, "stripped a markdown code fence")  # fence-stripped
    for probe in (sf, t):                          # balanced {...} span (prose-aware)
        span = _balanced_span(probe)
        if span and span != probe:
            add(span, "parsed the embedded object, ignoring surrounding text")
    base = sf
    starts = sorted({i for i in (base.find("{"), base.find("[")) if i >= 0})
    for i in starts:                               # bracket-balanced reconstructions
        fixed, n_br, n_co = _balance(base[i:])
        bits = []
        if n_br:
            bits.append(f"balanced {n_br} missing bracket(s)")
        if n_co:
            bits.append((f"removed {n_co} trailing comma(s)" if n_co > 1
                         else "removed a trailing comma"))
        add(fixed, "; ".join(bits) or "repaired the structure")
    return out


def _load_robust(text: str, object_pairs_hook=None):
    """Parse *text* into a dict/list, tolerating the noise LLMs actually add.

    Tries every candidate from :func:`_robust_candidates` (raw -> fence-stripped
    -> balanced span -> bracket-balanced reconstruction).  When several parse,
    the one that recovers the MOST content wins (so a truncated top-level array
    keeps every box instead of only its first balanced member).  Returns
    ``(obj, strict, note)``: *strict* is True only when the untouched string
    parsed as-is; on success *note* is a short phrase describing what was
    repaired (empty for the strict path); on failure *obj* is None and *note*
    carries the last parse error.
    """
    if not (text or "").strip():
        return None, False, "empty input"
    best = None                       # (obj, note, size)
    last_err = "unrecognised structure"
    for s, note in _robust_candidates(text):
        try:
            o = json.loads(s, object_pairs_hook=object_pairs_hook)
        except (ValueError, TypeError) as exc:
            last_err = str(exc)
            continue
        if isinstance(o, (dict, list)):
            size = len(json.dumps(o, ensure_ascii=False))
            if best is None or size > best[2]:
                best = (o, note, size)
    if best is None:
        return None, False, last_err
    obj, note, _size = best
    return obj, (note == ""), note


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
    # Fuzzy recovery (fences / prose / dropped brackets) - shared with the
    # overlay path, so a truncated response is saved repaired, not dropped.
    for cand, _note in _robust_candidates(raw):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
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


def companion_base(image_path: str, keep_image_ext: bool = False) -> str:
    """The ``<image_name>`` part of a companion caption path.

    Default (keep_image_ext=False) drops the image's own extension so
    ``cat.png`` -> ``cat``; with keep_image_ext=True it is preserved so
    ``cat.png`` -> ``cat.png`` (the caption then reads e.g. ``cat.png.md``).
    """
    return image_path if keep_image_ext else os.path.splitext(image_path)[0]


def companion_ext(fmt: str) -> str:
    """Default sidecar extension for a caption format.

    'json' -> .json (MOSS), 'ocr_json' -> .json (OCR Transcript),
    'md' -> .md (Markdown mode), anything else -> .txt (Booru / tag presets).
    """
    if fmt in ("json", "ocr_json"):
        return "json"
    if fmt == "md":
        return "md"
    return "txt"


def caption_path_for(image_path: str, fmt: str,
                     suffix: str = "", ext: str = "",
                     keep_image_ext: bool = False) -> str:
    """Sidecar caption path for an image.

    Named ``<image_name><suffix>.<ext>``.  By default the image's own extension
    is dropped (``cat.png`` -> ``cat.json``); *keep_image_ext* preserves it
    (``cat.png`` -> ``cat.png.json``).  When *ext* is empty the extension
    follows the preset format ('json' / 'md' / 'txt'); with *suffix*, *ext* and
    *keep_image_ext* all at their defaults this reproduces the original
    behaviour exactly.
    """
    root = companion_base(image_path, keep_image_ext)
    ext = (ext or "").strip().lstrip(".") or companion_ext(fmt)
    return f"{root}{suffix}.{ext}"


def save_caption(image_path: str, text: str, fmt: str,
                 suffix: str = "", ext: str = "",
                 keep_image_ext: bool = False) -> str:
    """Persist a caption next to its image; returns the written path.

    *fmt* still governs how the *content* is processed (parsed MOSS JSON vs
    fuzzy OCR JSON vs Markdown verbatim vs cleaned tags); *suffix* / *ext* /
    *keep_image_ext* only customise the resulting file name.
    """
    target = caption_path_for(image_path, fmt, suffix=suffix, ext=ext,
                              keep_image_ext=keep_image_ext)
    if fmt == "json":
        obj = parse_moss_json(text)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
    elif fmt == "ocr_json":
        # Fuzzy: find the first balanced JSON object and save it verbatim.
        obj = parse_fuzzy_json(text)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
    elif fmt == "md":
        # Markdown is written verbatim (no booru tag cleaning) - the model
        # already produced formatted text.
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(text.strip() + "\n")
    else:
        cleaned = clean_booru_tags(text)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(cleaned + "\n")
    return target


def load_existing_caption(image_path: str,
                          suffix: str = "", ext: str = "",
                          keep_image_ext: bool = False) -> tuple[str, str]:
    """Return (kind, text) for an existing sidecar caption.

    The companion name is ``<image_name><suffix>.<ext>``; *keep_image_ext*
    preserves the image's own extension in ``<image_name>`` (see
    :func:`companion_base`).  When *ext* is empty every sidecar type the app
    can write (json, txt, md) is tried in that order.  Returns ("", "") when
    no matching file exists or cannot be read.
    """
    import os
    base = companion_base(image_path, keep_image_ext) + (suffix or "")
    # With no explicit extension, search every sidecar type the app can write
    # (MOSS/OCR JSON, tag .txt, Markdown .md) in that order of preference.
    exts = [ext.strip().lstrip(".")] if ext else ["json", "txt", "md"]
    for e in exts:
        p = f"{base}.{e}"
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return e, fh.read()
            except OSError:
                pass
    return "", ""


# --------------------------------------------------------------------------
# Shared sidecar discovery (Caption Editor + Tab 4 bounding-box overlay)
# --------------------------------------------------------------------------
def _sidecar_groups(settings) -> list[tuple[str, str, bool]]:
    """``(suffix, ext, keep_image_ext)`` triples to try, load naming first."""
    if settings is None:
        return [("", "", False)]
    def grp(attr_s: str, attr_e: str) -> tuple[str, str, bool]:
        return (str(getattr(settings, attr_s, "") or ""),
                str(getattr(settings, attr_e, "") or ""),
                bool(getattr(settings, "caption_keep_image_ext", False)))
    out = [grp("caption_load_suffix", "caption_load_ext"),
           grp("caption_save_suffix", "caption_save_ext")]
    # Collapse identical groups (the historical default has both sides empty).
    dedup: list[tuple[str, str, bool]] = []
    for item in out:
        if item not in dedup:
            dedup.append(item)
    return dedup


def caption_candidate_paths(image_path: str, settings=None) -> list[str]:
    """Every sidecar path this app may read, in preference order.

    Load naming first, then save naming (a batch that saved as
    ``_transcript.json`` must be found even when the load naming is plain), and
    inside a group ``.json`` before ``.txt`` before ``.md`` - the order
    :func:`load_existing_caption` has always used.
    """
    out: list[str] = []
    for suffix, ext, keep in _sidecar_groups(settings):
        base = companion_base(image_path, keep) + suffix
        exts = [ext.strip().lstrip(".")] if ext else ["json", "txt", "md"]
        for e in exts:
            p = f"{base}.{e}"
            if p not in out:
                out.append(p)
    return out


def resolve_caption_file(image_path: str, settings=None) -> tuple[str, str]:
    """``(kind, text)`` of the first existing sidecar for *image_path*.

    One source of truth shared by the Caption Editor and the bounding-box
    overlay, so the panel and the drawing can never disagree about which file a
    caption came from.  Returns ("", "") when nothing readable exists.
    """
    for p in caption_candidate_paths(image_path, settings):
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return p.rsplit(".", 1)[-1].lower(), fh.read()
        except OSError:
            continue
    return "", ""


def caption_json_paths(image_path: str, settings=None) -> list[str]:
    """Existing ``.json`` sidecars only - the ones that can carry geometry."""
    return [p for p in caption_candidate_paths(image_path, settings)
            if p.lower().endswith(".json") and os.path.isfile(p)]


# --------------------------------------------------------------------------
# Fuzzy JSON extraction (OCR Transcript mode)
# --------------------------------------------------------------------------
def parse_fuzzy_json(raw: str) -> dict:
    """Extract the first balanced JSON object from a model response.

    Deliberately schema-agnostic: does NOT validate any keys or structure,
    so the system prompt can be freely reconfigured (different schema, extra
    fields, renamed sections) without touching this code.  The model is
    assumed to be faithful in returning exactly one JSON object.

    Handles: markdown fences, leading/trailing prose, thinking blocks before
    the actual answer.  Raises ValueError when no valid JSON object is found.
    """
    # Fuzzy recovery (fences / prose / dropped brackets) - shared with the
    # overlay path, so a truncated response is saved repaired, not dropped.
    for cand, _note in _robust_candidates(raw):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(
        "Model response did not contain a valid JSON object "
        "(OCR Transcript mode)")


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


#: Warnings collected by the *last* :func:`import_config` call (rejected preset
#: values etc.).  ``models.import_full_config`` drains and forwards them so the
#: user sees them at startup and after a Tab 3 import.
IMPORT_WARNINGS: list[str] = []


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
    IMPORT_WARNINGS.clear()
    for key, pd in (data.get("presets") or {}).items():
        if not isinstance(pd, dict):
            continue
        pd = dict(pd)                                # never mutate the input
        gp = pd.pop("params", None)
        gp = gp if isinstance(gp, dict) else {}
        pd["params"] = GenerationParams(**{
            k: _coerce_param(k, v) for k, v in gp.items() if k in param_names})
        preset = CaptionPreset(**{k: v for k, v in pd.items()
                                  if k in preset_names})
        # Preset-level *strings* never went through _coerce_param (only 'params'
        # did), so a hand-edited config could smuggle a nonsense coordinate
        # standard in.  Reject it to the default rather than drawing garbage.
        from app.core.boxes import (is_supported_colour_mode,
                                    is_supported_label_mode,
                                    is_supported_layout, is_supported_scale)
        if not is_supported_layout(preset.overlay_box_layout):
            IMPORT_WARNINGS.append(
                f"overlay_box_layout {preset.overlay_box_layout!r} not "
                "recognised; using 'yxyx' (ymin,xmin,ymax,xmax).")
            preset.overlay_box_layout = "yxyx"
        if not is_supported_scale(preset.overlay_coord_scale):
            IMPORT_WARNINGS.append(
                f"overlay_coord_scale {preset.overlay_coord_scale!r} not "
                "recognised; using '0-1000' (native normalized vision units).")
            preset.overlay_coord_scale = "0-1000"
        if not is_supported_label_mode(preset.overlay_labels):
            IMPORT_WARNINGS.append(
                f"overlay_labels {preset.overlay_labels!r} not recognised; "
                "using 'none'.")
            preset.overlay_labels = "none"
        if not is_supported_colour_mode(preset.overlay_colour_mode):
            IMPORT_WARNINGS.append(
                f"overlay_colour_mode {preset.overlay_colour_mode!r} not "
                "recognised; using 'vivid'.")
            preset.overlay_colour_mode = "vivid"
        if not isinstance(preset.overlay_any_preset, bool):
            IMPORT_WARNINGS.append(
                f"overlay_any_preset {preset.overlay_any_preset!r} is not a "
                "true/false value; coercing.")
            preset.overlay_any_preset = bool(preset.overlay_any_preset)
        presets[key] = preset
    # Merge in any built-in presets that are absent from the imported config.
    # This ensures new hardwired presets (e.g. OCR Transcript) always appear
    # in the dropdown even when loading a config exported before they existed.
    for key, default_preset in DEFAULT_PRESETS.items():
        if key not in presets:
            presets[key] = _clone(default_preset)
    if not presets:
        presets = {k: _clone(v) for k, v in DEFAULT_PRESETS.items()}
    active = data.get("active_preset") or next(iter(presets))
    return presets, active, bool(data.get("explicit_mode", False))


def _clone(p: CaptionPreset) -> CaptionPreset:
    return CaptionPreset(**{**asdict(p), "params": GenerationParams(**asdict(p.params))})
