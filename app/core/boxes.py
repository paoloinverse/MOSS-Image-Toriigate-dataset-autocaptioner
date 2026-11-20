"""Bounding-box extraction + presentation rules for the Dataset Explorer overlay.

Qt-free on purpose: every function here is plain data in / data out, so the
colour rule, the coordinate-standard handling and the dedupe semantics are all
headless-testable (see ``tests/test_bbox_overlay.py``).  The only Qt-aware
consumer is :mod:`app.ui.image_viewer`, which turns :class:`AnnotatedBox` records
into painter primitives.

Three ideas carry the module:

* **The coordinate standard is the user's choice, not a guess.**  MOSS-Image and
  OCR Transcript both ask the model for ``[y_min, x_min, y_max, x_max]`` in
  native 0-1000 relative units, so that is the default; the Tab 4 layout/scale
  dropdowns (persisted on the preset) cover *foreign* JSON whose spec cannot be
  foreseen from the file alone.
* **Colour is a pure function of geometry.**  ``md5`` of the product of the box
  coordinates, taken from the first three (most significant) digest bytes, then
  lifted into a bright (``vivid``) HSV band that keeps the md5 hue.  Canonicalise
  to 0-1000 integers *before* hashing so the same box gets the same colour
  whether the file stored it as 0-1, 0-100, 0-1000 or absolute pixels.
* **Boxes are unique unless coordinates *and* text collide.**
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

# Reuse the app's one-and-only box-detection vocabulary instead of inventing a
# second JSON dialect here (these are the primitives presets.py uses to harvest
# regions out of fuzzy model output).
from app.core.presets import (_balanced_span, _bare_box, _extract_text,
                               _is_box_key, _load_robust, _xywh_box)

__all__ = ["LAYOUTS", "SCALES", "DEFAULT_LAYOUT", "DEFAULT_SCALE",
           "LABEL_MODES", "COLOUR_MODES", "DEFAULT_LABELS",
           "DEFAULT_COLOUR_MODE",
           "BOX_BEARING_PRESET_KEYS", "AnnotatedBox", "BoxExtraction",
           "BoxSource", "box_colour", "coordinate_product", "extract_boxes",
           "resolve_and_extract",
           "infer_scale", "is_cjk", "is_supported_layout", "is_supported_scale",
           "preset_supports_boxes", "resolve_box_source", "vivid",
           "wrap_tokens"]

# ---------------------------------------------------------------------------
# Coordinate standards (Tab 4 dropdowns)
# ---------------------------------------------------------------------------
#: id -> (toolbar label, converter from the 4 raw values to (xmin, ymin, xmax,
#: ymax) *in the source unit*, where it typically comes from).  The converter
#: only reorders/derives corners; scaling to pixels is the scale's job.
LAYOUTS: dict[str, tuple[str, Any, str]] = {
    "yxyx": ("ymin,xmin,ymax,xmax  (MOSS / OCR spec)",
             lambda v: (v[1], v[0], v[3], v[2]),
             "this app's prompts, Gemini-style grounding"),
    "xyxy": ("xmin,ymin,xmax,ymax",
             lambda v: (v[0], v[1], v[2], v[3]),
             "COCO/VOC ordering, CVAT, Qwen-VL, ChatGLM-V"),
    "xywh": ("x,y,w,h  (top-left + size)",
             lambda v: (v[0], v[1], v[0] + v[2], v[1] + v[3]),
             "COCO native, Detectron2, MMDetection, Azure"),
    "cxcywh": ("cx,cy,w,h  (centre + size)",
               lambda v: (v[0] - v[2] / 2.0, v[1] - v[3] / 2.0,
                          v[0] + v[2] / 2.0, v[1] + v[3] / 2.0),
               "YOLO / diffusers exports"),
    "xxyy": ("xmin,xmax,ymin,ymax",
             lambda v: (v[0], v[2], v[1], v[3]),
             "a few legacy/proprietary exports"),
}
DEFAULT_LAYOUT = "yxyx"

#: id -> (toolbar label, divisor mapping one unit to a full image, or None for
#: absolute pixels).
SCALES: dict[str, tuple[str, float | None]] = {
    "auto": ("Auto (detect from values)", None),
    "0-1000": ("0-1000 normalized  (native vision)", 1000.0),
    "0-1": ("0-1 relative (fraction)", 1.0),
    "0-100": ("0-100 normalized", 100.0),
    "px": ("absolute pixels", None),
}
DEFAULT_SCALE = "0-1000"

#: Tab 4 presentation choices.  Kept here so the preset validation, the viewer
#: and the toolbar all agree on the same small vocabulary.
LABEL_MODES = ("none", "number", "text")
COLOUR_MODES = ("vivid", "raw")
DEFAULT_LABELS = "none"
DEFAULT_COLOUR_MODE = "vivid"

#: Presets whose output can carry geometry.  ``custom`` is included because only
#: the user knows what a custom prompt asked the model to emit; Booru (tags) and
#: Markdown (prose) carry none.
BOX_BEARING_PRESET_KEYS = frozenset({"moss", "ocr_transcript", "custom"})


def is_supported_layout(value) -> bool:
    return isinstance(value, str) and value in LAYOUTS


def is_supported_scale(value) -> bool:
    return isinstance(value, str) and value in SCALES


def is_supported_label_mode(value) -> bool:
    return isinstance(value, str) and value in LABEL_MODES


def is_supported_colour_mode(value) -> bool:
    return isinstance(value, str) and value in COLOUR_MODES


def preset_supports_boxes(preset, allow_any: bool = False) -> bool:
    """True when the overlay may draw boxes for *preset* (plan §3).

    ``allow_any`` is the Tab 4 power-user override: draw whatever geometry the
    JSON happens to carry even for presets whose prompt emits tags or prose.
    """
    if allow_any:
        return True
    if preset is None:
        return False
    if getattr(preset, "key", "") in BOX_BEARING_PRESET_KEYS:
        return True
    # Structural fallback: any future JSON-emitting preset qualifies for free.
    return getattr(preset, "output_format", "") in ("json", "ocr_json")


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
def vivid(rgb: tuple[int, int, int], s: float = 0.85, v: float = 1.0
          ) -> tuple[int, int, int]:
    """Keep the hue of *rgb*, force a bright band (``S`` / ``V``).

    Raw md5 bytes leave ~6.5 % of boxes with a luma below 0.18 - invisible
    against the viewer's dark background and muddy on a scan.  Forcing ``V=1.0``
    with a strong saturation lifts every colour into a bright band (min luma
    ~0.27, no dark outliers) while leaving the md5-derived hue untouched, so box
    identity stays exactly as deterministic as the raw rule.
    """
    import colorsys
    h = colorsys.rgb_to_hls(*[c / 255.0 for c in rgb])[0]
    l = v * (1.0 - s / 2.0)
    return tuple(round(c * 255) for c in colorsys.hls_to_rgb(h, l, s))


def coordinate_product(norm_box: Sequence[int]) -> int:
    """Product of the canonical 0-1000 corner values (order-invariant)."""
    p = 1
    for v in norm_box:
        p *= int(v)
    return p


def box_colour(norm_box: Sequence[int], mode: str = "vivid"
               ) -> tuple[int, int, int]:
    """md5 of the coordinate product, first three (most significant) bytes.

    *norm_box* must already be canonical 0-1000 **integers**: hashing raw floats
    is unstable (``0.1*0.2*0.3*0.4`` -> ``0.0024000000000000007``) and the 0-1
    and 0-1000 spellings of the same box would otherwise disagree.
    """
    digest = hashlib.md5(str(coordinate_product(norm_box)).encode("ascii")).digest()
    rgb = (digest[0], digest[1], digest[2])
    return vivid(rgb) if mode != "raw" else rgb


# ---------------------------------------------------------------------------
# Text shaping helpers (tooltip / labels) - Qt-free so they are testable
# ---------------------------------------------------------------------------
#: CJK + friends: scripts that write without word separators, so a whole
#: sentence is a single "word" to ``str.split()`` and would be elided away.
_CJK_RANGES = ((0x1100, 0x115F), (0x2E80, 0x303E), (0x3041, 0x33FF),
               (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xA000, 0xA4CF),
               (0xAC00, 0xD7A3), (0xF900, 0xFAFF), (0xFF00, 0xFF60),
               (0xFFE0, 0xFFE6))


def is_cjk(ch: str) -> bool:
    """True for CJK ideographs / kana / hangul and their punctuation."""
    if not ch:
        return False
    cp = ord(ch[0])
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def wrap_tokens(text: str) -> list[str]:
    """Split *text* into units a line may be broken between.

    Latin/Cyrillic words stay whole (trailing space kept so re-joining is
    lossless), every CJK character becomes its own token.  Without this a
    Japanese/Chinese string is one unbreakable "word": greedy wrapping then
    elides it to ``東京タワーの下…`` instead of showing the whole line.
    """
    tokens: list[str] = []
    buf: list[str] = []
    for ch in (text or ""):
        if ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            if ch == " ":                       # spaces are break points
                if tokens:
                    tokens[-1] += " "
            else:                               # \n, \t: forced break
                tokens.append("\n")
            continue
        if is_cjk(ch):
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
            continue
        buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


# ---------------------------------------------------------------------------
# Scale / layout maths
# ---------------------------------------------------------------------------
def infer_scale(values: Sequence[float]) -> str:
    """Best-effort scale guess from the largest magnitude seen in the file."""
    vals = [abs(float(v)) for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        return DEFAULT_SCALE
    top = max(vals)
    if top <= 1.0:
        return "0-1"
    if top <= 100.0:
        return "0-100"
    if top <= 1000.0:
        return "0-1000"
    return "px"


def _resolve_scale(scale: str, corners: Sequence[float], img_w: int, img_h: int
                   ) -> tuple[str, float]:
    """Resolve ``auto`` -> (scale actually applied, divisor; 0.0 for pixels)."""
    if scale != "auto":
        div = SCALES[scale][1]
        return scale, (float(div) if div is not None else 0.0)
    guess = infer_scale(corners)
    if guess == "px":
        return "px", 0.0
    return guess, float(SCALES[guess][1] or 1.0)


def corners_to_pixels(corners: tuple[float, float, float, float], scale: str,
                      divisor: float, img_w: int, img_h: int
                      ) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) in the source unit -> (x, y, w, h) in pixels."""
    x1, y1, x2, y2 = corners
    if scale != "px" and divisor:
        x1, x2 = x1 / divisor * img_w, x2 / divisor * img_w
        y1, y2 = y1 / divisor * img_h, y2 / divisor * img_h
    return x1, y1, (x2 - x1), (y2 - y1)


def corners_to_norm1000(corners: tuple[float, float, float, float], scale: str,
                        divisor: float, img_w: int, img_h: int
                        ) -> tuple[int, int, int, int]:
    """Canonical 0-1000 integer corners - the colour/dedupe identity of a box.

    Normalising *before* hashing is what makes a colour independent of the scale
    the file happened to use.
    """
    x1, y1, x2, y2 = corners
    if scale == "px":
        w, h = max(1, img_w), max(1, img_h)
        x1, x2 = x1 / w * 1000.0, x2 / w * 1000.0
        y1, y2 = y1 / h * 1000.0, y2 / h * 1000.0
    elif divisor:
        x1, x2 = x1 / divisor * 1000.0, x2 / divisor * 1000.0
        y1, y2 = y1 / divisor * 1000.0, y2 / divisor * 1000.0
    return (max(0, min(1000, round(x1))), max(0, min(1000, round(y1))),
            max(0, min(1000, round(x2))), max(0, min(1000, round(y2))))


# ---------------------------------------------------------------------------
# JSON harvesting
# ---------------------------------------------------------------------------
def _is_boxy(node) -> bool:
    """True when *node* looks like a box in any of the recognised spellings."""
    return _raw_box(node) is not None


def _merge_pairs(pairs):
    """``object_pairs_hook`` that keeps **repeated** box keys instead of the last.

    The OCR Transcript schema shows a region as two consecutive ``"bbox"``
    keys, and a plain ``json.loads`` silently throws every value but the last
    away — so half the boxes vanish before extraction ever sees them.  Only
    duplicates that are (or were merged from) boxes are collected into a list;
    everything else keeps stock ``json.loads`` last-wins semantics, so a
    repeated ``"text"`` key still yields a plain string for ``_extract_text``.
    """
    out: dict = {}
    for key, value in pairs:
        if key not in out:
            out[key] = value
            continue
        prev = out[key]
        if not (_is_boxy(prev) or _is_boxy(value)):
            out[key] = value                    # stock last-wins
            continue
        bucket = prev if (isinstance(prev, list) and not _bare_box(prev)) \
            else [prev]
        bucket.append(value)
        out[key] = bucket
    return out


def _point_pair(node) -> tuple[float, float, float, float] | None:
    """``[[x1,y1],[x2,y2]]`` -> flat corner tuple (labelme / SAM style)."""
    if (isinstance(node, (list, tuple)) and len(node) == 2
            and all(isinstance(p, (list, tuple)) and len(p) == 2
                    and all(isinstance(n, (int, float))
                            and not isinstance(n, bool) for n in p)
                    for p in node)):
        (x1, y1), (x2, y2) = node
        return (float(x1), float(y1), float(x2), float(y2))
    return None


def _raw_box(node) -> tuple[tuple[float, float, float, float], str] | None:
    """Any self-describing box shape -> (corner values in X-Y order, shape)."""
    if _bare_box(node):
        return (float(node[0]), float(node[1]), float(node[2]), float(node[3])), "flat"
    pp = _point_pair(node)
    if pp is not None:
        return pp, "points"
    cb = _xywh_box(node) if isinstance(node, dict) else None
    if cb is not None:
        return (float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3])), "corners"
    return None


def _own_box(node: dict):
    """The box a region dict carries itself: ``(key, (corners, shape))`` or None."""
    for k, v in node.items():
        if not _is_box_key(k):
            continue
        got = _raw_box(v)
        if got is not None:
            return str(k), got
    # The dict *is* the box: a bare corner object {"x_min":..,"y_min":..,...}.
    if not _extract_text(node):
        got = _raw_box(node)
        if got is not None and got[1] == "corners":
            return "", got
    return None


def _collect(node, path: str, out: list) -> None:
    """Every box below *node* (siblings such as ``face_bounding_box`` included)."""
    if isinstance(node, dict):
        for k, v in node.items():
            sub = f"{path}.{k}" if path else str(k)
            got = _raw_box(v) if _is_box_key(k) else None
            if got is not None:
                out.append((sub, got[0], got[1]))
            elif isinstance(v, (dict, list, tuple)):
                _collect(v, sub, out)
    elif isinstance(node, (list, tuple)):
        got = _raw_box(node)
        if got is not None:
            out.append((path, got[0], got[1]))
            return
        for i, v in enumerate(node):
            _collect(v, f"{path}.{i}", out)


@dataclass
class _Hit:
    box_id: str
    corners: tuple[float, float, float, float]
    shape: str
    text: str


def _walk(node, path: str, out: list[_Hit]) -> None:
    """Recursively harvest region records, keeping coordinates **raw**.

    Mirrors :func:`app.core.presets._walk_regions` but deliberately does *not*
    clamp or rescale (that helper folds everything into 0-1000, which would
    destroy absolute-pixel boxes on a 4000 px scan): layout and scale are applied
    later, from the user's dropdown choice.
    """
    if isinstance(node, dict):
        text = _extract_text(node)
        own = _own_box(node)
        if own is not None or text:
            sub: list = []
            _collect(node, path, sub)
            if own is not None:
                _key, (coords, shape) = own
                out.append(_Hit(path, coords, shape, text))
                for sp, c, sh in sub:
                    if tuple(c) != tuple(coords):
                        # Sibling box (e.g. a face inside a character record):
                        # inherits the region text so its tooltip says something.
                        out.append(_Hit(sp, c, sh, text))
            else:
                for sp, c, sh in sub:
                    out.append(_Hit(sp, c, sh, text))
            return
        for k, v in node.items():
            _walk(v, f"{path}.{k}" if path else str(k), out)
        return
    if isinstance(node, (list, tuple)):
        got = _raw_box(node)
        if got is not None:
            out.append(_Hit(path, got[0], got[1], ""))
            return
        for i, v in enumerate(node):
            _walk(v, f"{path}.{i}", out)


# ---------------------------------------------------------------------------
# Public records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnnotatedBox:
    """One drawable box: pixel rect + its text + its deterministic colour."""
    box_id: str
    rect: tuple[float, float, float, float]     # (x, y, w, h) in image pixels
    text: str
    rgb: tuple[int, int, int]
    order: int = 0
    group: int = 0          # >1 == several texts share these coordinates
    group_size: int = 1
    norm: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class BoxExtraction:
    boxes: list[AnnotatedBox] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    layout_used: str = DEFAULT_LAYOUT
    scale_used: str = DEFAULT_SCALE
    merged: int = 0             # duplicates removed (coords + text identical)
    parsed: bool = False
    repaired: bool = False      # the JSON had to be balanced/repaired first
    found: int = 0              # boxes harvested before dedupe (provenance)
    #: boxes per top-level JSON section, e.g. ``{'watermark': 2, 'texts': 1}``
    counts: dict = field(default_factory=dict)

    def provenance(self) -> str:
        """One-line 'where did these boxes come from' summary for the log."""
        if not self.parsed:
            return "unparsed JSON"
        parts = [f"{k or '(root)'}: {n}" for k, n in
                 sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        tail = f", {self.merged} duplicate(s) merged" if self.merged else ""
        return (f"{self.found} box(es) found -> {len(self.boxes)} drawn "
                f"[{', '.join(parts) if parts else 'none'}]{tail}")


@dataclass
class BoxSource:
    kind: str                   # "session" (latest run) | "disk" (sidecar)
    path: str                   # "" for an in-session result
    text: str
    alternatives: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return os.path.basename(self.path) if self.path else "latest run (unsaved)"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_boxes(json_text: str, img_w: int, img_h: int,
                  layout: str = DEFAULT_LAYOUT, scale: str = DEFAULT_SCALE,
                  colour_mode: str = "vivid") -> BoxExtraction:
    """Parse *json_text* into pixel-space :class:`AnnotatedBox` records.

    *layout* / *scale* come from the Tab 4 dropdowns (persisted on the preset).
    Inverted corners are auto-swapped and out-of-bounds boxes clamped, both
    reported as warnings - but a *plausible yet wrong* relative spec stays
    undetectable here by design (plan §4.3).
    """
    res = BoxExtraction()
    if not is_supported_layout(layout):
        res.warnings.append(f"layout {layout!r} not recognised; using "
                            f"{DEFAULT_LAYOUT!r}")
        layout = DEFAULT_LAYOUT
    if not is_supported_scale(scale):
        res.warnings.append(f"scale {scale!r} not recognised; using "
                            f"{DEFAULT_SCALE!r}")
        scale = DEFAULT_SCALE
    res.layout_used, res.scale_used = layout, scale
    if not (json_text or "").strip():
        return res
    # object_pairs_hook (_merge_pairs) is applied on every parse attempt: the OCR
    # schema literally repeats "bbox" inside one object, and a stock json.loads
    # would silently drop all but the last value - even in repaired JSON.
    data, strict, note = _load_robust(json_text, object_pairs_hook=_merge_pairs)
    if data is None:
        res.warnings.append(f"JSON parse error: {note}")
        return res
    res.parsed = True
    if not strict:
        # The model's JSON needed help (wrapper stripped / brackets balanced).
        # Say so, so the status line and log explain why a 'broken' file still
        # produced boxes instead of failing with a silent parse error.
        res.repaired = True
        res.warnings.insert(0, f"JSON auto-recovered ({note})")

    hits: list[_Hit] = []
    _walk(data, "", hits)
    if not hits:
        return res

    res.found = len(hits)
    for h in hits:
        section = (h.box_id or "").split(".")[0]
        res.counts[section] = res.counts.get(section, 0) + 1

    flat = [c for h in hits for c in h.corners]
    scale_used, divisor = _resolve_scale(scale, flat, img_w, img_h)
    res.scale_used = scale_used
    to_xyxy = LAYOUTS[layout][1]

    staged: list[tuple[tuple[int, int, int, int], str, AnnotatedBox]] = []
    for order, h in enumerate(hits):
        # 'points' / 'corners' shapes are X-Y by definition; a 'flat' 4-list
        # obeys the dropdown - which is the whole point of having it.
        corners = to_xyxy(h.corners) if h.shape == "flat" else h.corners
        swapped = corners[0] > corners[2] or corners[1] > corners[3]
        if swapped:
            x1, y1, x2, y2 = corners
            corners = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        norm = corners_to_norm1000(corners, scale_used, divisor, img_w, img_h)
        x, y, w, hgt = corners_to_pixels(corners, scale_used, divisor,
                                         img_w, img_h)
        if swapped:
            res.warnings.append(
                f"inverted corners in '{h.box_id}' (auto-corrected)")
        if x < -1 or y < -1 or x + w > img_w + 1 or y + hgt > img_h + 1:
            res.warnings.append(
                f"'{h.box_id}' lies outside the {img_w}x{img_h} image (clamped) "
                "- check the layout/scale choice")
        x = max(0.0, min(float(img_w), x))
        y = max(0.0, min(float(img_h), y))
        w = max(0.0, min(float(img_w) - x, w))
        hgt = max(0.0, min(float(img_h) - y, hgt))
        if w <= 0 or hgt <= 0:
            res.warnings.append(f"'{h.box_id}' has zero area")
        staged.append((norm, h.text,
                       AnnotatedBox(box_id=h.box_id, rect=(x, y, w, hgt),
                                    text=h.text, rgb=box_colour(norm, colour_mode),
                                    order=order, norm=norm)))

    # ---- dedupe: unique unless coordinates *and* text collide --------------
    seen: set = set()
    unique: list[AnnotatedBox] = []
    for norm, text, box in staged:
        key = (norm, text.strip())
        if key in seen:
            res.merged += 1
            continue
        seen.add(key)
        unique.append(box)
    # Several *different* texts on one identical rect -> a group: one colour
    # (colour derives from coordinates only), dashed stroke, numbered tooltip.
    by_norm: dict[tuple, list[AnnotatedBox]] = {}
    for box in unique:
        by_norm.setdefault(box.norm, []).append(box)
    final: list[AnnotatedBox] = []
    for members in by_norm.values():
        for gi, m in enumerate(members):
            final.append(replace(m, group=gi + 1, group_size=len(members)))
    final.sort(key=lambda b: b.order)
    res.boxes = final
    if len(res.warnings) > 6:
        extra = len(res.warnings) - 6
        res.warnings = res.warnings[:6] + [f"… {extra} more box warning(s)"]
    return res


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------
def resolve_box_source(image_path: str, settings=None,
                       session_cache: dict | None = None,
                       cache_time: float | None = None) -> BoxSource | None:
    """Where the box JSON for *image_path* comes from (plan §2).

    ① the latest analysis run kept in memory (present even when saving is off or
    uses a different file naming), ② a JSON sidecar on disk - newest ``mtime``
    when several exist, so the user is never silently shown a stale generation.

    The in-session result normally wins, but only while it really is the newest
    word on the image: pass ``cache_time`` (the moment that run was remembered)
    and a sidecar written **after** it - an edit in the Caption Editor, a save
    from another tool, a second batch run - takes over again instead of
    shadowing the file forever.
    """
    from app.core.presets import caption_json_paths
    cands = caption_json_paths(image_path, settings)
    newest = max(cands, key=lambda p: os.path.getmtime(p)) if cands else ""

    text = (session_cache or {}).get(image_path) or ""
    superseded = False
    if text.strip():
        if newest and cache_time:
            try:
                superseded = os.path.getmtime(newest) > cache_time + 1e-6
            except OSError:
                superseded = False
        if not superseded:
            return BoxSource("session", "", text,
                             [os.path.basename(p) for p in cands if p != newest])

    if not cands:
        return None
    try:
        with open(newest, "r", encoding="utf-8") as fh:
            body = fh.read()
    except OSError as exc:
        return BoxSource("disk", newest, "",
                         [f"could not read {os.path.basename(newest)}: {exc}"])
    alts = [os.path.basename(p) for p in cands if p != newest]
    if superseded:
        alts.insert(0, "in-session result superseded by a newer sidecar")
    return BoxSource("disk", newest, body, alts)


# ---------------------------------------------------------------------------
# Resolve the *best* box source (latest-run cache vs on-disk sidecar)
# ---------------------------------------------------------------------------
def _available_sources(image_path: str, settings=None,
                       session_cache: dict | None = None) -> list[BoxSource]:
    """Every box source that actually exists (in-memory run and/or sidecar).

    Unlike :func:`resolve_box_source` this returns *both* kinds so a caller can
    compare them; the ordering is not authoritative (recency is decided by
    resolve_box_source) - it merely lists what is available.
    """
    from app.core.presets import caption_json_paths
    cands = caption_json_paths(image_path, settings)
    newest = max(cands, key=lambda p: os.path.getmtime(p)) if cands else ""
    out: list[BoxSource] = []
    text = (session_cache or {}).get(image_path) or ""
    if text.strip():
        out.append(BoxSource("session", "", text,
                             [os.path.basename(p) for p in cands if p != newest]))
    if newest:
        try:
            with open(newest, "r", encoding="utf-8") as fh:
                body = fh.read()
        except OSError as exc:
            out.append(BoxSource("disk", newest, "",
                                 [f"could not read {os.path.basename(newest)}: {exc}"]))
        else:
            out.append(BoxSource("disk", newest, body,
                                 [os.path.basename(p) for p in cands if p != newest]))
    return out


def resolve_and_extract(image_path: str, settings=None,
                        session_cache: dict | None = None,
                        cache_time: float | None = None,
                        img_w: int = 0, img_h: int = 0,
                        layout: str = DEFAULT_LAYOUT, scale: str = DEFAULT_SCALE,
                        colour_mode: str = "vivid") -> tuple[BoxSource | None, BoxExtraction]:
    """Resolve the box JSON and extract boxes, preferring whatever yields boxes.

    The age-preferred source (:func:`resolve_box_source`) is used whenever it
    parses into at least one box - preserving the existing precedence exactly
    (latest run wins while fresh; a newer sidecar takes over again).  Only when
    that source produces *no* boxes do we fall through to the other available
    source, so a freshly saved, clean sidecar still shows its boxes instead of a
    broken in-session dump hiding them (the "nothing until I re-save" case).
    Returns ``(source, extraction)``; ``(None, empty)`` when there is no JSON at all.
    """
    primary = resolve_box_source(image_path, settings, session_cache,
                                 cache_time=cache_time)
    if primary is None:
        return None, BoxExtraction()
    res = extract_boxes(primary.text, img_w, img_h, layout=layout, scale=scale,
                        colour_mode=colour_mode)
    if res.boxes:
        return primary, res
    for src in _available_sources(image_path, settings, session_cache):
        if src.kind == primary.kind or not (src.text or "").strip():
            continue
        cand = extract_boxes(src.text, img_w, img_h, layout=layout, scale=scale,
                             colour_mode=colour_mode)
        if cand.boxes:
            cand.warnings.insert(0, f"'{primary.name}' had no boxes; "
                                    f"drawing '{src.name}' instead")
            return src, cand
    return primary, res
