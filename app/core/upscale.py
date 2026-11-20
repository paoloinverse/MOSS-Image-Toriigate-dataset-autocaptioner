"""In-RAM image upscaling applied right before a request is sent to the model.

This module is the *only* image-preprocessing step in this project (by
design): when Tab 2's "Upscale Before Analysis" toggle is on, each image is
loaded, resized with Lanczos or bicubic by independent X/Y factors of 1/2/4,
re-encoded IN MEMORY in its original format, and embedded into the payload as
a data-URL instead of the raw file bytes.  Source files are never modified
and nothing is written to disk; the work happens once per request.

Fail-safe contract: :func:`upscale_to_data_url` never raises.  Any problem
(Pillow not installed, unknown method/scales, undecodable file) yields
``None`` so the caller falls back to sending the original file bytes -
captioning must never be blocked by preprocessing.
"""
from __future__ import annotations

import base64
import io
import time

try:                                   # Pillow is an optional runtime dep
    from PIL import Image              # noqa: N813
except ImportError:                    # pragma: no cover - environment
    Image = None                       # type: ignore[assignment]

#: Upscaler id -> (display label, PIL resampling-filter attribute name).
#: The filter is resolved lazily so this module imports without Pillow.
METHODS: dict[str, tuple[str, str]] = {
    "lanczos": ("Lanczos", "LANCZOS"),
    "bicubic": ("Bicubic", "BICUBIC"),
}

#: Scale factors offered per axis (independent X / Y).
SCALE_OPTIONS: tuple[int, ...] = (1, 2, 4)

# Formats Pillow can re-encode; anything else is written out as PNG.
_SAVEABLE = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}
_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp",
         "BMP": "image/bmp", "TIFF": "image/tiff"}
_JPEG_QUALITY = 95                     # for JPEG / WebP re-encodes


def upscale_image(image, method_id: str,
                  scale_x: int, scale_y: int):
    """Resize *image* by (scale_x, scale_y) with the named resampler.

    (1, 1) is an identity no-op that returns the input untouched (no copy);
    any other combination returns a NEW image of size
    ``(width*scale_x, height*scale_y)``.  Raises ``KeyError`` for unknown
    method ids (callers validate against :data:`METHODS`).
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed - upscaling unavailable")
    if scale_x == 1 and scale_y == 1:
        return image
    resampler = getattr(Image, METHODS[method_id][1])
    return image.resize((image.width * scale_x, image.height * scale_y),
                        resampler)


def upscale_to_data_url(path: str, method_id: str = "lanczos",
                        scale_x: int = 2, scale_y: int = 2):
    """Upscale the image at *path* in RAM; return ``(data_url, report)``.

    The data URL is re-encoded in the source format (PNG stays lossless,
    JPEG/WebP at quality 95; exotic formats fall back to PNG; alpha is
    dropped when re-encoding as JPEG).  *report* is a short human summary
    for logging, e.g. ``"1024x768 -> 2048x1536 (Lanczos 2x/2x, 84 ms)"``.

    Returns ``None`` - never raises - when Pillow is missing, the method or
    scales are invalid, or the file cannot be read / decoded / re-encoded.
    """
    if Image is None:
        return None
    try:
        sx, sy = int(scale_x), int(scale_y)
    except (TypeError, ValueError):
        return None
    if method_id not in METHODS or sx < 1 or sy < 1:
        return None
    t0 = time.monotonic()
    try:
        with Image.open(path) as im:
            im.load()                            # decode now (fail fast)
            fmt = (im.format or "").upper()
            ow, oh = im.size
            out = upscale_image(im, method_id, sx, sy)
        save_fmt = fmt if fmt in _SAVEABLE else "PNG"
        if save_fmt == "JPEG" and out.mode not in ("RGB", "L"):
            out = out.convert("RGB")             # JPEG has no alpha channel
        buf = io.BytesIO()
        params = ({"quality": _JPEG_QUALITY}
                  if save_fmt in ("JPEG", "WEBP") else {})
        out.save(buf, format=save_fmt, **params)
        ms = (time.monotonic() - t0) * 1000.0
        report = (f"{ow}x{oh} -> {out.width}x{out.height} "
                  f"({METHODS[method_id][0]} {sx}x/{sy}x, {ms:.0f} ms)")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:{_MIME[save_fmt]};base64,{b64}", report
    except Exception:                            # noqa: BLE001 - fail-safe
        return None