"""Resource planning: dynamic KV-cache context sizing & VRAM/RAM spill math.

Core logic required by the spec:

    Available VRAM = (Total Selected VRAM) - (GGUF Model Size)
                     - (mmproj Size) - 500 MB (overhead)

The context size ``-c`` is then the *largest multiple of 1024* whose KV cache
(at the chosen quantisation, e.g. ``--cache-type-k q8_0``) fits into that
remaining VRAM.

If the model weights alone exceed VRAM, we compute how many transformer
layers can live on the GPU and push the remainder to system RAM via ``-ngl``
("Spilling to RAM", e.g. ``-ngl 35`` out of 40 layers).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.core.hardware import HardwareManager
from app.core.models import ArchInfo, ModelFile

KV_OVERHEAD_MB = 500          # fixed overhead reserved per spec
MIN_CONTEXT = 1024            # never plan below this
MAX_CONTEXT_CAP = 131072      # llama.cpp practical ceiling we will not exceed
MMPROJ_UNKNOWN_MB = 600.0     # conservative projector estimate when unknown

# ---------------------------------------------------------------------------
# Structured, verbose planning errors
# ---------------------------------------------------------------------------
class ResourcePlanError(Exception):
    """Raised when a resource plan cannot be built for the current setup.

    Carries a human-readable ``detail`` string (safe to show verbatim in the
    GUI) and a list of actionable ``suggestions`` so error reporting can tell
    the user exactly what to change instead of just crashing.
    """

    def __init__(self, message: str, detail: str = "", suggestions=None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.suggestions: list[str] = list(suggestions or [])

    def report(self) -> str:
        """Multi-line, log/GUI-friendly rendering of the failure."""
        lines = [f"[resources] ERROR: {self.message}"]
        if self.detail:
            lines.append(f"    reason : {self.detail}")
        for s in self.suggestions:
            lines.append(f"    fix    : {s}")
        return "\n".join(lines)

    @classmethod
    def from_exception(cls, exc: Exception, context: str = "") -> "ResourcePlanError":
        """Wrap any unexpected exception with location + remediation hints."""
        import traceback
        tb = traceback.format_exc(limit=4)
        detail = f"{type(exc).__name__}: {exc}"
        if context:
            detail = f"{context} -> {detail}"
        return cls(
            "Resource planning failed unexpectedly",
            detail=f"{detail}\n{tb}",
            suggestions=[
                "Check that the selected GGUF file is intact (re-run Download & Verify).",
                "Try a smaller quantization or select more GPUs in Tab 1.",
                "Report this full message if it persists after changing settings.",
            ],
        )


@dataclass
class ResourcePlan:
    context_tokens: int = MIN_CONTEXT
    n_gpu_layers: int = -1          # -ngl; -1 => CPU only
    full_offload: bool = False      # True when everything fits (-ngl 99)
    spilled_to_ram: bool = False    # some layers must run on CPU
    kv_bytes_per_token: int = 0
    available_vram_mb: int = 0
    model_mb: float = 0.0
    mmproj_mb: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"-c {self.context_tokens}",
            f"-ngl {self.n_gpu_layers if self.n_gpu_layers >= 0 else 0}",
            f"KV/token={self.kv_bytes_per_token/1024:.1f}KiB",
            f"freeVRAM={self.available_vram_mb}MiB",
            f"model={self.model_mb:.0f}MiB mmproj={self.mmproj_mb:.0f}MiB",
        ]
        if self.spilled_to_ram:
            parts.append("(spilling to RAM)")
        return " | ".join(parts)


def _file_size_mb(path) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def plan_resources(hw: HardwareManager, model_file: ModelFile,
                   arch: ArchInfo, cache_type: str = "q8_0",
                   mmproj_path=None) -> ResourcePlan:
    """Compute ``-c`` and ``-ngl`` for the current hardware/model selection.

    Raises :class:`ResourcePlanError` (with verbose detail + fix suggestions)
    when the inputs are invalid so callers can report actionable messages
    instead of crashing or silently producing nonsense plans.
    """
    try:
        return _plan_resources_inner(hw, model_file, arch, cache_type, mmproj_path)
    except ResourcePlanError:
        raise
    except Exception as exc:
        raise ResourcePlanError.from_exception(
            exc, context=f"plan_resources(model={getattr(model_file, 'filename', model_file)!r}, "
                         f"cache_type={cache_type!r})") from exc


def _plan_resources_inner(hw: HardwareManager, model_file: ModelFile,
                          arch: ArchInfo, cache_type: str,
                          mmproj_path) -> ResourcePlan:
    # ---- input validation (verbose, actionable) ---------------------------
    if model_file is None:
        raise ResourcePlanError(
            "No model file selected",
            detail="The quantization dropdown has no active GGUF entry.",
            suggestions=["Pick a model and quantization in Tab 1 first."])
    if arch is None:
        raise ResourcePlanError(
            "Model architecture is unknown",
            detail="config.json / GGUF metadata could not be parsed for KV sizing.",
            suggestions=["Re-download the GGUF, or check the HF repo id in the registry."])
    kv_bpt = arch.kv_bytes_per_token(cache_type)
    if kv_bpt <= 0:
        raise ResourcePlanError(
            f"Invalid KV-cache size computed for type '{cache_type}'",
            detail=f"arch(hidden={arch.hidden_size}, layers={arch.num_layers}, "
                   f"kv_heads={arch.num_kv_heads}, head_dim={arch.head_dim}) "
                   f"-> {kv_bpt} bytes/token",
            suggestions=["Select a supported cache type (f16/q8_0/q4_0/q4_1/fp8)."])

    plan = ResourcePlan()

    selected = hw.selected_gpus
    if not selected:
        # No GPU pool -> pure CPU inference with a modest context.
        plan.n_gpu_layers = 0
        plan.context_tokens = MIN_CONTEXT
        plan.notes.append("No GPUs selected: running CPU-only (-ngl 0).")
        return plan

    total_selected_vram = sum(g.total_vram_mb for g in selected)
    currently_used = sum(g.used_vram_mb for g in selected)
    safe_pool = total_selected_vram - currently_used
    if safe_pool <= 0:
        raise ResourcePlanError(
            "Selected GPU pool has no free VRAM",
            detail=(f"{len(selected)} GPU(s) selected, total={total_selected_vram} MiB, "
                    f"already used={currently_used} MiB."),
            suggestions=["Close other GPU workloads (games, browsers, training runs), "
                         "or select different GPUs in Tab 1."])

    model_mb = _file_size_mb(model_file.local_path()) or model_file.estimated_size_mb
    mmproj_mb = _file_size_mb(mmproj_path) if mmproj_path else 0.0
    if mmproj_mb == 0.0 and model_file.mmproj_filename:
        mp = model_file.mmproj_local_path()
        mmproj_mb = _file_size_mb(mp) if mp else 0.0
    # Vision projectors are small; assume ~600 MiB when unknown but needed.
    if model_file.mmproj_filename and mmproj_mb == 0.0:
        mmproj_mb = MMPROJ_UNKNOWN_MB
        plan.notes.append(
            f"mmproj '{model_file.mmproj_filename}' not found on disk; assuming "
            f"{MMPROJ_UNKNOWN_MB:.0f} MiB. Run 'Download & Verify' in Tab 1.")

    plan.model_mb = model_mb
    plan.mmproj_mb = mmproj_mb

    weights_need = model_mb + mmproj_mb + KV_OVERHEAD_MB
    remaining = safe_pool - weights_need

    kv_bpt = arch.kv_bytes_per_token(cache_type)
    plan.kv_bytes_per_token = max(1, kv_bpt)

    # ---- Spilling to RAM -------------------------------------------------
    if remaining < 0:
        # Weights do not fit: offload as many layers as possible to VRAM,
        # keep KV + activations headroom, spill the rest to system RAM.
        kv_reserve_mb = MIN_CONTEXT * kv_bpt / (1024 * 1024)
        activation_reserve_mb = 512
        usable_for_weights = max(0.0, safe_pool - KV_OVERHEAD_MB
                                 - kv_reserve_mb - activation_reserve_mb)
        mmproj_mb_eff = min(mmproj_mb, usable_for_weights)
        budget = max(0.0, usable_for_weights - mmproj_mb_eff)
        bytes_per_layer = model_mb * 1024 * 1024 / max(1, arch.num_layers)
        ngl = int(budget * 1024 * 1024 // bytes_per_layer) if bytes_per_layer else 0
        ngl = max(0, min(ngl, arch.num_layers))

        ram_avail_mb = hw.ram_available_bytes / (1024 * 1024)
        spill_mb = model_mb - (ngl * bytes_per_layer) / (1024 * 1024)
        if spill_mb > ram_avail_mb * 0.9:
            plan.notes.append(
                f"WARNING: {spill_mb:.0f} MiB would spill to RAM but only "
                f"{ram_avail_mb:.0f} MiB available - expect swapping/slowness. "
                "Fix: choose a smaller quantization (e.g. UD-IQ4_XS), add GPUs "
                "to the pool in Tab 1, or free system RAM.")
        else:
            plan.notes.append(
                f"Spill budget OK: {spill_mb:.0f} MiB of weights to system RAM "
                f"({ram_avail_mb:.0f} MiB currently available).")
        if ngl == 0:
            plan.notes.append(
                "NOTE: not even one layer fits on GPU; inference will be "
                "CPU-only and very slow. Consider a smaller quantization.")
        plan.n_gpu_layers = ngl
        plan.spilled_to_ram = ngl < arch.num_layers
        plan.available_vram_mb = 0
        plan.context_tokens = MIN_CONTEXT
        plan.notes.append(
            f"Model exceeds VRAM ({model_mb + mmproj_mb:.0f} MiB > {safe_pool} MiB): "
            f"offloading {ngl}/{arch.num_layers} layers to GPU (-ngl {ngl}), "
            f"rest spills to system RAM.")
        return plan

    # ---- Everything fits: maximise context -------------------------------
    plan.full_offload = True
    plan.n_gpu_layers = 99            # -ngl 99 => all layers on GPU
    plan.available_vram_mb = int(remaining)

    max_tokens = int(remaining * 1024 * 1024 // kv_bpt)
    ctx = (max_tokens // 1024) * 1024   # largest multiple of 1024 that fits
    ctx = max(MIN_CONTEXT, min(ctx, MAX_CONTEXT_CAP))
    plan.context_tokens = ctx
    plan.notes.append(
        f"Available VRAM after weights+{KV_OVERHEAD_MB}MB overhead: "
        f"{remaining:.0f} MiB -> -c {ctx} at {cache_type} KV "
        f"({kv_bpt/1024:.1f} KiB/token).")
    return plan


def shrink_context(current: int, step: int = 1024) -> int:
    """OOM recovery: reduce ``-c`` by one 1024-token step (floor at MIN)."""
    return max(MIN_CONTEXT, current - step)
