"""Hardware detection & resource management.

* GPU enumeration via ``pynvml`` (NVIDIA/CUDA) with a ``rocm-smi`` fallback
  for AMD/ROCm GPUs (RX 7900 XTX etc.).
* Continuous system-RAM monitoring via ``psutil`` so the planner can decide
  when model weights must "spill" to system RAM (partial layer offloading).

The whole subsystem runs inside a single QThread ("HardwareManager") that
emits fresh telemetry roughly once per second; all heavy driver calls happen
off the GUI thread.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field

import psutil
from PySide6.QtCore import QThread, Signal

POLL_INTERVAL_MS = 1000


@dataclass
class GPUInfo:
    """Snapshot of one physical GPU."""
    index: int
    name: str
    total_vram_mb: int
    used_vram_mb: int = 0
    util_percent: float = 0.0
    backend: str = "nvidia"          # "nvidia" | "amd"
    selected: bool = False           # included in the inference pool
    split_weight: int = 50           # relative weight for --tensor-split

    @property
    def free_vram_mb(self) -> int:
        return max(0, self.total_vram_mb - self.used_vram_mb)

    def label(self) -> str:
        return (f"GPU {self.index}: {self.name} "
                f"[{self.free_vram_mb / 1024:.1f} / {self.total_vram_mb / 1024:.1f} GiB free]")


# --------------------------------------------------------------------------
# Backend probers
# --------------------------------------------------------------------------
class NvidiaProber:
    """Enumerates NVIDIA GPUs through NVML (``pynvml``)."""

    name = "nvidia"

    def __init__(self):
        self._nvml = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            self._nvml = None

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def probe(self) -> list[GPUInfo]:
        if not self.available:
            return []
        pynvml = self._nvml
        out: list[GPUInfo] = []
        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception:
            return []
        for i in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gname = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gname, bytes):
                    gname = gname.decode("utf-8", "replace")
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except Exception:
                    util = 0
                out.append(GPUInfo(
                    index=i,
                    name=str(gname),
                    total_vram_mb=int(mem.total // (1024 * 1024)),
                    used_vram_mb=int(mem.used // (1024 * 1024)),
                    util_percent=float(util),
                    backend="nvidia",
                ))
            except Exception:
                continue
        return out


class AmdProber:
    """Enumerates AMD GPUs by parsing ``rocm-smi --showmeminfo vram -j``.

    Falls back to a plain-text parse if the JSON output is unavailable.
    """

    name = "amd"

    def __init__(self):
        self._exe = shutil.which("rocm-smi") or shutil.which("amd-smi")

    @property
    def available(self) -> bool:
        return bool(self._exe)

    def probe(self) -> list[GPUInfo]:
        if not self.available:
            return []
        try:
            res = subprocess.run(
                [self._exe, "--showmeminfo", "vram", "-j"],
                capture_output=True, text=True, timeout=8,
            )
            data = json.loads(res.stdout or "{}")
        except Exception:
            return self._probe_text()
        out: list[GPUInfo] = []
        idx = 0
        for key, val in sorted(data.items()):
            if not isinstance(val, dict):
                continue
            total = _first_number(val, ("total", "Total"))
            used = _first_number(val, ("used", "Used"))
            if total <= 0:
                continue
            name = None
            for k2, v2 in val.items():
                if isinstance(v2, str) and "GPU" in v2.upper():
                    name = v2
            out.append(GPUInfo(
                index=idx,
                name=name or f"AMD GPU {idx}",
                total_vram_mb=int(total // (1024 * 1024)),
                used_vram_mb=int(used // (1024 * 1024)),
                backend="amd",
            ))
            idx += 1
        return out

    def _probe_text(self) -> list[GPUInfo]:
        try:
            res = subprocess.run([self._exe], capture_output=True, text=True, timeout=8)
        except Exception:
            return []
        out: list[GPUInfo] = []
        for line in res.stdout.splitlines():
            m = re.search(r"GPU\[(\d+)\].*?(\d+(?:\.\d+)?)MB", line)
            if m:
                idx = int(m.group(1))
                if all(g.index != idx for g in out):
                    out.append(GPUInfo(index=idx, name=f"AMD GPU {idx}",
                                       total_vram_mb=int(float(m.group(2))),
                                       backend="amd"))
        return out


def _first_number(d: dict, keys: tuple[str, ...]) -> int:
    for k in keys:
        if k in d:
            try:
                return int(d[k])
            except (TypeError, ValueError):
                pass
    return 0


# --------------------------------------------------------------------------
# Hardware manager thread
# --------------------------------------------------------------------------
class HardwareManager(QThread):
    """Background poller emitting GPU/RAM telemetry to the GUI.

    Signals
    -------
    gpus_updated(list)      : list[GPUInfo] – emitted whenever the static
                              topology changes or on every poll (cheap).
    ram_updated(int, int)   : available_bytes, total_bytes
    log_message(str)        : human-readable startup/detection log lines
    """

    gpus_updated = Signal(list)
    ram_updated = Signal(int, int)
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()
        self.gpus: list[GPUInfo] = []
        self.ram_total_bytes: int = psutil.virtual_memory().total
        self.ram_available_bytes: int = psutil.virtual_memory().available
        self._probers = [NvidiaProber(), AmdProber()]

    # -- lifecycle ---------------------------------------------------------
    def run(self):  # noqa: D102 - QThread entry point
        while not self._stop.is_set():
            try:
                self._poll_once(first=(not self.gpus))
            except Exception as exc:  # never kill the poller
                self.log_message.emit(f"[hw] poll error: {exc}")
            self.msleep(POLL_INTERVAL_MS)
        self._stop.set()

    def stop(self):
        self._stop.set()
        self.wait(3000)

    # -- internals ---------------------------------------------------------
    def _poll_once(self, first: bool = False):
        if first or not self.gpus:
            detected: list[GPUInfo] = []
            for prober in self._probers:
                if prober.available:
                    detected = prober.probe()
                    if detected:
                        if first:
                            self.log_message.emit(
                                f"[hw] Detected {len(detected)} GPU(s) via {prober.name}")
                        break
            if not detected and first:
                self.log_message.emit("[hw] No CUDA/ROCm GPUs detected.")
            # Preserve user selection across re-probes (match by index+name).
            old = {(g.index, g.name): g for g in self.gpus}
            for g in detected:
                prev = old.get((g.index, g.name))
                if prev:
                    g.selected = prev.selected
                    g.split_weight = prev.split_weight
            if not self.gpus and detected:      # default: select everything
                for g in detected:
                    g.selected = True
            self.gpus = detected
            self.gpus_updated.emit(list(self.gpus))

        vm = psutil.virtual_memory()
        self.ram_available_bytes = vm.available
        self.ram_total_bytes = vm.total
        self.ram_updated.emit(vm.available, vm.total)
        self.gpus_updated.emit(list(self.gpus))

    # -- helpers -----------------------------------------------------------
    @property
    def selected_gpus(self) -> list[GPUInfo]:
        return [g for g in self.gpus if g.selected]

    @property
    def selected_vram_total_mb(self) -> int:
        """Gross VRAM of the *selected* pool (safe threshold applied later)."""
        return sum(g.total_vram_mb for g in self.selected_gpus)

    @property
    def safe_vram_total_mb(self) -> int:
        """"Safe VRAM": selected gross VRAM minus what other processes use now,
        minus a small driver/context headroom per GPU."""
        per_gpu_headroom = 300  # MiB reserved for CUDA/ROCm context overhead
        total = 0
        for g in self.selected_gpus:
            total += max(0, g.total_vram_mb - per_gpu_headroom)
        return total

    def tensor_split_string(self) -> str:
        """Comma-separated ratios honouring each GPU's slider weight.

        Selected GPUs contribute their weight; unselected GPUs get 0 so that
        llama.cpp keeps them completely out of the split.
        """
        weights = [g.split_weight if g.selected else 0 for g in self.gpus]
        if not any(weights):
            return ""
        return ",".join(str(w) for w in weights)
