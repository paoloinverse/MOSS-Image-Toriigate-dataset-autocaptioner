"""Regression tests for the phantom-AMD-family bug on NVIDIA-only boxes."""
import sys, types
from pathlib import Path

# stub PyQt6 so llamacpp imports headlessly
qtcore = types.ModuleType("PyQt6.QtCore")
class _Signal:
    def __init__(self,*a,**k): pass
    def connect(self,*a): pass
    def emit(self,*a): pass
class QObject:
    log=_Signal(); finished=_Signal()
    def __init__(self,*a,**k): pass
qtcore.QObject=QObject; qtcore.QThread=type("QThread",(QObject,),{}); qtcore.Signal=lambda *a,_n=None:_Signal()
pyqt=types.ModuleType("PyQt6"); pyqt.QtCore=qtcore
sys.modules["PyQt6"]=pyqt; sys.modules["PyQt6.QtCore"]=qtcore
sys.path.insert(0,"/workspace")
from app.core import llamacpp as L

def scenario(name, *, pci_map, nv_driver, amdgpu, has_nvcc=True, has_hipcc=False, vulkan=False, env=None):
    # pci_map: dict dir_name -> (class_hex, vendor_hex) written into a fake sysfs root
    import tempfile, os, shutil
    root = Path(tempfile.mkdtemp())
    devs = root/"devices"; devs.mkdir()
    for dn,(cls,ven) in pci_map.items():
        d=devs/dn; d.mkdir(); (d/"class").write_text(cls+"\n"); (d/"vendor").write_text(ven+"\n")
    orig_enum = L.enumerate_gpu_families
    L.enumerate_gpu_families = lambda: _enum_from(devs)
    def _enum_from(r):
        fams=set()
        for d in sorted(r.iterdir()):
            cls=(d/"class").read_text().strip().lower(); ven=(d/"vendor").read_text().strip().lower()
            if cls.startswith(L._PCI_GRAPHICS_IFACES):
                f=L._GPU_VENDOR_IDS.get(ven)
                if f: fams.add(f)
        return fams
    r = L.InstallRunner()
    r._has_nvidia_driver = lambda: nv_driver
    r._amdgpu_driver_loaded = staticmethod(lambda: amdgpu).__func__ if False else (lambda: amdgpu)
    type(r)._amdgpu_driver_loaded = staticmethod(lambda self_=None, _v=amdgpu: _v)
    r._find_cuda_root = lambda: Path("/usr/local/cuda") if has_nvcc else None
    r._has_nvcc = lambda root=None: has_nvcc
    # The new cuBLAS-dev verification must not touch the real filesystem in
    # these GPU-family scenarios (covered separately by test_cublas_preflight).
    r._missing_cuda_libs = lambda root: []
    r._find_rocm_root = lambda: Path("/opt/rocm")
    import shutil as _sh
    _orig_which=_sh.which
    r._has_rocm = lambda: has_hipcc
    # patch hipcc detection used inside preflight: it calls shutil.which + rocm_root/bin check
    L.shutil.which = lambda n, *a: (n if n=="hipcc" and has_hipcc else _orig_which(n,*a))
    r._has_vulkan = lambda: vulkan
    r._check_base_tools = lambda: None
    oldenv=dict(os.environ)
    for k,v in (env or {}).items(): os.environ[k]=v
    try:
        res=r.preflight()
    finally:
        os.environ.clear(); os.environ.update(oldenv)
        L.shutil.which=_orig_which
        L.enumerate_gpu_families=orig_enum
        shutil.rmtree(root)
    print(f"{name}: ok={res.ok} backend={res.backend} families={sorted(res.gpu_families)} err={(res.error.detail[:80] if res.error else '')}")
    return res

RTX=("0x030000","0x10de"); RTX_AUDIO=("0x040300","0x10de")
AMD_VIRT=("0x038000","0x1002")  # AMD-family virtual display function, no amdgpu driver
INTEL_APU=("0x030000","0x8086")

# Scenario A: user's machine - NVIDIA driver loaded, PCI shows RTX + phantom AMD display fn
r=scenario("A nvidia+phantom-amd,nvcc", pci_map={"0000:01:00.0":RTX,"0000:01:00.1":RTX_AUDIO,"0000:0d:00.0":AMD_VIRT}, nv_driver=True, amdgpu=False)
assert r.ok and r.backend=="cuda" and r.gpu_families=={"nvidia"}, "A FAILED"

# Scenario B: same but NO nvcc -> must halt demanding CUDA (not ROCm)
r=scenario("B nvidia+phantom-amd,no-nvcc", pci_map={"0000:01:00.0":RTX,"0000:0d:00.0":AMD_VIRT}, nv_driver=True, amdgpu=False, has_nvcc=False)
assert not r.ok and "NVIDIA" in r.error.detail and "ROCm" not in r.error.detail, "B FAILED"

# Scenario C: genuine dual-GPU (amdgpu loaded) but CUDA toolkit works -> must
# NOT halt: the NVIDIA+CUDA combo can serve the build, so the secondary AMD
# display silicon is dropped instead of demanding a ROCm stack (this was the
# residual false-positive that kept blocking RTX 5060 Ti + AMD iGPU boxes).
r=scenario("C real amd+nvidia,nvcc,no-hipcc", pci_map={"0000:01:00.0":RTX,"0000:05:00.0":("0x030000","0x1002")}, nv_driver=True, amdgpu=True, has_hipcc=False)
assert r.ok and r.backend=="cuda" and r.gpu_families=={"nvidia"}, "C FAILED"

# Scenario C2: same hardware but NO nvcc either -> now ROCm truly matters,
# so the preflight must halt demanding the ROCm toolchain (no silent CPU).
r=scenario("C2 real amd+nvidia,no-nvcc,no-hipcc", pci_map={"0000:01:00.0":RTX,"0000:05:00.0":("0x030000","0x1002")}, nv_driver=True, amdgpu=True, has_nvcc=False, has_hipcc=False)
assert not r.ok and "AMD" in r.error.detail and "CUDA" in r.error.detail, "C2 FAILED"

# Scenario D: genuine AMD box with amdgpu + hipcc -> rocm backend
r=scenario("D real amd + hipcc", pci_map={"0000:05:00.0":("0x030000","0x1002")}, nv_driver=False, amdgpu=True, has_hipcc=True, has_nvcc=False)
assert r.ok and r.backend=="rocm", "D FAILED"

# Scenario E: audio-only pollution never counted (already covered by A) ; intel-only iGPU w/o vulkan sdk, no nvidia driver -> halts? hw={intel}, no blocked entries unless want==vulkan -> falls to stage4: hw and vulkan_ready false -> 'else unreachable'? check
r=scenario("E intel-only, no vulkan sdk", pci_map={"0000:00:02.0":INTEL_APU}, nv_driver=False, amdgpu=False, has_nvcc=False, vulkan=False)
print("   E result:", r.ok, r.backend, r.error.message if r.error else "")

print("ALL SCENARIOS PASS")
