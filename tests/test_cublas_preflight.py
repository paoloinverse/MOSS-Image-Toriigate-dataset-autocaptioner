"""Regression tests for the 'CUDA::cublas target not found' handling.

Scenario reproduced from a real user log: an NVIDIA-repo minimal toolkit
(cuda-compiler-13-x) has nvcc, so stage 3 used to pass; cmake then died at
generate time with

    Target "ggml-cuda" links to: CUDA::cublas ... but the target was not found

The fixes under test:
  * preflight now verifies linkable libcublas.so inside the toolkit root and
    HALTS before any download, with copy-pasteable package names (both the
    distro split packages and the versioned cuda-toolkit-X-Y metapackage);
  * if the failure still slips through (e.g. a library llama.cpp starts
    requiring in the future), _stream() translates the cmake error text into
    the same concrete fix commands instead of the generic retry hint.
"""
import sys, types, tempfile, shutil
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


def make_fake_cuda(with_cublas: bool, with_curand: bool = True) -> Path:
    """Synthetic /usr/local/cuda-like root: nvcc present, libs optional."""
    root = Path(tempfile.mkdtemp())
    (root/"bin").mkdir()
    (root/"lib64").mkdir()
    nvcc = root/"bin"/"nvcc"
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.1, V13.1.115'\n")
    nvcc.chmod(0o755)
    if with_cublas:
        (root/"lib64"/"libcublas.so").write_text("")
    if with_curand:
        (root/"lib64"/"libcurand.so").write_text("")
    return root


def build_runner(pci_fams, cuda_root, missing_libs_real=True):
    r = L.InstallRunner()
    orig_enum = L.enumerate_gpu_families
    L.enumerate_gpu_families = lambda: set(pci_fams)
    r._check_base_tools = lambda: None
    r._has_nvidia_driver = lambda: True
    r._amdgpu_driver_loaded = staticmethod(lambda: False)
    type(r)._amdgpu_driver_loaded = staticmethod(lambda _v=False: _v)
    r._find_cuda_root = lambda: cuda_root
    r._has_nvcc = lambda root=None: cuda_root is not None and (cuda_root/"bin"/"nvcc").is_file()
    r._find_rocm_root = lambda: Path("/nonexistent-rocm")
    r._has_vulkan = lambda: False
    # ldconfig lookup must not depend on the sandbox's real libraries:
    if missing_libs_real:
        r._has_cuda_dev_lib = lambda name, root: (root is not None and
                                                  (Path(root)/"lib64"/f"{name}.so").exists())
    else:
        r._missing_cuda_libs = lambda root: []
    return r, orig_enum


try:
    # ---- 1. detection helper itself ---------------------------------------
    full = make_fake_cuda(True, True)
    lite = make_fake_cuda(True, False)          # cublas+? curand missing
    bare = make_fake_cuda(False, False)         # the user's case: compiler-only
    r, _ = build_runner({"nvidia"}, bare)
    assert r._missing_cuda_libs(bare) == ["libcublas", "libcurand", "libcudnn"], \
        f"bare root wrong: {r._missing_cuda_libs(bare)}"
    assert r._missing_cuda_libs(lite) == ["libcurand", "libcudnn"], \
        f"lite root wrong: {r._missing_cuda_libs(lite)}"
    assert r._missing_cuda_libs(full) == ["libcudnn"], \
        f"full root wrong: {r._missing_cuda_libs(full)}"
    assert r._cuda_toolkit_version(bare) == "13.1", r._cuda_toolkit_version(bare)
    print("1 _missing_cuda_libs/_cuda_toolkit_version: OK")

    # ---- 2. preflight halts BEFORE download on incomplete toolkit ----------
    r, orig_enum = build_runner({"nvidia"}, bare)
    try:
        res = r.preflight()
    finally:
        L.enumerate_gpu_families = orig_enum
    assert not res.ok and res.error is not None, "should have blocked"
    d = res.error.detail
    assert "incomplete" in d and "libcublas" in d, f"detail wrong: {d}"
    assert "ROCm" not in d, "must blame cuBLAS, not ROCm"
    joined = "\n".join(res.error.commands)
    assert "libcublas-dev" in joined, f"commands wrong: {joined}"
    assert "cuda-toolkit-13-1" in joined, f"versioned metapackage missing: {joined}"
    rep = res.error.report()
    assert "You do NOT need to reinstall CUDA itself" in rep, rep
    print("2 preflight halt + fix commands:", *res.error.commands, sep="\n   ")

    # ---- 3. complete toolkit proceeds to cuda backend ----------------------
    r, orig_enum = build_runner({"nvidia"}, full)
    try:
        res = r.preflight()
    finally:
        L.enumerate_gpu_families = orig_enum
    assert res.ok and res.backend == "cuda", (res.ok, res.backend,
                                              res.error.detail if res.error else "")
    print("3 complete toolkit -> backend=cuda: OK")

    # ---- 4. optional-lib gap alone must NOT block --------------------------
    r, orig_enum = build_runner({"nvidia"}, lite)
    try:
        res = r.preflight()
    finally:
        L.enumerate_gpu_families = orig_enum
    assert res.ok and res.backend == "cuda", "curand/cudnn are optional!"
    print("4 missing curand only -> no block: OK")

    # ---- 5. late cmake error translated by _stream -------------------------
    logs = []

    class _FakeStdout:
        def __init__(self, lines):
            self._it = iter(lines)

        def readline(self):
            return next(self._it, "")

    fake_proc = types.SimpleNamespace(
        stdout=_FakeStdout([l+"\n" for l in [
            "-- Configuring done (3.3s)",
            'CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:175 (target_link_libraries):',
            '  Target "ggml-cuda" links to:',
            "    CUDA::cublas",
            "  but the target was not found.",
            "-- Generating done (0.1s)",
            "CMake Generate step failed.  Build files cannot be regenerated correctly.",
        ]]),
        wait=lambda: 1)
    L.subprocess.Popen = lambda *a, **k: fake_proc
    r, _ = build_runner({"nvidia"}, bare)
    # PySide6 SignalInstance.connect is read-only on instances; patch the
    # class-level descriptor so emit() records into our list instead.
    sig_cls = type(r.log)
    orig_connect, orig_emit = sig_cls.connect, sig_cls.emit
    sig_cls.connect = lambda self, fn, *a: None
    sig_cls.emit = lambda self, *a: logs.append(a[0] if a else "")
    try:
        r._stream(["cmake", "-S", "src", "-B", "build"])
        raise AssertionError("AppError expected")
    except L.AppError as _e:
        err = _e
    finally:
        sig_cls.connect, sig_cls.emit = orig_connect, orig_emit
    blob = "\n".join(logs)
    assert "FIX: sudo apt install libcublas-dev" in blob, blob
    assert "cuda-toolkit-13-1" in blob or "cuda-toolkit-" in blob, blob
    assert "math libraries" in blob, blob
    assert ("cuBLAS dev files" in err.suggestions[0]
            or "libcublas-dev" in err.suggestions[0]), err.suggestions
    print("5 _stream translation: OK")

    print("ALL CUBLAS PREFLIGHT TESTS PASS")
finally:
    for p in (list(Path(tempfile.gettempdir()).glob("tmp*")) if False else []):
        pass
