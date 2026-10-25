"""Regression: configure must explicitly disable non-selected GPU backends.

Bug: on an NVIDIA box with CUDA toolkit installed but no Vulkan SDK, cmake
aborted with 'Could NOT find Vulkan (missing: Vulkan_LIBRARY
Vulkan_INCLUDE_DIR glslc)' because llama.cpp's CMake auto-enables the
Vulkan backend when it detects any partial Vulkan presence. The guided
installer only passed -DGGML_CUDA=ON and never -DGGML_VULKAN=OFF.
"""
import sys, types, tempfile, shutil
from pathlib import Path

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
from app.core.models import AppError

def make_runner(backend):
    """Build an InstallRunner whose run() stops right after configure."""
    root = Path(tempfile.mkdtemp())
    r = L.InstallRunner(dest_root=root)
    # fake a passing preflight + existing source tree
    pf = L.PreflightResult()
    pf.ok = True; pf.backend = backend; pf.reason = "test"
    pf.gpu_families = {"nvidia"} if backend == "cuda" else set()
    r.preflight = lambda: pf
    r._find_cuda_root = lambda: None
    src = root/"src"; (src/".git").mkdir(parents=True)
    pf.ok = True; pf.backend = backend; pf.reason = "test"   # set AFTER construction
    pf.gpu_families = {"nvidia"} if backend == "cuda" else set()
    cmds: list[list[str]] = []
    state = {"cfg": 0}
    def fake_stream(cmd):
        cmds.append(list(cmd))
        if cmd[0] == "cmake" and "-S" in cmd:
            state["cfg"] += 1
            raise AppError("STOP-TEST", detail="halted after configure")
        # git pull/clone and anything before configure: pretend success
    r._stream = fake_stream
    return r, cmds, root

cases = {
    "cuda":   ["-DGGML_CUDA=ON",  "-DGGML_HIP=OFF",   "-DGGML_VULKAN=OFF"],
    "rocm":   ["-DGGML_HIP=ON",   "-DGGML_CUDA=OFF",  "-DGGML_VULKAN=OFF"],
    "vulkan": ["-DGGML_VULKAN=ON","-DGGML_CUDA=OFF",  "-DGGML_HIP=OFF"],
    "cpu":    ["-DGGML_CUDA=OFF", "-DGGML_HIP=OFF",   "-DGGML_VULKAN=OFF"],
}
for backend, expected in cases.items():
    r, cmds, root = make_runner(backend)
    try:
        r.run()          # raises our STOP-TEST after the configure invocation
    except AppError as e:
        assert "STOP-TEST" in e.message or "halted after configure" in (e.detail or ""), f"{backend}: unexpected {e.message}/{e.detail}"
    cfg = next((c for c in cmds if c and c[0] == "cmake" and "-S" in c), None)
    assert cfg is not None, f"{backend}: no configure command captured"
    for flag in expected:
        assert flag in cfg, f"{backend}: missing {flag} in {cfg}"
    print(f"{backend}: configure flags OK -> {[f for f in cfg if f.startswith('-DGGML')]}")
    shutil.rmtree(root, ignore_errors=True)

# Retry-once behaviour: first configure fails with a stale cache present,
# second attempt must run after the build dir was wiped.
root = Path(tempfile.mkdtemp())
r = L.InstallRunner(dest_root=root)
pf = L.PreflightResult(); pf.ok=True; pf.backend="cuda"; pf.reason="t"; pf.gpu_families={"nvidia"}
r.preflight = lambda: pf
r._find_cuda_root = lambda: None
(src := root/"src"); (src/".git").mkdir(parents=True)
build = src/"build"; build.mkdir(); (build/"CMakeCache.txt").write_text("stale\n")
attempts = {"n": 0}; seen_cache = {}
def stream2(cmd):
    if cmd[0] == "cmake" and "-S" in cmd:
        attempts["n"] += 1
        seen_cache[attempts["n"]] = (build/"CMakeCache.txt").exists()
        if attempts["n"] == 1:
            raise AppError("configure blew up", detail="Configuring incomplete, errors occurred!")
        raise AppError("STOP-TEST-2")
    # git steps succeed silently
r._stream = stream2
try:
    r.run()
except AppError as e:
    assert "STOP-TEST-2" in e.message, e.message
assert attempts["n"] == 2, f"expected exactly one retry, got {attempts['n']}"
assert seen_cache[1] is True and seen_cache[2] is False, seen_cache
print("retry-once: stale cache cleared and configure retried exactly once OK")
shutil.rmtree(root, ignore_errors=True)

# No-cache case must NOT retry (fail fast).
root = Path(tempfile.mkdtemp())
r = L.InstallRunner(dest_root=root)
pf = L.PreflightResult(); pf.ok=True; pf.backend="cuda"; pf.reason="t"; pf.gpu_families={"nvidia"}
r.preflight = lambda: pf
r._find_cuda_root = lambda: None
(src := root/"src"); (src/".git").mkdir(parents=True)
n3 = {"c": 0}
def stream3(cmd):
    if cmd[0] == "cmake" and "-S" in cmd:
        n3["c"] += 1
        raise AppError("boom")
r._stream = stream3
try:
    r.run()
except AppError as e:
    assert "boom" in e.message
assert n3["c"] == 1, f"must not retry without a stale cache, got {n3['c']} attempts"
print("no-retry-without-cache OK")
shutil.rmtree(root, ignore_errors=True)

print("ALL CONFIGURE-FLAG TESTS PASS")
