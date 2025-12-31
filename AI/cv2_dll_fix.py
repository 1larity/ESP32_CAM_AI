import os, site, glob, importlib.util

# Keep DLL directory handles alive; Windows removes them when the handle is GC'd.
_DLL_HANDLES = []
_DLL_DIRS = set()

def _add_dir(d: str) -> None:
    if not d or not os.path.isdir(d) or d in _DLL_DIRS:
        return
    if hasattr(os, "add_dll_directory"):
        _DLL_HANDLES.append(os.add_dll_directory(d))
    else:
        os.environ["PATH"] = f"{d};{os.environ.get('PATH','')}"
    _DLL_DIRS.add(d)

def enable_opencv_cuda_dll_search() -> None:
    # cv2 package dir
    cv2_spec = importlib.util.find_spec("cv2")
    if cv2_spec and cv2_spec.submodule_search_locations:
        _add_dir(next(iter(cv2_spec.submodule_search_locations)))

    # CUDA Toolkit runtimes (support multiple installed versions, e.g., CUDA_PATH_V12_x)
    cuda_roots = [
        path
        for key, path in os.environ.items()
        if key.startswith("CUDA_PATH") and path and os.path.isdir(path)
    ]
    for root in cuda_roots:
        for sub in ("bin", os.path.join("bin", "x64"), os.path.join("lib", "x64")):
            _add_dir(os.path.join(root, sub))

    # Search common site-packages roots for NVIDIA DLL drops (cuDNN/cuBLAS)
    roots = [site.getusersitepackages()] + (site.getsitepackages() if hasattr(site, "getsitepackages") else [])
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat in ("cudnn*.dll", "cublas*.dll", "cublasLt*.dll"):
            for p in glob.glob(os.path.join(root, "**", pat), recursive=True):
                _add_dir(os.path.dirname(p))

    # Common standalone cuDNN installs (user-provided path)
    for cudnn_dir in (
        r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1",
    ):
        _add_dir(cudnn_dir)
