from __future__ import annotations

import importlib.util
import os
import site
import sys

# Cache pip-installed onvif-zeep modules separately so we can use them without shadowing
# the local AI/onvif package.
_PIP_ONVIF_MODULES: dict[str, object] = {}


def _load_onvif_zeep_camera():
    """
    Explicitly load onvif-zeep from site-packages, avoiding the local AI/onvif package.
    Returns ONVIFCamera class or None if not available.
    """
    if _PIP_ONVIF_MODULES:
        mod = _PIP_ONVIF_MODULES.get("onvif")
        if mod:
            return getattr(mod, "ONVIFCamera", None)

    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
    except Exception:
        pass

    ai_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    original_path = list(sys.path)
    original_onvif_modules = {
        k: v
        for k, v in sys.modules.items()
        if k == "onvif" or k.startswith("onvif.")
    }

    for base in candidates:
        if not base or not os.path.isdir(base):
            continue
        search_roots = {base, os.path.join(base, "Lib", "site-packages")}
        for root in search_roots:
            init_path = os.path.join(root, "onvif", "__init__.py")
            if not os.path.isfile(init_path):
                continue
            try:
                if os.path.samefile(
                    init_path, os.path.join(ai_dir, "onvif", "__init__.py")
                ):
                    continue
            except Exception:
                pass

            # Prepare an isolated module set so the pip package can import its siblings.
            backup_modules = {
                k: v
                for k, v in sys.modules.items()
                if k == "onvif" or k.startswith("onvif.")
            }
            for k in list(backup_modules.keys()):
                sys.modules.pop(k, None)
            try:
                cleaned_path = [root] + [
                    p for p in original_path if os.path.abspath(p) != ai_dir
                ]
                sys.path = cleaned_path
                spec = importlib.util.spec_from_file_location("onvif", init_path)
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                module.__path__ = [os.path.dirname(init_path)]
                sys.modules["onvif"] = module
                spec.loader.exec_module(module)
                camera = getattr(module, "ONVIFCamera", None)
                if camera:
                    pip_modules = {
                        k: v
                        for k, v in sys.modules.items()
                        if k == "onvif" or k.startswith("onvif.")
                    }
                    _PIP_ONVIF_MODULES.update(pip_modules)
                    return camera
            except Exception:
                continue
            finally:
                sys.path = original_path
                for k in list(sys.modules.keys()):
                    if k == "onvif" or k.startswith("onvif."):
                        sys.modules.pop(k, None)
                sys.modules.update(backup_modules)

    # Restore original modules if nothing was loaded.
    for k in list(sys.modules.keys()):
        if k == "onvif" or k.startswith("onvif."):
            sys.modules.pop(k, None)
    sys.modules.update(original_onvif_modules)
    return None


try:
    # Heavy optional dependency for full ONVIF support (pip package onvif-zeep)
    ONVIFCamera = _load_onvif_zeep_camera()
except Exception:  # pragma: no cover
    ONVIFCamera = None


__all__ = ["ONVIFCamera", "_PIP_ONVIF_MODULES", "_load_onvif_zeep_camera"]

