from __future__ import annotations

import os
import sys
from typing import Optional
from urllib.parse import urlparse

from .zeep_loader import ONVIFCamera, _PIP_ONVIF_MODULES, _load_onvif_zeep_camera


# ---------- optional zeep-based helper (richer compatibility) ---------- #
def try_onvif_zeep_stream(
    xaddr: str, user: Optional[str], password: Optional[str]
) -> tuple[list[dict], Optional[str], list[str]]:
    """
    Attempt to use onvif-zeep's ONVIFCamera to fetch profiles and stream URI.
    Returns (profiles, stream_uri, errors).
    profiles: list of {"token", "name"}.
    stream_uri: RTSP URL or None.
    errors: list of error strings.
    """
    errs: list[str] = []
    camera_cls = ONVIFCamera or _load_onvif_zeep_camera()
    if camera_cls is None:
        errs.append("onvif-zeep not installed")
        return [], None, errs
    # Determine WSDL directory from the pip-installed onvif package.
    wsdl_dir = None
    possible_mods = []
    if _PIP_ONVIF_MODULES:
        mod = _PIP_ONVIF_MODULES.get("onvif")
        if mod:
            possible_mods.append(mod)
    possible_mods.append(sys.modules.get("onvif"))
    ai_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    local_wsdl = os.path.join(ai_dir, "onvif", "wsdl")
    for mod in possible_mods:
        if mod and getattr(mod, "__file__", None):
            cand = os.path.join(os.path.dirname(mod.__file__), "wsdl")
            if os.path.isdir(cand):
                wsdl_dir = cand
                break
    if wsdl_dir is None and os.path.isdir(local_wsdl):
        wsdl_dir = local_wsdl
    if wsdl_dir is None or not os.path.isdir(wsdl_dir):
        errs.append(
            "onvif-zeep wsdl folder missing; copy WSDLs into AI/onvif/wsdl or site-packages/onvif/wsdl"
        )
        return [], None, errs

    parsed = urlparse(xaddr)
    host = parsed.hostname or ""
    if not host:
        errs.append("xaddr missing host")
        return [], None, errs
    port = parsed.port or 80
    path = parsed.path or "/onvif/device_service"
    # Temporarily install the pip onvif modules so ONVIFCamera internals can import siblings.
    backup_modules: Optional[dict[str, object]] = None
    if _PIP_ONVIF_MODULES:
        backup_modules = {k: sys.modules.get(k) for k in _PIP_ONVIF_MODULES.keys()}
        sys.modules.update(_PIP_ONVIF_MODULES)

    try:
        cam = camera_cls(host, port, user or "", password or "", wsdl_dir or None)
        media = cam.create_media_service()
        profiles_raw = media.GetProfiles()
        profiles: list[dict] = []
        for p in profiles_raw:
            token = getattr(p, "token", None)
            name = getattr(p, "Name", None) or token
            if token:
                profiles.append({"token": token, "name": name})
        stream_uri = None
        if profiles:
            for protocol in ("TCP", "RTSP"):
                try:
                    req = {
                        "StreamSetup": {
                            "Stream": "RTP-Unicast",
                            "Transport": {"Protocol": protocol},
                        },
                        "ProfileToken": profiles[0]["token"],
                    }
                    uri_obj = media.GetStreamUri(req)
                    uri_val = getattr(uri_obj, "Uri", None)
                    if uri_val:
                        stream_uri = uri_val
                        break
                except Exception as e:  # pragma: no cover - device specific
                    errs.append(f"zeep GetStreamUri {protocol}: {e}")
                    continue
        return profiles, stream_uri, errs
    except Exception as e:  # pragma: no cover - device specific
        errs.append(f"zeep init failed: {e}")
        return [], None, errs
    finally:
        if backup_modules is not None:
            for k, v in backup_modules.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


__all__ = ["try_onvif_zeep_stream"]

