from __future__ import annotations

from typing import Optional

from .client import OnvifAuthError, OnvifClient, OnvifError, try_onvif_zeep_stream
from .discovery import OnvifDiscoveryResult
from .rtsp import guess_fallback_urls


def enrich_onvif_device(
    res: OnvifDiscoveryResult, user: Optional[str], pwd: Optional[str]
) -> dict:
    info = {
        "xaddr": res.xaddr,
        "ip": res.ip,
        "auth_required": False,
        "name": res.ip,
        "model": None,
        "firmware": None,
        "profiles": [],
        "media_xaddr": None,
        "stream_uri": None,
        "user": user,
        "password": pwd,
        "errors": [],
        "fallback_urls": [],
        "zeep_used": False,
    }

    # Try zeep (onvif-zeep) first if available; helps picky cameras.
    profiles_zeep, stream_zeep, errs_zeep = try_onvif_zeep_stream(res.xaddr, user, pwd)
    if profiles_zeep or stream_zeep:
        info["profiles"] = profiles_zeep
        info["stream_uri"] = stream_zeep
        info["zeep_used"] = True
    if errs_zeep:
        info["errors"].extend(errs_zeep)

    client = OnvifClient(res.xaddr, username=user, password=pwd)
    try:
        dev = client.get_device_information()
        if dev:
            info["name"] = dev.manufacturer or dev.model or res.ip
            info["model"] = dev.model
            info["firmware"] = dev.firmware
    except OnvifAuthError:
        info["auth_required"] = True
    except OnvifError as e:
        info["errors"].append(f"device_info: {e}")

    media_xaddr = None
    try:
        caps = client.get_capabilities()
        media_xaddr = caps.media_xaddr
        info["media_xaddr"] = media_xaddr
    except OnvifAuthError:
        info["auth_required"] = True
    except OnvifError as e:
        info["errors"].append(f"capabilities: {e}")

    try:
        profiles = client.get_profiles(media_xaddr=media_xaddr or res.xaddr)
        info["profiles"] = [{"token": p.token, "name": p.name} for p in profiles]
        stream = None
        if profiles:
            stream = client.get_stream_uri(
                profiles[0].token, media_xaddr=media_xaddr or res.xaddr
            )
        info["stream_uri"] = stream
        info["fallback_urls"] = guess_fallback_urls(res.ip)
    except OnvifAuthError:
        info["auth_required"] = True
    except OnvifError as e:
        info["errors"].append(f"media: {e}")

    return info

