from __future__ import annotations


def format_onvif_label(info: dict) -> str:
    name = info.get("name") or info.get("model") or info.get("ip")
    stream = info.get("stream_uri") or "<no stream>"
    auth = "auth" if info.get("auth_required") else "open"
    profile_count = len(info.get("profiles") or [])
    return f"{name} | {stream} | profiles:{profile_count} | {auth}"


def render_onvif_details(info: dict) -> str:
    parts = []
    parts.append(f"IP: {info.get('ip')}")
    parts.append(f"XAddr: {info.get('xaddr')}")
    if info.get("media_xaddr"):
        parts.append(f"Media XAddr: {info.get('media_xaddr')}")
    parts.append(f"Name/Model: {info.get('name')} / {info.get('model')}")
    if info.get("firmware"):
        parts.append(f"Firmware: {info.get('firmware')}")
    profiles = info.get("profiles") or []
    if profiles:
        prof_lines = []
        for p in profiles:
            if isinstance(p, dict):
                prof_lines.append(
                    f"- {p.get('name') or p.get('token')} (token={p.get('token')})"
                )
            else:
                prof_lines.append(f"- {p}")
        parts.append("Profiles:\n" + "\n".join(prof_lines))
    parts.append(f"Stream URI: {info.get('stream_uri') or '<none>'}")
    if info.get("error"):
        parts.append(f"Error: {info.get('error')}")
    errs = info.get("errors") or []
    if errs:
        parts.append("Errors:")
        parts.extend(f"- {e}" for e in errs)
    if info.get("auth_required"):
        parts.append("Auth required: yes")
    fallbacks = info.get("fallback_urls") or []
    if fallbacks:
        parts.append("Fallback RTSP guesses:")
        parts.extend(f"- {u}" for u in fallbacks)
    return "\n".join(parts)


__all__ = ["format_onvif_label", "render_onvif_details"]

