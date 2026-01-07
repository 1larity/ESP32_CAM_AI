from __future__ import annotations

from typing import Optional
from urllib.parse import quote, urlsplit, urlunsplit


def guess_fallback_urls(ip: str) -> list[str]:
    if not ip:
        return []
    return [
        f"rtsp://{ip}:554/Streaming/Channels/101",
        f"rtsp://{ip}:554/Streaming/Channels/102",
        f"rtsp://{ip}:554/live/ch0",
        f"rtsp://{ip}:554/live/ch1",
    ]


def inject_auth(url: str | None, user: Optional[str], pwd: Optional[str]) -> str | None:
    if not url or not user or not pwd:
        return url
    try:
        parsed = urlsplit(url)
        if parsed.username:
            return url  # already has creds
        hostname = parsed.hostname or ""
        if not hostname:
            return url
        host = (
            f"[{hostname}]"
            if ":" in hostname and not hostname.startswith("[")
            else hostname
        )
        port = f":{parsed.port}" if parsed.port else ""
        userinfo = f"{quote(user, safe='')}:{quote(pwd, safe='')}@"
        netloc = f"{userinfo}{host}{port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except Exception:
        return url

