from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import requests


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ip: str
    port: int
    path: str
    status_code: int
    auth_required: bool
    elapsed_ms: int
    name: Optional[str] = None


def probe_esp32_cam(
    ip: str,
    *,
    session: requests.Session,
    port: int = 80,
    timeout: float = 0.6,
    path: str = "/api/status",
) -> ProbeResult | None:
    """
    Probe a single IP for an ESP32-CAM by checking ONLY the status endpoint.

    Hit conditions:
      - Any 2xx response
      - Any 401 response (treated as 'auth required' but still a hit)

    Name extraction:
      - If 2xx and JSON contains 'name' or 'camera'
    """
    url = f"http://{ip}:{port}{path}"
    t0 = time.perf_counter()
    try:
        r = session.get(url, timeout=timeout)
    except Exception:
        return None
    elapsed_ms = int((time.perf_counter() - t0) * 1000.0)

    if (200 <= r.status_code < 300) or (r.status_code == 401):
        name: Optional[str] = None
        if 200 <= r.status_code < 300:
            try:
                data = r.json()
                name = data.get("name") or data.get("camera")
            except Exception:
                name = None

        return ProbeResult(
            ip=ip,
            port=port,
            path=path,
            status_code=r.status_code,
            auth_required=(r.status_code == 401),
            elapsed_ms=elapsed_ms,
            name=name,
        )

    return None
