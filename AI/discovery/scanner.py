from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .probe import ProbeResult, probe_esp32_cam


ProgressCb = Callable[[int, int, str], None]
ResultCb = Callable[[ProbeResult], None]


@dataclass(slots=True)
class ScanConfig:
    max_workers: int = 64
    timeout: float = 0.6
    port: int = 80
    path: str = "/api/status"


class RangeScanner:
    """
    Concurrent range scanner.

    Uses ThreadPoolExecutor so the scan feels asynchronous:
    - many probes in flight
    - progress ticks as each IP completes
    - results appear as soon as any worker finds a hit
    """

    def __init__(self, *, config: ScanConfig | None = None, stop_event: threading.Event | None = None) -> None:
        self.cfg = config or ScanConfig()
        self.stop_event = stop_event or threading.Event()
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def stop(self) -> None:
        self.stop_event.set()

    def scan_range(
        self,
        subnet_prefix: str,
        start: int,
        end: int,
        *,
        on_progress: ProgressCb,
        on_result: ResultCb,
    ) -> None:
        ips = [f"{subnet_prefix}{i}" for i in range(start, end + 1)]
        total = len(ips)
        if total == 0:
            return

        # Each task uses its own Session (requests.Session is not thread-safe).
        def task(ip: str) -> tuple[str, ProbeResult | None]:
            if self.stop_event.is_set():
                return ip, None
            sess = requests.Session()
            sess.headers.update({"User-Agent": "ESP32-CAM-Discovery/1.0"})
            res = probe_esp32_cam(
                ip,
                session=sess,
                port=self.cfg.port,
                timeout=self.cfg.timeout,
                path=self.cfg.path,
            )
            try:
                sess.close()
            except Exception:
                pass
            return ip, res

        done = 0
        workers = min(self.cfg.max_workers, total)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(task, ip) for ip in ips]

            for fut in as_completed(futures):
                if self.stop_event.is_set():
                    break

                ip, res = fut.result()
                done += 1

                if res is not None:
                    key = f"{res.ip}:{res.port}"
                    with self._lock:
                        if key not in self._seen:
                            self._seen.add(key)
                            on_result(res)

                on_progress(done, total, ip)
