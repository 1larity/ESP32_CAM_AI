# stream.py
# Unified capture with robust timeouts, retries, and MJPEG fallback.
from __future__ import annotations
import threading
import queue
import time
from typing import Optional, Tuple, List
from urllib.parse import urlparse, urlunparse

import cv2 as cv
import numpy as np
import requests

from settings import CameraSettings
from utils import monotonic_ms

# Silence noisy OpenCV FFmpeg warnings (e.g., intermittent stream timeouts).
try:
    cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


class StreamCapture:
    def __init__(self, cam: CameraSettings):
        self.cam = cam
        self._stop = threading.Event()
        # Keep only the freshest frame; drop stale frames aggressively to avoid UI stalls.
        self._q: "queue.Queue[Tuple[bool, Optional[np.ndarray], int]]" = queue.Queue(maxsize=1)
        self._t: Optional[threading.Thread] = None
        self.last_backend = "init"

    def start(self):
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        try:
            ok, frame, ts = self._q.get(timeout=0.25)
            return ok, frame, ts
        except queue.Empty:
            return False, None, 0

    # ---------- internals ----------
    def _run(self):
        while not self._stop.is_set():
            success = False
            for url in self._candidate_urls():
                parsed = urlparse(url)
                try:
                    if parsed.scheme in ("rtsp",):
                        ok = self._run_opencv(url)
                        success = success or ok
                        if ok:
                            break
                    elif parsed.scheme in ("http", "https"):
                        ok = self._run_opencv(url)
                        if not ok:
                            ok = self._run_mjpeg(url)
                        success = success or ok
                        if ok:
                            break
                    else:
                        self._fail_once("bad-url")
                except Exception:
                    self._fail_once("exception")
                    continue
            if not success:
                self._fail_once("all-candidates")
            try:
                self._sleep_with_cancel(1.0)
            except Exception:
                self._sleep_with_cancel(1.0)

    def _candidate_urls(self) -> List[str]:
        """
        Return candidate URLs to try (primary, alt_streams, and common substream variants).
        """
        seen: set[str] = set()
        urls: List[str] = []

        def add(u: Optional[str]) -> None:
            if not u:
                return
            u = u.strip()
            if not u or u in seen:
                return
            seen.add(u)
            urls.append(u)

        add(self.cam.effective_url())
        for u in getattr(self.cam, "alt_streams", []) or []:
            add(u)

        # Heuristic variants (e.g., main/sub stream swaps)
        base = self.cam.effective_url()
        if base:
            variants = [
                ("/101", "/102"),
                ("/102", "/101"),
                ("/Streaming/Channels/101", "/Streaming/Channels/102"),
                ("/Streaming/Channels/1", "/Streaming/Channels/2"),
                ("/live/ch0", "/live/ch1"),
                ("/ch0", "/ch1"),
            ]
            for old, new in variants:
                if old in base:
                    add(base.replace(old, new, 1))
        return urls

    def _run_opencv(self, url: str) -> bool:
        cap = None
        self.last_backend = "cv-ffmpeg"
        # Basic auth inline if provided
        u = url
        if self.cam.user and self.cam.password:
            p = urlparse(url)
            netloc = f"{self.cam.user}:{self.cam.password}@{p.hostname or ''}"
            if p.port:
                netloc += f":{p.port}"
            u = urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
        try:
            cap = cv.VideoCapture(u, cv.CAP_FFMPEG)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                return False
            self._offer(True, frame, monotonic_ms())
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                self._offer(True, frame, monotonic_ms())
            cap.release()
        finally:
            if cap is not None:
                cap.release()
        return True

    def _run_mjpeg(self, url: str) -> bool:
        self.last_backend = "http-mjpeg"
        headers = {
            "Connection": "keep-alive",
            "Accept": "multipart/x-mixed-replace, image/jpeg, */*",
            "User-Agent": "ESP32-CAM-AI-Viewer/1.0",
        }
        auth_obj = None
        if self.cam.user and self.cam.password:
            auth_obj = requests.auth.HTTPBasicAuth(self.cam.user, self.cam.password)
        try:
            with requests.get(url, stream=True, auth=auth_obj, timeout=(5, 15), headers=headers) as r:
                r.raise_for_status()
                buf = bytearray()
                for chunk in r.iter_content(chunk_size=4096):
                    if self._stop.is_set():
                        return True
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    # crude JPEG scan
                    while True:
                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9")
                        if start != -1 and end != -1 and end > start:
                            jpg = bytes(buf[start:end+2])
                            del buf[:end+2]
                            frame = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv.IMREAD_COLOR)
                            if frame is not None:
                                self._offer(True, frame, monotonic_ms())
                            continue
                        break
            return True
        except requests.exceptions.ReadTimeout:
            self._fail_once("timeout")
            return False
        except Exception:
            self._fail_once("http-error")
            return False

    def _offer(self, ok: bool, frame: Optional[np.ndarray], ts_ms: int):
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put((ok, frame, ts_ms))

    def _fail_once(self, tag: str):
        self.last_backend = f"disconnected:{tag}"
        self._offer(False, None, 0)

    def _sleep_with_cancel(self, sec: float):
        t0 = time.time()
        while not self._stop.is_set() and time.time() - t0 < sec:
            time.sleep(0.05)
