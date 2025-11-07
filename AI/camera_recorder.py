
"""
CameraRecorder: pre‑event buffered video recorder for PySide6 apps.

- Ring buffer of (ts_ms, frame_bgr) in RAM.
- On start(): flushes last N seconds then appends live frames.
- Background writer thread with target FPS and codec fallback.
- Safe write to .tmp then rename on stop().
"""

from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple, List

import numpy as np
import cv2


@dataclass
class RecorderConfig:
    out_dir: Path
    prebuffer_sec: int = 5
    fps: int = 20
    codec_primary: str = "MJPG"
    codec_fallback: str = "mp4v"
    max_ram_mb: int = 256  # soft cap for ring buffer


class CameraRecorder:
    def __init__(self, name: str, cfg: RecorderConfig):
        self.name = name
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.out_dir / ".tmp").mkdir(parents=True, exist_ok=True)

        self._ring: Deque[Tuple[int, np.ndarray]] = deque()
        self._ring_lock = threading.Lock()
        self._ring_bytes = 0

        self._writer = None
        self._writer_path_tmp: Optional[Path] = None
        self._writer_path_final: Optional[Path] = None
        self._writer_lock = threading.Lock()
        self._running = False
        self._write_thread: Optional[threading.Thread] = None
        self._q: Deque[Tuple[int, np.ndarray]] = deque()
        self._q_lock = threading.Lock()

        self._last_written_ts = 0
        self._target_dt = int(1000 / max(1, self.cfg.fps))

        self._w = None
        self._h = None

    # ---- ingestion ----
    def ingest_frame(self, frame_bgr: np.ndarray, ts_ms: Optional[int] = None):
        if frame_bgr is None or frame_bgr.size == 0:
            return
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)

        h, w = frame_bgr.shape[:2]
        if self._w is None:
            self._w, self._h = int(w), int(h)

        # update ring buffer
        b = int(frame_bgr.nbytes) + 16
        with self._ring_lock:
            self._ring.append((ts_ms, frame_bgr.copy()))
            self._ring_bytes += b
            # trim by time
            cutoff = ts_ms - self.cfg.prebuffer_sec * 1000
            while self._ring and self._ring[0][0] < cutoff:
                _, old = self._ring.popleft()
                self._ring_bytes -= int(old.nbytes) + 16
            # trim by RAM
            cap_bytes = self.cfg.max_ram_mb * 1024 * 1024
            while self._ring and self._ring_bytes > cap_bytes:
                _, old = self._ring.popleft()
                self._ring_bytes -= int(old.nbytes) + 16

        # enqueue to writer if recording
        if self._running:
            with self._q_lock:
                self._q.append((ts_ms, frame_bgr.copy()))

    # ---- control ----
    def start(self) -> Path:
        if self._running:
            return self._writer_path_final or self._writer_path_tmp or Path()

        now = time.localtime()
        stamp = time.strftime("%Y%m%d-%H%M%S", now)
        base = f"{self._sanitize(self.name)}_{stamp}"
        tmp = self.cfg.out_dir / ".tmp" / f"{base}.avi"
        final = self.cfg.out_dir / f"{base}.avi"
        self._writer_path_tmp = tmp
        self._writer_path_final = final

        # open writer lazily after we know size
        if self._w is None or self._h is None:
            # fallback default if no frames yet
            self._w, self._h = 640, 480

        w_even = self._w - (self._w % 2)
        h_even = self._h - (self._h % 2)
        if w_even <= 0 or h_even <= 0:
            w_even, h_even = 640, 480

        self._open_writer(tmp, w_even, h_even)

        # seed queue with ring buffer content near target FPS
        seed = []
        with self._ring_lock:
            seed = list(self._ring)
        seed = self._resample(seed, self.cfg.fps)
        with self._q_lock:
            self._q.extend(seed)

        self._running = True
        self._last_written_ts = 0
        self._write_thread = threading.Thread(target=self._writer_loop, name=f"{self.name}-recorder", daemon=True)
        self._write_thread.start()
        return final

    def stop(self) -> Optional[Path]:
        if not self._running:
            return self._writer_path_final
        self._running = False
        # wait for thread
        if self._write_thread:
            self._write_thread.join(timeout=5.0)
        self._write_thread = None

        # close writer
        with self._writer_lock:
            try:
                if self._writer:
                    self._writer.release()
            finally:
                self._writer = None

        # rename tmp -> final if tmp exists and is non-empty
        try:
            if self._writer_path_tmp and self._writer_path_tmp.exists():
                if self._writer_path_tmp.stat().st_size > 0:
                    if self._writer_path_final:
                        try:
                            self._writer_path_final.unlink()
                        except FileNotFoundError:
                            pass
                        self._writer_path_tmp.replace(self._writer_path_final)
                        return self._writer_path_final
        except Exception:
            pass
        return self._writer_path_tmp

    # ---- internals ----
    def _open_writer(self, path: Path, w: int, h: int):
        fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec_primary)
        writer = cv2.VideoWriter(str(path), fourcc, float(self.cfg.fps), (w, h))
        if not writer.isOpened():
            # fallback to mp4 in tmp folder
            alt = path.with_suffix(".mp4")
            fourcc2 = cv2.VideoWriter_fourcc(*self.cfg.codec_fallback)
            writer2 = cv2.VideoWriter(str(alt), fourcc2, float(self.cfg.fps), (w, h))
            if writer2.isOpened():
                self._writer_path_tmp = alt
                writer = writer2
            else:
                raise RuntimeError("Failed to open VideoWriter for both MJPG and mp4v")
        with self._writer_lock:
            self._writer = writer

    def _writer_loop(self):
        target_dt = self._target_dt
        while self._running or self._q:
            item = None
            with self._q_lock:
                if self._q:
                    item = self._q.popleft()
            if item is None:
                time.sleep(0.005)
                continue

            ts, frame = item
            # resample live to target FPS using timestamps
            if self._last_written_ts == 0 or ts - self._last_written_ts >= target_dt:
                self._write_frame(frame)
                self._last_written_ts = ts
            else:
                # drop frame
                continue

    def _write_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return
        h, w = frame.shape[:2]
        w_even = w - (w % 2)
        h_even = h - (h % 2)
        if (w_even != w) or (h_even != h):
            frame = cv2.resize(frame, (w_even, h_even), interpolation=cv2.INTER_AREA)
        with self._writer_lock:
            if self._writer:
                self._writer.write(frame)

    @staticmethod
    def _resample(samples: List[Tuple[int, np.ndarray]], fps: int):
        """Return a list of frames at ~fps from timestamped samples."""
        if not samples:
            return []
        target_dt = int(1000 / max(1, fps))
        out: List[Tuple[int, np.ndarray]] = []
        last_ts = 0
        for ts, frame in samples:
            if not out:
                out.append((ts, frame))
                last_ts = ts
            else:
                if ts - last_ts >= target_dt:
                    out.append((ts, frame))
                    last_ts = ts
        return out

    @staticmethod
    def _sanitize(name: str) -> str:
        keep = []
        for ch in name:
            if ch.isalnum() or ch in ("-", "_"):
                keep.append(ch)
            elif ch == " ":
                keep.append("_")
        return "".join(keep)[:64]
