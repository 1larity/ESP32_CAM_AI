# recorder.py
# Per-camera prebuffer recorder with flush-on-start.
from __future__ import annotations
from collections import deque
from pathlib import Path
import cv2 as cv
from typing import Optional
import numpy as np
from utils import timestamp_name, ensure_dir

class PrebufferRecorder:
    def __init__(self, cam_name: str, out_dir: Path, fps: int = 25, pre_ms: int = 3000):
        self.cam_name = cam_name
        self.out_dir = Path(out_dir)
        self.fps = fps
        self.pre_ms = pre_ms
        self.buf = deque()  # (ts_ms, bgr)
        self.writer: Optional[cv.VideoWriter] = None
        self.size = None

    def on_frame(self, bgr: np.ndarray, ts_ms: int):
        self.size = (bgr.shape[1], bgr.shape[0])
        self.buf.append((ts_ms, bgr.copy()))
        # Trim buffer
        while self.buf and ts_ms - self.buf[0][0] > self.pre_ms:
            self.buf.popleft()
        if self.writer is not None:
            self.writer.write(bgr)

    def start(self):
        if self.writer is not None:
            return
        ensure_dir(self.out_dir)
        fname = f"{self.cam_name}_{timestamp_name()}.avi"
        path = str(self.out_dir / fname)
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        self.writer = cv.VideoWriter(path, fourcc, self.fps, self.size)
        for _, b in self.buf:
            self.writer.write(b)

    def stop(self):
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None

    def close(self):
        self.stop()
        self.buf.clear()
