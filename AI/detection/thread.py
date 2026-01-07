from __future__ import annotations

import threading
import time
from collections import deque

from PySide6 import QtCore

from utils import monotonic_ms, debug
from .config import DetectorConfig
from .engine import DetectorEngine


class DetectorThread(QtCore.QThread):
    resultsReady = QtCore.Signal(object)

    def __init__(self, cfg: DetectorConfig, name: str):
        super().__init__()
        self.cfg = cfg
        self.name = name
        # Latest frames handed off from UI thread; deque(maxlen=1) keeps only newest.
        self._frames = deque(maxlen=1)
        self._stop = threading.Event()
        self._profile_next_ms = 0
        self._backoff_until = 0

        self._engine = DetectorEngine(cfg, name)

    def submit_frame(self, *args) -> None:
        if len(args) == 2:
            bgr, ts_ms = args
        elif len(args) == 3:
            _, bgr, ts_ms = args
        else:
            raise TypeError(
                "submit_frame expected (bgr, ts_ms) or (name, bgr, ts_ms), "
                f"got {len(args)} positional arguments"
            )

        # Drop older frames; detector only needs the most recent snapshot.
        self._frames.append((bgr, ts_ms))

    def stop(self, wait_ms: int = 0) -> None:
        self._stop.set()
        if wait_ms and self.isRunning() and QtCore.QThread.currentThread() != self:
            if not self.wait(wait_ms):
                print(
                    f"[Detector:{getattr(self, 'name', '')}] stop(): thread did not exit within {wait_ms} ms"
                )

    def run(self) -> None:
        next_due = 0
        while not self._stop.is_set():
            now = monotonic_ms()
            if now < getattr(self, "_backoff_until", 0):
                time.sleep(0.05)
                continue
            if now < next_due:
                time.sleep(max(0, (next_due - now) / 1000.0))
                continue
            next_due = now + self.cfg.interval_ms

            if not self._frames:
                continue

            snap_bgr, ts_ms = self._frames.pop()
            bgr = snap_bgr.copy()

            pkt, yolo_skipped, end_ms = self._engine.process_frame(bgr, ts_ms)

            # Dynamic backoff: if run exceeded 400ms, pause detection briefly.
            t_run = int(pkt.timing_ms.get("yolo", 0) + pkt.timing_ms.get("faces", 0))
            if t_run > 400:
                self._backoff_until = end_ms + min(t_run, 1500)
            elif yolo_skipped:
                # If we skipped YOLO due to contention, yield a short pause.
                self._backoff_until = end_ms + 150

            # Throttled profiling to help diagnose stalls without flooding logs.
            now_ms = monotonic_ms()
            if now_ms >= self._profile_next_ms:
                debug(
                    f"[Detector {self.name}] yolo={len(pkt.yolo)} pets={len(pkt.pets)} "
                    f"faces={len(pkt.faces)} "
                    f"t_yolo={pkt.timing_ms.get('yolo', 0)}ms "
                    f"t_faces={pkt.timing_ms.get('faces', 0)}ms "
                    f"yolo_skipped={yolo_skipped} "
                    f"face_mode={self._engine.face_mode_label()}"
                )
                self._profile_next_ms = now_ms + 2000

            self.resultsReady.emit(pkt)


__all__ = ["DetectorThread", "DetectorConfig"]

