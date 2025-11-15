# detectors/thread.py
from __future__ import annotations
import os
import threading
import time
from typing import Optional, Tuple

import numpy as np
from PyQt6 import QtCore

from .config import DetectorConfig
from .packet import DetectionPacket
from .yolo_backend import load_yolo, run_yolo
from .faces_backend import FaceBackend
from utils import monotonic_ms


class DetectorThread(QtCore.QThread):
    resultsReady = QtCore.pyqtSignal(object)

    def __init__(self, cfg: DetectorConfig, name: str):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self._latest: Optional[Tuple[np.ndarray, int]] = None
        self._lock = threading.RLock()
        self._stop = threading.Event()

        self.models_dir = os.path.dirname(self.cfg.yolo_model)
        self._net = load_yolo(self.cfg)
        self._faces = FaceBackend(self.cfg, self.models_dir)

        print(
            f"[Detector:{self.name}] init: "
            f"net={'OK' if self._net is not None else 'NONE'}, "
            f"face={'OK' if self._faces.cascade is not None else 'NONE'}, "
            f"lbph={'OK' if self._faces.lbph.recognizer is not None else 'NONE'}, "
            f"yolo_conf={self.cfg.yolo_conf}"
        )

    def submit_frame(self, cam_name: str, bgr: np.ndarray, ts_ms: int):
        if cam_name != self.name:
            return
        with self._lock:
            self._latest = (bgr.copy(), ts_ms)

    def stop(self):
        self._stop.set()

    def run(self):
        next_due = 0
        while not self._stop.is_set():
            now = monotonic_ms()
            if now < next_due:
                time.sleep(max(0, (next_due - now) / 1000.0))
                continue
            next_due = now + self.cfg.interval_ms

            with self._lock:
                snap = self._latest
            if snap is None:
                continue

            bgr, ts_ms = snap
            H, W = bgr.shape[:2]
            pkt = DetectionPacket(self.name, ts_ms, (W, H))

            t0 = monotonic_ms()
            run_yolo(self._net, self.cfg, bgr, pkt)
            t1 = monotonic_ms()
            self._faces.run(bgr, pkt)
            t2 = monotonic_ms()

            pkt.timing_ms["yolo"] = int(t1 - t0)
            pkt.timing_ms["faces"] = int(t2 - t1)

            self.resultsReady.emit(pkt)
