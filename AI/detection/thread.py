from __future__ import annotations
import os
import threading
from collections import deque
import time
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal, Slot
from utils import monotonic_ms, debug
from .core import run_yolo
from .lbph import load_lbph, run_faces
from .packet import DetectionPacket

# Limit concurrent YOLO runs to reduce CPU contention across cameras.
YOLO_SEMAPHORE = threading.Semaphore(1)

# Keep OpenCV from spinning up many worker threads; helps reduce contention.
try:
    cv2.setNumThreads(1)
except Exception:
    pass


@dataclass
class DetectorConfig:
    yolo_model: str
    yolo_conf: float = 0.35
    yolo_nms: float = 0.45
    interval_ms: int = 100
    face_cascade: Optional[str] = None

    @classmethod
    def from_app(cls, app_cfg):
        m = app_cfg.models_dir
        return cls(
            yolo_model=str((m / "yolov8n.onnx").resolve()),
            yolo_conf=app_cfg.thresh_yolo,
            yolo_nms=0.45,
            interval_ms=getattr(app_cfg, "detect_interval_ms", 100),
            face_cascade=str((m / "haarcascade_frontalface_default.xml").resolve()),
        )


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
        self._last_yolo = None  # (ts_ms, yolo_boxes, pet_boxes)

        self.models_dir = os.path.dirname(self.cfg.yolo_model)

        self._net = None
        if os.path.exists(self.cfg.yolo_model):
            try:
                self._net = cv2.dnn.readNetFromONNX(self.cfg.yolo_model)
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as e:
                print(f"[Detector:{self.name}] YOLO load failed: {e}")
                self._net = None
        else:
            print(f"[Detector:{self.name}] YOLO model not found at {self.cfg.yolo_model}")

        self._face = None
        if self.cfg.face_cascade and os.path.exists(self.cfg.face_cascade):
            try:
                self._face = cv2.CascadeClassifier(self.cfg.face_cascade)
            except Exception as e:
                print(f"[Detector:{self.name}] Haar load failed: {e}")
                self._face = None
        else:
            print(f"[Detector:{self.name}] Haar cascade not found at {self.cfg.face_cascade}")

        self._rec, self._labels = load_lbph(self.models_dir)

        print(
            f"[Detector:{self.name}] init: net={'OK' if self._net is not None else 'NONE'}, "
            f"face={'OK' if self._face is not None else 'NONE'}, "
            f"lbph={'OK' if self._rec is not None else 'NONE'}, "
            f"yolo_conf={self.cfg.yolo_conf}"
        )

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
                print(f"[Detector:{getattr(self, 'name', '')}] stop(): thread did not exit within {wait_ms} ms")
   
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
            H, W = bgr.shape[:2]
            pkt = DetectionPacket(self.name, ts_ms, (W, H))
            t_start = monotonic_ms()
            t0 = t_start

            # Reuse last YOLO results if we must skip and they are fresh.
            reuse_yolo = False
            yolo_skipped = False

            if self._net is not None:
                # Try to acquire YOLO slot; wait briefly before skipping.
                acquired = YOLO_SEMAPHORE.acquire(timeout=0.05)
                if acquired:
                    try:
                        yolo_boxes, pet_boxes, t_yolo = run_yolo(
                            self._net, bgr, self.cfg.yolo_conf, self.cfg.yolo_nms
                        )
                        pkt.yolo.extend(yolo_boxes)
                        pkt.pets.extend(pet_boxes)
                        pkt.timing_ms["yolo_core"] = t_yolo
                        self._last_yolo = (ts_ms, yolo_boxes, pet_boxes)
                    except Exception as e:
                        print(f"[Detector:{self.name}] YOLO error: {e}")
                    finally:
                        YOLO_SEMAPHORE.release()
                else:
                    pkt.timing_ms["yolo_core"] = 0  # skipped to avoid contention
                    yolo_skipped = True
                    if self._last_yolo:
                        last_ts, last_yolo_boxes, last_pet_boxes = self._last_yolo
                        if ts_ms - last_ts <= 2000:
                            reuse_yolo = True
                            pkt.yolo.extend(last_yolo_boxes)
                            pkt.pets.extend(last_pet_boxes)

            t1 = monotonic_ms()

            if self._face is not None:
                face_boxes, t_faces = run_faces(
                    bgr, self._face, self._rec, self._labels
                )
                pkt.faces.extend(face_boxes)
                pkt.timing_ms["faces_core"] = t_faces

            t2 = monotonic_ms()
            pkt.timing_ms["yolo"] = int(t1 - t0)
            pkt.timing_ms["faces"] = int(t2 - t1)

            # Dynamic backoff: if run exceeded 400ms, pause detection briefly.
            t_run = t2 - t_start
            if t_run > 400:
                self._backoff_until = t2 + min(t_run, 1500)
            elif yolo_skipped:
                # If we skipped YOLO due to contention, yield a short pause.
                self._backoff_until = t2 + 150

            # Throttled profiling to help diagnose stalls without flooding logs.
            now_ms = monotonic_ms()
            if now_ms >= self._profile_next_ms:
                debug(
                    f"[Detector {self.name}] yolo={len(pkt.yolo)} pets={len(pkt.pets)} "
                    f"faces={len(pkt.faces)} "
                    f"t_yolo={pkt.timing_ms.get('yolo', 0)}ms "
                    f"t_faces={pkt.timing_ms.get('faces', 0)}ms "
                    f"yolo_skipped={yolo_skipped}"
                )
                self._profile_next_ms = now_ms + 2000

            self.resultsReady.emit(pkt)
