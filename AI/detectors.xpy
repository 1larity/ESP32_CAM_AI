# detectors.py
# YOLO (ONNX) + Haar cascade + LBPH recognition, aligned to original monolithic MDI app.

from __future__ import annotations
import os
import threading
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PyQt6 import QtCore

from utils import monotonic_ms

# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------

@dataclass
class DetBox:
    cls: str
    score: float
    xyxy: Tuple[int, int, int, int]


@dataclass
class DetectionPacket:
    name: str
    ts_ms: int
    size: Tuple[int, int]
    yolo: List[DetBox] = field(default_factory=list)
    faces: List[DetBox] = field(default_factory=list)
    pets: List[DetBox] = field(default_factory=list)
    timing_ms: Dict[str, int] = field(default_factory=dict)


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


COCO_ID_TO_NAME: Dict[int, str] = {0: "person", 15: "cat", 16: "dog"}


def _letterbox(img: np.ndarray, new_shape=640, color=114):
    """Match original YOLODetector._letterbox: square 640x640 with padding."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape, new_shape, 3), color, np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


# -------------------------------------------------------------------------
# Detector thread
# -------------------------------------------------------------------------

class DetectorThread(QtCore.QThread):
    # Emit DetectionPacket as a generic Python object
    resultsReady = QtCore.pyqtSignal(object)

    def __init__(self, cfg: DetectorConfig, name: str):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self._latest: Optional[Tuple[np.ndarray, int]] = None
        self._lock = threading.RLock()
        self._stop = threading.Event()

        # Derive models_dir from YOLO path (matches existing layout)
        self.models_dir = os.path.dirname(self.cfg.yolo_model)

        # YOLO model
        self._net = None
        if os.path.exists(self.cfg.yolo_model):
            try:
                # match original: readNetFromONNX + CPU target
                self._net = cv2.dnn.readNetFromONNX(self.cfg.yolo_model)
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as e:
                print(f"[Detector:{self.name}] YOLO load failed: {e}")
                self._net = None
        else:
            print(f"[Detector:{self.name}] YOLO model not found at {self.cfg.yolo_model}")

        # Face cascade
        self._face = None
        if self.cfg.face_cascade and os.path.exists(self.cfg.face_cascade):
            try:
                self._face = cv2.CascadeClassifier(self.cfg.face_cascade)
            except Exception as e:
                print(f"[Detector:{self.name}] Haar load failed: {e}")
                self._face = None
        else:
            print(f"[Detector:{self.name}] Haar cascade not found at {self.cfg.face_cascade}")

        # LBPH recogniser + labels (as in EnrollmentService)
        self._rec = None
        self._labels: Dict[int, str] = {}
        self._load_lbph()

        print(
            f"[Detector:{self.name}] init: net={'OK' if self._net is not None else 'NONE'}, "
            f"face={'OK' if self._face is not None else 'NONE'}, "
            f"lbph={'OK' if self._rec is not None else 'NONE'}, "
            f"yolo_conf={self.cfg.yolo_conf}"
        )

    def _load_lbph(self):
        model_path = os.path.join(self.models_dir, "lbph_faces.xml")
        labels_path = os.path.join(self.models_dir, "labels_faces.json")
        try:
            if os.path.exists(model_path):
                # requires opencv-contrib-python
                self._rec = cv2.face.LBPHFaceRecognizer_create()  # type: ignore[attr-defined]
                self._rec.read(model_path)
            if os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as fp:
                    m = json.load(fp)
                # stored as {name: id}; invert
                self._labels = {int(v): k for k, v in m.items()}
        except Exception as e:
            print(f"[Detector:{self.name}] LBPH disabled ({e})")
            self._rec = None
            self._labels = {}

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

            # --- YOLO, following original YOLODetector.detect semantics ---
            if self._net is not None:
                try:
                    img, r, dx, dy = _letterbox(bgr, new_shape=640)
                except Exception as e:
                    print(f"[Detector:{self.name}] letterbox error: {e}")
                    img, r, dx, dy = _letterbox(bgr, new_shape=640)

                blob = cv2.dnn.blobFromImage(
                    img, 1 / 255.0, (640, 640), swapRB=True, crop=False
                )
                self._net.setInput(blob)
                out = self._net.forward()
                out = np.squeeze(out)

                if out.ndim == 2 and out.shape[0] in (84, 85):
                    out = out.T
                elif out.ndim == 3:
                    o = out[0]
                    out = o.T if o.shape[0] in (84, 85) else o

                boxes: List[Tuple[float, float, float, float]] = []
                scores: List[float] = []
                ids: List[int] = []

                for det in out:
                    det = np.asarray(det).ravel()
                    if det.shape[0] < 5:
                        continue
                    cx, cy, w, h = det[:4]
                    if det.shape[0] >= 85:
                        obj = float(det[4])
                        cls_scores = det[5:]
                        c = int(np.argmax(cls_scores))
                        conf = obj * float(cls_scores[c])
                    else:
                        c = int(det[4])
                        conf = float(det[5]) if det.shape[0] > 5 else 0.0
                    if conf < self.cfg.yolo_conf:
                        continue
                    boxes.append((float(cx), float(cy), float(w), float(h)))
                    scores.append(conf)
                    ids.append(c)

                # Map back to original image coordinates and filter by COCO classes of interest
                for (cx, cy, w, h), conf, cid in zip(boxes, scores, ids):
                    if cid not in COCO_ID_TO_NAME:
                        continue
                    cx0 = (cx - dx) / r
                    cy0 = (cy - dy) / r
                    w0 = w / r
                    h0 = h / r
                    x1 = max(0, int(cx0 - w0 / 2))
                    y1 = max(0, int(cy0 - h0 / 2))
                    x2 = min(W - 1, int(cx0 + w0 / 2))
                    y2 = min(H - 1, int(cy0 + h0 / 2))
                    label = COCO_ID_TO_NAME[cid]
                    box = DetBox(label, float(conf), (x1, y1, x2, y2))
                    pkt.yolo.append(box)
                    if label in ("cat", "dog"):
                        pkt.pets.append(box)

            t1 = monotonic_ms()

            # --- Faces + LBPH; matches original FaceDB.detect_faces / recognize_roi ---
            if self._face is not None:
                try:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    try:
                        eq = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
                    except Exception:
                        eq = cv2.equalizeHist(gray)

                    minsz = max(40, int(0.12 * min(gray.shape[:2])))
                    faces = self._face.detectMultiScale(eq, 1.1, 4, minSize=(minsz, minsz))
                    if len(faces) == 0:
                        faces = self._face.detectMultiScale(eq, 1.05, 3, minSize=(minsz, minsz))

                    for (fx, fy, fw, fh) in faces:
                        name = "face"
                        score = 0.6
                        if self._rec is not None:
                            try:
                                roi = gray[fy:fy + fh, fx:fx + fw]
                                roi = cv2.resize(roi, (160, 160))
                                pred, dist = self._rec.predict(roi)
                                if 0 <= pred < len(self._labels) and dist <= 95.0:
                                    label_name = self._labels.get(int(pred), "face")
                                    name = label_name
                                    score = max(0.0, min(1.0, 1.0 - (dist / 95.0)))
                                else:
                                    name = "unknown"
                                    score = 0.4
                                print(
                                    f"[Detector:{self.name}] LBPH pred={pred} "
                                    f"name={self._labels.get(int(pred), '?')} dist={dist:.1f} -> {name}"
                                )
                            except Exception as e:
                                print(f"[Detector:{self.name}] LBPH predict error: {e}")
                                name = "face"
                                score = 0.6

                        x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
                        pkt.faces.append(DetBox(name, float(score), (x1, y1, x2, y2)))

                except Exception as e:
                    print(f"[Detector:{self.name}] face error: {e}")

            t2 = monotonic_ms()
            pkt.timing_ms["yolo"] = int(t1 - t0)
            pkt.timing_ms["faces"] = int(t2 - t1)

            self.resultsReady.emit(pkt)