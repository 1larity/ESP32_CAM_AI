# detectors.py
# Letterbox-correct YOLO, plus face recognition inference using LBPH.
from __future__ import annotations
import os, threading, time, json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import cv2 as cv, numpy as np
from PyQt6 import QtCore
from utils import monotonic_ms

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
            interval_ms=app_cfg.detect_interval_ms,
            face_cascade=str((m / "haarcascade_frontalface_default.xml").resolve()),
        )

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

COCO_ID_TO_NAME = {0: "person", 15: "cat", 16: "dog"}

def _letterbox(img: np.ndarray, new_shape=(640, 640), color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(new_shape[1]/h, new_shape[0]/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    pad_top = (new_shape[1]-nh)//2
    pad_left = (new_shape[0]-nw)//2
    resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_LINEAR)
    canvas = np.full((new_shape[1], new_shape[0], 3), color, dtype=resized.dtype)
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
    return canvas, r, pad_left, pad_top

class DetectorThread(QtCore.QThread):
    resultsReady = QtCore.pyqtSignal(DetectionPacket)

    def __init__(self, cfg: DetectorConfig, name: str):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self._latest = None
        self._lock = threading.RLock()
        self._stop = threading.Event()

        # YOLO
        self._net = None
        if os.path.exists(self.cfg.yolo_model):
            try:
                self._net = cv.dnn.readNet(self.cfg.yolo_model)
            except Exception:
                self._net = None

        # Faces: detector
        self._face = None
        if self.cfg.face_cascade and os.path.exists(self.cfg.face_cascade):
            self._face = cv.CascadeClassifier(self.cfg.face_cascade)

        # Faces: recognizer
        self._rec = None
        self._labels: Dict[int, str] = {}
        self._load_face_recognizer()

    def _load_face_recognizer(self):
        try:
            model_path = os.path.join(os.path.dirname(self.cfg.yolo_model), "lbph_faces.xml")
            labels_path = os.path.join(os.path.dirname(self.cfg.yolo_model), "labels_faces.json")
            if os.path.exists(model_path):
                self._rec = cv.face.LBPHFaceRecognizer_create()
                self._rec.read(model_path)
            if os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as fp:
                    m = json.load(fp)
                # stored as {name: id}; invert
                self._labels = {int(v): k for k, v in m.items()}
        except Exception:
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
                time.sleep(max(0, (next_due - now)/1000.0))
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

            # --- YOLO with true letterbox mapping back to original image ---
            if self._net is not None:
                try:
                    lb, r, padl, padt = _letterbox(bgr, (640, 640))
                    blob = cv.dnn.blobFromImage(lb, 1/255.0, (640, 640), swapRB=True, crop=False)
                    self._net.setInput(blob)
                    out = self._net.forward()
                    boxes, scores, ids = self._parse_yolov8(out)  # in 640-letterboxed space
                    # Map back to original image using r and pads
                    mapped = []
                    for (cx, cy, ww, hh), sc, cid in boxes:
                        # Accept both normalized [0..1] and 640-pixel coords
                        mx = max(abs(cx), abs(cy), abs(ww), abs(hh))
                        if mx <= 1.5:  # normalized
                            cx, cy, ww, hh = cx*640.0, cy*640.0, ww*640.0, hh*640.0
                        # de-letterbox
                        x1 = (cx - ww/2) - padl
                        y1 = (cy - hh/2) - padt
                        x2 = (cx + ww/2) - padl
                        y2 = (cy + hh/2) - padt
                        x1 = int(np.clip(x1 / r, 0, W-1))
                        y1 = int(np.clip(y1 / r, 0, H-1))
                        x2 = int(np.clip(x2 / r, 0, W-1))
                        y2 = int(np.clip(y2 / r, 0, H-1))
                        bw, bh = x2 - x1, y2 - y1
                        if bw <= 2 or bh <= 2:
                            continue
                        mapped.append(([x1, y1, bw, bh], sc, cid))
                    if mapped:
                        nms_boxes = [m[0] for m in mapped]
                        nms_scores = [float(m[1]) for m in mapped]
                        idxs = cv.dnn.NMSBoxes(nms_boxes, nms_scores, self.cfg.yolo_conf, self.cfg.yolo_nms)
                        if len(idxs) > 0:
                            for i in idxs.flatten():
                                x, y, bw, bh = nms_boxes[i]
                                cid = int(mapped[i][2])
                                pkt.yolo.append(DetBox(COCO_ID_TO_NAME.get(cid, str(cid)),
                                                       float(nms_scores[i]), (x, y, x+bw, y+bh)))
                except Exception:
                    pass

            pkt.timing_ms["det"] = monotonic_ms() - t0

            # --- Haar faces + LBPH recognition ---
            if self._face is not None:
                try:
                    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
                    faces = self._face.detectMultiScale(gray, 1.2, 4, minSize=(32, 32))
                    for (fx, fy, fw, fh) in faces:
                        name = "face"
                        score = 1.0
                        if self._rec is not None:
                            try:
                                roi = cv.resize(gray[fy:fy+fh, fx:fx+fw], (128, 128), interpolation=cv.INTER_AREA)
                                lab, conf = self._rec.predict(roi)
                                # LBPH: lower conf is better. Empirical threshold ~65–85.
                                if conf <= 75 and lab in self._labels:
                                    name = self._labels[lab]
                                    # Map LBPH conf to 0..1 score for display
                                    score = float(max(0.0, min(1.0, 1.0 - (conf / 100.0))))
                            except Exception:
                                pass
                        pkt.faces.append(DetBox(name, score, (fx, fy, fx+fw, fy+fh)))
                except Exception:
                    pass

            self.resultsReady.emit(pkt)

    def _parse_yolov8(self, out: np.ndarray):
        """Return list of (cx,cy,w,h),score,class_id in the 640x640 letterbox space.
        Handles (1,84,N) and (1,N,85). Does NOT scale to image here."""
        out = out
        dets = []
        if out.ndim == 3 and out.shape[1] >= 84:  # (1,84,N)
            a = out[0]
            xywh = a[0:4, :]
            cls = a[4:, :]
            cls_ids = np.argmax(cls, axis=0)
            cls_scores = cls.max(axis=0)
            sel = cls_scores >= self.cfg.yolo_conf
            idxs = np.where(sel)[0]
            for i in idxs:
                cx, cy, ww, hh = [float(v) for v in xywh[:, i]]
                dets.append(((cx, cy, ww, hh), float(cls_scores[i]), int(cls_ids[i])))
        elif out.ndim == 3 and out.shape[2] >= 85:  # (1,N,85)
            a = out[0]
            xywh = a[:, 0:4]
            cls = a[:, 5:]
            cls_ids = np.argmax(cls, axis=1)
            cls_scores = cls.max(axis=1)
            sel = cls_scores >= self.cfg.yolo_conf
            idxs = np.where(sel)[0]
            for i in idxs:
                cx, cy, ww, hh = [float(v) for v in xywh[i]]
                dets.append(((cx, cy, ww, hh), float(cls_scores[i]), int(cls_ids[i])))
        return dets
