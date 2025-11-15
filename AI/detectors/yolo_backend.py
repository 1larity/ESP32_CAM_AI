# detectors/yolo_backend.py
from __future__ import annotations
from typing import List, Tuple
import os

import cv2
import numpy as np

from .packet import DetBox, DetectionPacket
from .config import DetectorConfig, COCO_ID_TO_NAME, letterbox_square


def load_yolo(cfg: DetectorConfig):
    if not os.path.exists(cfg.yolo_model):
        print(f"[YOLO] model not found at {cfg.yolo_model}")
        return None
    try:
        net = cv2.dnn.readNetFromONNX(cfg.yolo_model)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except Exception as e:
        print(f"[YOLO] load failed: {e}")
        return None


def run_yolo(net, cfg: DetectorConfig, bgr, pkt: DetectionPacket) -> int:
    if net is None:
        return 0
    H, W = bgr.shape[:2]
    img, r, dx, dy = letterbox_square(bgr, new_shape=640)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
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
        if conf < cfg.yolo_conf:
            continue
        boxes.append((float(cx), float(cy), float(w), float(h)))
        scores.append(conf)
        ids.append(c)

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

    # no easy split timing here; we return 0 and let caller time externally
    return 0
