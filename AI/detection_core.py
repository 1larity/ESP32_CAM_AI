from __future__ import annotations

from typing import Dict, List, Tuple
import time

import cv2
import numpy as np

from detection_packet import DetBox

# Classes we care about from COCO
COCO_ID_TO_NAME: Dict[int, str] = {0: "person", 15: "cat", 16: "dog"}


def run_yolo(
    net: cv2.dnn_Net,
    bgr: np.ndarray,
    conf_thresh: float,
    nms_thresh: float,  # kept for future use, not currently applied
) -> Tuple[List[DetBox], List[DetBox], int]:
    """
    Run YOLO ONNX model and return (all_yolo_boxes, pet_boxes, elapsed_ms).

    This is a functional extraction of the YOLO block from DetectorThread.run.
    """
    start = time.monotonic()

    H, W = bgr.shape[:2]

    # Letterbox to 640x640 like original _letterbox implementation
    img, r, dx, dy = _letterbox(bgr, new_shape=640)

    blob = cv2.dnn.blobFromImage(
        img, 1 / 255.0, (640, 640), swapRB=True, crop=False
    )
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
        if conf < conf_thresh:
            continue
        boxes.append((float(cx), float(cy), float(w), float(h)))
        scores.append(conf)
        ids.append(c)

    # Map back to original image coordinates and filter by COCO classes of interest
    yolo_boxes: List[DetBox] = []
    pet_boxes: List[DetBox] = []

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
        yolo_boxes.append(box)
        if label in ("cat", "dog"):
            pet_boxes.append(box)

    elapsed_ms = int((time.monotonic() - start) * 1000.0)
    return yolo_boxes, pet_boxes, elapsed_ms


def _letterbox(img: np.ndarray, new_shape=640, color=114):
    """
    Match original YOLODetector._letterbox: square 640x640 with padding.
    Returns (padded_image, scale, dx, dy).
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape, new_shape, 3), color, np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top
