# detectors/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

# COCO subset we care about
COCO_ID_TO_NAME: Dict[int, str] = {0: "person", 15: "cat", 16: "dog"}


@dataclass
class DetectorConfig:
    yolo_model: str
    yolo_conf: float = 0.35
    yolo_nms: float = 0.45
    interval_ms: int = 100
    face_cascade: str | None = None

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


def letterbox_square(
    img, new_shape: int = 640, color: int = 114
) -> tuple:
    """Square letterbox to new_shape x new_shape with padding."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape, new_shape, 3), color, np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top
