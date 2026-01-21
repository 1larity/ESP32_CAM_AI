from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DetectorConfig:
    yolo_model: str
    yolo_conf: float = 0.35
    yolo_nms: float = 0.45
    interval_ms: int = 100
    face_cascade: Optional[str] = None
    face_model: Optional[str] = None
    use_lbph: bool = True
    use_gpu: bool = False

    @classmethod
    def from_app(cls, app_cfg):
        m = app_cfg.models_dir
        yolo_path = m / "yolo11n.onnx"
        return cls(
            yolo_model=str(Path(yolo_path).resolve()),
            yolo_conf=app_cfg.thresh_yolo,
            yolo_nms=0.45,
            interval_ms=getattr(app_cfg, "detect_interval_ms", 100),
            face_cascade=str((m / "haarcascade_frontalface_default.xml").resolve()),
            face_model=str(
                Path(getattr(app_cfg, "face_model", (m / "face_yunet.onnx"))).resolve()
            ),
            use_lbph=not getattr(app_cfg, "ignore_enrollment_models", False),
            use_gpu=bool(getattr(app_cfg, "use_gpu", False)),
        )
