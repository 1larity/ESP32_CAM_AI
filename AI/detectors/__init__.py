# detectors/__init__.py
from .packet import DetBox, DetectionPacket
from .config import DetectorConfig, COCO_ID_TO_NAME
from .thread import DetectorThread

__all__ = [
    "DetBox",
    "DetectionPacket",
    "DetectorConfig",
    "COCO_ID_TO_NAME",
    "DetectorThread",
]
