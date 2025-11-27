from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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
