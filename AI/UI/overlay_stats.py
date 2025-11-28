# overlay_stats.py
# FPS counter + simple YOLO stats helper (no drawing here).

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import time


@dataclass
class YoloStats:
    faces: int = 0
    known_faces: int = 0
    pets: int = 0
    total: int = 0


class FpsCounter:
    """Simple EMA-based FPS counter per camera."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._last_time: float | None = None
        self._fps: float = 0.0
        self._alpha: float = alpha

    def update(self) -> float:
        now = time.monotonic()
        if self._last_time is None:
            self._last_time = now
            return self._fps
        dt = now - self._last_time
        self._last_time = now
        if dt <= 0:
            return self._fps
        inst = 1.0 / dt
        if self._fps == 0.0:
            self._fps = inst
        else:
            self._fps = self._fps * (1.0 - self._alpha) + inst * self._alpha
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


def compute_yolo_stats(boxes: Iterable[object]) -> YoloStats:
    """Compute simple stats from DetBox-like objects (faces/pets/known)."""
    stats = YoloStats()
    for b in boxes:
        label = (
            getattr(b, "label", None)
            or getattr(b, "name", None)
            or getattr(b, "cls", "")
        )
        label = str(label).lower()

        stats.total += 1

        # Pets
        if label in ("dog", "cat", "bird"):
            stats.pets += 1

        # Faces / persons:
        # - treat anything that isn't a pet as a face/person
        # - "face"/"unknown" are unknown faces
        if label not in ("dog", "cat", "bird"):
            stats.faces += 1
            if label not in ("face", "unknown"):
                stats.known_faces += 1

    return stats
