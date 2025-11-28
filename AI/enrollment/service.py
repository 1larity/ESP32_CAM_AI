# AI/enrollment/service.py
# Singleton service for face enrollment and LBPH training.

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional, Dict, Any, List

import numpy as np
from PyQt6 import QtCore

from settings import BASE_DIR
from detection.lbph import train_lbph_models
from .capture import capture_enrollment_sample


class EnrollmentService(QtCore.QObject):
    """Collect face crops for one person and train LBPH models."""

    _inst: Optional["EnrollmentService"] = None
    status_changed = QtCore.pyqtSignal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.face_dir: Path = BASE_DIR / "data" / "faces"
        self.models_dir: Path = BASE_DIR / "models"
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.active: bool = False
        self.target_name: str = ""
        self.samples_needed: int = 0
        self.samples_got: int = 0
        self._last_save_ms: int = 0
        self._last_gray: Optional[np.ndarray] = None
        self.target_cam: Optional[str] = None  # if not None, only this camera is used
        self._existing_count: int = 0  # how many images already on disk for this person
    # ------------------------------------------------------------------ helper probably overkill
    def rebuild_lbph_model_from_disk(self) -> bool:
        """Rebuild LBPH models from all face images on disk."""
        return train_lbph_models(str(self.face_dir), str(self.models_dir))

    # ------------------------------------------------------------------ Singleton

    @classmethod
    def instance(cls) -> "EnrollmentService":
        if cls._inst is None:
            cls._inst = EnrollmentService()
        return cls._inst

    # ------------------------------------------------------------------ Public API

    def start(self, name: str, n: int, cam_name: Optional[str] = None) -> None:
        """Start an enrollment session collecting *n* samples for *name*."""
        self.target_name = name.strip()
        if not self.target_name:
            return

        self.target_cam = cam_name or None

        person_dir = self.face_dir / self.target_name
        person_dir.mkdir(parents=True, exist_ok=True)

        # Count existing images so we don't overwrite them,
        # but DO NOT count them towards this session's target.
        existing: List[Path] = []
        for pat in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
            existing.extend(person_dir.glob(pat))
        self._existing_count = len(existing)

        self.samples_got = 0
        self.samples_needed = max(1, int(n))
        self._last_save_ms = 0
        self._last_gray = None

        self.active = True
        self._emit_status()

    def end(self) -> None:
        """Abort the current enrollment session without training."""
        self.active = False
        self._emit_status()

    def on_detections(self, cam_name: str, bgr: np.ndarray, pkt) -> None:
        """Called from CameraWidget._on_detections for each detection packet."""
        if not self.active or not self.target_name:
            return

        now_ms = time.monotonic_ns() // 1_000_000

        new_last_ms, new_last_gray, saved = capture_enrollment_sample(
            cam_name=cam_name,
            target_cam=self.target_cam,
            target_name=self.target_name,
            bgr=bgr,
            pkt=pkt,
            face_dir=self.face_dir,
            existing_count=self._existing_count,
            samples_got=self.samples_got,
            last_save_ms=self._last_save_ms,
            last_gray=self._last_gray,
            now_ms=now_ms,
        )
        if not saved:
            return

        self._last_save_ms = new_last_ms
        self._last_gray = new_last_gray
        self.samples_got += 1
        self._emit_status()

        if self.samples_got >= self.samples_needed:
            self.active = False
            self._emit_status()
            self._train_and_emit()

    # ------------------------------------------------------------------ Internals

    def _emit_status(self) -> None:
        folder = self.face_dir / self.target_name if self.target_name else self.face_dir
        payload: Dict[str, Any] = {
            "active": self.active,
            "name": self.target_name,
            "got": self.samples_got,
            "need": self.samples_needed,
            "folder": str(folder),
            "done": (
                not self.active
                and self.samples_got >= self.samples_needed
                and self.samples_needed > 0
            ),
            "cam": self.target_cam,
        }
        self.status_changed.emit(payload)

    def _train_and_emit(self) -> bool:
        ok = train_lbph_models(str(self.face_dir), str(self.models_dir))
        if not ok:
            return False

        # Emit final "done" snapshot with folder pointing at face root
        self.status_changed.emit(
            {
                "active": False,
                "name": self.target_name,
                "got": self.samples_got,
                "need": self.samples_needed,
                "folder": str(self.face_dir),
                "done": True,
                "cam": self.target_cam,
            }
        )
        return True
