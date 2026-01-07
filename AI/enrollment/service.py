# AI/enrollment/service.py
# Singleton service for face enrollment and LBPH training.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal

from settings import BASE_DIR
from .service_control import start as _start, stop as _stop
from .service_detections import on_detections as _on_detections
from .state import _emit_status, set_unknown_capture
from .training_service import rebuild_lbph_model_from_disk, _train_now
from .unknowns_service import (
    _bootstrap_auto_unknowns,
    _count_unknown,
    _maybe_save_unknowns,
    _promote_unknown,
)


class EnrollmentService(QtCore.QObject):
    """
    Collect face crops for one person and train LBPH models.

    The typical call flow is:

      1. UI calls start(name, total_samples, target_cam)
      2. Video / detection layer calls on_detections(cam_name, bgr, pkt)
         for every DetectionPacket.
      3. Service saves cropped face ROIs into data/faces/<name>/...
      4. When enough samples are collected, it triggers LBPH training.

    Status updates are emitted via the `status_changed` signal as dicts with keys:
      - active: bool
      - target_name: str
      - samples_needed: int
      - samples_got: int
      - existing_count: int   (images already on disk for this person)
      - done: bool
      - last_error: Optional[str]
    """

    _inst: Optional["EnrollmentService"] = None
    status_changed = Signal(dict)

    # ------------------------------------------------------------------ lifecycle / state

    def __init__(self) -> None:
        super().__init__()

        # Where face crops and models are stored
        self.face_dir: Path = BASE_DIR / "data" / "faces"
        self.pet_dir: Path = BASE_DIR / "data" / "pets"
        self.unknown_face_dir: Path = BASE_DIR / "data" / "unknown_faces"
        self.unknown_pet_dir: Path = BASE_DIR / "data" / "unknown_pets"
        self.models_dir: Path = BASE_DIR / "models"
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self.pet_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_face_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_pet_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Enrollment state
        self.active: bool = False
        self.target_name: str = ""
        self.samples_needed: int = 0
        self.samples_got: int = 0

        # Sampling debounce / last-saved frame
        self._last_save_ms: int = 0
        self._last_gray: Optional[np.ndarray] = None

        # Optional camera filter (only sample from this camera if set)
        self.target_cam: Optional[str] = None

        # Number of images already on disk when enrollment started
        self._existing_count: int = 0

        # Unknown collection settings
        self.collect_unknown_faces: bool = False
        self.collect_unknown_pets: bool = False
        self._last_unknown_face: Dict[str, int] = {}
        self._last_unknown_pet: Dict[str, int] = {}
        self.unknown_capture_limit: int = 50
        self.auto_train_unknowns: bool = False
        self._auto_label_idx: int = 1

    # ------------------------------------------------------------------ singleton access

    @classmethod
    def instance(cls) -> "EnrollmentService":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    # ------------------------------------------------------------------ public API (control)

    start = _start
    stop = _stop

    # ------------------------------------------------------------------ public API (frame hook)

    on_detections = _on_detections

    # Attach extracted helpers (keeps this file small).
    _emit_status = _emit_status
    set_unknown_capture = set_unknown_capture

    _maybe_save_unknowns = _maybe_save_unknowns
    _count_unknown = _count_unknown
    _promote_unknown = _promote_unknown
    _bootstrap_auto_unknowns = _bootstrap_auto_unknowns

    rebuild_lbph_model_from_disk = rebuild_lbph_model_from_disk
    _train_now = _train_now


__all__ = ["EnrollmentService"]
