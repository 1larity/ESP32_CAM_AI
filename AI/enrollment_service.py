# enrollment_service.py
# Progress signals, reliable saving, debounce, and final LBPH training + labels.

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Optional, Dict, Any, List

import cv2 as cv
import numpy as np
from PyQt6 import QtCore

from settings import BASE_DIR


class EnrollmentService(QtCore.QObject):
    """Central service for face enrollment and LBPH training."""

    _inst: Optional["EnrollmentService"] = None

    # Emitted on any state change:
    #   {
    #     "active": bool,
    #     "name": str,
    #     "got": int,
    #     "need": int,
    #     "folder": str,
    #     "done": bool,
    #     "cam": Optional[str],
    #   }
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
        # If not None, only this camera's detections are used
        self.target_cam: Optional[str] = None
        self._existing_count: int = 0  # how many images already on disk for this person

    @classmethod
    def instance(cls) -> "EnrollmentService":
        if cls._inst is None:
            cls._inst = EnrollmentService()
        return cls._inst

    # ------------------------------------------------------------------ API

    def start(self, name: str, n: int, cam_name: Optional[str] = None) -> None:
        """
        Start an enrollment session.

        - name: person label.
        - n: number of new samples to collect in this session.
        - cam_name: if provided, only that camera will be accepted.
        """
        self.target_name = name.strip()
        if not self.target_name:
            return

        self.target_cam = cam_name or None

        person_dir = self.face_dir / self.target_name
        person_dir.mkdir(parents=True, exist_ok=True)

        # Count existing images so we know where to start numbering,
        # but DO NOT count them towards this session's target.
        existing: List[Path] = []
        for pat in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
            existing.extend(person_dir.glob(pat))
        self._existing_count = len(existing)

        # For this run we want "n new samples"
        self.samples_got = 0
        self.samples_needed = max(1, int(n))
        self._last_save_ms = 0
        self._last_gray = None

        # Always start a session, even if there are already plenty on disk.
        self.active = True
        self._emit()

    def end(self) -> None:
        """Abort or finish the current enrollment session without training."""
        self.active = False
        self._emit()

    # ----------------------------------------------------------------- Hooks

    def on_detections(self, cam_name: str, bgr: np.ndarray, pkt) -> None:
        """Called from CameraWidget._on_detections for each detection packet."""
        if not self.active or not self.target_name:
            return

        # If a specific camera is selected, ignore others
        if self.target_cam is not None and cam_name != self.target_cam:
            return

        # pick largest face
        best = None
        best_area = 0
        for f in pkt.faces:
            x1, y1, x2, y2 = f.xyxy
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        if best is None:
            return

        x1, y1, x2, y2 = best
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return
        roi = cv.resize(roi, (128, 128), interpolation=cv.INTER_AREA)

        # Simple debounce: don't save more than ~4 fps per cam
        now_ms = time.monotonic_ns() // 1_000_000
        if self._last_gray is not None and now_ms - self._last_save_ms < 250:
            return

        self._last_gray = roi

        person_dir = self.face_dir / self.target_name
        person_dir.mkdir(parents=True, exist_ok=True)

        # Base index after any existing files so we don't overwrite them.
        base_index = self._existing_count + self.samples_got + 1
        next_index = base_index
        while True:
            out_path = person_dir / f"{self.target_name}_{next_index:04d}.png"
            if not out_path.exists():
                break
            next_index += 1

        cv.imwrite(str(out_path), roi)

        # Count only NEW samples this session
        self.samples_got += 1
        self._last_save_ms = now_ms
        self._emit()

        if self.samples_got >= self.samples_needed:
            self.active = False
            self._emit()
            self._train_lbph()

    # ----------------------------------------------------------------- Internals

    def _emit(self) -> None:
        payload: Dict[str, Any] = {
            "active": self.active,
            "name": self.target_name,
            "got": self.samples_got,
            "need": self.samples_needed,
            "folder": str(self.face_dir / self.target_name) if self.target_name else str(self.face_dir),
            "done": (not self.active and self.samples_got >= self.samples_needed and self.samples_needed > 0),
            "cam": self.target_cam,
        }
        self.status_changed.emit(payload)

    def _train_lbph(self) -> bool:
        """Scan all person folders and train LBPH model + labels.

        Returns True if training succeeded, False otherwise.
        """
        subs = [p for p in self.face_dir.iterdir() if p.is_dir()]
        imgs: List[np.ndarray] = []
        labels: List[int] = []
        label_map: Dict[str, int] = {}
        next_id = 0

        for p in sorted(subs):
            label_map[p.name] = next_id
            label_id = next_id
            next_id += 1
            files: List[Path] = []
            for pat in ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
                files.extend(sorted(p.glob(pat)))
            for f in files:
                im = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                imgs.append(im)
                labels.append(label_id)

        if not imgs:
            return False

        try:
            rec = cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        except Exception:
            return False

        rec.train(imgs, np.array(labels))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        rec.write(str(self.models_dir / "lbph_faces.xml"))
        with open(self.models_dir / "labels_faces.json", "w", encoding="utf-8") as fp:
            json.dump(label_map, fp, indent=2)

        # Emit final done status
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
