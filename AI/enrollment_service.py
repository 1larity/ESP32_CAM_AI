# enrollment_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import json

import cv2 as cv
import numpy as np
from PyQt6 import QtCore

from detectors import DetectionPacket
from settings import BASE_DIR


@dataclass
class _EnrollState:
    active: bool = False
    kind: str = "face"          # only "face" for now
    name: str = ""
    samples_needed: int = 0
    samples_got: int = 0
    faces_dir: str = ""
    status: str = "Idle"


class EnrollmentService(QtCore.QObject):
    """
    Singleton service handling face enrollment and LBPH model training.

    Usage:
      svc = EnrollmentService.instance()
      svc.begin_face(name, n_samples, cam_name)
      ...
      svc.on_detections(cam_name, bgr, pkt)  # called by CameraWidget
    """
    status_changed = QtCore.pyqtSignal()

    _instance: Optional["EnrollmentService"] = None

    def __init__(self):
        super().__init__()
        self.state = _EnrollState()
        self.faces_dir = str(Path(BASE_DIR) / "data" / "faces")
        self.models_dir = str(Path(BASE_DIR) / "models")
        self.cam_filter: Optional[str] = None

    # ------------------------------------------------------------------ #
    # singleton access
    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls) -> "EnrollmentService":
        if cls._instance is None:
            cls._instance = EnrollmentService()
        return cls._instance

    # ------------------------------------------------------------------ #
    # public API from UI
    # ------------------------------------------------------------------ #
    def begin_face(self, name: str, n: int, cam_name: Optional[str] = None):
        """
        Called by EnrollDialog. Starts collecting face samples.
        """
        self.start(name, n, cam_name)

    def start(self, name: str, n: int, cam_name: Optional[str] = None):
        s = self.state
        s.active = True
        s.kind = "face"
        s.name = name
        s.samples_needed = n
        s.samples_got = 0
        s.status = f"Collecting samples for {name} ({n} needed)"
        self.cam_filter = cam_name
        self.status_changed.emit()

    def end(self):
        s = self.state
        s.active = False
        s.status = "Idle"
        self.cam_filter = None
        self.status_changed.emit()

    # ------------------------------------------------------------------ #
    # properties / helpers
    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        return self.state.active

    @property
    def status_text(self) -> str:
        return self.state.status

    @property
    def samples_needed(self) -> int:
        return self.state.samples_needed

    @property
    def samples_got(self) -> int:
        return self.state.samples_got

    def _emit_status(self, text: str):
        self.state.status = text
        self.status_changed.emit()

    # ------------------------------------------------------------------ #
    # main hook from CameraWidget
    # ------------------------------------------------------------------ #
    def on_detections(self, cam_name: str, bgr, pkt: DetectionPacket):
        """
        Called from CameraWidget._on_detections on each detection packet.
        We pick the largest face ROI and save it as a grayscale PNG sample.
        """
        s = self.state
        if not s.active:
            return

        # Only accept packets from the selected camera, if any
        if self.cam_filter is not None and cam_name != self.cam_filter:
            return

        if not pkt.faces:
            return

        # choose largest face by area
        best_face = None
        best_area = 0
        for f in pkt.faces:
            x1, y1, x2, y2 = f.xyxy
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_face = f

        if best_face is None:
            return

        x1, y1, x2, y2 = best_face.xyxy
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), bgr.shape[1])
        y2 = min(int(y2), bgr.shape[0])
        if x2 <= x1 or y2 <= y1:
            return

        roi = bgr[y1:y2, x1:x2]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (128, 128))

        faces_root = Path(self.faces_dir)
        faces_root.mkdir(parents=True, exist_ok=True)
        person_dir = faces_root / s.name
        person_dir.mkdir(parents=True, exist_ok=True)

        idx = s.samples_got + 1
        out_path = person_dir / f"{idx:04d}.png"
        cv.imwrite(str(out_path), gray)

        s.samples_got += 1
        self._emit_status(
            f"Collected {s.samples_got}/{s.samples_needed} for {s.name}"
        )

        if s.samples_got >= s.samples_needed:
            self._emit_status("Training LBPH model…")
            ok = self._maybe_train_and_save()
            if ok:
                self._emit_status("Training complete.")
            else:
                self._emit_status("Training failed or no data.")
            self.end()

    # ------------------------------------------------------------------ #
    # training helpers
    # ------------------------------------------------------------------ #
    def _maybe_train_and_save(self) -> bool:
        """
        Train LBPH face recogniser from faces_dir and save into models_dir.
        Returns True on success, False otherwise.
        """
        faces_root = Path(self.faces_dir)
        if not faces_root.exists():
            self._emit_status("No faces directory to train from.")
            return False

        images: List[np.ndarray] = []
        labels: List[int] = []
        label_map: dict[str, int] = {}
        label_id = 0

        for person_dir in sorted(faces_root.iterdir()):
            if not person_dir.is_dir():
                continue
            person_name = person_dir.name
            if person_name not in label_map:
                label_map[person_name] = label_id
                label_id += 1
            lbl = label_map[person_name]
            for img_path in sorted(person_dir.glob("*.png")):
                img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                labels.append(lbl)

        if not images:
            self._emit_status("No images found for training.")
            return False

        recognizer = cv.face.LBPHFaceRecognizer_create()
        labels_np = np.array(labels, dtype=np.int32)
        recognizer.train(images, labels_np)

        models_dir = Path(self.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "lbph_faces.xml"
        recognizer.write(str(model_path))

        labels_path = models_dir / "labels_faces.json"
        with labels_path.open("w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)

        self._emit_status(f"Trained LBPH model with {len(label_map)} labels.")
        return True

    def _train_lbph(self) -> bool:
        """
        Public helper used by the menu action 'Rebuild face model from disk…'.
        """
        return self._maybe_train_and_save()
