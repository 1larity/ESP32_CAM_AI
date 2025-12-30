# AI/enrollment/service.py
# Singleton service for face enrollment and LBPH training.

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional, Dict, Any

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Signal

from settings import BASE_DIR
from detection.lbph import train_lbph_models
from .capture import capture_enrollment_sample


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
        self.unknown_face_dir: Path = BASE_DIR / "data" / "unknown_faces"
        self.unknown_pet_dir: Path = BASE_DIR / "data" / "unknown_pets"
        self.models_dir: Path = BASE_DIR / "models"
        self.face_dir.mkdir(parents=True, exist_ok=True)
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

    # ------------------------------------------------------------------ status dispatch

    def _emit_status(self, **kwargs: Any) -> None:
        """
        Helper to emit a status dict.

        Base payload:
          active, target_name, samples_needed, samples_got, existing_count, done, last_error
        Extra keys may be merged in via kwargs.
        """
        data: Dict[str, Any] = {
            "active": self.active,
            "target_name": self.target_name,
            "samples_needed": self.samples_needed,
            "samples_got": self.samples_got,
            "existing_count": self._existing_count,
            "done": self.samples_got >= self.samples_needed and self.samples_needed > 0,
            "last_error": None,
        }
        data.update(kwargs)
        self.status_changed.emit(data)

    # ------------------------------------------------------------------ config

    def set_unknown_capture(
        self, faces: bool, pets: bool, limit: int | None = None, auto_train: bool | None = None
    ) -> None:
        self.collect_unknown_faces = bool(faces)
        self.collect_unknown_pets = bool(pets)
        if limit is not None:
            self.unknown_capture_limit = max(1, int(limit))
        if auto_train is not None:
            self.auto_train_unknowns = bool(auto_train)
        if self.auto_train_unknowns:
            self._bootstrap_auto_unknowns()

    # ------------------------------------------------------------------ public API (control)

    def start(
        self,
        name: str,
        total_samples: int,
        target_cam: Optional[str] = None,
    ) -> None:
        """
        Begin an enrollment session.

        name:
          Person / label name. Must be non-empty.
        total_samples:
          Number of samples to capture in this session.
        target_cam:
          If not None, only frames from this camera name will be accepted.
        """
        # Stop any existing session
        self.stop()

        self.target_name = name.strip()
        if not self.target_name:
            self._emit_status(last_error="Name is empty")
            return

        self.samples_needed = max(1, int(total_samples))
        self.samples_got = 0
        self._last_save_ms = 0
        self._last_gray = None
        self.target_cam = target_cam

        # Count any existing images so filenames continue from there
        person_dir = self.face_dir / self.target_name
        person_dir.mkdir(parents=True, exist_ok=True)
        self._existing_count = len(list(person_dir.glob("*.png")))

        self.active = True
        self._emit_status()

    def stop(self) -> None:
        """
        Stop enrollment without triggering training.

        This is used when the user cancels / aborts enrollment.
        """
        if self.active:
            self.active = False
            self._emit_status()

    # ------------------------------------------------------------------ public API (frame hook)

    def on_detections(self, cam_name: str, bgr: np.ndarray, pkt) -> None:
        """
        Entry point from the video layer.

        Called from UI/CameraWidgetVideo for each `DetectionPacket`.

        cam_name:
          Name of the camera that produced this frame.
        bgr:
          BGR frame as a NumPy array.
        pkt:
          DetectionPacket with at least `.faces` containing DetBox objects.
        """
        if bgr is None or pkt is None:
            return

        now_ms = int(time.time() * 1000)

        # Collect unknowns even when not actively enrolling
        self._maybe_save_unknowns(cam_name, bgr, pkt, now_ms)

        if not self.active:
            return

        # Respect camera filter if set
        if self.target_cam is not None and cam_name != self.target_cam:
            return

        # Delegate the heavy lifting (face selection, debouncing, path creation)
        new_last_save_ms, last_gray, saved = capture_enrollment_sample(
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

        # Update internal state from helper
        self._last_save_ms = new_last_save_ms
        self._last_gray = last_gray

        if not saved:
            # Nothing new persisted for this frame
            return

        # One more sample captured
        self.samples_got += 1
        done = self.samples_got >= self.samples_needed

        self._emit_status(done=done)

        if done:
            # Freeze further sampling and kick off training
            self.active = False
            self._train_now()

    # ------------------------------------------------------------------ unknown capture

    def _maybe_save_unknowns(self, cam_name: str, bgr: np.ndarray, pkt, now_ms: int) -> None:
        """Persist unknown faces/pets into queue folders for later training."""
        if self.collect_unknown_faces:
            for face in getattr(pkt, "faces", []) or []:
                label = (face.cls or "").lower()
                if label in ("", "face", "unknown", "person"):
                    last = self._last_unknown_face.get(cam_name, 0)
                    if now_ms - last < 800:
                        continue
                    # enforce per-cam limit
                    if self._count_unknown(self.unknown_face_dir / cam_name) >= self.unknown_capture_limit:
                        continue
                    self._last_unknown_face[cam_name] = now_ms
                    try:
                        x1, y1, x2, y2 = map(int, face.xyxy)
                        roi = bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        if roi.size == 0:
                            continue
                        out_dir = self.unknown_face_dir / cam_name
                        out_dir.mkdir(parents=True, exist_ok=True)
                        fname = f"{cam_name}_{now_ms}.jpg"
                        from cv2 import imwrite
                        imwrite(str(out_dir / fname), roi)
                        if self.auto_train_unknowns:
                            self._promote_unknown(roi, cam_name, is_pet=False)
                    except Exception:
                        continue

        if self.collect_unknown_pets:
            for pet in getattr(pkt, "pets", []) or []:
                last = self._last_unknown_pet.get(cam_name, 0)
                if now_ms - last < 800:
                    continue
                if self._count_unknown(self.unknown_pet_dir / cam_name) >= self.unknown_capture_limit:
                    continue
                self._last_unknown_pet[cam_name] = now_ms
                try:
                    x1, y1, x2, y2 = map(int, pet.xyxy)
                    roi = bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if roi.size == 0:
                        continue
                    out_dir = self.unknown_pet_dir / cam_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{cam_name}_{now_ms}_{pet.cls}.jpg"
                    from cv2 import imwrite
                    imwrite(str(out_dir / fname), roi)
                    if self.auto_train_unknowns:
                        self._promote_unknown(roi, cam_name, is_pet=True)
                except Exception:
                    continue

    def _count_unknown(self, path: Path) -> int:
        if not path.exists():
            return 0
        return len(list(path.glob("*.jpg")))

    def _promote_unknown(self, roi, cam_name: str, is_pet: bool) -> None:
        """
        Save unknown into auto_* folder for provisional training and kick rebuild.
        """
        try:
            from cv2 import imwrite
            label_prefix = "auto_pet" if is_pet else "auto_person"
            out_dir = self.face_dir / f"{label_prefix}_{cam_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{label_prefix}_{self._auto_label_idx:05d}.jpg"
            self._auto_label_idx += 1
            imwrite(str(out_dir / fname), roi)
            # Kick rebuild asynchronously
            QtCore.QTimer.singleShot(0, self._train_now)
        except Exception:
            pass

    def _bootstrap_auto_unknowns(self) -> None:
        """
        If auto-train is enabled, seed auto_person/auto_pet folders from any existing
        unknown capture queues so recognition can start without waiting for new saves.
        """
        try:
            import cv2
        except Exception:
            return

        promoted = False

        # Faces
        for cam_dir in self.unknown_face_dir.glob("*"):
            if not cam_dir.is_dir():
                continue
            dest = self.face_dir / f"auto_person_{cam_dir.name}"
            if dest.exists() and any(dest.glob("*.jpg")):
                continue
            files = sorted(cam_dir.glob("*.jpg"))
            dest.mkdir(parents=True, exist_ok=True)
            for f in files[: min(20, len(files))]:
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                fname = f"{dest.name}_{self._auto_label_idx:05d}.jpg"
                self._auto_label_idx += 1
                cv2.imwrite(str(dest / fname), img)
                promoted = True

        # Pets
        for cam_dir in self.unknown_pet_dir.glob("*"):
            if not cam_dir.is_dir():
                continue
            dest = self.face_dir / f"auto_pet_{cam_dir.name}"
            if dest.exists() and any(dest.glob("*.jpg")):
                continue
            files = sorted(cam_dir.glob("*.jpg"))
            dest.mkdir(parents=True, exist_ok=True)
            for f in files[: min(20, len(files))]:
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                fname = f"{dest.name}_{self._auto_label_idx:05d}.jpg"
                self._auto_label_idx += 1
                cv2.imwrite(str(dest / fname), img)
                promoted = True

        if promoted:
            QtCore.QTimer.singleShot(0, self._train_now)

    # ------------------------------------------------------------------ training

    def rebuild_lbph_model_from_disk(self) -> bool:
        """
        Scan `self.face_dir` and train LBPH models into `self.models_dir`.

        This is used both after enrollment completes and when the user
        requests a manual "rebuild from disk" via the controller.
        """
        ok = train_lbph_models(str(self.face_dir), str(self.models_dir))
        if not ok:
            self._emit_status(last_error="No faces found on disk for training.")
        else:
            self._emit_status()
        return ok

    def _train_now(self) -> None:
        """
        Train LBPH models from all faces on disk and update status.
        """
        self.rebuild_lbph_model_from_disk()
