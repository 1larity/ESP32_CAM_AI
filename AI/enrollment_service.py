# enrollment_service.py
# Progress signals, reliable saving, debounce, and final LBPH training + labels.
from __future__ import annotations
from pathlib import Path
import json
import time
import cv2 as cv
import numpy as np
from PyQt6 import QtCore

from settings import BASE_DIR

class EnrollmentService(QtCore.QObject):
    _inst = None

    # Emitted on any state change:
    #   {"active":bool,"name":str,"got":int,"need":int,"folder":str,"done":bool}
    status_changed = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.active = False
        self.target_name = ""
        self.samples_needed = 0
        self.samples_got = 0
        self._last_save_ms = 0
        self._last_gray = None
        self.face_dir = BASE_DIR / "data" / "faces"
        self.models_dir = BASE_DIR / "models"
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def instance(cls) -> "EnrollmentService":
        if cls._inst is None:
            cls._inst = EnrollmentService()
        return cls._inst

    # ---- session control ----
    def start(self, name: str, n: int):
        self.target_name = name.strip()
        self.samples_needed = max(1, int(n))
        self.samples_got = 0
        self._last_save_ms = 0
        self._last_gray = None
        self.active = True
        (self.face_dir / self.target_name).mkdir(parents=True, exist_ok=True)
        self._emit()

    def end(self):
        self.active = False
        self._emit()

    # Called from CameraWidget._on_detections on each detection packet
    def on_detections(self, cam_name: str, bgr: np.ndarray, pkt: DetectionPacket):
        if not self.active or not self.target_name:
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
        self.samples_got += 1

        person_dir = self.face_dir / self.target_name
        person_dir.mkdir(parents=True, exist_ok=True)
        out_path = person_dir / f"{self.target_name}_{self.samples_got:04d}.png"
        cv.imwrite(str(out_path), roi)

        self._last_save_ms = now_ms
        self._emit()

        if self.samples_got >= self.samples_needed:
            self.active = False
            self._emit()
            self._train_lbph()

    # ---- internals ----
    def _emit(self):
        self.status_changed.emit({
            "active": self.active,
            "name": self.target_name,
            "got": self.samples_got,
            "need": self.samples_needed,
            "folder": str(self.face_dir / self.target_name),
            "done": (not self.active and self.samples_got >= self.samples_needed)
        })

    def _train_lbph(self):
        # Aggregate all persons; write model + labels
        subs = [p for p in (self.face_dir).iterdir() if p.is_dir()]
        imgs = []; labels = []; label_map = {}; next_id = 0
        for p in sorted(subs):
            label_map[p.name] = next_id
            for f in sorted(list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))):
                im = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                imgs.append(im); labels.append(next_id)
            next_id += 1
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
        self.status_changed.emit({"active": False, "name": self.target_name, "got": self.samples_got,
                                  "need": self.samples_needed, "folder": str(self.face_dir), "done": True})
        return True
