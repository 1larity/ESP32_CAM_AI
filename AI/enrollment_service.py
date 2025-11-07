# enrollment_service.py
# Progress signals, reliable saving, debounce, and final LBPH training + labels.
from __future__ import annotations
from pathlib import Path
import json
import time
import cv2 as cv
import numpy as np
from PyQt6 import QtCore
from detectors import DetectionPacket
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
        self.samples_needed = 20
        self.samples_got = 0
        self._last_save_ms = 0
        self._min_gap_ms = 150  # avoid saving identical consecutive frames
        self._last_gray = None  # for quick similarity pruning

        # Discover faces root compatible with legacy layout
        candidates = [
            BASE_DIR / "data" / "faces",
            BASE_DIR / "data" / "enroll" / "faces",
            BASE_DIR / "faces",
        ]
        for c in candidates:
            if c.exists():
                self.face_dir = c
                break
        else:
            self.face_dir = BASE_DIR / "data" / "faces"
        self.models_dir = BASE_DIR / "models"

    @classmethod
    def instance(cls) -> "EnrollmentService":
        if cls._inst is None:
            cls._inst = EnrollmentService()
        return cls._inst

    # ---- API ----
    def begin_face(self, name: str, n: int):
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

        # debounce by time
        now_ms = pkt.ts_ms or int(time.monotonic() * 1000)
        if now_ms - self._last_save_ms < self._min_gap_ms:
            return

        x1, y1, x2, y2 = best
        h, w = bgr.shape[:2]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return

        crop = bgr[y1:y2, x1:x2]
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, (128, 128), interpolation=cv.INTER_AREA)

        # simple similarity prune vs last saved
        if self._last_gray is not None:
            diff = cv.absdiff(gray, self._last_gray)
            if float(cv.mean(diff)[0]) < 2.0:
                # too similar to last; skip this frame
                return

        self._last_gray = gray
        idx = self.samples_got + 1
        out = (self.face_dir / self.target_name / f"{idx:03d}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(out), gray)
        self.samples_got += 1
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
            return
        try:
            rec = cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        except Exception:
            return
        rec.train(imgs, np.array(labels))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        rec.write(str(self.models_dir / "lbph_faces.xml"))
        with open(self.models_dir / "labels_faces.json", "w", encoding="utf-8") as fp:
            json.dump(label_map, fp, indent=2)
        # notify completion
        self.status_changed.emit({
            "active": False, "name": self.target_name, "got": self.samples_got,
            "need": self.samples_needed, "folder": str(self.face_dir / self.target_name), "done": True
        })
