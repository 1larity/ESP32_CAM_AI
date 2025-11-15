# detectors/faces_backend.py
from __future__ import annotations
from typing import Optional
import os

import cv2 as cv

from .packet import DetBox, DetectionPacket
from .config import DetectorConfig
from .lbph_model import LBPHModel


class FaceBackend:
    def __init__(self, cfg: DetectorConfig, models_dir: str):
        self.cascade: Optional[cv.CascadeClassifier] = None
        self.lbph = LBPHModel.load(models_dir)
        if cfg.face_cascade and os.path.exists(cfg.face_cascade):
            try:
                self.cascade = cv.CascadeClassifier(cfg.face_cascade)
            except Exception as e:
                print(f"[Face] Haar load failed: {e}")
                self.cascade = None
        else:
            print(f"[Face] Haar not found at {cfg.face_cascade}")

    def run(self, bgr, pkt: DetectionPacket):
        if self.cascade is None:
            return
        try:
            gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
            try:
                eq = cv.createCLAHE(2.0, (8, 8)).apply(gray)
            except Exception:
                eq = cv.equalizeHist(gray)

            minsz = max(40, int(0.12 * min(gray.shape[:2])))
            faces = self.cascade.detectMultiScale(eq, 1.1, 4, minSize=(minsz, minsz))
            if len(faces) == 0:
                faces = self.cascade.detectMultiScale(eq, 1.05, 3, minSize=(minsz, minsz))

            for (fx, fy, fw, fh) in faces:
                roi = gray[fy:fy + fh, fx:fx + fw]
                name, score = self.lbph.predict_name(roi)
                x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
                pkt.faces.append(DetBox(name, float(score), (x1, y1, x2, y2)))
        except Exception as e:
            print(f"[Face] error: {e}")

