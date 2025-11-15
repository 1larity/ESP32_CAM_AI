# detectors/lbph_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import json
import os

import cv2 as cv
import numpy as np


@dataclass
class LBPHModel:
    recognizer: Optional[cv.face_LBPHFaceRecognizer]  # type: ignore
    labels: Dict[int, str]
    accept_conf: float = 95.0  # distance threshold

    @classmethod
    def load(cls, models_dir: str) -> "LBPHModel":
        model_path = os.path.join(models_dir, "lbph_faces.xml")
        labels_path = os.path.join(models_dir, "labels_faces.json")
        rec = None
        labels: Dict[int, str] = {}
        try:
            if os.path.exists(model_path):
                rec = cv.face.LBPHFaceRecognizer_create()  # type: ignore[attr-defined]
                rec.read(model_path)
            if os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as fp:
                    m = json.load(fp)
                labels = {int(v): k for k, v in m.items()}
        except Exception as e:
            print(f"[LBPH] disabled: {e}")
            rec = None
            labels = {}
        return cls(rec, labels)

    def predict_name(self, roi_gray) -> Tuple[str, float]:
        if self.recognizer is None:
            return "face", 0.6
        try:
            roi = cv.resize(roi_gray, (160, 160))
            pred, dist = self.recognizer.predict(roi)
            if 0 <= pred and dist <= self.accept_conf and pred in self.labels:
                name = self.labels[int(pred)]
                score = max(0.0, min(1.0, 1.0 - (dist / self.accept_conf)))
            else:
                name = "unknown"
                score = 0.4
            print(f"[LBPH] pred={pred} name={self.labels.get(int(pred), '?')} dist={dist:.1f} -> {name}")
            return name, float(score)
        except Exception as e:
            print(f"[LBPH] predict error: {e}")
            return "face", 0.6
