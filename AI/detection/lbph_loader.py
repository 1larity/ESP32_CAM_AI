from __future__ import annotations

from typing import Dict, Optional, Tuple
import os
import json

import cv2


def load_lbph(models_dir: str) -> Tuple[Optional[object], Dict[int, str]]:
    """
    Load LBPH face recogniser and label mapping from the given models directory.
    Functional extraction of DetectorThread._load_lbph.
    """
    rec = None
    labels: Dict[int, str] = {}

    model_path = os.path.join(models_dir, "lbph_faces.xml")
    labels_path = os.path.join(models_dir, "labels_faces.json")
    try:
        if os.path.exists(model_path):
            # requires opencv-contrib-python
            rec = cv2.face.LBPHFaceRecognizer_create()  # type: ignore[attr-defined]
            rec.read(model_path)
        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as fp:
                m = json.load(fp)
            # stored as {name: id}; invert
            labels = {int(v): k for k, v in m.items()}
    except Exception as e:
        print(f"[LBPH] disabled ({e})")
        rec = None
        labels = {}

    return rec, labels


__all__ = ["load_lbph"]

