from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import os
import json
import time

import cv2
import numpy as np

from .packet import DetBox


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


def run_faces(
    bgr: np.ndarray,
    cascade: Optional[cv2.CascadeClassifier],
    rec: Optional[object],
    labels: Dict[int, str],
) -> Tuple[List[DetBox], int]:
    """
    Run Haar cascade + optional LBPH recognition.
    Returns (face_boxes, elapsed_ms).
    """
    start = time.monotonic()
    faces_out: List[DetBox] = []

    if cascade is None:
        return faces_out, 0

    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        try:
            eq = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        except Exception:
            eq = cv2.equalizeHist(gray)

        minsz = max(40, int(0.12 * min(gray.shape[:2])))
        faces = cascade.detectMultiScale(eq, 1.1, 4, minSize=(minsz, minsz))

        for (fx, fy, fw, fh) in faces:
            name = "face"
            score = 0.6
            if rec is not None:
                try:
                    roi = gray[fy:fy + fh, fx:fx + fw]
                    roi = cv2.resize(roi, (160, 160))
                    pred, dist = rec.predict(roi)
                    if 0 <= pred < len(labels) and dist <= 95.0:
                        label_name = labels.get(int(pred), "face")
                        name = label_name
                        score = max(0.0, min(1.0, 1.0 - (dist / 95.0)))
                    else:
                        name = "unknown"
                        score = 0.4
                    print(
                        f"[LBPH] pred={pred} name={labels.get(int(pred), '?')} "
                        f"dist={dist:.1f} -> {name}"
                    )
                except Exception as e:
                    print(f"[LBPH] predict error: {e}")
                    name = "face"
                    score = 0.6

            x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
            faces_out.append(DetBox(name, float(score), (x1, y1, x2, y2)))
    except Exception as e:
        print(f"[Faces] error: {e}")

    elapsed_ms = int((time.monotonic() - start) * 1000.0)
    return faces_out, elapsed_ms
