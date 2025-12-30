from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
import json
import time
import traceback

import cv2
import numpy as np

from .packet import DetBox

# Default distance threshold for LBPH recognition. Larger values are more lenient.
LBPH_DEFAULT_THRESHOLD = 140.0

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
            if rec is not None and labels:
                try:
                    roi = gray[fy:fy + fh, fx:fx + fw]
                    roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
                    pred, dist = rec.predict(roi)
                    threshold = LBPH_DEFAULT_THRESHOLD
                    if 0 <= pred and dist <= threshold:
                        label_name = labels.get(int(pred), "face")
                        name = label_name
                        # map distance → [0.3, 1.0]
                        score = max(0.3, min(1.0, (threshold - dist) / threshold + 0.3))
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


def train_lbph_models(face_dir: str, models_dir: str) -> bool:
    """Scan all person folders under face_dir and train LBPH model + labels.

    Models are written to models_dir / "lbph_faces.xml" and
    models_dir / "labels_faces.json" (stored as {name: id}).
    Returns True if training succeeded, False otherwise.
    """
    base = Path(face_dir)
    if not base.exists():
        return False

    subs = [p for p in base.iterdir() if p.is_dir()]
    imgs: List[np.ndarray] = []
    labels_arr: List[int] = []
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
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            if im.shape != (128, 128):
                im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
            imgs.append(im)
            labels_arr.append(label_id)

    if not imgs:
        return False

    try:
        rec = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )  # type: ignore[attr-defined]
    except Exception:
        return False

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    tmp_model = models_path / "lbph_faces.xml.tmp"
    final_model = models_path / "lbph_faces.xml"
    tmp_labels = models_path / "labels_faces.json.tmp"
    final_labels = models_path / "labels_faces.json"

    try:
        if len(imgs) != len(labels_arr):
            raise ValueError(
                f"LBPH training: image/label count mismatch {len(imgs)} != {len(labels_arr)}"
            )

        # Ensure labels are int32 1D
        labels_np = np.asarray(labels_arr, dtype=np.int32).reshape(-1)

        # Train can throw cv2.error / ValueError depending on inputs
        rec.train(imgs, labels_np)

        # Atomic model write
        try:
            if tmp_model.exists():
                tmp_model.unlink()
        except Exception:
            pass
        rec.write(str(tmp_model))
        os.replace(str(tmp_model), str(final_model))

        # Atomic labels write
        try:
            if tmp_labels.exists():
                tmp_labels.unlink()
        except Exception:
            pass
        with open(tmp_labels, "w", encoding="utf-8") as fp:
            json.dump(label_map, fp, indent=2)
        os.replace(str(tmp_labels), str(final_labels))

        return True

    except Exception:
        # Clean up temp files, but never crash caller
        try:
            if tmp_model.exists():
                tmp_model.unlink()
        except Exception:
            pass
        try:
            if tmp_labels.exists():
                tmp_labels.unlink()
        except Exception:
            pass

        print("[LBPH] training failed:")
        print(traceback.format_exc())
        return False
