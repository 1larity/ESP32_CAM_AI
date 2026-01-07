from __future__ import annotations

from typing import Dict, List
from pathlib import Path
import os
import json
import traceback

import cv2
import numpy as np


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
        # Pets are stored under data/pets; ignore any legacy auto_pet_* folders that may
        # still exist under data/faces from older versions.
        if p.name.startswith("auto_pet_"):
            continue
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
        # If the user purged all samples, clear any previous on-disk model so
        # we don't keep recognising stale labels.
        try:
            models_path = Path(models_dir)
            for f in ("lbph_faces.xml", "labels_faces.json"):
                p = models_path / f
                if p.exists():
                    p.unlink()
        except Exception:
            pass
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


__all__ = ["train_lbph_models"]

