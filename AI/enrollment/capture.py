# AI/enrollment/capture.py
# Helpers for selecting faces and saving enrollment samples.

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np


def _pick_largest_face(pkt) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) for the largest face in pkt.faces, or None."""
    best = None
    best_area = 0
    for f in getattr(pkt, "faces", []):
        x1, y1, x2, y2 = f.xyxy
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best


def capture_enrollment_sample(
    cam_name: str,
    target_cam: Optional[str],
    target_name: str,
    bgr: np.ndarray,
    pkt,
    face_dir: Path,
    existing_count: int,
    samples_got: int,
    last_save_ms: int,
    last_gray: Optional[np.ndarray],
    now_ms: int,
    min_interval_ms: int = 250,
) -> Tuple[int, Optional[np.ndarray], bool]:
    """Try to capture and persist a single face sample.

    Returns (new_last_save_ms, new_last_gray, did_save).
    """
    if not target_name:
        return last_save_ms, last_gray, False

    # Optional camera filter
    if target_cam is not None and cam_name != target_cam:
        return last_save_ms, last_gray, False

    xyxy = _pick_largest_face(pkt)
    if xyxy is None:
        return last_save_ms, last_gray, False

    x1, y1, x2, y2 = xyxy
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return last_save_ms, last_gray, False

    roi = cv.resize(roi, (128, 128), interpolation=cv.INTER_AREA)

    # Simple debounce: don't save more than ~4 fps per cam
    if last_gray is not None and now_ms - last_save_ms < min_interval_ms:
        return last_save_ms, last_gray, False

    person_dir = face_dir / target_name
    person_dir.mkdir(parents=True, exist_ok=True)

    # Base index after any existing files so we don't overwrite them.
    base_index = existing_count + samples_got + 1
    next_index = base_index
    while True:
        out_path = person_dir / f"{target_name}_{next_index:04d}.png"
        if not out_path.exists():
            break
        next_index += 1

    cv.imwrite(str(out_path), roi)
    return now_ms, roi, True
