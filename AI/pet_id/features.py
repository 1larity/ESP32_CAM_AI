from __future__ import annotations

from typing import Tuple

import numpy as np


def crop_xyxy_with_pad(
    bgr: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    *,
    pad_frac: float = 0.08,
) -> np.ndarray | None:
    """
    Crop a BGR ROI from an image with a fractional padding around the box.
    """
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    if x2 <= x1 or y2 <= y1:
        return None

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * float(pad_frac)))
    pad_y = int(round(bh * float(pad_frac)))

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    roi = bgr[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return None
    return roi


def _lbp8(gray: np.ndarray) -> np.ndarray:
    """
    8-neighbour LBP for a grayscale image. Returns uint8 codes (0..255).
    """
    g = gray.astype(np.uint8, copy=False)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return np.zeros((0, 0), dtype=np.uint8)

    c = g[1:-1, 1:-1]
    lbp = np.zeros_like(c, dtype=np.uint8)
    lbp |= ((g[:-2, :-2] >= c).astype(np.uint8)) << 7
    lbp |= ((g[:-2, 1:-1] >= c).astype(np.uint8)) << 6
    lbp |= ((g[:-2, 2:] >= c).astype(np.uint8)) << 5
    lbp |= ((g[1:-1, 2:] >= c).astype(np.uint8)) << 4
    lbp |= ((g[2:, 2:] >= c).astype(np.uint8)) << 3
    lbp |= ((g[2:, 1:-1] >= c).astype(np.uint8)) << 2
    lbp |= ((g[2:, :-2] >= c).astype(np.uint8)) << 1
    lbp |= ((g[1:-1, :-2] >= c).astype(np.uint8)) << 0
    return lbp


def extract_pet_embedding(
    bgr: np.ndarray,
    *,
    size: int = 96,
    hist_bins: Tuple[int, int, int] = (8, 8, 8),
) -> np.ndarray | None:
    """
    Compute a lightweight appearance embedding for a pet ROI.

    This is intentionally local-only and model-free:
      - HSV 3D histogram (color)
      - LBP histogram (texture)
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    try:
        import cv2
    except Exception:
        return None

    try:
        img = cv2.resize(bgr, (int(size), int(size)), interpolation=cv2.INTER_AREA)
    except Exception:
        return None

    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(hist_bins), [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten().astype(np.float32, copy=False)
    except Exception:
        return None

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = _lbp8(gray)
        if lbp.size == 0:
            return None
        lbp_hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float32, copy=False)
        lbp_hist /= float(lbp_hist.sum() + 1e-6)
    except Exception:
        return None

    feat = np.concatenate([hist, lbp_hist]).astype(np.float32, copy=False)
    n = float(np.linalg.norm(feat) + 1e-6)
    feat /= n
    return feat


__all__ = ["crop_xyxy_with_pad", "extract_pet_embedding"]

