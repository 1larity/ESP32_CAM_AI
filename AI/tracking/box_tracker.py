from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List


XYXY = Tuple[int, int, int, int]


def iou(a: XYXY, b: XYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = float(a_area + b_area - inter)
    return float(inter / denom) if denom > 0 else 0.0


@dataclass
class SingleTrack:
    """A tiny IOU-based single-target tracker (detection-driven).

    Keeps continuity between detector packets by picking the best IOU match.
    """

    bbox: Optional[XYXY] = None
    label: str = ""
    score: float = 0.0
    last_seen_ts_ms: int = 0

    def clear(self) -> None:
        self.bbox = None
        self.label = ""
        self.score = 0.0
        self.last_seen_ts_ms = 0

    def update(
        self,
        candidates: Iterable[Tuple[str, float, XYXY]],
        ts_ms: int,
        iou_min: float,
        prefer_labels: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, float, XYXY]]:
        cand_list = list(candidates)
        if not cand_list:
            return None

        # If we have an active bbox, try to keep it via IOU.
        if self.bbox is not None:
            best = None
            best_iou = 0.0
            for lbl, sc, bb in cand_list:
                v = iou(self.bbox, bb)
                if v > best_iou:
                    best_iou = v
                    best = (lbl, sc, bb)
            if best is not None and best_iou >= float(iou_min):
                self.label, self.score, self.bbox = best
                self.last_seen_ts_ms = int(ts_ms)
                return best

        # Otherwise pick a fresh target.
        # Priority 1: preferred labels (in order)
        if prefer_labels:
            for pref in prefer_labels:
                pref = (pref or "").strip().lower()
                if not pref:
                    continue
                best = None
                best_sc = -1.0
                for lbl, sc, bb in cand_list:
                    if (lbl or "").strip().lower() == pref and float(sc) > best_sc:
                        best = (lbl, sc, bb)
                        best_sc = float(sc)
                if best is not None:
                    self.label, self.score, self.bbox = best
                    self.last_seen_ts_ms = int(ts_ms)
                    return best

        # Priority 2: largest score * sqrt(area) (stable “salience” heuristic)
        best = None
        best_val = -1.0
        for lbl, sc, bb in cand_list:
            x1, y1, x2, y2 = bb
            area = max(1, (x2 - x1) * (y2 - y1))
            val = float(sc) * (area ** 0.5)
            if val > best_val:
                best_val = val
                best = (lbl, sc, bb)
        if best is not None:
            self.label, self.score, self.bbox = best
            self.last_seen_ts_ms = int(ts_ms)
        return best
