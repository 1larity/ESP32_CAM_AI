from __future__ import annotations

from typing import Dict, List, Tuple

from .types import PetTrack


def _area(xyxy: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = xyxy
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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
    ua = _area(a)
    ub = _area(b)
    denom = float(ua + ub - inter)
    return float(inter / denom) if denom > 0 else 0.0


class PetTracker:
    def __init__(self, *, expire_ms: int, iou_thresh: float) -> None:
        self._expire_ms = int(expire_ms)
        self._iou_thresh = float(iou_thresh)
        self._next_id = 1
        self._tracks: Dict[int, PetTrack] = {}

    def assign(self, *, ts_ms: int, boxes: List[Tuple[int, int, int, int]]) -> List[PetTrack]:
        """
        Assign boxes to tracks (greedy IoU).

        Returns a list of PetTrack objects aligned with `boxes` order.
        """
        now = int(ts_ms)
        # Expire old tracks
        for tid in list(self._tracks.keys()):
            tr = self._tracks[tid]
            if now - int(tr.last_seen_ts_ms) > self._expire_ms:
                self._tracks.pop(tid, None)

        tracks = list(self._tracks.values())
        assigned_track_ids: set[int] = set()
        out: List[PetTrack] = []

        for box in boxes:
            best: PetTrack | None = None
            best_iou = 0.0
            for tr in tracks:
                if tr.track_id in assigned_track_ids:
                    continue
                s = iou(tr.last_xyxy, box)
                if s > best_iou:
                    best_iou = s
                    best = tr

            if best is not None and best_iou >= self._iou_thresh:
                # Update existing track
                best.last_xyxy = box
                best.last_seen_ts_ms = now
                best.last_area = _area(box)
                assigned_track_ids.add(best.track_id)
                out.append(best)
                continue

            # New track
            tid = self._next_id
            self._next_id += 1
            tr = PetTrack(
                track_id=tid,
                last_xyxy=box,
                last_seen_ts_ms=now,
                last_area=_area(box),
            )
            self._tracks[tid] = tr
            assigned_track_ids.add(tid)
            out.append(tr)

        return out


__all__ = ["PetTracker", "iou"]

