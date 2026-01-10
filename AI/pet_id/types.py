from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PetIdConfig:
    """
    Local (no-cloud) pet identification tuning parameters.
    """

    enabled: bool = True
    min_box_px: int = 72
    reid_interval_ms: int = 1200
    track_expire_ms: int = 2500
    iou_match_thresh: float = 0.25
    sim_thresh: float = 0.78
    sim_margin: float = 0.06
    include_auto_labels: bool = False
    max_samples_per_label: int = 200


@dataclass
class PetTrack:
    track_id: int
    last_xyxy: Tuple[int, int, int, int]
    last_seen_ts_ms: int
    last_area: int = 0
    label: str = "unknown"
    label_sim: float = 0.0
    last_reid_ts_ms: int = 0
    pending_label: Optional[str] = None
    pending_sim: float = 0.0
    pending_count: int = 0


__all__ = ["PetIdConfig", "PetTrack"]

