from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from settings import BASE_DIR

from .features import crop_xyxy_with_pad, extract_pet_embedding
from .gallery import PetGallery, load_or_build_gallery
from .tracker import PetTracker
from .types import PetIdConfig, PetTrack


class PetIdService:
    """
    Local pet identification service.

    - No cloud / no external process
    - Uses data under AI/data/pets/<name>/ for enrolled pets
    - Annotates DetBox objects in pkt.pets with:
        - id_label: str
        - id_conf: float (cosine similarity 0..1)
        - id_track: int
    """

    _inst: Optional["PetIdService"] = None

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._gallery: Optional[PetGallery] = None
        self._gallery_next_check = 0.0
        self._trackers: Dict[str, PetTracker] = {}

    @classmethod
    def instance(cls) -> "PetIdService":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def _ensure_gallery(self, cfg: PetIdConfig, models_dir: Path) -> PetGallery:
        """
        Load the pet gallery (from cache if possible), rebuilding at most every few seconds.
        """
        now = time.monotonic()
        if self._gallery is not None and now < self._gallery_next_check:
            return self._gallery

        pets_root = BASE_DIR / "data" / "pets"
        gallery = load_or_build_gallery(
            models_dir=Path(models_dir),
            pets_root=pets_root,
            include_auto=bool(cfg.include_auto_labels),
            max_samples_per_label=int(cfg.max_samples_per_label),
        )
        self._gallery = gallery
        self._gallery_next_check = now + 3.0
        return gallery

    def _tracker_for(self, cam_name: str, cfg: PetIdConfig) -> PetTracker:
        tr = self._trackers.get(cam_name)
        if tr is None:
            tr = PetTracker(expire_ms=int(cfg.track_expire_ms), iou_thresh=float(cfg.iou_match_thresh))
            self._trackers[cam_name] = tr
        return tr

    def annotate_packet(
        self,
        cam_name: str,
        bgr: np.ndarray,
        pkt: object,
        *,
        models_dir: Path,
        cfg: PetIdConfig | None = None,
    ) -> None:
        """
        Annotate pkt.pets DetBox objects with per-pet identity labels.
        Safe to call even when no pets are enrolled; it will no-op.
        """
        if pkt is None or bgr is None:
            return
        pets = getattr(pkt, "pets", None) or []
        if not pets:
            return

        config = cfg or PetIdConfig()
        if not getattr(config, "enabled", True):
            return

        valid: list[tuple[object, tuple[int, int, int, int]]] = []
        for d in pets:
            try:
                x1, y1, x2, y2 = map(int, getattr(d, "xyxy", (0, 0, 0, 0)))
            except Exception:
                x1, y1, x2, y2 = 0, 0, 0, 0
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                try:
                    setattr(d, "id_label", "unknown")
                    setattr(d, "id_conf", 0.0)
                except Exception:
                    pass
                continue
            valid.append((d, (x1, y1, x2, y2)))
        if not valid:
            return

        with self._lock:
            gallery = self._ensure_gallery(config, models_dir=Path(models_dir))
            if gallery is None or len(getattr(gallery, "labels", ()) or ()) == 0:
                # Still attach unknown metadata so downstream (unknown capture) can behave sensibly.
                for d in pets:
                    try:
                        setattr(d, "id_label", "unknown")
                        setattr(d, "id_conf", 0.0)
                    except Exception:
                        pass
                return

            tracker = self._tracker_for(cam_name, config)
            boxes = [xyxy for _, xyxy in valid]
            tracks = tracker.assign(ts_ms=int(getattr(pkt, "ts_ms", 0) or 0), boxes=boxes)

            for (det, _), tr in zip(valid, tracks):
                self._annotate_det_for_track(
                    det=det,
                    track=tr,
                    bgr=bgr,
                    gallery=gallery,
                    cfg=config,
                )

    def _annotate_det_for_track(
        self,
        *,
        det: object,
        track: PetTrack,
        bgr: np.ndarray,
        gallery: PetGallery,
        cfg: PetIdConfig,
    ) -> None:
        try:
            x1, y1, x2, y2 = map(int, getattr(det, "xyxy", track.last_xyxy))
        except Exception:
            x1, y1, x2, y2 = track.last_xyxy
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        min_side = min(w, h)

        ts_ms = int(getattr(track, "last_seen_ts_ms", 0) or 0)

        # Default annotations: carry forward track label (even if unknown) for stability.
        label = str(getattr(track, "label", "unknown") or "unknown")
        conf = float(getattr(track, "label_sim", 0.0) or 0.0)

        should_reid = False
        if min_side >= int(cfg.min_box_px):
            # Periodic re-ID per track to keep CPU bounded.
            if int(ts_ms) - int(getattr(track, "last_reid_ts_ms", 0) or 0) >= int(cfg.reid_interval_ms):
                should_reid = True
            # If we previously had no useful label, try again sooner.
            if label in ("", "unknown"):
                should_reid = True

        if should_reid:
            roi = crop_xyxy_with_pad(bgr, (x1, y1, x2, y2))
            feat = extract_pet_embedding(roi) if roi is not None else None
            if feat is not None:
                best_label, best_sim, second_sim = gallery.match(feat)
                best_label = str(best_label or "unknown")
                best_sim = float(best_sim)
                second_sim = float(second_sim)

                accepted = (
                    best_label not in ("", "unknown")
                    and best_sim >= float(cfg.sim_thresh)
                    and (best_sim - second_sim) >= float(cfg.sim_margin)
                )

                if accepted:
                    self._maybe_update_track_label(track, best_label, best_sim)
                else:
                    # If we had a label already, keep it unless we are clearly not confident.
                    if label in ("", "unknown"):
                        track.label = "unknown"
                        track.label_sim = max(0.0, min(1.0, best_sim))
                track.last_reid_ts_ms = ts_ms

                label = str(getattr(track, "label", "unknown") or "unknown")
                conf = float(getattr(track, "label_sim", 0.0) or 0.0)

        try:
            setattr(det, "id_track", int(getattr(track, "track_id", 0) or 0))
        except Exception:
            pass
        try:
            setattr(det, "id_label", label)
            setattr(det, "id_conf", float(max(0.0, min(1.0, conf))))
        except Exception:
            pass

    def _maybe_update_track_label(self, track: PetTrack, new_label: str, new_sim: float) -> None:
        cur = str(getattr(track, "label", "unknown") or "unknown")
        if cur in ("", "unknown"):
            track.label = new_label
            track.label_sim = float(new_sim)
            track.pending_label = None
            track.pending_count = 0
            return

        if new_label == cur:
            # Refresh confidence
            track.label_sim = max(float(track.label_sim), float(new_sim))
            track.pending_label = None
            track.pending_count = 0
            return

        # Require 2 consistent matches to switch away from an existing label.
        pend = getattr(track, "pending_label", None)
        if pend != new_label:
            track.pending_label = new_label
            track.pending_sim = float(new_sim)
            track.pending_count = 1
            return

        track.pending_count = int(getattr(track, "pending_count", 0) or 0) + 1
        track.pending_sim = max(float(getattr(track, "pending_sim", 0.0) or 0.0), float(new_sim))
        if track.pending_count >= 2:
            track.label = new_label
            track.label_sim = float(track.pending_sim)
            track.pending_label = None
            track.pending_count = 0


__all__ = ["PetIdService"]
