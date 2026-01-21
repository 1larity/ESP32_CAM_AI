from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from detectors import DetectionPacket
from utils import monotonic_ms

from .box_tracker import SingleTrack

XYXY = Tuple[int, int, int, int]


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _center(bb: XYXY) -> Tuple[float, float]:
    x1, y1, x2, y2 = bb
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _area(bb: XYXY) -> int:
    x1, y1, x2, y2 = bb
    return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class _Cfg:
    enabled: bool
    source: str  # any|yolo|faces|pets
    classes: List[str]

    deadzone: float
    kp_pan: float
    kp_tilt: float

    max_vel: float
    min_vel: float
    smooth_alpha: float

    iou_min: float
    lost_ms: int
    manual_hold_ms: int

    zoom_enabled: bool
    zoom_target_frac: float


class PtzAutoTracker:
    """PTZ auto-track controller.

    Consumes DetectionPacket boxes and drives widget PTZ velocity to keep a
    selected target centered. The CameraWidget already maps pan/tilt to match
    display orientation; therefore this controller outputs screen-style:
      - pan > 0 = move right
      - tilt > 0 = move up
    """

    def __init__(self, widget):
        self.w = widget
        self.track = SingleTrack()
        self._pan = 0.0
        self._tilt = 0.0
        self._zoom = 0.0
        self._active = False
        self._last_packet_ms = 0  # **CHANGED** watchdog for detector stalls

    # -------------------------
    # Public API
    # -------------------------

    def status_text(self) -> str:
        if not self._cfg().enabled:
            return ""
        if not bool(getattr(self.w, "_ptz_available", False)):
            return "AT:off"
        if self._active and self.track.bbox is not None:
            lbl = (self.track.label or "").strip()
            return f"AT:{lbl or 'target'}"
        return "AT:on"

    def watchdog(self) -> None:
        """Stop PTZ if detections stall while a non-zero velocity is active."""
        cfg = self._cfg()
        if not cfg.enabled:
            self._safe_stop()
            return
        if not bool(getattr(self.w, "_ptz_available", False)):
            self._safe_stop()
            return
        # **CHANGED** if we haven't received a detector packet recently, stop to prevent runaway.
        stale_ms = int(getattr(getattr(self.w, "cam_cfg", None), "autotrack_stale_stop_ms", 900) or 900)
        if self._active and self._last_packet_ms:
            if (monotonic_ms() - int(self._last_packet_ms)) > max(100, stale_ms):
                self._safe_stop()

    def on_packet(self, pkt: DetectionPacket) -> None:
        cfg = self._cfg()

        self._last_packet_ms = int(monotonic_ms())  # **CHANGED** update watchdog timestamp

        # If disabled, ensure we are stopped.
        if not cfg.enabled:
            self._safe_stop()
            return

        if not bool(getattr(self.w, "_ptz_available", False)):
            self._safe_stop()
            return

        if not bool(getattr(self.w, "_ai_enabled", False)):
            # Auto-track is detection-driven; if AI is off, stop.
            self._safe_stop()
            return

        # Pause after manual PTZ input.
        now = monotonic_ms()
        if now < int(getattr(self.w, "_autotrack_manual_until_ms", 0) or 0):
            self._safe_stop()
            return

        # Update / select target
        dets = list(self._candidates(pkt, cfg))
        ts_ms = int(getattr(pkt, "ts_ms", 0) or 0)

        picked = self.track.update(
            dets,
            ts_ms=ts_ms,
            iou_min=cfg.iou_min,
            prefer_labels=self._prefer_labels(cfg),
        )

        if picked is None:
            if self.track.last_seen_ts_ms and (ts_ms - self.track.last_seen_ts_ms) > int(cfg.lost_ms):
                self.track.clear()
            else:
                self.track.clear()
            self._safe_stop()  # **CHANGED** stop immediately when no target this packet (prevents endstop runaways)
            return

        lbl, sc, bb = picked

        # Compute normalized offsets from center (-1..+1)
        w, h = pkt.size
        cx, cy = _center(bb)
        ox = (cx - (w * 0.5)) / (w * 0.5) if w > 0 else 0.0
        oy = (cy - (h * 0.5)) / (h * 0.5) if h > 0 else 0.0

        dz = float(cfg.deadzone)
        if abs(ox) < dz:
            ox = 0.0
        if abs(oy) < dz:
            oy = 0.0

        # Proportional controller:
        # - pan follows ox (right is +)
        # - tilt is inverted because image y-down, command wants y-up
        pan = _clamp(cfg.kp_pan * ox, -cfg.max_vel, cfg.max_vel)
        tilt = _clamp(-cfg.kp_tilt * oy, -cfg.max_vel, cfg.max_vel)

        zoom = 0.0
        if cfg.zoom_enabled and bool(getattr(self.w, "_ptz_has_zoom", False)):
            frac = _area(bb) / float(max(1, w * h))
            target = float(cfg.zoom_target_frac)
            # Simple bang-bang with proportional-ish scaling
            err = _clamp((target - frac) / max(1e-6, target), -1.0, 1.0)
            zoom = _clamp(0.5 * err, -cfg.max_vel, cfg.max_vel)

        # Smooth output to avoid jitter.
        a = _clamp(cfg.smooth_alpha, 0.0, 1.0)
        self._pan = (a * pan) + ((1.0 - a) * self._pan)
        self._tilt = (a * tilt) + ((1.0 - a) * self._tilt)
        self._zoom = (a * zoom) + ((1.0 - a) * self._zoom)

        # Enforce a minimum movement magnitude for "dumb" PTZ heads that ignore
        # small velocities (they often have a deadband and only respond above a threshold).
        min_v = float(cfg.min_vel)
        if min_v > 0.0:
            if self._pan != 0.0:
                self._pan = _clamp(
                    (1.0 if self._pan > 0 else -1.0) * max(abs(self._pan), min_v), -cfg.max_vel, cfg.max_vel
                )
            if self._tilt != 0.0:
                self._tilt = _clamp(
                    (1.0 if self._tilt > 0 else -1.0) * max(abs(self._tilt), min_v), -cfg.max_vel, cfg.max_vel
                )
            if self._zoom != 0.0:
                self._zoom = _clamp(
                    (1.0 if self._zoom > 0 else -1.0) * max(abs(self._zoom), min_v), -cfg.max_vel, cfg.max_vel
                )

        # If everything is zero, stop.
        if self._pan == 0.0 and self._tilt == 0.0 and self._zoom == 0.0:
            self._safe_stop()
            return

        self._active = True
        self._send_velocity(self._pan, self._tilt, self._zoom)

    # -------------------------
    # Internals
    # -------------------------

    def _cfg(self) -> _Cfg:
        cam = getattr(self.w, "cam_cfg", None)

        def _get(name: str, default):
            try:
                return getattr(cam, name)
            except Exception:
                return default

        enabled = bool(_get("autotrack_enabled", False))
        source = str(_get("autotrack_source", "any") or "any").strip().lower()
        classes = str(_get("autotrack_classes", "person,dog,cat") or "person,dog,cat")
        cls_list = [c.strip().lower() for c in classes.split(",") if c.strip()]
        return _Cfg(
            enabled=enabled,
            source=source if source in ("any", "yolo", "faces", "pets") else "any",
            classes=cls_list,
            deadzone=float(_get("autotrack_deadzone", 0.06) or 0.06),
            kp_pan=float(_get("autotrack_kp_pan", 0.70) or 0.70),
            kp_tilt=float(_get("autotrack_kp_tilt", 0.70) or 0.70),
            max_vel=_clamp(float(_get("autotrack_max_vel", 0.70) or 0.70), 0.10, 1.00),
            min_vel=_clamp(float(_get("autotrack_min_vel", 0.00) or 0.00), 0.0, 1.00),
            smooth_alpha=_clamp(float(_get("autotrack_smooth_alpha", 0.45) or 0.45), 0.0, 1.0),
            iou_min=_clamp(float(_get("autotrack_iou_min", 0.12) or 0.12), 0.0, 1.0),
            lost_ms=int(_get("autotrack_lost_ms", 1500) or 1500),
            manual_hold_ms=int(_get("autotrack_manual_hold_ms", 1200) or 1200),
            zoom_enabled=bool(_get("autotrack_zoom_enabled", False)),
            zoom_target_frac=_clamp(float(_get("autotrack_zoom_target_frac", 0.28) or 0.28), 0.02, 0.90),
        )

    def _prefer_labels(self, cfg: _Cfg) -> Optional[List[str]]:
        # If user specified classes, we keep that order as preference.
        return cfg.classes or None

    def _candidates(self, pkt: DetectionPacket, cfg: _Cfg) -> Iterable[Tuple[str, float, XYXY]]:
        boxes: List[Tuple[object, str]] = []  # **CHANGED** keep source kind for class matching
        if cfg.source in ("any", "yolo") and getattr(pkt, "yolo", None):
            boxes.extend([(b, "yolo") for b in (getattr(pkt, "yolo", []) or [])])
        if cfg.source in ("any", "faces") and getattr(pkt, "faces", None):
            boxes.extend([(b, "faces") for b in (getattr(pkt, "faces", []) or [])])
        if cfg.source in ("any", "pets") and getattr(pkt, "pets", None):
            boxes.extend([(b, "pets") for b in (getattr(pkt, "pets", []) or [])])

        for b, kind in boxes:
            try:
                lbl = str(getattr(b, "cls", "") or "").strip()
                lbl_l = lbl.lower()
                sc = float(getattr(b, "score", 0.0) or 0.0)
                bb = getattr(b, "xyxy", None)
                if not bb or len(bb) != 4:
                    continue
                xyxy = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                if cfg.classes:
                    classes_set = {c.strip().lower() for c in cfg.classes if c}
                    if ("any" in classes_set) or ("all" in classes_set) or ("*" in classes_set):
                        pass
                    elif lbl_l in classes_set:
                        pass
                    elif kind == "faces" and bool(
                        classes_set & {"face", "faces", "person", "people"}
                    ):
                        pass
                    elif kind == "pets" and bool(classes_set & {"pet", "pets"}):
                        pass
                    else:
                        continue
                yield (lbl, sc, xyxy)
            except Exception:
                continue

    def _safe_stop(self) -> None:
        if self._active:
            self._active = False
            self._pan = self._tilt = self._zoom = 0.0
            self._send_velocity(0.0, 0.0, 0.0)

    def _send_velocity(self, pan: float, tilt: float, zoom: float) -> None:
        # Mark as driving so the CameraWidget wrapper doesn't treat it as manual control.
        try:
            setattr(self.w, "_autotrack_driving", True)
            self.w._ptz_set_velocity(float(pan), float(tilt), float(zoom))
        finally:
            try:
                setattr(self.w, "_autotrack_driving", False)
            except Exception:
                pass
