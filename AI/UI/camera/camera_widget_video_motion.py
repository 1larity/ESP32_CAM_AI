from __future__ import annotations

import cv2

from detectors import DetectionPacket
from utils import monotonic_ms


def attach_video_motion_handlers(cls) -> None:
    """Inject motion/orientation + auto-record helpers into CameraWidget."""

    def _motion_settings(self) -> tuple[bool, int]:
        cam_motion = getattr(self.cam_cfg, "record_motion", None)
        app_motion = getattr(self.app_cfg, "record_motion", False)
        record_motion = app_motion if cam_motion is None else bool(cam_motion)

        cam_sens = getattr(self.cam_cfg, "motion_sensitivity", None)
        app_sens = getattr(self.app_cfg, "motion_sensitivity", 50)
        try:
            sensitivity = int(app_sens if cam_sens is None else cam_sens)
        except Exception:
            sensitivity = 50
        sensitivity = max(0, min(100, sensitivity))
        return record_motion, sensitivity

    def _update_rec_indicator(self, recording: bool) -> None:
        self._rec_indicator_on = bool(recording)
        # ensure overlay refreshes when state changes
        self._overlay_cache_dirty = True

    def _on_orientation_changed(self) -> None:
        # Stop any ongoing recording to avoid size mismatch errors.
        if hasattr(self, "_recorder"):
            self._recorder.close()
        self._auto_recording_active = False
        self._auto_record_deadline = 0
        self._update_rec_indicator(False)
        if hasattr(self, "btn_rec"):
            self.btn_rec.setText("REC")
        # Reset motion buffers so first frame after change re-seeds baseline.
        if hasattr(self, "_motion_prev"):
            del self._motion_prev
        self._last_bgr_for_motion = None
        self._last_bgr = None
        self._last_pkt = None
        self._overlay_cache_dirty = True

    def _apply_orientation(self, frame):
        rot = int(getattr(self.cam_cfg, "rotation_deg", 0) or 0)
        if rot in (90, 180, 270):
            if rot == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rot == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if getattr(self.cam_cfg, "flip_horizontal", False):
            frame = cv2.flip(frame, 1)
        if getattr(self.cam_cfg, "flip_vertical", False):
            frame = cv2.flip(frame, 0)
        return frame

    # ----------------------------
    # Auto recording helpers
    # ----------------------------

    def _evaluate_auto_record(self, pkt: DetectionPacket) -> None:
        now_ms = monotonic_ms()
        triggers = []

        # Person detection (known/unknown via faces/yolo)
        if getattr(self.app_cfg, "record_person", False):
            if any(
                (f.cls or "").lower() not in ("", "face", "unknown", "person")
                for f in getattr(pkt, "faces", []) or []
            ):
                triggers.append("person")
            if any(b.cls == "person" for b in getattr(pkt, "yolo", []) or []):
                triggers.append("person")

        if getattr(self.app_cfg, "record_unknown_person", False):
            if any(
                (f.cls or "").lower() in ("", "face", "unknown", "person")
                for f in getattr(pkt, "faces", []) or []
            ):
                triggers.append("unknown_person")

        if getattr(self.app_cfg, "record_pet", False):
            if any(b.cls in ("dog", "cat") for b in getattr(pkt, "pets", []) or []):
                triggers.append("pet")

        if getattr(self.app_cfg, "record_unknown_pet", False):
            # We don't have labels beyond cat/dog; treat any pet as unknown if flag set.
            if getattr(pkt, "pets", None):
                triggers.append("unknown_pet")

        record_motion, _ = self._motion_settings()
        if record_motion:
            if self._motion_trigger():
                triggers.append("motion")

        if triggers:
            self._auto_record_deadline = now_ms + 5000
            if self._recorder.writer is None:
                self._recorder.start()
                self._auto_recording_active = True
                self._update_rec_indicator(True)
        else:
            if self._auto_recording_active and now_ms > self._auto_record_deadline:
                self._recorder.stop()
                self._auto_recording_active = False
                self._update_rec_indicator(False)

    def _motion_trigger(self) -> bool:
        if self._last_bgr_for_motion is None:
            return False
        _, sensitivity = self._motion_settings()
        if sensitivity == 0:
            return False
        try:
            # downscale for speed
            small = cv2.resize(self._last_bgr_for_motion, (64, 36))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if not hasattr(self, "_motion_prev"):
                self._motion_prev = gray
                return False
            diff = cv2.absdiff(gray, self._motion_prev)
            self._motion_prev = gray

            # Mean intensity change across the frame
            score = float(np.mean(diff))
            # Fraction of pixels above a small per-pixel delta; catches localized motion
            pix_delta = 4.0 + (100 - sensitivity) * 0.1  # smaller delta at higher sensitivity
            active_frac = float((diff > pix_delta).mean())

            # Higher sensitivity -> lower thresholds
            mean_threshold = 5.0 + (100 - sensitivity) * 0.15  # range ~5..20
            active_threshold = 0.25 - (sensitivity * 0.002)    # range ~0.25..0.05
            active_threshold = max(0.02, min(0.5, active_threshold))

            return score > mean_threshold or active_frac > active_threshold
        except Exception:
            return False

    def _evaluate_motion_only(self) -> None:
        record_motion, _ = self._motion_settings()
        if not record_motion:
            return
        now_ms = monotonic_ms()
        if self._motion_trigger():
            self._auto_record_deadline = now_ms + 5000
            if self._recorder.writer is None:
                self._recorder.start()
                self._auto_recording_active = True
                self._update_rec_indicator(True)
        else:
            if self._auto_recording_active and now_ms > self._auto_record_deadline:
                self._recorder.stop()
                self._auto_recording_active = False
                self._update_rec_indicator(False)

    cls._motion_settings = _motion_settings
    cls._update_rec_indicator = _update_rec_indicator
    cls._on_orientation_changed = _on_orientation_changed
    cls._apply_orientation = _apply_orientation
    cls._evaluate_auto_record = _evaluate_auto_record
    cls._motion_trigger = _motion_trigger
    cls._evaluate_motion_only = _evaluate_motion_only

