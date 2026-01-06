from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Slot

from detectors import DetectionPacket
from enrollment import EnrollmentService
from utils import debug, monotonic_ms


def attach_video_loop_handlers(cls) -> None:
    """Inject frame polling + detector callback helpers into CameraWidget."""

    def _poll_frame(self) -> None:
        ok, frame, ts_ms = self._capture.read()
        if not ok or frame is None:
            return

        frame = self._apply_orientation(frame)
        self._last_bgr = frame

        # Frame profiling (throttled) to spot stalls without spamming logs.
        if not hasattr(self, "_frame_stats"):
            self._frame_stats = {"last_ts": ts_ms, "next_log": monotonic_ms() + 2000}
        dt = ts_ms - self._frame_stats["last_ts"]
        self._frame_stats["last_ts"] = ts_ms
        now_ms = monotonic_ms()
        if now_ms >= self._frame_stats["next_log"]:
            fps = 1000.0 / dt if dt > 0 else 0.0
            debug(
                f"[Cam {self.cam_cfg.name}] frame dt={dt}ms (~{fps:.1f} fps) "
                f"rec={'on' if self._recorder.writer else 'off'} "
                f"ai={'on' if getattr(self, '_ai_enabled', False) else 'off'} "
                f"backend={getattr(self._capture, 'last_backend', '?')}"
            )
            self._frame_stats["next_log"] = now_ms + 2000

        # Hand off frame to recorder (recorder buffers a copy internally).
        self._recorder.on_frame(frame, ts_ms)

        # For motion sensitivity (optional)
        self._last_bgr_for_motion = frame
        record_motion, _ = self._motion_settings()
        if record_motion:
            self._evaluate_motion_only()

        # Send the latest frame to the detector thread; detector copies on its side.
        if getattr(self, "_ai_enabled", False):
            self._detector.submit_frame(self.cam_cfg.name, frame, ts_ms)

        # Use last detection packet for a short window to reduce overlay flicker
        pkt_for_frame: Optional[DetectionPacket] = None
        if getattr(self, "_last_pkt", None) is not None:
            age = ts_ms - getattr(self, "_last_pkt_ts", 0)
            if 0 <= age <= getattr(self, "_overlay_ttl_ms", 0):
                pkt_for_frame = self._last_pkt

        self._update_pixmap(frame, pkt_for_frame)

    @Slot(object)
    def _on_detections(self, pkt_obj) -> None:
        if not getattr(self, "_ai_enabled", False):
            return

        pkt = pkt_obj
        if not isinstance(pkt, DetectionPacket):
            return
        if pkt.name != self.cam_cfg.name:
            return

        # Presence log
        self._presence.update(pkt)

        # Auto-recording triggers
        self._evaluate_auto_record(pkt)

        # Remember last packet for flicker-free overlays
        self._last_pkt = pkt
        self._last_pkt_ts = pkt.ts_ms

        # MQTT state publish (counts + recognized names)
        self._publish_mqtt_state(pkt)

        # **CHANGED** overlays depend on new detections; invalidate cache
        self._invalidate_overlay_cache()

        if self._last_bgr is not None:
            EnrollmentService.instance().on_detections(self.cam_cfg.name, self._last_bgr, pkt)
            self._update_pixmap(self._last_bgr, pkt)

    # ----------------------------
    # Actions
    # ----------------------------

    def _snapshot(self) -> None:
        if self._last_bgr is None:
            return
        import cv2
        import time as _time

        fname = f"{self.cam_cfg.name}_{_time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = self.app_cfg.output_dir / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self._last_bgr)

    def _toggle_recording(self) -> None:
        if self._recorder.writer is None:
            self._recorder.start()
            self.btn_rec.setText("STOP")
            self._update_rec_indicator(True)
        else:
            self._recorder.stop()
            self.btn_rec.setText("REC")
            self._update_rec_indicator(False)

    def _publish_mqtt_state(self, pkt: DetectionPacket) -> None:
        if not getattr(self, "_mqtt", None):
            return
        if not getattr(self._mqtt, "connected", False):
            return
        person_count = 0
        pet_count = 0
        recognized = []
        for b in getattr(pkt, "faces", []) or []:
            label = (b.cls or "").strip()
            if label and label.lower() not in ("face", "unknown"):
                person_count += 1
                recognized.append(label)
            else:
                person_count += 1
        for b in getattr(pkt, "yolo", []) or []:
            if b.cls == "person":
                person_count += 1
            if b.cls in ("dog", "cat"):
                pet_count += 1
        for b in getattr(pkt, "pets", []) or []:
            if b.cls in ("dog", "cat"):
                pet_count += 1

        topic_base = getattr(self, "_mqtt_topic", None) or (self.cam_cfg.name or "cam").replace(
            " ", "_"
        )
        try:
            self._mqtt.publish(f"{topic_base}/counts/person", str(person_count), retain=True)
            self._mqtt.publish(f"{topic_base}/counts/pet", str(pet_count), retain=True)
            names = ",".join(sorted(set(recognized)))
            self._mqtt.publish(f"{topic_base}/recognized", names, retain=True)
        except Exception as e:
            print(f"[MQTT] detection publish error: {e}")

    # Bind injected methods
    cls._poll_frame = _poll_frame
    cls._on_detections = _on_detections
    cls._publish_mqtt_state = _publish_mqtt_state
    cls._snapshot = _snapshot
    cls._toggle_recording = _toggle_recording

