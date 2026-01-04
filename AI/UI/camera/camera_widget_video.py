# camera/camera_widget_video.py
# Video polling + detection handling + overlay rendering (with cached overlay layer)

from __future__ import annotations
import time
from typing import Optional, Tuple
from urllib.parse import urlparse
import cv2
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Slot
from detectors import DetectionPacket
from enrollment import EnrollmentService
from settings import save_settings
from stream import StreamCapture
from utils import debug, monotonic_ms
from .camera_widget_overlay_layer import attach_overlay_layer


def attach_video_handlers(cls) -> None:
    """Inject frame / detector / overlay / HUD helpers into CameraWidget."""

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

    # Attach overlay helpers from dedicated module
    attach_overlay_layer(cls)

    # ----------------------------
    # Frame loop + compositing
    # ----------------------------

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

    # ----------------------------
    # Detector callback
    # ----------------------------

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

        topic_base = getattr(self, "_mqtt_topic", None) or (self.cam_cfg.name or "cam").replace(" ", "_")
        try:
            self._mqtt.publish(f"{topic_base}/counts/person", str(person_count), retain=True)
            self._mqtt.publish(f"{topic_base}/counts/pet", str(pet_count), retain=True)
            names = ",".join(sorted(set(recognized)))
            self._mqtt.publish(f"{topic_base}/recognized", names, retain=True)
        except Exception as e:
            print(f"[MQTT] detection publish error: {e}")

    # ----------------------------
    # Bind injected methods
    # ----------------------------

    cls._poll_frame = _poll_frame
    cls._on_detections = _on_detections
    cls._publish_mqtt_state = _publish_mqtt_state

    cls._snapshot = _snapshot
    cls._toggle_recording = _toggle_recording

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
            if any((f.cls or "").lower() not in ("", "face", "unknown", "person") for f in getattr(pkt, "faces", []) or []):
                triggers.append("person")
            if any(b.cls == "person" for b in getattr(pkt, "yolo", []) or []):
                triggers.append("person")

        if getattr(self.app_cfg, "record_unknown_person", False):
            if any((f.cls or "").lower() in ("", "face", "unknown", "person") for f in getattr(pkt, "faces", []) or []):
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

    # ----------------------------
    # Stream variants (main/sub)
    # ----------------------------

    def _compute_stream_variants(self) -> list[tuple[str, str]]:
        """
        Build a deduped list of (label, url) variants (e.g., main/substream).
        """
        seen: set[str] = set()
        variants: list[tuple[str, str]] = []

        def add(url: Optional[str], label: str) -> None:
            if not url:
                return
            url = url.strip()
            if not url or url in seen:
                return
            seen.add(url)
            variants.append((label, url))

        current = getattr(self.cam_cfg, "stream_url", None)
        add(current, "Primary")

        for u in getattr(self.cam_cfg, "alt_streams", []) or []:
            add(u, "Alt")

        # Heuristic variants for common ONVIF/RTSP layouts
        if current:
            repls = [
                ("/101", "/102"),
                ("/102", "/101"),
                ("/Streaming/Channels/101", "/Streaming/Channels/102"),
                ("/Streaming/Channels/1", "/Streaming/Channels/2"),
                ("/live/ch0", "/live/ch1"),
                ("/ch0", "/ch1"),
            ]
            for old, new in repls:
                if old in current:
                    add(current.replace(old, new, 1), f"Variant {new}")
        return variants

    def _rebuild_stream_menu(self) -> None:
        if not hasattr(self, "menu_stream"):
            return
        self.menu_stream.clear()
        variants = self._compute_stream_variants()
        if not variants:
            act = self.menu_stream.addAction("No variants found")
            act.setEnabled(False)
            return
        for label, url in variants:
            act = self.menu_stream.addAction(f"{label}: {url}")
            act.setData(url)
            act.triggered.connect(lambda _=False, u=url: self._apply_stream_url(u))
        self.menu_stream.addSeparator()
        act_custom = self.menu_stream.addAction("Custom URL...")
        act_custom.triggered.connect(self._prompt_custom_stream)

    def _prompt_custom_stream(self) -> None:
        txt, ok = QtWidgets.QInputDialog.getText(self, "Stream URL", "Enter stream URL:")
        if ok and txt:
            self._apply_stream_url(txt.strip())

    def _apply_stream_url(self, url: str) -> None:
        url = (url or "").strip()
        if not url or url == getattr(self.cam_cfg, "stream_url", None):
            return
        # Pause polling and swap the capture backend.
        if hasattr(self, "_frame_timer"):
            self._frame_timer.stop()
        try:
            self._capture.stop()
        except Exception:
            pass
        self.cam_cfg.stream_url = url
        try:
            # Persist immediately so NVR-friendly substream sticks.
            save_settings(self.app_cfg)
        except Exception:
            pass
        self._capture = StreamCapture(self.cam_cfg)
        self._last_bgr = None
        self._last_pkt = None
        self._overlay_cache_dirty = True
        self._capture.start()
        if hasattr(self, "_frame_timer"):
            self._frame_timer.start()
        # recompute overlay cache after swap
        self._overlay_cache_pixmap = None

    cls._motion_settings = _motion_settings
    cls._update_rec_indicator = _update_rec_indicator
    cls._on_orientation_changed = _on_orientation_changed
    cls._apply_orientation = _apply_orientation
    cls._evaluate_auto_record = _evaluate_auto_record
    cls._motion_trigger = _motion_trigger
    cls._evaluate_motion_only = _evaluate_motion_only
    cls._compute_stream_variants = _compute_stream_variants
    cls._rebuild_stream_menu = _rebuild_stream_menu
    cls._apply_stream_url = _apply_stream_url
    cls._prompt_custom_stream = _prompt_custom_stream
