# camera_widget_video.py
# Frame polling, detector, recorder integration and overlays/HUD drawing.

from __future__ import annotations

from typing import Optional
import time

from PyQt6 import QtCore, QtGui

from detectors import DetectionPacket
from enrollment import EnrollmentService
from utils import qimage_from_bgr
from UI.overlays import draw_overlays


def attach_video_handlers(cls) -> None:
    """Inject frame / recorder / overlay handlers into CameraWidget."""

    # ------------------------------------------------------------------ frame / detector / recorder

    def _poll_frame(self) -> None:
        ok, frame, ts_ms = self._capture.read()
        if not ok or frame is None:
            return

        self._last_bgr = frame
        self._last_ts = ts_ms

        # Feed recorder
        self._recorder.on_frame(frame, ts_ms)

        # Feed detector only if AI enabled
        if self._ai_enabled:
            self._detector.submit_frame(self.cam_cfg.name, frame, ts_ms)

        # Use last detection packet for a short window to reduce overlay flicker
        pkt_for_frame: Optional[DetectionPacket] = None
        if self._last_pkt is not None:
            age = ts_ms - self._last_pkt_ts
            if 0 <= age <= self._overlay_ttl_ms:
                pkt_for_frame = self._last_pkt

        # Draw current frame (optionally with last overlays)
        self._update_pixmap(frame, pkt_for_frame)

    def _update_pixmap(self, bgr, pkt: Optional[DetectionPacket]) -> None:
        qimg = qimage_from_bgr(bgr)
        pixmap = QtGui.QPixmap.fromImage(qimg)

        # always paint; detections depend on AI, HUD does not
        painter = QtGui.QPainter(pixmap)
        try:
            # Detection boxes/labels only if:
            # - we have a packet
            # - AI is enabled
            # - at least one detection overlay type is active
            if pkt is not None and self._ai_enabled:
                detections_enabled = (
                    getattr(self._overlays, "yolo", False)
                    or getattr(self._overlays, "faces", False)
                    or getattr(self._overlays, "pets", False)
                    or getattr(self._overlays, "tracks", False)
                )
                if detections_enabled:
                    # Temporarily suppress HUD inside overlays.py; we draw HUD ourselves
                    orig_hud = getattr(self._overlays, "hud", False)
                    self._overlays.hud = False
                    draw_overlays(painter, pkt, self._overlays)
                    self._overlays.hud = orig_hud

            # HUD (camera name + date/time) is independent of AI on/off
            if getattr(self._overlays, "hud", False):
                self._draw_hud(painter)
        finally:
            painter.end()

        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))

    @QtCore.pyqtSlot(object)
    def _on_detections(self, pkt_obj) -> None:
        if not self._ai_enabled:
            return

        pkt = pkt_obj
        if not isinstance(pkt, DetectionPacket):
            return
        if pkt.name != self.cam_cfg.name:
            return

        # Presence log
        self._presence.update(pkt)

        # Remember last packet for flicker-free overlays
        self._last_pkt = pkt
        self._last_pkt_ts = pkt.ts_ms

        if self._last_bgr is not None:
            # Feed enrollment service for this camera (or any, depending on configuration)
            EnrollmentService.instance().on_detections(
                self.cam_cfg.name, self._last_bgr, pkt
            )
            # Draw overlays immediately on top of the last frame
            self._update_pixmap(self._last_bgr, pkt)

    # ------------------------------------------------------------------ HUD

    def _draw_hud(self, p: QtGui.QPainter) -> None:
        """
        Draw camera name + current wall-clock date/time in the top-left corner.
        This does not depend on AI or detection packets.
        """
        text = f"{self.cam_cfg.name}  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        margin = 6
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        font = p.font()
        font.setPointSize(max(font.pointSize(), 9))
        p.setFont(font)

        # Simple text background
        metrics = QtGui.QFontMetrics(font)
        w = metrics.horizontalAdvance(text) + margin * 2
        h = metrics.height() + margin * 2
        rect = QtCore.QRect(margin, margin, w, h)

        bg = QtGui.QColor(0, 0, 0, 128)
        p.fillRect(rect, bg)
        p.drawText(
            rect.adjusted(margin, margin // 2, -margin, 0),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            text,
        )

    # ------------------------------------------------------------------ recording / snapshot helpers

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
        # PrebufferRecorder exposes writer; use that as "is recording" flag.
        if self._recorder.writer is None:
            self._recorder.start()
            self.btn_rec.setText("■ STOP")
        else:
            self._recorder.stop()
            self.btn_rec.setText("● REC")

    # Bind helpers onto the CameraWidget class
    cls._poll_frame = _poll_frame
    cls._update_pixmap = _update_pixmap
    cls._on_detections = _on_detections
    cls._draw_hud = _draw_hud
    cls._snapshot = _snapshot
    cls._toggle_recording = _toggle_recording
