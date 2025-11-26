# camera_widget_video.py
# Frame polling, recorder integration and HUD drawing.

from __future__ import annotations

from typing import Optional

import time

from PyQt6 import QtCore, QtGui

from detectors import DetectionPacket
from utils import qimage_from_bgr
from UI.overlays import draw_overlays


def attach_video_handlers(cls) -> None:
    """Inject frame / recorder / HUD handlers into CameraWidget."""

    def _poll_frame(self) -> None:
        ok, frame, ts_ms = self._capture.read()
        if not ok or frame is None:
            return

        self._last_bgr = frame
        self._last_ts = ts_ms

        # Feed recorder
        self._recorder.on_frame(frame, ts_ms)

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

        painter = QtGui.QPainter(pixmap)
        try:
            # Detection overlays only if:
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
                    # Temporarily suppress HUD inside overlays.draw_overlays;
                    # we draw HUD ourselves below using wall-clock time.
                    orig_hud = getattr(self._overlays, "hud", False)
                    self._overlays.hud = False
                    try:
                        draw_overlays(painter, pkt, self._overlays)
                    finally:
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

        # Cache packet + timestamp for flicker-free overlays
        self._last_pkt = pkt
        self._last_pkt_ts = pkt.ts_ms

        # Presence bus
        self._presence.update(pkt)

        # Recorder events
        self._recorder.on_detections(pkt)

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

    def _draw_hud(self, p: QtGui.QPainter) -> None:
        """Draw camera name + current wall-clock date/time in the top-left corner."""
        text = f"{self.cam_cfg.name}  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        margin = 6
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        font = p.font()
        font.setPointSize(max(font.pointSize(), 9))
        p.setFont(font)
        metrics = p.fontMetrics()
        rect = metrics.tightBoundingRect(text)
        rect = rect.adjusted(-margin, -margin // 2, margin, margin // 2)
        bg = QtGui.QColor(0, 0, 0, 160)
        p.fillRect(rect, bg)
        p.drawText(
            rect.adjusted(margin, margin // 2, -margin, 0),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            text,
        )

    # Bind helpers onto the class
    cls._poll_frame = _poll_frame
    cls._update_pixmap = _update_pixmap
    cls._on_detections = _on_detections
    cls._snapshot = _snapshot
    cls._toggle_recording = _toggle_recording
    cls._draw_hud = _draw_hud
