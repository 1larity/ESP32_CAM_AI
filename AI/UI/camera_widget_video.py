# AI/UI/camera_widget_video.py
# Video polling + detection handling + overlay rendering (with cached overlay layer)

from __future__ import annotations
import time
from typing import Optional, Tuple
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Slot
from detectors import DetectionPacket
from enrollment import EnrollmentService
from utils import qimage_from_bgr
from UI.overlays import draw_overlays
from UI.overlay_stats import FpsCounter, compute_yolo_stats, YoloStats


def attach_video_handlers(cls) -> None:
    """Inject frame / detector / overlay / HUD helpers into CameraWidget."""

    # ----------------------------
    # Overlay caching helpers
    # ----------------------------

    def _overlay_toggles_key(self) -> Tuple[bool, bool, bool, bool, bool, bool]:
        ov = self._overlays
        return (
            bool(getattr(ov, "hud", False)),
            bool(getattr(ov, "yolo", False)),
            bool(getattr(ov, "faces", False)),
            bool(getattr(ov, "pets", False)),
            bool(getattr(ov, "tracks", False)),
            bool(getattr(ov, "stats", False)),
        )

    def _ensure_overlay_cache(self, w: int, h: int) -> None:
        # Lazy init cache fields
        if not hasattr(self, "_overlay_cache_pixmap"):
            self._overlay_cache_pixmap = None
            self._overlay_cache_key = None
            self._overlay_cache_dirty = True
            self._hud_cache_next_ms = 0
            self._stats_cache_next_ms = 0

        if (
            self._overlay_cache_pixmap is None
            or self._overlay_cache_pixmap.size() != QtCore.QSize(w, h)
        ):
            self._overlay_cache_pixmap = QtGui.QPixmap(w, h)
            self._overlay_cache_pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            self._overlay_cache_dirty = True
            self._overlay_cache_key = None

    def _render_overlay_cache(
        self, pkt: Optional[DetectionPacket], w: int, h: int, now_ms: int
    ) -> None:
        """
        Render overlays into a transparent pixmap layer.

        Design:
          - Base video is rendered every frame (QPixmap from QImage)
          - Overlays are cached and only redrawn when "something changes"
        """
        self._overlay_cache_pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(self._overlay_cache_pixmap)
        try:
            # Detection overlays (cached)
            if pkt is not None and getattr(self, "_ai_enabled", False):
                detections_enabled = (
                    getattr(self._overlays, "yolo", False)
                    or getattr(self._overlays, "faces", False)
                    or getattr(self._overlays, "pets", False)
                    or getattr(self._overlays, "tracks", False)
                )
                if detections_enabled:
                    # overlays.py may draw HUD; we handle HUD here
                    orig_hud = getattr(self._overlays, "hud", False)
                    self._overlays.hud = False
                    draw_overlays(painter, pkt, self._overlays)
                    self._overlays.hud = orig_hud

            # HUD overlay (cached, updated on cadence by cache invalidation logic)
            if getattr(self._overlays, "hud", False):
                self._draw_hud(painter)

            # Stats overlay (cached, but cadence-limited by cache invalidation logic)
            if getattr(self._overlays, "stats", False) and getattr(self, "_ai_enabled", False):
                if not hasattr(self, "_fps_counter"):
                    self._fps_counter = FpsCounter()

                fps = self._fps_counter.update()

                boxes = []
                if pkt is not None:
                    boxes.extend(getattr(pkt, "faces", []) or [])
                    boxes.extend(getattr(pkt, "pets", []) or [])
                stats = compute_yolo_stats(boxes)

                self._draw_stats_line(painter, fps, stats, w, h)

        finally:
            painter.end()

    def _invalidate_overlay_cache(self) -> None:
        """Call this whenever overlay toggles / AI toggles / visual settings change."""
        if not hasattr(self, "_overlay_cache_dirty"):
            self._overlay_cache_dirty = True
        else:
            self._overlay_cache_dirty = True

    # ----------------------------
    # Frame loop + compositing
    # ----------------------------

    def _poll_frame(self) -> None:
        ok, frame, ts_ms = self._capture.read()
        if not ok or frame is None:
            return

        self._last_bgr = frame

        # Copy for recorder
        self._recorder.on_frame(frame.copy(), ts_ms)

        # Copy for detector
        if getattr(self, "_ai_enabled", False):
            self._detector.submit_frame(self.cam_cfg.name, frame.copy(), ts_ms)

        # Use last detection packet for a short window to reduce overlay flicker
        pkt_for_frame: Optional[DetectionPacket] = None
        if getattr(self, "_last_pkt", None) is not None:
            age = ts_ms - getattr(self, "_last_pkt_ts", 0)
            if 0 <= age <= getattr(self, "_overlay_ttl_ms", 0):
                pkt_for_frame = self._last_pkt

        self._update_pixmap(frame, pkt_for_frame)

    def _update_pixmap(self, bgr, pkt: Optional[DetectionPacket]) -> None:
        """
        Render base video every frame, overlay layer only when needed,
        then composite base + overlay layer.
        """
        qimg = qimage_from_bgr(bgr)
        base = QtGui.QPixmap.fromImage(qimg)

        w = base.width()
        h = base.height()
        now_ms = int(time.time() * 1000)

        self._ensure_overlay_cache(w, h)

        toggles_key = self._overlay_toggles_key()

        # Detection identity key (must change when overlay content changes)
        pkt_key = None
        if pkt is not None:
            pkt_key = (getattr(pkt, "ts_ms", None), getattr(pkt, "seq", None))

        # HUD changes on second boundary (only matters if HUD enabled)
        hud_sec = int(time.time()) if getattr(self._overlays, "hud", False) else None

        # **CHANGED** throttle stats overlay updates (otherwise caching gives little benefit)
        stats_tick = None
        if getattr(self._overlays, "stats", False) and getattr(self, "_ai_enabled", False):
            if now_ms >= getattr(self, "_stats_cache_next_ms", 0):
                self._stats_cache_next_ms = now_ms + 125  # ~8 Hz
                self._overlay_cache_dirty = True
            stats_tick = self._stats_cache_next_ms

        # **CHANGED** redraw HUD only once per second
        if getattr(self._overlays, "hud", False):
            if now_ms >= getattr(self, "_hud_cache_next_ms", 0):
                self._hud_cache_next_ms = (hud_sec + 1) * 1000
                self._overlay_cache_dirty = True

        cache_key = (w, h, toggles_key, pkt_key, hud_sec, stats_tick)

        if getattr(self, "_overlay_cache_dirty", True) or getattr(self, "_overlay_cache_key", None) != cache_key:
            self._render_overlay_cache(pkt, w, h, now_ms)
            self._overlay_cache_key = cache_key
            self._overlay_cache_dirty = False

        # Composite cached overlays onto base pixmap
        painter = QtGui.QPainter(base)
        try:
            painter.drawPixmap(0, 0, self._overlay_cache_pixmap)
        finally:
            painter.end()

        self._pixmap_item.setPixmap(base)
        self._scene.setSceneRect(QtCore.QRectF(base.rect()))

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

        # Remember last packet for flicker-free overlays
        self._last_pkt = pkt
        self._last_pkt_ts = pkt.ts_ms

        # **CHANGED** overlays depend on new detections; invalidate cache
        self._invalidate_overlay_cache()

        if self._last_bgr is not None:
            EnrollmentService.instance().on_detections(self.cam_cfg.name, self._last_bgr, pkt)
            self._update_pixmap(self._last_bgr, pkt)

    # ----------------------------
    # Overlay drawing primitives
    # ----------------------------

    def _draw_hud(self, p: QtGui.QPainter) -> None:
        text = f"{self.cam_cfg.name}  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        margin = 6

        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        font = p.font()
        font.setPointSize(max(font.pointSize(), 9))
        p.setFont(font)

        fm = QtGui.QFontMetrics(font)
        rect = fm.boundingRect(text)
        rect = QtCore.QRectF(margin, margin, rect.width() + 8, rect.height() + 4)

        bg = QtGui.QColor(0, 0, 0, 128)
        p.fillRect(rect, bg)

        p.drawText(
            rect.adjusted(4, 0, -4, 0),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _draw_stats_line(
        self, p: QtGui.QPainter, fps: float, stats: YoloStats, width: int, height: int
    ) -> None:
        margin = 6
        text = (
            f"FPS: {fps:4.1f} | "
            f"faces: {stats.faces} ({stats.known_faces} known) | "
            f"pets: {stats.pets} | total: {stats.total}"
        )

        font = p.font()
        fm = QtGui.QFontMetrics(font)
        text_width = fm.horizontalAdvance(text)
        text_height = fm.height()

        x = margin
        y = height - margin
        rect = QtCore.QRectF(x - 4, y - text_height - 2, text_width + 8, text_height + 4)

        bg = QtGui.QColor(0, 0, 0, 128)
        p.fillRect(rect, bg)
        p.setPen(QtGui.QColor(255, 255, 255))
        p.drawText(
            rect.adjusted(4, 0, -4, 0),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            text,
        )

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
            self.btn_rec.setText("■ STOP")
        else:
            self._recorder.stop()
            self.btn_rec.setText("● REC")

    # ----------------------------
    # Bind injected methods
    # ----------------------------

    cls._overlay_toggles_key = _overlay_toggles_key
    cls._ensure_overlay_cache = _ensure_overlay_cache
    cls._render_overlay_cache = _render_overlay_cache
    cls._invalidate_overlay_cache = _invalidate_overlay_cache

    cls._poll_frame = _poll_frame
    cls._update_pixmap = _update_pixmap
    cls._on_detections = _on_detections

    cls._draw_hud = _draw_hud
    cls._draw_stats_line = _draw_stats_line

    cls._snapshot = _snapshot
    cls._toggle_recording = _toggle_recording
