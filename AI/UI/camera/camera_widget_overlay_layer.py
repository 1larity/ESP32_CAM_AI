# camera_widget_overlay_layer.py
# Overlay cache + HUD/stats drawing helpers for CameraWidget video rendering.
from __future__ import annotations

import time
from typing import Optional, Tuple

from PySide6 import QtCore, QtGui

from ..overlay_stats import compute_yolo_stats, YoloStats, FpsCounter
from ..overlays import draw_overlays
from utils import qimage_from_bgr


def attach_overlay_layer(cls) -> None:
    """Attach overlay cache/drawing helpers to CameraWidget."""

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

    def _overlay_scale_factor(self, w: int, h: int) -> float:
        """Scale overlays relative to video resolution so text remains readable."""
        base = max(1, min(w, h))
        # Quantize to suppress tiny stream size jitters (common on some ONVIF RTSP feeds).
        scale = round(base / 480.0, 2)
        return max(1.0, min(3.0, scale))

    def _ensure_overlay_cache(self, w: int, h: int) -> None:
        # Lazy init cache fields
        if not hasattr(self, "_overlay_cache_pixmap"):
            self._overlay_cache_pixmap = None
            self._overlay_cache_key = None
            self._overlay_cache_dirty = True
            self._hud_cache_next_ms = 0
            self._stats_cache_next_ms = 0

        new_size = QtCore.QSize(w, h)
        if (
            self._overlay_cache_pixmap is None
            or self._overlay_cache_pixmap.size() != new_size
        ):
            self._overlay_cache_pixmap = QtGui.QPixmap(w, h)
            self._overlay_cache_pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            self._overlay_cache_dirty = True
            self._overlay_cache_key = None
        # Set scale once per stream; keep stable to avoid flicker unless unlocked.
        if not hasattr(self, "_overlay_scale_locked"):
            self._overlay_scale_locked = False
        if (not getattr(self, "_overlay_scale_locked", False)) or not hasattr(self, "_overlay_scale"):
            self._overlay_scale = self._overlay_scale_factor(w, h)
            self._overlay_scale_locked = True

    def _render_overlay_cache(
        self, pkt, w: int, h: int, now_ms: int
    ) -> None:
        """
        Render overlays into a transparent pixmap layer.

        Design:
          - Base video is rendered every frame (QPixmap from QImage)
          - Overlays are cached and only redrawn when "something changes"
        """
        self._overlay_cache_pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        scale = float(getattr(self, "_overlay_scale", 1.0) or 1.0)

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
                    draw_overlays(painter, pkt, self._overlays, scale=scale)
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

            # Recording indicator overlay (placed on video, not toolbar)
            if getattr(self, "_rec_indicator_on", False):
                self._draw_rec_indicator(painter, w, h)

        finally:
            painter.end()

    def _invalidate_overlay_cache(self) -> None:
        """Call this whenever overlay toggles / AI toggles / visual settings change."""
        self._overlay_cache_dirty = True

    def _update_pixmap(self, bgr, pkt) -> None:
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

        # Throttle stats overlay updates (otherwise caching gives little benefit)
        stats_tick = None
        if getattr(self._overlays, "stats", False) and getattr(self, "_ai_enabled", False):
            if now_ms >= getattr(self, "_stats_cache_next_ms", 0):
                self._stats_cache_next_ms = now_ms + 125  # ~8 Hz
                self._overlay_cache_dirty = True
            stats_tick = self._stats_cache_next_ms

        # Redraw HUD once per second
        if getattr(self._overlays, "hud", False):
            if now_ms >= getattr(self, "_hud_cache_next_ms", 0):
                self._hud_cache_next_ms = (hud_sec + 1) * 1000 if hud_sec is not None else now_ms
                self._overlay_cache_dirty = True

        cache_key = (w, h, toggles_key, pkt_key, hud_sec, stats_tick)

        if getattr(self, "_overlay_cache_dirty", True) or getattr(self, "_overlay_cache_key", None) != cache_key:
            self._render_overlay_cache(pkt, w, h, now_ms)
            self._overlay_cache_key = cache_key
            self._overlay_cache_dirty = False

        painter = QtGui.QPainter(base)
        try:
            painter.drawPixmap(0, 0, self._overlay_cache_pixmap)
        finally:
            painter.end()

        self._pixmap_item.setPixmap(base)
        self._scene.setSceneRect(QtCore.QRectF(base.rect()))

    def _draw_hud(self, p: QtGui.QPainter) -> None:
        text = f"{self.cam_cfg.name}  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        scale = float(getattr(self, "_overlay_scale", 1.0) or 1.0)
        margin = int(6 * scale)

        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        font = p.font()
        base_pt = font.pointSize() if font.pointSize() > 0 else 9
        font.setPointSize(int(max(base_pt, 9 * scale)))
        p.setFont(font)

        fm = QtGui.QFontMetrics(font)
        rect = fm.boundingRect(text)
        rect = QtCore.QRectF(margin, margin, rect.width() + 8 * scale, rect.height() + 4 * scale)

        bg = QtGui.QColor(0, 0, 0, 128)
        p.fillRect(rect, bg)

        p.drawText(
            rect.adjusted(4 * scale, 0, -4 * scale, 0),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _draw_stats_line(
        self, p: QtGui.QPainter, fps: float, stats: YoloStats, width: int, height: int
    ) -> None:
        scale = float(getattr(self, "_overlay_scale", 1.0) or 1.0)
        margin = int(6 * scale)
        text = (
            f"FPS: {fps:4.1f} | "
            f"faces: {stats.faces} ({stats.known_faces} known) | "
            f"pets: {stats.pets} | total: {stats.total}"
        )

        font = p.font()
        base_pt = font.pointSize() if font.pointSize() > 0 else 9
        font.setPointSize(int(max(base_pt, 9 * scale)))
        fm = QtGui.QFontMetrics(font)
        text_width = fm.horizontalAdvance(text)
        text_height = fm.height()

        x = margin
        y = height - margin
        rect = QtCore.QRectF(x - 4 * scale, y - text_height - 2 * scale, text_width + 8 * scale, text_height + 4 * scale)

        bg = QtGui.QColor(0, 0, 0, 128)
        p.fillRect(rect, bg)
        p.setPen(QtGui.QColor(255, 255, 255))
        p.drawText(
            rect.adjusted(4 * scale, 0, -4 * scale, 0),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            text,
        )

    def _draw_rec_indicator(self, p: QtGui.QPainter, w: int, h: int) -> None:
        scale = float(getattr(self, "_overlay_scale", 1.0) or 1.0)
        margin = int(10 * scale)
        box_w, box_h = int(76 * scale), int(26 * scale)
        rect = QtCore.QRectF(w - box_w - margin, margin, box_w, box_h)
        bg = QtGui.QColor(180, 0, 0, 180)
        p.fillRect(rect, bg)

        dot_r = int(6 * scale)
        dot_center = QtCore.QPointF(rect.left() + 12 * scale, rect.center().y())
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 80, 80)))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(dot_center, dot_r, dot_r)

        p.setPen(QtGui.QPen(QtGui.QColor(255, 230, 230)))
        font = p.font()
        base_pt = font.pointSize() if font.pointSize() > 0 else 9
        font.setPointSize(int(max(base_pt, 9 * scale)))
        font.setBold(True)
        p.setFont(font)
        p.drawText(
            rect.adjusted(24 * scale, 0, -6 * scale, 0),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            "REC",
        )

    cls._overlay_toggles_key = _overlay_toggles_key
    cls._ensure_overlay_cache = _ensure_overlay_cache
    cls._overlay_scale_factor = _overlay_scale_factor
    cls._render_overlay_cache = _render_overlay_cache
    cls._invalidate_overlay_cache = _invalidate_overlay_cache
    cls._update_pixmap = _update_pixmap
    cls._draw_hud = _draw_hud
    cls._draw_stats_line = _draw_stats_line
    cls._draw_rec_indicator = _draw_rec_indicator
