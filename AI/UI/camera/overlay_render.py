from __future__ import annotations

import time

from PySide6 import QtCore, QtGui

from ..overlay_stats import compute_yolo_stats, FpsCounter
from ..overlays import draw_overlays
from utils import qimage_from_bgr
from .overlay_cache import compute_overlay_cache_key


def _render_overlay_cache(self, pkt, w: int, h: int, now_ms: int) -> None:
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

    cache_key = compute_overlay_cache_key(self, pkt, w, h, now_ms)

    if getattr(self, "_overlay_cache_dirty", True) or getattr(
        self, "_overlay_cache_key", None
    ) != cache_key:
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


__all__ = ["_render_overlay_cache", "_update_pixmap"]

