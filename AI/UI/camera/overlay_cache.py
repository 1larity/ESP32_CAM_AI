from __future__ import annotations

import time
from typing import Tuple

from PySide6 import QtCore, QtGui


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

    new_size = QtCore.QSize(w, h)
    if self._overlay_cache_pixmap is None or self._overlay_cache_pixmap.size() != new_size:
        self._overlay_cache_pixmap = QtGui.QPixmap(w, h)
        self._overlay_cache_pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        self._overlay_cache_dirty = True
        self._overlay_cache_key = None
    # Set scale once per camera; keep stable to avoid flicker/resizes.
    if not hasattr(self, "_overlay_scale"):
        self._overlay_scale = None
    if not hasattr(self, "_overlay_scale_set"):
        self._overlay_scale_set = False
    if not self._overlay_scale_set:
        stored = getattr(self.cam_cfg, "overlay_scale", None)
        if stored is not None:
            self._overlay_scale = float(stored)
            self._overlay_scale_set = True
        elif w > 0 and h > 0:
            self._overlay_scale = self._overlay_scale_factor(w, h)
            self._overlay_scale_set = True
            try:
                self.cam_cfg.overlay_scale = float(self._overlay_scale)
            except Exception:
                pass


def _invalidate_overlay_cache(self) -> None:
    """Call this whenever overlay toggles / AI toggles / visual settings change."""
    self._overlay_cache_dirty = True


def compute_overlay_cache_key(self, pkt, w: int, h: int, now_ms: int) -> tuple:
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
            self._hud_cache_next_ms = (
                (hud_sec + 1) * 1000 if hud_sec is not None else now_ms
            )
            self._overlay_cache_dirty = True

    return (w, h, toggles_key, pkt_key, hud_sec, stats_tick)


__all__ = [
    "_overlay_toggles_key",
    "_ensure_overlay_cache",
    "_invalidate_overlay_cache",
    "compute_overlay_cache_key",
]

