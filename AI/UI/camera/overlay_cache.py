from __future__ import annotations

from pathlib import Path
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
        self._overlay_text_px = None
        self._overlay_text_px_set = False

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

    # Set text size once per stream based on video height.
    # Do not recalculate during the session unless explicitly reset (e.g., ONVIF stream swap).
    if not hasattr(self, "_overlay_text_px"):
        self._overlay_text_px = None
    if not hasattr(self, "_overlay_text_px_set"):
        self._overlay_text_px_set = False
    if not self._overlay_text_px_set and h > 0:
        pct = getattr(self.cam_cfg, "overlay_text_pct", 4.0)
        try:
            pct = float(pct)
        except Exception:
            pct = 4.0
        # Guardrails: keep within a sane UI range.
        pct = max(1.0, min(12.0, pct))
        try:
            self._overlay_text_px = max(8, int(round(float(h) * (pct / 100.0))))
            self._overlay_text_px_set = True
        except Exception:
            self._overlay_text_px = None
            self._overlay_text_px_set = False


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

    # Face params (face_recog.json) may affect overlay rendering (e.g., showing box sizes).
    face_params_mtime = 0
    try:
        models_dir = getattr(getattr(self, "app_cfg", None), "models_dir", None)
        if models_dir:
            face_params_mtime = int((Path(models_dir) / "face_recog.json").stat().st_mtime_ns)
    except Exception:
        face_params_mtime = 0

    return (w, h, toggles_key, pkt_key, hud_sec, stats_tick, face_params_mtime)


__all__ = [
    "_overlay_toggles_key",
    "_ensure_overlay_cache",
    "_invalidate_overlay_cache",
    "compute_overlay_cache_key",
]
