from __future__ import annotations

from .overlay_cache import _ensure_overlay_cache, _invalidate_overlay_cache, _overlay_toggles_key
from .overlay_draw import (
    _draw_hud,
    _draw_rec_indicator,
    _draw_stats_line,
    _overlay_scale_factor,
)
from .overlay_render import _render_overlay_cache, _update_pixmap


def attach_overlay_layer(cls) -> None:
    """Attach overlay cache/drawing helpers to CameraWidget."""

    cls._overlay_toggles_key = _overlay_toggles_key
    cls._ensure_overlay_cache = _ensure_overlay_cache
    cls._overlay_scale_factor = _overlay_scale_factor
    cls._render_overlay_cache = _render_overlay_cache
    cls._invalidate_overlay_cache = _invalidate_overlay_cache
    cls._update_pixmap = _update_pixmap
    cls._draw_hud = _draw_hud
    cls._draw_stats_line = _draw_stats_line
    cls._draw_rec_indicator = _draw_rec_indicator
