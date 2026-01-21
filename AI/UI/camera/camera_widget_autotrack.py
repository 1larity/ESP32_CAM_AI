from __future__ import annotations

from utils import monotonic_ms
from settings import save_settings

# UI glue for PTZ auto-tracking (moving object tracking).
# This module wraps some PTZ methods to:
#   - pause auto-track after any manual PTZ action
#   - expose a small status string for overlays
#   - provide a toggle handler usable by menu actions / dialogs


def attach_autotrack_handlers(cls) -> None:
    """Attach auto-track helpers to CameraWidget."""

    if getattr(cls, "__autotrack_attached", False):
        return
    cls.__autotrack_attached = True  # type: ignore[attr-defined]

    # -------------------------
    # Helpers
    # -------------------------

    def _autotrack_note_manual(self) -> None:
        """Pause auto-track after manual PTZ input."""
        try:
            hold = int(getattr(self.cam_cfg, "autotrack_manual_hold_ms", 1200) or 1200)
        except Exception:
            hold = 1200
        self._autotrack_manual_until_ms = int(monotonic_ms()) + max(0, hold)

    def _autotrack_status_text(self) -> str:
        at = getattr(self, "_autotrack", None)
        if at is None:
            return ""
        try:
            return str(at.status_text() or "")
        except Exception:
            return ""

    def _on_autotrack_toggled(self, checked: bool) -> None:
        self.cam_cfg.autotrack_enabled = bool(checked)
        # Persist immediately (consistent with other per-cam settings)
        try:
            save_settings(self.app_cfg)
        except Exception:
            pass
        try:
            self._invalidate_overlay_cache()
        except Exception:
            pass

    cls._autotrack_note_manual = _autotrack_note_manual
    cls._autotrack_status_text = _autotrack_status_text
    cls._on_autotrack_toggled = _on_autotrack_toggled

    # -------------------------
    # Wrappers: mark manual PTZ
    # -------------------------

    if hasattr(cls, "_ptz_set_velocity") and not getattr(cls, "__autotrack_wrapped", False):
        cls.__autotrack_wrapped = True  # type: ignore[attr-defined]

        _orig_set_velocity = cls._ptz_set_velocity

        def _ptz_set_velocity(self, pan: float, tilt: float, zoom: float) -> None:
            # If this call is not coming from the auto-tracker, treat it as manual.
            if not bool(getattr(self, "_autotrack_driving", False)):
                # Only note manual if there is *some* movement request.
                if abs(float(pan)) > 1e-6 or abs(float(tilt)) > 1e-6 or abs(float(zoom)) > 1e-6:
                    try:
                        self._autotrack_note_manual()
                    except Exception:
                        pass
            return _orig_set_velocity(self, pan, tilt, zoom)

        cls._ptz_set_velocity = _ptz_set_velocity

    # Also wrap “jump” commands that don't go through set_velocity
    for _name in ("_ptz_enqueue_home", "_ptz_enqueue_preset", "_ptz_enqueue_stop"):
        if hasattr(cls, _name):
            orig = getattr(cls, _name)

            def _wrap(orig_fn):
                def _f(self, *a, **kw):
                    if not bool(getattr(self, "_autotrack_driving", False)):
                        try:
                            self._autotrack_note_manual()
                        except Exception:
                            pass
                    return orig_fn(self, *a, **kw)

                return _f

            setattr(cls, _name, _wrap(orig))
