from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets

from settings import save_settings
from stream import StreamCapture


def attach_video_stream_handlers(cls) -> None:
    """Inject stream variant helpers into CameraWidget."""

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

    cls._compute_stream_variants = _compute_stream_variants
    cls._rebuild_stream_menu = _rebuild_stream_menu
    cls._apply_stream_url = _apply_stream_url
    cls._prompt_custom_stream = _prompt_custom_stream

