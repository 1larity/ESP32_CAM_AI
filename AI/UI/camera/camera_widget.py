# camera/camera_widget.py
from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse
import requests

from PySide6 import QtCore, QtGui, QtWidgets

from settings import AppSettings, CameraSettings, save_settings
from .camera_settings_dialog import CameraSettingsDialog

# Helper initialiser / attach functions
from .camera_widget_init import init_camera_widget
from .camera_widget_video import attach_video_handlers
from .camera_widget_overlays import attach_overlay_handlers
from .camera_widget_view import attach_view_handlers

# Map framesize enum -> (label, width, height) per esp32-camera sensor.h
FRAME_SIZES = {
    0: ("96x96", 96, 96),
    1: ("QQVGA", 160, 120),
    2: ("QCIF", 176, 144),
    3: ("HQVGA", 240, 176),
    4: ("240x240", 240, 240),
    5: ("QVGA", 320, 240),
    6: ("CIF", 352, 288),
    7: ("HVGA", 480, 320),
    8: ("VGA", 640, 480),
    9: ("SVGA", 800, 600),
    10: ("XGA", 1024, 768),
    11: ("SXGA", 1280, 1024),
    12: ("UXGA", 1600, 1200),
    13: ("QXGA", 2048, 1536),
}


class CameraWidget(QtWidgets.QWidget):
    """
    One camera widget.

    Responsibilities are split into helper modules:
      - init_camera_widget(self)               → build UI, state, wiring
      - attach_video_handlers(CameraWidget)    → frame polling, recorder, HUD, detections handler
      - attach_overlay_handlers(CameraWidget)  → AI / overlay toggles
      - attach_view_handlers(CameraWidget)     → fit / lock helpers
    """

    # Class-level guard: ensures injected handlers exist before init wiring connects signals.
    _handlers_attached: bool = False

    def __init__(
        self,
        cam_cfg: CameraSettings,
        app_cfg: AppSettings,
        parent: Optional[QtWidgets.QWidget] = None,
        mqtt_service=None,
    ) -> None:
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.app_cfg = app_cfg
        self._mqtt = mqtt_service

        # IMPORTANT:
        # Ensure injected methods (including _on_detections) exist BEFORE init_camera_widget()
        # connects signals to them. This avoids startup crashes if module import ordering causes
        # attach_* not to have run yet.
        if not self.__class__._handlers_attached:
            attach_video_handlers(self.__class__)
            attach_overlay_handlers(self.__class__)
            attach_view_handlers(self.__class__)
            self.__class__._handlers_attached = True

        # Delegate all heavy init work
        init_camera_widget(self)

    # Lifecycle entry points used by MainWindow
    def start(self) -> None:
        self._capture.start()
        self._detector.start()
        self._frame_timer.start()

    def stop(self) -> None:
        self._frame_timer.stop()
        self._capture.stop()
        self._detector.stop(wait_ms=2000)
        self._recorder.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self.stop()
        event.accept()

    # ------------------------------------------------------------------ #
    # Info dialog
    # ------------------------------------------------------------------ #

    def _api_status_url(self) -> str | None:
        """
        Build the API status URL for this camera based on its stream URL.
        Defaults to http://<host>:80/api/status.
        """
        parsed = urlparse(self.cam_cfg.effective_url())
        host = parsed.hostname
        if not host:
            return None
        port = 80
        return f"http://{host}:{port}/api/status"

    def _show_info(self) -> None:
        url = self._api_status_url()
        if not url:
            QtWidgets.QMessageBox.warning(
                self, "Camera Info", "Cannot determine API URL for this camera."
            )
            return

        try:
            auth = None
            if self.cam_cfg.user and self.cam_cfg.password:
                auth = requests.auth.HTTPBasicAuth(self.cam_cfg.user, self.cam_cfg.password)
            resp = requests.get(url, auth=auth, timeout=3)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Camera Info",
                f"Failed to fetch camera info from {url}:\n{e}",
            )
            return

        ip = data.get("ip") or urlparse(url).hostname or "n/a"
        fs_code = data.get("framesize")
        fs_name, fs_w, fs_h = FRAME_SIZES.get(fs_code, (str(fs_code), None, None))
        ptz = data.get("ptz") or {}
        pan = ptz.get("pan", "n/a")
        tilt = ptz.get("tilt", "n/a")

        text = (
            f"IP: {ip}\n"
            f"Framesize: {fs_name} (code {fs_code})\n"
            f"Pixels: {fs_w if fs_w else 'n/a'} x {fs_h if fs_h else 'n/a'}\n"
            f"PTZ: pan={pan}, tilt={tilt}"
        )

        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle(f"Camera Info - {self.cam_cfg.name}")
        dlg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        dlg.setText(text)
        dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dlg.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        dlg.exec()

    def _sync_flash_from_camera(self) -> None:
        """
        Query camera /api/status for flash state and align local mode/level so we
        don't override the hardware state on startup.
        """
        url = self._api_status_url()
        if not url:
            return
        try:
            auth = None
            if self.cam_cfg.user and self.cam_cfg.password:
                auth = requests.auth.HTTPBasicAuth(self.cam_cfg.user, self.cam_cfg.password)
            resp = requests.get(url, auth=auth, timeout=2.0)
            resp.raise_for_status()
            data = resp.json()
            flash_on = bool(data.get("flash", False))
            level = int(data.get("flash_level", 0) or 0)
            level = max(0, min(255, level))
            self._flash_level = level
            self.cam_cfg.flash_level = level
            self._flash_mode = "on" if (flash_on and level > 0) else "off"
            self.cam_cfg.flash_mode = self._flash_mode
        except Exception:
            # Best-effort; keep existing config if camera unreachable.
            return

    def _open_camera_settings(self) -> None:
        dlg = CameraSettingsDialog(self.cam_cfg, self.app_cfg, self, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            dlg.apply()
            save_settings(self.app_cfg)
            # make sure any motion toggle/sensitivity changes apply immediately
            self._overlay_cache_dirty = True


# Keep module-level attachment too (harmless with the class guard above).
attach_video_handlers(CameraWidget)
attach_overlay_handlers(CameraWidget)
attach_view_handlers(CameraWidget)
