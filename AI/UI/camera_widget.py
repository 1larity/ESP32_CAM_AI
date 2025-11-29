# camera_widget.py
# Thin wrapper around the per-camera widget. Most behaviour lives in
# helper modules (camera_widget_init / _video / _overlays / _view).

from __future__ import annotations

from typing import Optional

from PySide6 import QtGui, QtWidgets

from settings import AppSettings, CameraSettings

# Helper initialiser / attach functions
from UI.camera_widget_init import init_camera_widget
from UI.camera_widget_video import attach_video_handlers
from UI.camera_widget_overlays import attach_overlay_handlers
from UI.camera_widget_view import attach_view_handlers


class CameraWidget(QtWidgets.QWidget):
    """
    One camera widget.

    Responsibilities are split into helper modules:
      - init_camera_widget(self)              → build UI, state, wiring
      - attach_video_handlers(CameraWidget)   → frame polling, recorder, HUD
      - attach_overlay_handlers(CameraWidget) → AI / overlay toggles
      - attach_view_handlers(CameraWidget)    → fit / lock helpers
    """

    def __init__(
        self,
        cam_cfg: CameraSettings,
        app_cfg: AppSettings,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.app_cfg = app_cfg

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
        self._detector.stop()
        self._recorder.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self.stop()
        event.accept()


# Attach heavy-weight behaviour from helper modules
attach_video_handlers(CameraWidget)
attach_overlay_handlers(CameraWidget)
attach_view_handlers(CameraWidget)
