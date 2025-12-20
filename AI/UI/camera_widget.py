# camera_widget.py
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
    ) -> None:
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.app_cfg = app_cfg

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


# Keep module-level attachment too (harmless with the class guard above).
attach_video_handlers(CameraWidget)
attach_overlay_handlers(CameraWidget)
attach_view_handlers(CameraWidget)
