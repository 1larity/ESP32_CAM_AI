from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Callable, Sequence
from PySide6 import QtCore, QtGui, QtWidgets

class StartupDialog(QtWidgets.QDialog):
    """
    Simple static loader dialog shown while wiring cameras.
    """

    def __init__(
        self,
        cams: Sequence[object],
        loader: Callable[[object], None],
        parent: QtWidgets.QWidget | None = None,
        version: str | None = None,
        preflight: Callable[["StartupDialog"], None] | None = None,
        initial_status: str | None = None,
        preflight_delay_ms: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.CustomizeWindowHint
        )
        self.setModal(True)

        # Size to 20% of available screen and center it; keep a sensible minimum.
        screen_geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        w = max(480, int(screen_geo.width() * 0.2))
        h = max(240, int(screen_geo.height() * 0.2))
        self.setFixedSize(w, h)
        centered = QtWidgets.QStyle.alignedRect(
            QtCore.Qt.LayoutDirection.LeftToRight,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            self.size(),
            screen_geo,
        )
        self.setGeometry(centered)

        self.cams = list(cams)
        self.loader = loader
        self._idx = 0
        self._started = False
        self.version = version
        self._preflight = preflight
        self._initial_status = initial_status
        self._preflight_delay_ms = max(0, int(preflight_delay_ms))

        self._build_ui()

    def _build_ui(self) -> None:
        from startup_dialog_ui import build_startup_dialog_ui

        ui = build_startup_dialog_ui(
            self,
            cams=self.cams,
            version=self.version,
            initial_status=self._initial_status,
            image_path=Path(__file__).resolve().parent / "loadscreen.png",
        )
        self.img = ui.img
        self.overlay = ui.overlay
        self.lbl_version = ui.lbl_version
        self.lbl_status = ui.lbl_status
        self.pb = ui.pb
        self._img_orig = ui.img_orig

    def _refresh_image_and_overlay(self) -> None:
        if self._img_orig is not None:
            pm = self._img_orig.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.img.setPixmap(pm)
        self._update_overlay_geometry()

    def _update_overlay_geometry(self) -> None:
        """Keep overlay full-size and status wide enough to avoid cropping."""
        w = self.width()
        h = self.height()
        self.overlay.setGeometry(0, 0, w, h)
        self.lbl_status.setMinimumWidth(int(w * 0.9))
        self.pb.setMinimumWidth(int(w * 0.85))

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._refresh_image_and_overlay()
        if not self._started:
            self._started = True
            self.pb.setMaximum(max(1, len(self.cams)))
            if self._preflight:
                # Run preflight on the next tick so the dialog can paint first,
                # then begin camera loading.
                QtCore.QTimer.singleShot(
                    0, lambda: self._run_preflight_then_tick()
                )
            else:
                QtCore.QTimer.singleShot(self._preflight_delay_ms, self._tick)

    def _run_preflight_then_tick(self) -> None:
        if not self._preflight:
            QtCore.QTimer.singleShot(self._preflight_delay_ms, self._tick)
            return
        try:
            self._preflight(self)
        except Exception:
            pass
        QtCore.QTimer.singleShot(self._preflight_delay_ms, self._tick)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        super().closeEvent(event)

    def _tick(self) -> None:
        if self._idx >= len(self.cams):
            self.pb.setValue(self.pb.maximum())
            self.accept()
            return

        cam = self.cams[self._idx]
        label = getattr(cam, "name", f"Camera {self._idx + 1}")
        self.lbl_status.setText(f"Connecting to {label} ({self._idx + 1}/{len(self.cams)})")
        self.pb.setValue(self._idx)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)

        try:
            self.loader(cam)
        except Exception:
            pass

        self._idx += 1
        QtCore.QTimer.singleShot(0, self._tick)

    def update_status(self, text: str) -> None:
        """Update status label and process events so it paints immediately."""
        self.lbl_status.setText(text)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)


def _pump_events(ms: int) -> None:
    """Wait for ms while letting the UI repaint."""
    end = time.time() + (ms / 1000.0)
    while time.time() < end:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        QtCore.QThread.msleep(50)


def main() -> None:
    from mdi_app import APP_VERSION, maybe_run_profile_cli

    maybe_run_profile_cli()

    # Keep imports light until after the loader is visible.
    from PySide6 import QtWidgets
    from settings import load_settings
    import utils
    from utils import DebugMode
    from mqtt_client import MqttService
    from mqtt_init import init_mqtt
    from CUDA_init import init_cuda
    from model_init import init_models
    from startup_flow import StartupFlow

    # Debug flags (set here)
    # - `utils.DEBUG_MODE` writes to `AI/logs/debug.log`
    # - `utils.PTZ_DEBUG_MODE` writes to `AI/logs/ptz_debug.log`
    # Options: DebugMode.OFF | DebugMode.PRINT | DebugMode.LOG | DebugMode.BOTH
    utils.DEBUG_MODE = DebugMode.OFF
    utils.PTZ_DEBUG_MODE = DebugMode.PRINT

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP32-CAM AI Viewer")
    app_cfg = load_settings()

    mqtt = MqttService(app_cfg)
    flow = StartupFlow(
        app_cfg=app_cfg,
        mqtt=mqtt,
        pump_events=_pump_events,
        init_mqtt=init_mqtt,
        init_cuda=init_cuda,
        init_models=init_models,
    )

    dlg = StartupDialog(
        app_cfg.cameras,
        loader=flow.load_camera,
        parent=None,
        version=APP_VERSION,
        preflight=flow.preflight,
        initial_status="Starting...",
        preflight_delay_ms=0,
    )
    dlg.exec()

    win = flow.ensure_main_window()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
