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
        self.setStyleSheet(
            """
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0c1428, stop:1 #1f2f55);
                color: #e8eaf6;
                border: 1px solid #2f4068;
            }
            QLabel#status {
                color: #dde4f2;
                font-size: 14px;
            }
            QLabel#version {
                color: #b7c7e6;
                font-size: 12px;
            }
            QProgressBar {
                background: #0d172a;
                border: 1px solid #3a4b72;
                border-radius: 6px;
                height: 18px;
                text-align: center;
                color: #e0f7ff;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #66ccff, stop:1 #2b86ff);
            }
            """
        )

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Optional overlay image (expected at AI/loadscreen.png)
        self.img = QtWidgets.QLabel(self)
        self.img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._img_orig: QtGui.QPixmap | None = None
        img_path = Path(__file__).resolve().parent / "loadscreen.png"
        if img_path.exists():
            pm = QtGui.QPixmap(str(img_path))
            if not pm.isNull():
                self._img_orig = pm
                self.img.setPixmap(pm)
        if self._img_orig is None:
            self.img.setVisible(False)

        # Overlay area for status/progress on top of the image
        self.overlay = QtWidgets.QWidget(self.img)
        overlay_lay = QtWidgets.QVBoxLayout(self.overlay)
        overlay_lay.setContentsMargins(16, 12, 16, 24)

        # Top-right version label
        self.lbl_version = QtWidgets.QLabel(self.overlay)
        self.lbl_version.setObjectName("version")
        self.lbl_version.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        if self.version:
            self.lbl_version.setText(f"v{self.version}")
        else:
            self.lbl_version.setVisible(False)

        top_lay = QtWidgets.QHBoxLayout()
        top_lay.addStretch(1)
        top_lay.addWidget(self.lbl_version)
        overlay_lay.addLayout(top_lay)
        overlay_lay.addStretch(1)

        self.lbl_status = QtWidgets.QLabel(self._initial_status or "Preparing...", self.overlay)
        self.lbl_status.setObjectName("status")
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.lbl_status.setWordWrap(True)
        overlay_lay.addWidget(
            self.lbl_status, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )

        self.pb = QtWidgets.QProgressBar(self.overlay)
        self.pb.setRange(0, max(1, len(self.cams)))
        self.pb.setValue(0)
        self.pb.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        overlay_lay.addWidget(
            self.pb, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )

        lay.addWidget(self.img)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
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

    # Enable debug (prints + logs to AI/logs/debug.log)
    utils.DEBUG_MODE = DebugMode.BOTH

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP32-CAM AI Viewer")
    app_cfg = load_settings()

    mqtt = MqttService(app_cfg)

    # Lazily create the main window on first camera load (after loader shows).
    win_holder: dict[str, QtWidgets.QMainWindow | None] = {"win": None}

    def _ensure_main_window() -> QtWidgets.QMainWindow:
        if win_holder["win"] is None:
            from UI.main_window import MainWindow  # deferred to avoid heavy imports before loader

            win_holder["win"] = MainWindow(app_cfg, load_on_init=False, mqtt_service=mqtt)
        return win_holder["win"]  # type: ignore[return-value]

    def _safe_update(dlg: StartupDialog, text: str) -> None:
        try:
            if dlg.isVisible():
                dlg.update_status(text)
        except Exception:
            pass

    def _preflight(dlg: StartupDialog) -> None:
        _safe_update(dlg, "Starting...")
        _pump_events(2000)  # brief pause after showing the loader

        if getattr(app_cfg, "mqtt_enabled", False):
            host = getattr(app_cfg, "mqtt_host", None)
            port = getattr(app_cfg, "mqtt_port", 8883)
            if host:
                _safe_update(dlg, f"MQTT: connecting to {host}:{port}...")
            else:
                _safe_update(dlg, "MQTT: enabled but host not set; skipping connect.")
            try:
                mqtt.add_on_connect(lambda _svc: _safe_update(dlg, "MQTT: connected"))
                mqtt.start()
            except Exception as e:
                _safe_update(dlg, f"MQTT: failed to start ({e})")
        else:
            _safe_update(dlg, "MQTT: disabled; skipping")

        def _status_cb(msg: str) -> None:
            _safe_update(dlg, msg)

        # CUDA + model check
        try:
            _safe_update(dlg, "Checking CUDA...")
            import cv2_dll_fix

            cv2_dll_fix.enable_opencv_cuda_dll_search()
            import cv2

            cuda_ok = False
            detail = ""
            if hasattr(cv2, "cuda"):
                try:
                    cnt = cv2.cuda.getCudaEnabledDeviceCount()
                    cuda_ok = bool(cnt and cnt > 0)
                    detail = f"devices={cnt}"
                except Exception as e:
                    detail = str(e)
            else:
                detail = "cv2.cuda missing"

            msg = "CUDA detected; GPU acceleration enabled"
            if not cuda_ok:
                msg = "CUDA not available; using CPU"
                if detail:
                    msg += f" ({detail})"
            _safe_update(dlg, msg)
        except Exception as e:
            _safe_update(dlg, f"CUDA check failed; using CPU ({e})")

        try:
            from models import ModelManager

            ModelManager.ensure_models(app_cfg, status_cb=_status_cb)
        except Exception as e:
            _safe_update(dlg, f"Model check failed: {e}")

    def _load_camera(cam_obj: object) -> None:
        win = _ensure_main_window()
        win._add_camera_window(cam_obj)

    dlg = StartupDialog(
        app_cfg.cameras,
        loader=_load_camera,
        parent=None,
        version=APP_VERSION,
        preflight=_preflight,
        initial_status="Starting...",
        preflight_delay_ms=0,
    )
    dlg.exec()

    win = _ensure_main_window()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
