from __future__ import annotations

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
        img_path = Path(__file__).resolve().parent.parent / "loadscreen.png"
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
        overlay_lay.setContentsMargins(24, 0, 24, 24)
        overlay_lay.addStretch(1)

        self.lbl_status = QtWidgets.QLabel("Preparing...", self.overlay)
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
            QtCore.QTimer.singleShot(0, self._tick)

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
