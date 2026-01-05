from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PySide6 import QtCore, QtGui, QtWidgets


STARTUP_DIALOG_STYLESHEET = """
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


@dataclass(frozen=True)
class StartupDialogUI:
    img: QtWidgets.QLabel
    overlay: QtWidgets.QWidget
    lbl_version: QtWidgets.QLabel
    lbl_status: QtWidgets.QLabel
    pb: QtWidgets.QProgressBar
    img_orig: QtGui.QPixmap | None


def build_startup_dialog_ui(
    dialog: QtWidgets.QDialog,
    *,
    cams: Sequence[object],
    version: str | None,
    initial_status: str | None,
    image_path: Path,
) -> StartupDialogUI:
    dialog.setStyleSheet(STARTUP_DIALOG_STYLESHEET)

    lay = QtWidgets.QVBoxLayout(dialog)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)

    # overlay image AI/loadscreen.png
    img = QtWidgets.QLabel(dialog)
    img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    img_orig: QtGui.QPixmap | None = None
    if image_path.exists():
        pm = QtGui.QPixmap(str(image_path))
        if not pm.isNull():
            img_orig = pm
            img.setPixmap(pm)
    if img_orig is None:
        img.setVisible(False)

    # Overlay area for status/progress on top of the image
    overlay = QtWidgets.QWidget(img)
    overlay_lay = QtWidgets.QVBoxLayout(overlay)
    overlay_lay.setContentsMargins(16, 12, 16, 24)

    # Top-right version label
    lbl_version = QtWidgets.QLabel(overlay)
    lbl_version.setObjectName("version")
    lbl_version.setAlignment(
        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
    )
    if version:
        lbl_version.setText(f"v{version}")
    else:
        lbl_version.setVisible(False)

    top_lay = QtWidgets.QHBoxLayout()
    top_lay.addStretch(1)
    top_lay.addWidget(lbl_version)
    overlay_lay.addLayout(top_lay)
    overlay_lay.addStretch(1)

    lbl_status = QtWidgets.QLabel(initial_status or "Preparing...", overlay)
    lbl_status.setObjectName("status")
    lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    lbl_status.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Preferred,
    )
    lbl_status.setWordWrap(True)
    overlay_lay.addWidget(lbl_status, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

    pb = QtWidgets.QProgressBar(overlay)
    pb.setRange(0, max(1, len(cams)))
    pb.setValue(0)
    pb.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Fixed,
    )
    overlay_lay.addWidget(pb, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

    lay.addWidget(img)

    return StartupDialogUI(
        img=img,
        overlay=overlay,
        lbl_version=lbl_version,
        lbl_status=lbl_status,
        pb=pb,
        img_orig=img_orig,
    )

