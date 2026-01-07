from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtWidgets


@dataclass(frozen=True)
class OnvifDiscoveryDialogUI:
    btn_scan: QtWidgets.QPushButton
    btn_stop: QtWidgets.QPushButton
    btn_fetch_auth: QtWidgets.QPushButton
    btn_add: QtWidgets.QPushButton
    list: QtWidgets.QListWidget
    lbl_status: QtWidgets.QLabel
    progress: QtWidgets.QProgressBar
    details: QtWidgets.QPlainTextEdit


def build_onvif_discovery_dialog_ui(dialog: QtWidgets.QDialog) -> OnvifDiscoveryDialogUI:
    dialog.setWindowTitle("Discover ONVIF Cameras")

    btn_scan = QtWidgets.QPushButton("Scan")
    btn_stop = QtWidgets.QPushButton("Stop")
    btn_stop.setEnabled(False)
    btn_fetch_auth = QtWidgets.QPushButton("Fetch with credentialsƒ?İ")
    btn_fetch_auth.setEnabled(False)
    btn_add = QtWidgets.QPushButton("Add Selected")
    btn_add.setEnabled(False)

    list_widget = QtWidgets.QListWidget()

    lbl_status = QtWidgets.QLabel("Idle")
    progress = QtWidgets.QProgressBar()
    progress.setMaximum(0)
    progress.setValue(0)

    details = QtWidgets.QPlainTextEdit()
    details.setReadOnly(True)
    details.setMaximumBlockCount(500)

    btns = QtWidgets.QHBoxLayout()
    btns.addWidget(btn_scan)
    btns.addWidget(btn_stop)
    btns.addWidget(btn_fetch_auth)
    btns.addWidget(btn_add)

    lay = QtWidgets.QVBoxLayout(dialog)
    lay.addWidget(
        QtWidgets.QLabel(
            "WS-Discovery will broadcast on the local network and query camera capabilities."
        )
    )
    lay.addLayout(btns)
    lay.addWidget(lbl_status)
    lay.addWidget(progress)
    lay.addWidget(list_widget)
    lay.addWidget(QtWidgets.QLabel("Camera details"))
    lay.addWidget(details)

    return OnvifDiscoveryDialogUI(
        btn_scan=btn_scan,
        btn_stop=btn_stop,
        btn_fetch_auth=btn_fetch_auth,
        btn_add=btn_add,
        list=list_widget,
        lbl_status=lbl_status,
        progress=progress,
        details=details,
    )

