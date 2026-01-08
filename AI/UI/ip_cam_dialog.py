# ip_cam_dialog.py
from __future__ import annotations
from typing import Optional
from PySide6 import QtWidgets
from settings import CameraSettings

class AddIpCameraDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Camera by IP")
        self._app_cfg = app_cfg
        self._camera: Optional[CameraSettings] = None

        self.edit_name = QtWidgets.QLineEdit()
        self.edit_ip   = QtWidgets.QLineEdit()
        self.edit_user = QtWidgets.QLineEdit()
        self.edit_pass = QtWidgets.QLineEdit(); self.edit_pass.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.edit_token = QtWidgets.QLineEdit()

        # Sensible defaults
        default_name = f"Cam-{len(self._app_cfg.cameras) + 1}"
        self.edit_name.setText(default_name)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.edit_name)
        form.addRow("IP address", self.edit_ip)
        form.addRow("Username (optional)", self.edit_user)
        form.addRow("Password (optional)", self.edit_pass)
        form.addRow("Token (optional)", self.edit_token)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(btns)

    def _on_accept(self):
        name = self.edit_name.text().strip()
        ip   = self.edit_ip.text().strip()
        user = self.edit_user.text().strip() or None
        pw   = self.edit_pass.text().strip() or None
        token = self.edit_token.text().strip() or None

        if not ip:
            QtWidgets.QMessageBox.warning(self, "Add Camera", "Enter an IP address.")
            return
        if not name:
            name = ip

        cam = CameraSettings.from_ip(name=name, host=ip, user=user, password=pw, token=token)
        cam.record_motion = False
        self._camera = cam
        self.accept()

    def get_camera(self) -> Optional[CameraSettings]:
        return self._camera
