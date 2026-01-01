# mqtt_settings.py
# Dialog to edit MQTT / Home Assistant broker settings.
from __future__ import annotations
from typing import Optional
from pathlib import Path
from PySide6 import QtWidgets, QtCore
from settings import AppSettings, save_settings


class MqttSettingsDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg: AppSettings, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("MQTT Settings")
        self._app_cfg = app_cfg

        self.chk_enabled = QtWidgets.QCheckBox("Enable MQTT publishing")
        self.chk_enabled.setChecked(bool(app_cfg.mqtt_enabled))

        self.edit_host = QtWidgets.QLineEdit(app_cfg.mqtt_host or "")
        self.spin_port = QtWidgets.QSpinBox()
        self.spin_port.setRange(1, 65535)
        self.spin_port.setValue(int(app_cfg.mqtt_port or 8883))

        self.edit_client_id = QtWidgets.QLineEdit(app_cfg.mqtt_client_id or "")
        self.edit_base_topic = QtWidgets.QLineEdit(app_cfg.mqtt_base_topic or "esp32_cam_ai")
        self.edit_discovery = QtWidgets.QLineEdit(app_cfg.mqtt_discovery_prefix or "homeassistant")

        self.chk_tls = QtWidgets.QCheckBox("Use TLS (recommended)")
        self.chk_tls.setChecked(bool(app_cfg.mqtt_tls))

        self.edit_ca = QtWidgets.QLineEdit(app_cfg.mqtt_ca_path or "")
        self.btn_ca = QtWidgets.QToolButton()
        self.btn_ca.setText("Browse")
        self.btn_ca.clicked.connect(self._choose_ca_file)

        ca_row = QtWidgets.QHBoxLayout()
        ca_row.addWidget(self.edit_ca, 1)
        ca_row.addWidget(self.btn_ca)

        self.chk_insecure = QtWidgets.QCheckBox("Allow self-signed / skip hostname check (insecure)")
        self.chk_insecure.setChecked(bool(app_cfg.mqtt_insecure))

        self.spin_keepalive = QtWidgets.QSpinBox()
        self.spin_keepalive.setRange(5, 600)
        self.spin_keepalive.setValue(int(app_cfg.mqtt_keepalive or 60))

        self.edit_user = QtWidgets.QLineEdit(app_cfg.mqtt_user or "")
        self.edit_pass = QtWidgets.QLineEdit()
        self.edit_pass.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.chk_clear_password = QtWidgets.QCheckBox("Clear stored password")

        form = QtWidgets.QFormLayout()
        form.addRow(self.chk_enabled)
        form.addRow("Host", self.edit_host)
        form.addRow("Port", self.spin_port)
        form.addRow("Client ID (optional)", self.edit_client_id)
        form.addRow("Base topic", self.edit_base_topic)
        form.addRow("Discovery prefix", self.edit_discovery)
        form.addRow(self.chk_tls)
        form.addRow("CA certificate", ca_row)
        form.addRow(self.chk_insecure)
        form.addRow("Keepalive (s)", self.spin_keepalive)
        form.addRow("Username", self.edit_user)
        form.addRow("Password (leave blank to keep)", self.edit_pass)
        form.addRow(self.chk_clear_password)

        info = QtWidgets.QLabel(
            "Password is stored encrypted locally (not synced). "
            "Leave password blank to keep the existing one, or tick 'Clear' to remove."
        )
        info.setWordWrap(True)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(info)
        layout.addWidget(btns)

    def _choose_ca_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CA certificate", "", "Certificates (*.crt *.pem);;All Files (*)")
        if path:
            self.edit_ca.setText(path)

    def _on_accept(self):
        host = self.edit_host.text().strip()
        if self.chk_enabled.isChecked() and not host:
            QtWidgets.QMessageBox.warning(self, "MQTT Settings", "Host is required when MQTT is enabled.")
            return

        cfg = self._app_cfg
        cfg.mqtt_enabled = self.chk_enabled.isChecked()
        cfg.mqtt_host = host or None
        cfg.mqtt_port = int(self.spin_port.value())
        cfg.mqtt_client_id = self.edit_client_id.text().strip() or None
        cfg.mqtt_base_topic = self.edit_base_topic.text().strip() or "esp32_cam_ai"
        cfg.mqtt_discovery_prefix = self.edit_discovery.text().strip() or "homeassistant"
        cfg.mqtt_tls = self.chk_tls.isChecked()
        ca_path = self.edit_ca.text().strip()
        cfg.mqtt_ca_path = ca_path or None
        cfg.mqtt_insecure = self.chk_insecure.isChecked()
        cfg.mqtt_keepalive = int(self.spin_keepalive.value())
        cfg.mqtt_user = self.edit_user.text().strip() or None

        new_pwd = self.edit_pass.text()
        if new_pwd:
            cfg.mqtt_password = new_pwd
        elif self.chk_clear_password.isChecked():
            cfg.mqtt_password = None
        # else leave existing password intact

        save_settings(cfg)
        self.accept()
