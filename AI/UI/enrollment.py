# enrollment.py
# Enrollment dialog UI, talks to EnrollmentController.

from __future__ import annotations
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Slot
from enrollment_service import EnrollmentController
from settings import AppSettings


class EnrollDialog(QtWidgets.QDialog):
    """
    Modal dialog for face/pet enrollment.
    """

    def __init__(self, app_cfg: AppSettings, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.app_cfg = app_cfg
        self.ctrl = EnrollmentController()

        self.setWindowTitle("Enroll faces / pets")
        self.resize(480, 320)

        self._build_ui()
        self._wire_signals()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit(self)
        self.cam_combo = QtWidgets.QComboBox(self)
        self.samples_spin = QtWidgets.QSpinBox(self)
        self.samples_spin.setRange(1, 500)
        self.samples_spin.setValue(40)

        form.addRow("Name / label", self.name_edit)
        form.addRow("Camera", self.cam_combo)
        form.addRow("Samples this session", self.samples_spin)
        layout.addLayout(form)

        # Camera selector (None => accept from any camera)
        self.cam_combo.addItem("All cameras", None)
        for cam in getattr(self.app_cfg, "cameras", []) or []:
            name = getattr(cam, "name", None)
            if name:
                self.cam_combo.addItem(str(name), str(name))

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start", self)
        self.btn_stop = QtWidgets.QPushButton("Abort", self)
        self.btn_close = QtWidgets.QPushButton("Close", self)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_close.clicked.connect(self.reject)

    def _wire_signals(self) -> None:
        self.ctrl.status_changed.connect(self._on_status)

    # ------------------------------------------------------------------ slots

    @Slot()
    def _on_start(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Name required", "Please enter a name/label.")
            return

        total = self.samples_spin.value()
        target_cam = self.cam_combo.currentData()
        if target_cam is not None:
            target_cam = str(target_cam)
        self.ctrl.start(name=name, total_samples=total, target_cam=target_cam)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Collecting samples…")
    @Slot()
    def _on_stop(self) -> None:
        self.ctrl.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    @Slot(dict)
    def _on_status(self, data: dict) -> None:
        active = data.get("active", False)
        done = data.get("done", False)
        name = data.get("target_name", "")
        need = data.get("samples_needed", 0)
        got = data.get("samples_got", 0)
        existing = data.get("existing_count", 0)
        err = data.get("last_error")

        if err:
            self.status_label.setText(f"Error: {err}")
        else:
            self.status_label.setText(f"Enrolling '{name}' - existing {existing}, this session {got}/{need}")
        if need > 0:
            pct = int((got / max(1, need)) * 100)
            self.progress.setValue(min(max(pct, 0), 100))
        else:
            self.progress.setValue(0)
        if done:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.status_label.setText(f"Done {got}/{need}. Training saved models to models/lbphfaces.xml")
            self.progress.setValue(100)
        elif not active:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
