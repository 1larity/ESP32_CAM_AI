# enrollment.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore
from pathlib import Path
from settings import AppSettings, BASE_DIR
from enrollment_service import EnrollmentService


class EnrollDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Enrollment")
        self.app_cfg = app_cfg
        self.svc = EnrollmentService.instance()
        self.svc.status_changed.connect(self._on_status)

        self.name = QtWidgets.QLineEdit()
        self.target = QtWidgets.QSpinBox()
        self.target.setRange(5, 200)
        self.target.setValue(25)

        # Camera selection
        self.cam_combo = QtWidgets.QComboBox()
        cam_names = [c.name for c in self.app_cfg.cameras]
        if cam_names:
            self.cam_combo.addItems(cam_names)
        else:
            self.cam_combo.addItem("(no cameras)")
            self.cam_combo.setEnabled(False)

        self.status = QtWidgets.QLabel("Idle")
        self.pb = QtWidgets.QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)

        btn_start = QtWidgets.QPushButton("Start Face Enrollment")
        btn_stop = QtWidgets.QPushButton("Stop")
        btn_start.clicked.connect(self._start)
        btn_stop.clicked.connect(self._stop)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Camera", self.cam_combo)
        form.addRow("Samples target", self.target)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.status)
        lay.addWidget(self.pb)
        lay.addWidget(btn_start)
        lay.addWidget(btn_stop)

    def _start(self):
        nm = self.name.text().strip()
        if not nm:
            QtWidgets.QMessageBox.warning(self, "Enroll", "Enter a name.")
            return
        if not self.cam_combo.isEnabled() or self.cam_combo.count() == 0:
            QtWidgets.QMessageBox.warning(self, "Enroll", "No cameras available for enrollment.")
            return
        cam_name = self.cam_combo.currentText().strip() or None
        # force faces dir under BASE_DIR/data/faces
        faces_root = Path(BASE_DIR) / "data" / "faces"
        faces_root.mkdir(parents=True, exist_ok=True)
        self.svc.faces_dir = str(faces_root)
        self.svc.begin_face(nm, int(self.target.value()), cam_name)
        self.status.setText("Running…")

    def _stop(self):
        self.svc.end()

    @QtCore.pyqtSlot()
    def _on_status(self):
        st = self.svc.status_text
        self.status.setText(st)
        if self.svc.samples_needed:
            pct = int(100 * self.svc.samples_got / self.svc.samples_needed)
        else:
            pct = 0
        self.pb.setValue(pct)
