# AI/UI/enrollment.py
# Enrollment dialog: pick person name, camera and sample count, show progress.

from __future__ import annotations
from PyQt6 import QtWidgets, QtCore

from settings import AppSettings
from enrollment_service import EnrollmentService


class EnrollDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enrollment")
        self.svc = EnrollmentService.instance()
        self._app_cfg = app_cfg

        # Inputs
        self.name = QtWidgets.QLineEdit()

        # Camera picker: Any camera + one entry per configured camera
        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.addItem("Any camera", None)
        for cam in self._app_cfg.cameras:
            self.cam_combo.addItem(cam.name, cam.name)

        self.target = QtWidgets.QSpinBox()
        self.target.setRange(5, 400)
        self.target.setValue(40)

        self.btn_start = QtWidgets.QPushButton("Start Face Enrollment")
        # Text reflects that sampling stops automatically; this aborts the run.
        self.btn_stop = QtWidgets.QPushButton("Abort")

        self.lbl_status = QtWidgets.QLabel("Idle")
        self.pb = QtWidgets.QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Camera", self.cam_combo)
        form.addRow("Samples", self.target)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(btns)
        lay.addWidget(self.lbl_status)
        lay.addWidget(self.pb)

        # Wiring
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._abort)
        self.svc.status_changed.connect(self._on_status)

        # preset last used name if dialog reopened quickly
        self._on_status(
            {
                "active": self.svc.active,
                "name": self.svc.target_name,
                "got": self.svc.samples_got,
                "need": self.svc.samples_needed,
                "folder": "",
                "done": False,
                "cam": getattr(self.svc, "target_cam", None),
            }
        )

    def _start(self):
        nm = self.name.text().strip()
        if not nm:
            QtWidgets.QMessageBox.warning(self, "Enrollment", "Enter a name.")
            return
        cam_name = self.cam_combo.currentData()
        self.svc.start(nm, int(self.target.value()), cam_name=cam_name)

    def _abort(self):
        # Explicitly abort current enrollment session
        self.svc.end()

    @QtCore.pyqtSlot(dict)
    def _on_status(self, st: dict):
        got = int(st.get("got", 0))
        need = max(1, int(st.get("need", 1)))
        pct = int(round(100.0 * got / need))
        self.pb.setValue(pct)

        nm = st.get("name", "")
        active = bool(st.get("active", False))
        done = bool(st.get("done", False))
        folder = st.get("folder", "")
        cam = st.get("cam", None)

        if nm and not self.name.text().strip():
            self.name.setText(nm)

        # best-effort reflect camera back into combo, if present
        if cam is not None:
            idx = self.cam_combo.findData(cam)
            if idx >= 0:
                self.cam_combo.setCurrentIndex(idx)

        if active:
            cam_txt = f" on {cam}" if cam else ""
            self.lbl_status.setText(f"Collecting {got}/{need}{cam_txt} → {folder}")
        elif done:
            self.lbl_status.setText(
                f"Done {got}/{need}. Training saved to models/lbph_faces.xml."
            )
        else:
            self.lbl_status.setText("Idle")
