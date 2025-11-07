# enrollment.py
# Progress UI with label and progress bar, bound to EnrollmentService signals.
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore
from settings import AppSettings
from enrollment_service import EnrollmentService

class EnrollDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enrollment")
        self.svc = EnrollmentService.instance()

        self.name = QtWidgets.QLineEdit()
        self.target = QtWidgets.QSpinBox(); self.target.setRange(5, 400); self.target.setValue(40)
        self.btn_start = QtWidgets.QPushButton("Start Face Enrollment")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.lbl_status = QtWidgets.QLabel("Idle")
        self.pb = QtWidgets.QProgressBar(); self.pb.setRange(0, 100); self.pb.setValue(0)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Samples", self.target)
        btns = QtWidgets.QHBoxLayout(); btns.addWidget(self.btn_start); btns.addWidget(self.btn_stop)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form); lay.addLayout(btns); lay.addWidget(self.lbl_status); lay.addWidget(self.pb)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self.svc.end)
        self.svc.status_changed.connect(self._on_status)

        # preset last used name if dialog reopened quickly
        self._on_status({
            "active": self.svc.active, "name": self.svc.target_name,
            "got": self.svc.samples_got, "need": self.svc.samples_needed,
            "folder": "", "done": False
        })

    def _start(self):
        nm = self.name.text().strip()
        if not nm:
            QtWidgets.QMessageBox.warning(self, "Enrollment", "Enter a name.")
            return
        self.svc.begin_face(nm, int(self.target.value()))

    @QtCore.pyqtSlot(dict)
    def _on_status(self, st: dict):
        got = int(st.get("got", 0)); need = max(1, int(st.get("need", 1)))
        pct = int(round(100.0 * got / need))
        self.pb.setValue(pct)
        nm = st.get("name", "")
        active = bool(st.get("active", False))
        done = bool(st.get("done", False))
        if nm and not self.name.text().strip():
            self.name.setText(nm)
        if active:
            self.lbl_status.setText(f"Collecting {got}/{need} → {st.get('folder','')}")
        elif done:
            self.lbl_status.setText(f"Done {got}/{need}. Training saved to models/lbph_faces.xml.")
        else:
            self.lbl_status.setText("Idle")
