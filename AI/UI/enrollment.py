# enrollment.py
# Progress UI with label and progress bar, bound to EnrollmentService signals.

from __future__ import annotations

from PyQt6 import QtWidgets, QtCore

from settings import AppSettings
from enrollment import EnrollmentService


class EnrollDialog(QtWidgets.QDialog):
    """Simple UI wrapper around EnrollmentService with progress feedback."""

    def __init__(self, app_cfg: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enrollment")
        self.svc = EnrollmentService.instance()
        self._app_cfg = app_cfg

        # Inputs
        self.name = QtWidgets.QLineEdit()

        self.target = QtWidgets.QSpinBox()
        self.target.setRange(5, 400)
        self.target.setValue(40)

        # Camera picker: Any camera + each configured camera
        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.addItem("Any camera", None)
        for cam in self._app_cfg.cameras:
            self.cam_combo.addItem(cam.name, cam.name)

        self.btn_start = QtWidgets.QPushButton("Start")
        # Sampling stops automatically; this explicitly aborts the session.
        self.btn_abort = QtWidgets.QPushButton("Abort")
        # Explicit close button as requested
        self.btn_close = QtWidgets.QPushButton("Close")

        self.lbl_status = QtWidgets.QLabel("Idle")
        self.pb = QtWidgets.QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Camera", self.cam_combo)
        form.addRow("Samples (new)", self.target)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_abort)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(btns)
        lay.addWidget(self.lbl_status)
        lay.addWidget(self.pb)

        # Wiring
        self.btn_start.clicked.connect(self._start)
        self.btn_abort.clicked.connect(self._abort)
        self.btn_close.clicked.connect(self.accept)
        self.svc.status_changed.connect(self._on_status)

        # Preset last used state if dialog reopened quickly
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

    # ----------------------------------------------------------------- Slots

    def _start(self) -> None:
        nm = self.name.text().strip()
        if not nm:
            QtWidgets.QMessageBox.warning(self, "Enrollment", "Enter a name.")
            return
        cam_name = self.cam_combo.currentData()
        # Uses new EnrollmentService.start(name, n, cam_name)
        self.svc.start(nm, int(self.target.value()), cam_name=cam_name)

    def _abort(self) -> None:
        # Abort the current enrollment session
        self.svc.end()

    @QtCore.pyqtSlot(dict)
    def _on_status(self, st: dict) -> None:
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

        # Reflect active camera back into combo if known
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
