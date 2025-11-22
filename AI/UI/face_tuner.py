# face_tuner.py
from __future__ import annotations
from PyQt6 import QtWidgets, QtCore
from face_params import FaceParams

class FaceRecTunerDialog(QtWidgets.QDialog):
    def __init__(self, models_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face Recognizer Tuner")
        self.models_dir = models_dir
        self.params = FaceParams.load(models_dir)

        layout = QtWidgets.QFormLayout(self)

        self.s_conf = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_conf.setRange(40, 200)
        self.s_conf.setValue(int(self.params.accept_conf))
        self.s_conf.setTickInterval(5); self.s_conf.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("Accept confidence (≤):", self.s_conf)

        self.s_roi = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_roi.setRange(64, 256)
        self.s_roi.setValue(int(self.params.roi_size))
        self.s_roi.setTickInterval(16); self.s_roi.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("ROI size (px):", self.s_roi)

        self.s_min = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_min.setRange(24, 160)
        self.s_min.setValue(int(self.params.min_face_px))
        self.s_min.setTickInterval(8); self.s_min.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("Min face size (px):", self.s_min)

        self.cb_eq = QtWidgets.QCheckBox("Equalize histogram")
        self.cb_eq.setChecked(self.params.eq_hist)
        layout.addRow(self.cb_eq)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel |
            QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply_only)
        layout.addRow(btns)

    def _apply_only(self):
        self._collect_and_save()
        QtWidgets.QMessageBox.information(self, "Face Tuner", "Parameters saved. Live detectors will pick up changes on next frame.")

    def accept(self):
        self._collect_and_save()
        super().accept()

    def _collect_and_save(self):
        self.params.accept_conf = float(self.s_conf.value())
        self.params.roi_size = int(self.s_roi.value())
        self.params.min_face_px = int(self.s_min.value())
        self.params.eq_hist = bool(self.cb_eq.isChecked())
        self.params.save(self.models_dir)
