from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from pet_params import PetParams


class PetRecTunerDialog(QtWidgets.QDialog):
    def __init__(self, models_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pet Recognizer Tuner")
        self.models_dir = models_dir
        self.params = PetParams.load(models_dir)

        layout = QtWidgets.QFormLayout(self)

        self.s_thresh = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_thresh.setRange(40, 99)
        self.s_thresh.setValue(int(round(float(self.params.sim_thresh) * 100.0)))
        self.s_thresh.setTickInterval(1)
        self.s_thresh.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.s_thresh.setToolTip("Similarity threshold; lower is more lenient.")
        layout.addRow(
            "Match threshold (%):",
            self._wrap_slider(self.s_thresh, suffix="%"),
        )

        self.s_min = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_min.setRange(16, 200)
        self.s_min.setValue(int(self.params.min_box_px))
        self.s_min.setTickInterval(8)
        self.s_min.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.s_min.setToolTip("Ignore pet detections smaller than this (min side in pixels).")
        layout.addRow("Min pet size (px):", self._wrap_slider(self.s_min, suffix=" px"))

        self.cb_auto = QtWidgets.QCheckBox("Include auto_pet_* folders (experimental)")
        self.cb_auto.setChecked(bool(getattr(self.params, "include_auto_labels", False)))
        layout.addRow(self.cb_auto)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
        ).clicked.connect(self._apply_only)
        layout.addRow(btns)

    def _apply_only(self):
        self._collect_and_save()
        QtWidgets.QMessageBox.information(
            self,
            "Pet Tuner",
            "Parameters saved. Pet ID will pick up changes on the next detection.",
        )

    def accept(self):
        self._collect_and_save()
        super().accept()

    def _collect_and_save(self):
        self.params.sim_thresh = float(self.s_thresh.value()) / 100.0
        self.params.min_box_px = int(self.s_min.value())
        self.params.include_auto_labels = bool(self.cb_auto.isChecked())
        self.params.save(self.models_dir)

    def _wrap_slider(
        self, slider: QtWidgets.QSlider, *, suffix: str = ""
    ) -> QtWidgets.QWidget:
        """
        Place a numeric label to the right of a slider and keep it synced.
        """
        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider)

        def fmt(v: int) -> str:
            return f"{int(v)}{suffix}"

        lbl = QtWidgets.QLabel(fmt(int(slider.value())))
        lbl.setFixedWidth(70)
        lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        h.addWidget(lbl)
        slider.valueChanged.connect(lambda v: lbl.setText(fmt(v)))
        return container


__all__ = ["PetRecTunerDialog"]

