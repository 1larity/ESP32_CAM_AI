from __future__ import annotations

from PySide6 import QtWidgets, QtCore


class SecurityRecordingDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Security Recording")
        self.app_cfg = app_cfg

        layout = QtWidgets.QFormLayout(self)

        self.cb_person = QtWidgets.QCheckBox("Record people")
        self.cb_person.setChecked(getattr(app_cfg, "record_person", False))
        layout.addRow(self.cb_person)

        self.cb_unknown_person = QtWidgets.QCheckBox("Record unknown people")
        self.cb_unknown_person.setChecked(getattr(app_cfg, "record_unknown_person", False))
        layout.addRow(self.cb_unknown_person)

        self.cb_pet = QtWidgets.QCheckBox("Record pets")
        self.cb_pet.setChecked(getattr(app_cfg, "record_pet", False))
        layout.addRow(self.cb_pet)

        self.cb_unknown_pet = QtWidgets.QCheckBox("Record unknown pets")
        self.cb_unknown_pet.setChecked(getattr(app_cfg, "record_unknown_pet", False))
        layout.addRow(self.cb_unknown_pet)

        self.cb_motion = QtWidgets.QCheckBox("Record motion (no detection)")
        self.cb_motion.setChecked(getattr(app_cfg, "record_motion", False))
        layout.addRow(self.cb_motion)

        self.s_motion = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_motion.setRange(0, 100)
        self.s_motion.setValue(int(getattr(app_cfg, "motion_sensitivity", 50)))
        self.s_motion.setTickInterval(5)
        self.s_motion.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("Motion sensitivity:", self._wrap_slider(self.s_motion))

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _wrap_slider(self, slider: QtWidgets.QSlider) -> QtWidgets.QWidget:
        c = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(c)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider)
        lbl = QtWidgets.QLabel(str(int(slider.value())))
        lbl.setFixedWidth(40)
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(lbl)
        slider.valueChanged.connect(lambda v: lbl.setText(str(int(v))))
        return c

    def apply(self):
        self.app_cfg.record_person = self.cb_person.isChecked()
        self.app_cfg.record_unknown_person = self.cb_unknown_person.isChecked()
        self.app_cfg.record_pet = self.cb_pet.isChecked()
        self.app_cfg.record_unknown_pet = self.cb_unknown_pet.isChecked()
        self.app_cfg.record_motion = self.cb_motion.isChecked()
        self.app_cfg.motion_sensitivity = int(self.s_motion.value())
