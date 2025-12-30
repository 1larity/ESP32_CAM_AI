from __future__ import annotations

from PySide6 import QtWidgets, QtCore


class UnknownCaptureDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unknown Capture & Auto-train")
        self.app_cfg = app_cfg

        layout = QtWidgets.QFormLayout(self)

        self.cb_faces = QtWidgets.QCheckBox("Collect unknown faces")
        self.cb_faces.setChecked(getattr(app_cfg, "collect_unknown_faces", False))
        layout.addRow(self.cb_faces)

        self.cb_pets = QtWidgets.QCheckBox("Collect unknown pets")
        self.cb_pets.setChecked(getattr(app_cfg, "collect_unknown_pets", False))
        layout.addRow(self.cb_pets)

        self.s_limit = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_limit.setRange(20, 200)
        self.s_limit.setValue(int(getattr(app_cfg, "unknown_capture_limit", 50)))
        self.s_limit.setTickInterval(10)
        self.s_limit.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("Per-class cap (images):", self._wrap_slider(self.s_limit))

        self.cb_auto_train = QtWidgets.QCheckBox("Auto-train unknowns with provisional IDs")
        self.cb_auto_train.setChecked(getattr(app_cfg, "auto_train_unknowns", False))
        layout.addRow(self.cb_auto_train)

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
        self.app_cfg.collect_unknown_faces = self.cb_faces.isChecked()
        self.app_cfg.collect_unknown_pets = self.cb_pets.isChecked()
        self.app_cfg.unknown_capture_limit = int(self.s_limit.value())
        self.app_cfg.auto_train_unknowns = self.cb_auto_train.isChecked()
