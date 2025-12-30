from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class CameraSettingsDialog(QtWidgets.QDialog):
    """
    Per-camera settings (motion recording, LED, overlays).
    """

    def __init__(self, cam_cfg, app_defaults, widget, parent=None):
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.app_defaults = app_defaults
        self.widget = widget
        self.setWindowTitle(f"Camera Settings - {getattr(cam_cfg, 'name', '')}")

        layout = QtWidgets.QFormLayout(self)

        # Motion recording toggle
        default_motion = getattr(cam_cfg, "record_motion", None)
        if default_motion is None:
            default_motion = getattr(app_defaults, "record_motion", False)
        self.cb_motion = QtWidgets.QCheckBox("Record motion (no detection)")
        self.cb_motion.setChecked(bool(default_motion))
        layout.addRow(self.cb_motion)

        # Motion sensitivity slider
        default_sens = getattr(cam_cfg, "motion_sensitivity", None)
        if default_sens is None:
            default_sens = getattr(app_defaults, "motion_sensitivity", 50)
        self.s_motion = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_motion.setRange(0, 100)
        self.s_motion.setValue(int(default_sens))
        self.s_motion.setTickInterval(5)
        self.s_motion.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("Motion sensitivity:", self._wrap_slider(self.s_motion))

        # LED / flash controls
        self.cb_flash = QtWidgets.QComboBox()
        self.cb_flash.addItems(["Off", "On", "Auto"])
        mode_idx = {"off": 0, "on": 1, "auto": 2}.get(
            getattr(widget, "_flash_mode", "off").lower(), 0
        )
        self.cb_flash.setCurrentIndex(mode_idx)

        self.s_flash = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_flash.setRange(0, 255)
        self.s_flash.setSingleStep(4)
        self.s_flash.setValue(int(getattr(widget, "_flash_level", 128)))
        self.s_flash.setTickInterval(8)
        self.s_flash.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addRow("LED mode:", self.cb_flash)
        layout.addRow("LED level:", self._wrap_slider(self.s_flash, width=60))

        # AI controls
        self.cb_ai_enabled = QtWidgets.QCheckBox("Enable AI")
        self.cb_ai_enabled.setChecked(bool(getattr(widget, "_ai_enabled", True)))
        layout.addRow(self.cb_ai_enabled)

        self.cb_ai_yolo = QtWidgets.QCheckBox("YOLO")
        self.cb_ai_yolo.setChecked(bool(getattr(widget._overlays, "yolo", True)))
        layout.addRow(self.cb_ai_yolo)

        self.cb_ai_faces = QtWidgets.QCheckBox("Faces")
        self.cb_ai_faces.setChecked(bool(getattr(widget._overlays, "faces", True)))
        layout.addRow(self.cb_ai_faces)

        self.cb_ai_pets = QtWidgets.QCheckBox("Pets")
        self.cb_ai_pets.setChecked(bool(getattr(widget._overlays, "pets", True)))
        layout.addRow(self.cb_ai_pets)

        # Orientation controls
        self.cb_rotation = QtWidgets.QComboBox()
        self.cb_rotation.addItems(["0°", "90°", "180°", "270°"])
        rot = int(getattr(cam_cfg, "rotation_deg", 0) or 0)
        if rot in (0, 90, 180, 270):
            self.cb_rotation.setCurrentIndex((0, 90, 180, 270).index(rot))
        self.cb_flip_h = QtWidgets.QCheckBox("Flip horizontally (mirror)")
        self.cb_flip_h.setChecked(bool(getattr(cam_cfg, "flip_horizontal", False)))
        self.cb_flip_v = QtWidgets.QCheckBox("Flip vertically")
        self.cb_flip_v.setChecked(bool(getattr(cam_cfg, "flip_vertical", False)))
        layout.addRow("Rotation:", self.cb_rotation)
        layout.addRow(self.cb_flip_h)
        layout.addRow(self.cb_flip_v)

        # Overlay controls
        self.cb_overlay_det = QtWidgets.QCheckBox("Show detections (boxes + labels)")
        any_det = (
            getattr(widget._overlays, "yolo", False)
            or getattr(widget._overlays, "faces", False)
            or getattr(widget._overlays, "pets", False)
        )
        self.cb_overlay_det.setChecked(any_det)
        layout.addRow(self.cb_overlay_det)

        self.cb_overlay_hud = QtWidgets.QCheckBox("Show HUD (name + time)")
        self.cb_overlay_hud.setChecked(bool(getattr(widget._overlays, "hud", True)))
        layout.addRow(self.cb_overlay_hud)

        self.cb_overlay_stats = QtWidgets.QCheckBox("Show stats (FPS + counts)")
        self.cb_overlay_stats.setChecked(bool(getattr(widget._overlays, "stats", True)))
        layout.addRow(self.cb_overlay_stats)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _wrap_slider(self, slider: QtWidgets.QSlider, width: int = 40) -> QtWidgets.QWidget:
        c = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(c)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider)
        lbl = QtWidgets.QLabel(str(int(slider.value())))
        lbl.setFixedWidth(width)
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(lbl)
        slider.valueChanged.connect(lambda v: lbl.setText(str(int(v))))
        return c

    def apply(self):
        # Motion
        self.cam_cfg.record_motion = self.cb_motion.isChecked()
        self.cam_cfg.motion_sensitivity = int(self.s_motion.value())

        # LED / flash
        mode_text = self.cb_flash.currentText()
        level = int(self.s_flash.value())
        self.widget._flash_level = level
        self.widget.cam_cfg.flash_level = level
        self.widget._flash_mode = mode_text.lower()
        self.widget.cam_cfg.flash_mode = mode_text.lower()
        # sync hidden controls so internal helpers work
        self.widget.cb_flash.blockSignals(True)
        self.widget.cb_flash.setCurrentText(mode_text)
        self.widget.cb_flash.blockSignals(False)
        self.widget.s_flash.blockSignals(True)
        self.widget.s_flash.setValue(level)
        self.widget.s_flash.blockSignals(False)
        self.widget._apply_flash_mode(initial=False)

        # AI
        ai_enabled = self.cb_ai_enabled.isChecked()
        ai_yolo = self.cb_ai_yolo.isChecked()
        ai_faces = self.cb_ai_faces.isChecked()
        ai_pets = self.cb_ai_pets.isChecked()
        self.widget.cam_cfg.ai_enabled = ai_enabled
        self.widget.cam_cfg.ai_yolo = ai_yolo
        self.widget.cam_cfg.ai_faces = ai_faces
        self.widget.cam_cfg.ai_pets = ai_pets

        for act, val in (
            (self.widget.act_ai_enabled, ai_enabled),
            (self.widget.act_ai_yolo, ai_yolo),
            (self.widget.act_ai_faces, ai_faces),
            (self.widget.act_ai_pets, ai_pets),
        ):
            act.blockSignals(True)
            act.setChecked(val)
            act.blockSignals(False)

        self.widget._on_ai_toggled(ai_enabled)
        self.widget._on_ai_yolo_toggled(ai_yolo)
        self.widget._on_ai_faces_toggled(ai_faces)
        self.widget._on_ai_pets_toggled(ai_pets)

        # Orientation
        rot_val = (0, 90, 180, 270)[self.cb_rotation.currentIndex()]
        flip_h = self.cb_flip_h.isChecked()
        flip_v = self.cb_flip_v.isChecked()
        changed_orientation = (
            rot_val != int(getattr(self.cam_cfg, "rotation_deg", 0) or 0)
            or flip_h != bool(getattr(self.cam_cfg, "flip_horizontal", False))
            or flip_v != bool(getattr(self.cam_cfg, "flip_vertical", False))
        )
        self.cam_cfg.rotation_deg = rot_val
        self.cam_cfg.flip_horizontal = flip_h
        self.cam_cfg.flip_vertical = flip_v
        if changed_orientation:
            self.widget._on_orientation_changed()

        # Overlays
        det_on = self.cb_overlay_det.isChecked()
        hud_on = self.cb_overlay_hud.isChecked()
        stats_on = self.cb_overlay_stats.isChecked()
        self.widget._on_overlay_master_toggled(det_on)
        self.widget._on_overlay_hud_toggled(hud_on)
        self.widget._on_overlay_stats_toggled(stats_on)
        # keep toggle actions in sync
        self.widget.act_overlay_detections.setChecked(det_on)
        self.widget.act_overlay_hud.setChecked(hud_on)
        self.widget.act_overlay_stats.setChecked(stats_on)
