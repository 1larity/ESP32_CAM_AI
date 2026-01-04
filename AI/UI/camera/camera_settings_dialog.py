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

        # Stream variants (show only when applicable)
        self._stream_variants = []
        self._custom_stream_url = None
        try:
            if hasattr(widget, "_compute_stream_variants"):
                self._stream_variants = widget._compute_stream_variants()
        except Exception:
            self._stream_variants = []
        # Dedup by URL, keep first label
        uniq = []
        seen = set()
        for label, url in self._stream_variants:
            if not url or url in seen:
                continue
            seen.add(url)
            uniq.append((label, url))
        self._stream_variants = uniq
        show_stream = False
        current_url = getattr(cam_cfg, "stream_url", "") or ""
        if current_url.startswith(("rtsp://", "http://", "https://")):
            show_stream = len(self._stream_variants) > 1 or bool(getattr(cam_cfg, "alt_streams", None))

        if show_stream:
            self.cb_stream = QtWidgets.QComboBox()
            for label, url in self._stream_variants:
                self.cb_stream.addItem(f"{label}: {url}", userData=url)
            # ensure current URL is present and selected
            if current_url and current_url not in seen:
                self.cb_stream.addItem(f"Current: {current_url}", userData=current_url)
            for idx in range(self.cb_stream.count()):
                if self.cb_stream.itemData(idx) == current_url:
                    self.cb_stream.setCurrentIndex(idx)
                    break
            btn_custom = QtWidgets.QPushButton("Custom stream...")
            btn_custom.clicked.connect(self._prompt_custom_stream)
            stream_row = QtWidgets.QHBoxLayout()
            stream_row.setContentsMargins(0, 0, 0, 0)
            stream_row.addWidget(self.cb_stream, 1)
            stream_row.addWidget(btn_custom)
            layout.addRow("Stream:", stream_row)

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

    def _prompt_custom_stream(self) -> None:
        txt, ok = QtWidgets.QInputDialog.getText(self, "Stream URL", "Enter stream URL:", text=self.cam_cfg.stream_url)
        if ok and txt:
            url = txt.strip()
            if url:
                self._custom_stream_url = url
                if hasattr(self, "cb_stream"):
                    idx = self.cb_stream.findData(url)
                    if idx == -1:
                        self.cb_stream.addItem(f"Custom: {url}", userData=url)
                        idx = self.cb_stream.count() - 1
                    self.cb_stream.setCurrentIndex(idx)

    def apply(self):
        # Motion
        self.cam_cfg.record_motion = self.cb_motion.isChecked()
        self.cam_cfg.motion_sensitivity = int(self.s_motion.value())

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

        # Stream selection (if applicable)
        if hasattr(self, "cb_stream"):
            selected = self._custom_stream_url
            if not selected:
                selected = self.cb_stream.currentData()
            if selected:
                selected = selected.strip()
            if selected and selected != getattr(self.widget.cam_cfg, "stream_url", None):
                self.widget._apply_stream_url(selected)
            # Persist alt streams as any other known variants except primary
            all_urls = []
            for i in range(self.cb_stream.count()):
                u = self.cb_stream.itemData(i)
                if u:
                    all_urls.append(u)
            primary = getattr(self.widget.cam_cfg, "stream_url", None)
            self.widget.cam_cfg.alt_streams = [u for u in all_urls if u and u != primary]
