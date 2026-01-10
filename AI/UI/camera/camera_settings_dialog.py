from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ha_discovery import publish_discovery


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

        # Camera name
        self.edit_name = QtWidgets.QLineEdit(getattr(cam_cfg, "name", "") or "")
        layout.addRow("Name", self.edit_name)

        # Motion recording toggle
        default_motion = getattr(cam_cfg, "record_motion", None)
        if default_motion is None:
            default_motion = getattr(app_defaults, "record_motion", False)
        self.cb_motion = QtWidgets.QCheckBox("Record motion (no detection)")
        self.cb_motion.setChecked(bool(default_motion))
        layout.addRow(self.cb_motion)

        # MQTT per-camera toggle
        self.cb_mqtt = QtWidgets.QCheckBox("Send MQTT messages")
        self.cb_mqtt.setChecked(bool(getattr(cam_cfg, "mqtt_publish", True)))
        layout.addRow(self.cb_mqtt)

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
        # Stream selection UI is ONVIF-only.
        if show_stream and not bool(getattr(cam_cfg, "is_onvif", False)):
            show_stream = False

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

        # PTZ controls (ONVIF)
        if bool(getattr(cam_cfg, "is_onvif", False)):
            self.cb_ptz_hide_zoom = QtWidgets.QCheckBox("Hide PTZ zoom buttons")
            self.cb_ptz_hide_zoom.setChecked(bool(getattr(cam_cfg, "ptz_disable_zoom", False)))
            self.cb_ptz_hide_presets = QtWidgets.QCheckBox("Hide PTZ presets button")
            self.cb_ptz_hide_presets.setChecked(bool(getattr(cam_cfg, "ptz_disable_presets", False)))
            layout.addRow(self.cb_ptz_hide_zoom)
            layout.addRow(self.cb_ptz_hide_presets)

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

        # Overlay text size (per camera)
        self.s_text_pct = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.s_text_pct.setRange(1, 12)
        self.s_text_pct.setTickInterval(1)
        self.s_text_pct.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        cur_pct = getattr(cam_cfg, "overlay_text_pct", 4.0)
        try:
            cur_pct = float(cur_pct)
        except Exception:
            cur_pct = 4.0
        cur_pct = max(1.0, min(12.0, cur_pct))
        self.s_text_pct.setValue(int(round(cur_pct)))
        layout.addRow("Overlay text size (% of video height):", self._wrap_slider(self.s_text_pct, width=55))

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self) -> None:  # type: ignore[override]
        new_name = (self.edit_name.text() or "").strip()
        if not new_name:
            QtWidgets.QMessageBox.warning(self, "Camera Settings", "Camera name cannot be empty.")
            return
        old_name = (getattr(self.cam_cfg, "name", "") or "").strip()
        if new_name != old_name:
            for c in getattr(self.app_defaults, "cameras", []) or []:
                if c is self.cam_cfg:
                    continue
                if (getattr(c, "name", "") or "").strip() == new_name:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Camera Settings",
                        "A camera with that name already exists.",
                    )
                    return
        super().accept()

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
        # Rename (if requested)
        old_name = getattr(self.cam_cfg, "name", "") or ""
        new_name = (self.edit_name.text() or "").strip()
        renamed = False
        if new_name and new_name != old_name:
            renamed = True
            # Clear old retained MQTT topics before switching topic base.
            try:
                if bool(getattr(self.cam_cfg, "mqtt_publish", True)):
                    self.widget._publish_mqtt_cleared_state()
            except Exception:
                pass

            self.cam_cfg.name = new_name
            try:
                if getattr(self.widget, "_subwindow", None) is not None:
                    self.widget._subwindow.setWindowTitle(new_name)
            except Exception:
                pass
            try:
                if hasattr(self.widget, "_recorder"):
                    self.widget._recorder.cam_name = new_name
            except Exception:
                pass

            # Persist window geometry under the new name.
            try:
                geo = getattr(self.app_defaults, "window_geometries", None) or {}
                if old_name in geo and new_name not in geo:
                    geo[new_name] = geo.pop(old_name)
                    self.app_defaults.window_geometries = geo
            except Exception:
                pass

            # Update camera-specific MQTT topic base for future publishes.
            try:
                new_topic = new_name.replace(" ", "_")
                self.widget._mqtt_topic = new_topic
                if hasattr(self.widget, "_presence"):
                    self.widget._presence.cam = new_name
                    self.widget._presence._mqtt_topic = new_topic
            except Exception:
                pass

            try:
                mqtt = getattr(self.widget, "_mqtt", None)
                if mqtt is not None and getattr(mqtt, "connected", False):
                    publish_discovery(
                        mqtt,
                        getattr(self.app_defaults, "cameras", []) or [],
                        getattr(self.app_defaults, "mqtt_discovery_prefix", "homeassistant"),
                        getattr(mqtt, "base_topic", getattr(self.app_defaults, "mqtt_base_topic", "esp32_cam_ai")),
                    )
            except Exception:
                pass

        # Motion
        self.cam_cfg.record_motion = self.cb_motion.isChecked()
        self.cam_cfg.motion_sensitivity = int(self.s_motion.value())

        # MQTT
        mqtt_on = bool(self.cb_mqtt.isChecked())
        if hasattr(self.widget, "_apply_mqtt_publish"):
            try:
                self.widget._apply_mqtt_publish(mqtt_on)
            except Exception:
                pass
        else:
            self.cam_cfg.mqtt_publish = mqtt_on

        if renamed and mqtt_on:
            try:
                self.widget._publish_mqtt_snapshot()
            except Exception:
                pass

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

        # PTZ controls (ONVIF)
        if hasattr(self, "cb_ptz_hide_zoom") and hasattr(self, "cb_ptz_hide_presets"):
            hide_zoom = bool(self.cb_ptz_hide_zoom.isChecked())
            hide_presets = bool(self.cb_ptz_hide_presets.isChecked())
            changed_ptz_ui = (
                hide_zoom != bool(getattr(self.cam_cfg, "ptz_disable_zoom", False))
                or hide_presets != bool(getattr(self.cam_cfg, "ptz_disable_presets", False))
            )
            self.cam_cfg.ptz_disable_zoom = hide_zoom
            self.cam_cfg.ptz_disable_presets = hide_presets
            if changed_ptz_ui:
                try:
                    detected_zoom = bool(
                        getattr(self.widget, "_ptz_detected_has_zoom", getattr(self.widget, "_ptz_has_zoom", False))
                    )
                    detected_presets = bool(
                        getattr(
                            self.widget,
                            "_ptz_detected_has_presets",
                            getattr(self.widget, "_ptz_supports_presets", False),
                        )
                    )
                    self.widget._ptz_has_zoom = detected_zoom and (not hide_zoom)
                    self.widget._ptz_supports_presets = detected_presets and (not hide_presets)
                except Exception:
                    pass
                try:
                    self.widget._ptz_stop()
                except Exception:
                    pass
                try:
                    self.widget._invalidate_overlay_cache()
                except Exception:
                    pass

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

        # Overlay text size (% of video height)
        try:
            new_pct = float(self.s_text_pct.value())
        except Exception:
            new_pct = 4.0
        old_pct = getattr(self.cam_cfg, "overlay_text_pct", 4.0)
        try:
            old_pct = float(old_pct)
        except Exception:
            old_pct = 4.0
        new_pct = max(1.0, min(12.0, new_pct))
        if abs(new_pct - old_pct) > 0.01:
            self.cam_cfg.overlay_text_pct = new_pct
            try:
                # Recompute once from current stream resolution on next paint.
                self.widget._overlay_text_px = None
                self.widget._overlay_text_px_set = False
            except Exception:
                pass
            try:
                self.widget._invalidate_overlay_cache()
            except Exception:
                pass

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
