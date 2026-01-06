from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
from settings import AppSettings, CameraSettings, save_settings
from models import ModelManager
from enrollment import EnrollmentService
from UI.enrollment import EnrollDialog
from UI.image_manager import ImageManagerDialog
from UI.events_pane import EventsPane
from UI.discovery_dialog import DiscoveryDialog
from UI.ip_cam_dialog import AddIpCameraDialog
from UI.onvif_dialog import OnvifDiscoveryDialog
from UI.camera import CameraWidget
from UI.mqtt_settings import MqttSettingsDialog
from UI.unknown_capture_dialog import UnknownCaptureDialog
from ha_discovery import publish_discovery

from UI.face_rebuild import FaceRebuildController
from UI.main_window_menus import build_menus
from UI.person_tools import archive_person_folder, purge_auto_unknowns, restore_person_folder

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_cfg: AppSettings, *, load_on_init: bool = True, mqtt_service=None):
        super().__init__()
        self.app_cfg = app_cfg
        self._mqtt = mqtt_service

        # Background face rebuild state
        self._face_rebuild = FaceRebuildController(self)

        # MDI area
        self.mdi = QtWidgets.QMdiArea()
        self.mdi.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.mdi.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setCentralWidget(self.mdi)
        self.setWindowTitle("ESP32-CAM AI Viewer")

        # Events pane dock
        self.events_pane = EventsPane(self.app_cfg.logs_dir, parent=self)
        self.dock_events = QtWidgets.QDockWidget("Events", self)
        self.dock_events.setObjectName("eventsDock")
        self.dock_events.setWidget(self.events_pane)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_events
        )
        self.dock_events.hide()

        # Restore geometry/state if persisted
        if getattr(self.app_cfg, "window_geometry", None):
            try:
                self.restoreGeometry(QtCore.QByteArray.fromHex(self.app_cfg.window_geometry.encode()))
            except Exception:
                pass
        if getattr(self.app_cfg, "window_state", None):
            try:
                self.restoreState(QtCore.QByteArray.fromHex(self.app_cfg.window_state.encode()))
            except Exception:
                pass

        self._build_menus()
        if self._mqtt is not None:
            disc_fn = lambda _=None: publish_discovery(
                self._mqtt,
                self.app_cfg.cameras,
                getattr(self.app_cfg, "mqtt_discovery_prefix", "homeassistant"),
                getattr(self._mqtt, "base_topic", getattr(self.app_cfg, "mqtt_base_topic", "esp32_cam_ai")),
            )
            self._mqtt.add_on_connect(lambda _: disc_fn())
            if getattr(self._mqtt, "connected", False):
                disc_fn()
        if load_on_init:
            self._load_initial_cameras()

    # ------------------------------------------------------------------ #
    # camera windows
    # ------------------------------------------------------------------ #

    def _load_initial_cameras(self) -> None:
        for cam in self.app_cfg.cameras:
            self._add_camera_window(cam)

    def _add_camera_window(self, cam_cfg: CameraSettings) -> None:
        try:
            w = CameraWidget(cam_cfg, self.app_cfg, self, mqtt_service=self._mqtt)
        except Exception as e:
            print(f"[MainWindow] Failed to init camera {getattr(cam_cfg, 'name', '')}: {e}")
            return

        sub = QtWidgets.QMdiSubWindow()
        sub.setWidget(w)
        sub.setWindowTitle(cam_cfg.name)
        # remove Qt icon from cam windows
        sub.setWindowIcon(QtGui.QIcon())

        self.mdi.addSubWindow(sub)

        # Restore per-camera geometry/state if available (re-apply after addSubWindow)
        geom_rec = (self.app_cfg.window_geometries or {}).get(cam_cfg.name)
        if geom_rec and len(geom_rec) >= 5:
            x, y, w_geom, h_geom, maximized = map(int, geom_rec[:5])
            if maximized:
                sub.showMaximized()
            elif w_geom >= 200 and h_geom >= 200:
                sub.setGeometry(x, y, w_geom, h_geom)

        # remember our QMdiSubWindow in the widget so fit_window_to_video
        # can correctly size the outer frame
        w._subwindow = sub

        w.start()
        sub.show()

    # ------------------------------------------------------------------ #
    # camera adding
    # ------------------------------------------------------------------ #

    def _add_camera_url_dialog(self) -> None:
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Add Camera", "Enter RTSP or HTTP stream URL:"
        )
        if ok and text:
            cam_cfg = CameraSettings(
                name=f"Custom-{len(self.app_cfg.cameras) + 1}",
                stream_url=text,
            )
            self.app_cfg.cameras.append(cam_cfg)
            self._add_camera_window(cam_cfg)
            save_settings(self.app_cfg)

    def _add_camera_ip_dialog(self) -> None:
        dlg = AddIpCameraDialog(self.app_cfg, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cam_cfg = dlg.get_camera()
            if cam_cfg is not None:
                self.app_cfg.cameras.append(cam_cfg)
                self._add_camera_window(cam_cfg)

    # ------------------------------------------------------------------ #
    # events pane
    # ------------------------------------------------------------------ #

    def _toggle_events_pane(self) -> None:
        if self.dock_events.isVisible():
            self.dock_events.hide()
        else:
            self.dock_events.show()

    # ------------------------------------------------------------------ #
    # global view actions
    # ------------------------------------------------------------------ #

    def _fit_all(self) -> None:
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.fit_to_window()

    def _100_all(self) -> None:
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.zoom_100()

    def _resize_all_to_video(self) -> None:
        """
        Resize every camera subwindow to match its video at the
        current zoom level.
        """
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.fit_window_to_video()

    # ------------------------------------------------------------------ #
    # close / persistence
    # ------------------------------------------------------------------ #

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.stop()
        if self._mqtt is not None:
            try:
                self._mqtt.stop()
            except Exception:
                pass
        try:
            self.app_cfg.window_geometry = bytes(self.saveGeometry().toHex()).decode()
            self.app_cfg.window_state = bytes(self.saveState().toHex()).decode()
            # Persist per-camera MDI subwindow geometry/state
            geo: dict[str, list[int]] = {}
            for sub in self.mdi.subWindowList():
                try:
                    name = sub.windowTitle()
                    r = sub.geometry()
                    geo[name] = [r.x(), r.y(), r.width(), r.height(), int(sub.isMaximized())]
                except Exception:
                    continue
            self.app_cfg.window_geometries = geo
        except Exception:
            pass
        save_settings(self.app_cfg)
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # dialogs
    # ------------------------------------------------------------------ #

    def _open_enrollment(self) -> None:
        dlg = EnrollDialog(self.app_cfg, self)
        dlg.exec()
        # After enrollment, rebuild LBPH model from disk with progress
        self._start_face_rebuild("Rebuilding face model after enrollment")

    def _open_image_manager(self) -> None:
        dlg = ImageManagerDialog(self.app_cfg, self)
        dlg.exec()
        # After image management changes, rebuild LBPH model from disk with progress
        self._start_face_rebuild("Rebuilding face model after image changes")

    def _open_discovery(self) -> None:
        dlg = DiscoveryDialog(self.app_cfg, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cam_info = dlg.selected_camera()
            if cam_info:
                cam_cfg = CameraSettings(
                    name=cam_info.get("name"),
                    stream_url=cam_info.get("stream_url"),
                    user=cam_info.get("user"),
                    password=cam_info.get("password"),
                )
                self.app_cfg.cameras.append(cam_cfg)
                self._add_camera_window(cam_cfg)
                save_settings(self.app_cfg)

    def _open_onvif_discovery(self) -> None:
        dlg = OnvifDiscoveryDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cam_cfg = dlg.selected_camera()
            if cam_cfg is not None:
                self.app_cfg.cameras.append(cam_cfg)
                self._add_camera_window(cam_cfg)
                save_settings(self.app_cfg)

    def _open_mqtt_settings(self) -> None:
        dlg = MqttSettingsDialog(self.app_cfg, self)
        dlg.exec()

    def _remove_camera_dialog(self) -> None:
        names = [getattr(c, "name", "") for c in self.app_cfg.cameras]
        if not names:
            QtWidgets.QMessageBox.information(self, "Remove Camera", "No cameras configured.")
            return
        name, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Remove Camera",
            "Select camera to remove:",
            names,
            editable=False,
        )
        if not ok or not name:
            return
        cams = [c for c in self.app_cfg.cameras if getattr(c, "name", None) != name]
        if len(cams) == len(self.app_cfg.cameras):
            QtWidgets.QMessageBox.information(
                self, "Remove Camera", f"No camera named '{name}' found."
            )
            return
        # Stop and close the camera window if it is open
        for sub in list(self.mdi.subWindowList()):
            if sub.windowTitle() == name:
                w = sub.widget()
                if isinstance(w, CameraWidget):
                    try:
                        w.stop()
                    except Exception:
                        pass
                sub.close()
        # Remove any stored geometry for this camera
        geo = self.app_cfg.window_geometries or {}
        if name in geo:
            try:
                del geo[name]
                self.app_cfg.window_geometries = geo
            except Exception:
                pass
        # Persist updated camera list
        self.app_cfg.cameras = cams
        save_settings(self.app_cfg)
        QtWidgets.QMessageBox.information(
            self, "Remove Camera", f"Removed camera '{name}'."
        )

    def _rename_camera_dialog(self) -> None:
        names = [getattr(c, "name", "") for c in self.app_cfg.cameras]
        if not names:
            QtWidgets.QMessageBox.information(self, "Rename Camera", "No cameras configured.")
            return
        old, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Rename Camera",
            "Select camera to rename:",
            names,
            editable=False,
        )
        if not ok or not old:
            return
        new, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rename Camera",
            f"Enter new name for '{old}':",
        )
        if not ok:
            return
        new = new.strip()
        if not new:
            QtWidgets.QMessageBox.warning(self, "Rename Camera", "Name cannot be empty.")
            return
        # Ensure unique
        if any(getattr(c, "name", "") == new for c in self.app_cfg.cameras):
            QtWidgets.QMessageBox.warning(self, "Rename Camera", "A camera with that name already exists.")
            return
        # Update config and any open window
        for c in self.app_cfg.cameras:
            if getattr(c, "name", "") == old:
                c.name = new
        for sub in list(self.mdi.subWindowList()):
            if sub.windowTitle() == old:
                sub.setWindowTitle(new)
                w = sub.widget()
                if hasattr(w, "cam_cfg"):
                    try:
                        w.cam_cfg.name = new
                    except Exception:
                        pass
        save_settings(self.app_cfg)
        QtWidgets.QMessageBox.information(self, "Rename Camera", f"Renamed '{old}' to '{new}'.")

    def _open_unknown_capture_dialog(self) -> None:
        dlg = UnknownCaptureDialog(self.app_cfg, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            dlg.apply()
            EnrollmentService.instance().set_unknown_capture(
                faces=self.app_cfg.collect_unknown_faces,
                pets=self.app_cfg.collect_unknown_pets,
                limit=getattr(self.app_cfg, "unknown_capture_limit", 50),
                auto_train=getattr(self.app_cfg, "auto_train_unknowns", False),
            )
            save_settings(self.app_cfg)

    # ------------------------------------------------------------------ #
    # menus
    # ------------------------------------------------------------------ #

    def _build_menus(self) -> None:
        build_menus(self)

    # ------------------------------------------------------------------ #
    # face model rebuild with progress
    # ------------------------------------------------------------------ #

    def _start_face_rebuild(self, title: str) -> None:
        self._face_rebuild.start(title)

    # Toggle handlers
    def _on_unknown_faces_toggled(self, checked: bool) -> None:
        self.app_cfg.collect_unknown_faces = bool(checked)
        svc = EnrollmentService.instance()
        svc.set_unknown_capture(
            faces=self.app_cfg.collect_unknown_faces,
            pets=self.app_cfg.collect_unknown_pets,
            limit=getattr(self.app_cfg, "unknown_capture_limit", 50),
            auto_train=getattr(self.app_cfg, "auto_train_unknowns", False),
        )
        save_settings(self.app_cfg)

    def _on_unknown_pets_toggled(self, checked: bool) -> None:
        self.app_cfg.collect_unknown_pets = bool(checked)
        svc = EnrollmentService.instance()
        svc.set_unknown_capture(
            faces=self.app_cfg.collect_unknown_faces,
            pets=self.app_cfg.collect_unknown_pets,
            limit=getattr(self.app_cfg, "unknown_capture_limit", 50),
            auto_train=getattr(self.app_cfg, "auto_train_unknowns", False),
        )
        save_settings(self.app_cfg)

    def _on_ignore_enroll_toggled(self, checked: bool) -> None:
        self.app_cfg.ignore_enrollment_models = bool(checked)
        save_settings(self.app_cfg)
        QtWidgets.QMessageBox.information(
            self,
            "Ignore Enrollment Models",
            "Setting will take effect on next detector restart. Restart app to reload without LBPH.",
        )

    def _on_use_gpu_toggled(self, checked: bool) -> None:
        self.app_cfg.use_gpu = bool(checked)
        save_settings(self.app_cfg)
        QtWidgets.QMessageBox.information(
            self,
            "YOLO GPU",
            "Setting will take effect on next detector restart. Restart the app to switch backend.",
        )

    def _archive_person_folder(self) -> None:
        archive_person_folder(self, self._start_face_rebuild)

    def _restore_person_folder(self) -> None:
        restore_person_folder(self, self._start_face_rebuild)

    def _purge_auto_unknowns(self) -> None:
        purge_auto_unknowns(self, self._start_face_rebuild)
