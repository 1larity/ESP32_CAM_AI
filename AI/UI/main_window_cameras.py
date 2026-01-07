from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from settings import CameraSettings, save_settings
from UI.camera import CameraWidget
from UI.ip_cam_dialog import AddIpCameraDialog


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


def _add_camera_url_dialog(self) -> None:
    text, ok = QtWidgets.QInputDialog.getText(self, "Add Camera", "Enter RTSP or HTTP stream URL:")
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
        QtWidgets.QMessageBox.information(self, "Remove Camera", f"No camera named '{name}' found.")
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
    QtWidgets.QMessageBox.information(self, "Remove Camera", f"Removed camera '{name}'.")


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

