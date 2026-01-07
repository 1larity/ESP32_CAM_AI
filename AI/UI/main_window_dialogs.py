from __future__ import annotations

from PySide6 import QtWidgets

from enrollment import EnrollmentService
from settings import CameraSettings, save_settings
from UI.discovery_dialog import DiscoveryDialog
from UI.enrollment import EnrollDialog
from UI.image_manager import ImageManagerDialog
from UI.mqtt_settings import MqttSettingsDialog
from UI.onvif_dialog import OnvifDiscoveryDialog
from UI.unknown_capture_dialog import UnknownCaptureDialog


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

