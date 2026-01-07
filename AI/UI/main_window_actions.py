from __future__ import annotations

from PySide6 import QtWidgets

from enrollment import EnrollmentService
from settings import save_settings
from UI.camera import CameraWidget
from UI.person_tools import archive_person_folder, purge_auto_unknowns, restore_person_folder


def _toggle_events_pane(self) -> None:
    if self.dock_events.isVisible():
        self.dock_events.hide()
    else:
        self.dock_events.show()


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

