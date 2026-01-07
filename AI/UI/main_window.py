from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ha_discovery import publish_discovery
from settings import AppSettings
from UI.events_pane import EventsPane
from UI.face_rebuild import FaceRebuildController
from UI.main_window_actions import (
    _100_all,
    _archive_person_folder,
    _fit_all,
    _on_ignore_enroll_toggled,
    _on_unknown_faces_toggled,
    _on_unknown_pets_toggled,
    _on_use_gpu_toggled,
    _purge_auto_unknowns,
    _resize_all_to_video,
    _restore_person_folder,
    _toggle_events_pane,
)
from UI.main_window_cameras import (
    _add_camera_ip_dialog,
    _add_camera_url_dialog,
    _add_camera_window,
    _load_initial_cameras,
    _remove_camera_dialog,
    _rename_camera_dialog,
)
from UI.main_window_dialogs import (
    _open_discovery,
    _open_enrollment,
    _open_image_manager,
    _open_mqtt_settings,
    _open_onvif_discovery,
    _open_unknown_capture_dialog,
)
from UI.main_window_menus import build_menus
from UI.main_window_persistence import closeEvent as _close_event


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_cfg: AppSettings, *, load_on_init: bool = True, mqtt_service=None):
        super().__init__()
        self.app_cfg = app_cfg
        self._mqtt = mqtt_service

        # Background face rebuild state
        self._face_rebuild = FaceRebuildController(self)

        # MDI area
        self.mdi = QtWidgets.QMdiArea()
        self.mdi.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.mdi.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setCentralWidget(self.mdi)
        self.setWindowTitle("ESP32-CAM AI Viewer")

        # Events pane dock
        self.events_pane = EventsPane(self.app_cfg.logs_dir, parent=self)
        self.dock_events = QtWidgets.QDockWidget("Events", self)
        self.dock_events.setObjectName("eventsDock")
        self.dock_events.setWidget(self.events_pane)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
            self.dock_events,
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

    def _build_menus(self) -> None:
        build_menus(self)

    def _start_face_rebuild(self, title: str) -> None:
        self._face_rebuild.start(title)

    # Attach handlers from helper modules to keep MainWindow small.
    _load_initial_cameras = _load_initial_cameras
    _add_camera_window = _add_camera_window
    _add_camera_url_dialog = _add_camera_url_dialog
    _add_camera_ip_dialog = _add_camera_ip_dialog
    _remove_camera_dialog = _remove_camera_dialog
    _rename_camera_dialog = _rename_camera_dialog

    _open_enrollment = _open_enrollment
    _open_image_manager = _open_image_manager
    _open_discovery = _open_discovery
    _open_onvif_discovery = _open_onvif_discovery
    _open_mqtt_settings = _open_mqtt_settings
    _open_unknown_capture_dialog = _open_unknown_capture_dialog

    _toggle_events_pane = _toggle_events_pane
    _fit_all = _fit_all
    _100_all = _100_all
    _resize_all_to_video = _resize_all_to_video

    _on_unknown_faces_toggled = _on_unknown_faces_toggled
    _on_unknown_pets_toggled = _on_unknown_pets_toggled
    _on_ignore_enroll_toggled = _on_ignore_enroll_toggled
    _on_use_gpu_toggled = _on_use_gpu_toggled

    _archive_person_folder = _archive_person_folder
    _restore_person_folder = _restore_person_folder
    _purge_auto_unknowns = _purge_auto_unknowns

    closeEvent = _close_event

