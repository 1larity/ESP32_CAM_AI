from __future__ import annotations
from typing import Optional
from PyQt6 import QtCore, QtGui, QtWidgets

from settings import AppSettings, CameraSettings, save_settings
from utils import open_folder_or_warn
from models import ModelManager
from enrollment import EnrollmentService
from UI.enrollment import EnrollDialog
from UI.image_manager import ImageManagerDialog
from UI.events_pane import EventsPane
from UI.discovery_dialog import DiscoveryDialog
from UI.ip_cam_dialog import AddIpCameraDialog
from UI.camera_widget import CameraWidget
from enrollment import get_enrollment_service


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_cfg: AppSettings):
        super().__init__()
        self.app_cfg = app_cfg

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
        self.dock_events.setWidget(self.events_pane)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_events
        )
        self.dock_events.hide()

        self._build_menus()
        self._load_initial_cameras()

    # ------------------------------------------------------------------ #
    # camera windows
    # ------------------------------------------------------------------ #

    def _load_initial_cameras(self) -> None:
        for cam in self.app_cfg.cameras:
            self._add_camera_window(cam)

    def _add_camera_window(self, cam_cfg: CameraSettings) -> None:
        w = CameraWidget(cam_cfg, self.app_cfg, self)

        sub = QtWidgets.QMdiSubWindow()
        sub.setWidget(w)
        sub.setWindowTitle(cam_cfg.name)
        # remove Qt icon from cam windows
        sub.setWindowIcon(QtGui.QIcon())
        self.mdi.addSubWindow(sub)

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
        save_settings(self.app_cfg)
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # dialogs
    # ------------------------------------------------------------------ #

    def _open_enrollment(self) -> None:
        dlg = EnrollDialog(self.app_cfg, self)
        dlg.exec()
        # After enrollment, rebuild LBPH model from disk (silent)
        self._rebuild_faces_silent()

    def _open_image_manager(self) -> None:
        dlg = ImageManagerDialog(self.app_cfg, self)
        dlg.exec()
        # After image management changes, rebuild LBPH model from disk (silent)
        self._rebuild_faces_silent()

    def _open_discovery(self) -> None:
        dlg = DiscoveryDialog(self.app_cfg, self)
        dlg.exec()

    # ------------------------------------------------------------------ #
    # menus
    # ------------------------------------------------------------------ #

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        # File
        m_file = menubar.addMenu("File")
        act_add_ip = m_file.addAction("Add Camera by IP…")
        act_add_ip.triggered.connect(self._add_camera_ip_dialog)
        act_add_url = m_file.addAction("Add Camera by URL…")
        act_add_url.triggered.connect(self._add_camera_url_dialog)
        m_file.addSeparator()
        act_save = m_file.addAction("Save Settings")
        act_save.triggered.connect(lambda: save_settings(self.app_cfg))
        m_file.addSeparator()
        act_exit = m_file.addAction("Exit")
        act_exit.triggered.connect(self.close)

        # Cameras
        m_cams = menubar.addMenu("Cameras")
        m_cams.addAction("Enroll faces / pets…").triggered.connect(
            self._open_enrollment
        )
        m_cams.addAction("Image manager…").triggered.connect(
            self._open_image_manager
        )
        m_cams.addSeparator()
        m_cams.addAction("Discover ESP32-CAMs…").triggered.connect(
            self._open_discovery
        )

        # Tools
        m_tools = menubar.addMenu("Tools")
        # NOTE: per request, removed: Open config/data/models folder
        m_tools.addAction("Open recordings folder").triggered.connect(
            lambda: open_folder_or_warn(self, self.app_cfg.output_dir)
        )
        m_tools.addAction("Open logs folder").triggered.connect(
            lambda: open_folder_or_warn(self, self.app_cfg.logs_dir)
        )
        m_tools.addSeparator()
        m_tools.addAction("Fetch default models…").triggered.connect(
            lambda: ModelManager.fetch_defaults(self, self.app_cfg)
        )
        act_rebuild_faces = QtGui.QAction("Rebuild face model from disk…", self)
        act_rebuild_faces.triggered.connect(self._rebuild_faces)
        m_tools.addAction(act_rebuild_faces)

        # View
        m_view = menubar.addMenu("View")
        act_events = m_view.addAction("Events pane")
        act_events.triggered.connect(self._toggle_events_pane)
        m_view.addSeparator()
        m_view.addAction("Tile Subwindows").triggered.connect(
            self.mdi.tileSubWindows
        )
        m_view.addAction("Cascade Subwindows").triggered.connect(
            self.mdi.cascadeSubWindows
        )
        m_view.addSeparator()
        m_view.addAction("Fit All to Window").triggered.connect(self._fit_all)
        m_view.addAction("100% All").triggered.connect(self._100_all)
        m_view.addAction("Resize windows to video size").triggered.connect(
            self._resize_all_to_video
        )

    # ------------------------------------------------------------------ #
    # face model rebuild
    # ------------------------------------------------------------------ #

    def _rebuild_faces_silent(self) -> bool:
        """
        Rebuild LBPH face model from the images on disk, without
        displaying any UI messages. Returns True if a model was built.
        """
        svc = get_enrollment_service()
        return svc.rebuild_lbph_model_from_disk()

    def _rebuild_faces(self) -> None:
        """
        Rebuild LBPH face model from the images on disk and show
        a message box with the result.
        """
        ok = self._rebuild_faces_silent()
        if ok:
            QtWidgets.QMessageBox.information(
                self,
                "Rebuild Face Model",
                "LBPH model rebuilt from disk samples.",
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Rebuild Face Model",
                "No face samples found to rebuild.",
            )
