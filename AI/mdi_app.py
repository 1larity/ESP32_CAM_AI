# mdi_app.py
from __future__ import annotations
import sys
from PyQt6 import QtCore, QtGui, QtWidgets

from settings import AppSettings, CameraSettings, load_settings, save_settings
from utils import open_folder_or_warn
from image_manager import ImageManagerDialog
from models import ModelManager
from enrollment import EnrollDialog
from enrollment_service import EnrollmentService
from events_pane import EventsPane
from discovery_dialog import DiscoveryDialog
from ip_cam_dialog import AddIpCameraDialog
from camera_widget import CameraWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_cfg: AppSettings):
        super().__init__()
        self.app_cfg = app_cfg
        self.setWindowTitle("ESP32-CAM AI Viewer")
        self.resize(1280, 800)

        self.mdi = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi)

        self.events_pane = EventsPane(self.app_cfg.logs_dir, parent=self)
        self.dock_events = QtWidgets.QDockWidget("Events", self)
        self.dock_events.setWidget(self.events_pane)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_events)
        self.dock_events.hide()

        self._build_menus()
        self._load_initial_cameras()

    # cameras
    def _load_initial_cameras(self):
        for cam in self.app_cfg.cameras:
            self._add_camera_window(cam)

    def _add_camera_window(self, cam_cfg: CameraSettings):
        w = CameraWidget(cam_cfg, self.app_cfg, self)
        sub = QtWidgets.QMdiSubWindow()
        sub.setWidget(w)
        sub.setWindowTitle(cam_cfg.name)
        sub.setWindowIcon(QtGui.QIcon())  # no Qt icon
        self.mdi.addSubWindow(sub)
        w.start()
        sub.show()

    def _add_camera_url_dialog(self):
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

    def _add_camera_ip_dialog(self):
        dlg = AddIpCameraDialog(self.app_cfg, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cam_cfg = dlg.get_camera()
            if cam_cfg is not None:
                self.app_cfg.cameras.append(cam_cfg)
                self._add_camera_window(cam_cfg)
                save_settings(self.app_cfg)

    # view / tools
    def _toggle_events_pane(self):
        self.dock_events.setVisible(not self.dock_events.isVisible())

    def _fit_all(self):
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.fit_to_window()

    def _100_all(self):
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.zoom_100()

    def _resize_all_to_video(self):
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget) and w._last_bgr is not None:
                h, width = w._last_bgr.shape[:2]
                sub.resize(width + 40, h + 80)

    def closeEvent(self, event: QtGui.QCloseEvent):
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.stop()
        save_settings(self.app_cfg)
        super().closeEvent(event)

    def _build_menus(self):
        mb = self.menuBar()

        m_file = mb.addMenu("File")
        m_file.addAction("Add Camera by IP…").triggered.connect(self._add_camera_ip_dialog)
        m_file.addAction("Add Camera by URL…").triggered.connect(self._add_camera_url_dialog)
        m_file.addSeparator()
        m_file.addAction("Save Settings").triggered.connect(lambda: save_settings(self.app_cfg))
        m_file.addSeparator()
        m_file.addAction("Exit").triggered.connect(self.close)

        m_tools = mb.addMenu("Tools")
        m_tools.addAction("Enrollment…").triggered.connect(self._open_enrollment)
        m_tools.addAction("Image Manager…").triggered.connect(self._open_image_manager)
        m_tools.addSeparator()
        m_tools.addAction("Open models folder").triggered.connect(
            lambda: open_folder_or_warn(self, self.app_cfg.models_dir)
        )
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
        m_tools.addSeparator()
        m_tools.addAction("Discover ESP32-CAMs…").triggered.connect(self._discover_esp32)

        act_rebuild_faces = QtGui.QAction("Rebuild face model from disk…", self)
        act_rebuild_faces.triggered.connect(self._rebuild_faces)
        m_tools.addAction(act_rebuild_faces)

        m_view = mb.addMenu("View")
        m_view.addAction("Events pane").triggered.connect(self._toggle_events_pane)
        m_view.addSeparator()
        m_view.addAction("Tile Subwindows").triggered.connect(self.mdi.tileSubWindows)
        m_view.addAction("Cascade Subwindows").triggered.connect(self.mdi.cascadeSubWindows)
        m_view.addSeparator()
        m_view.addAction("Fit All").triggered.connect(self._fit_all)
        m_view.addAction("100% All").triggered.connect(self._100_all)
        m_view.addAction("Resize windows to video size").triggered.connect(self._resize_all_to_video)

    # dialogs / tools
    def _open_enrollment(self):
        EnrollDialog(self.app_cfg, self).exec()

    def _open_image_manager(self):
        ImageManagerDialog(self.app_cfg, self).exec()

    def _discover_esp32(self):
        DiscoveryDialog(self).exec()

    def _rebuild_faces(self):
        svc = EnrollmentService.instance()
        try:
            ok = svc._train_lbph()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Rebuild Face Model", f"Failed:\n{e}")
            return
        if ok:
            QtWidgets.QMessageBox.information(
                self, "Rebuild Face Model", "LBPH model rebuilt from disk samples."
            )
        else:
            QtWidgets.QMessageBox.information(
                self, "Rebuild Face Model", "No face samples found to rebuild."
            )


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP32-CAM AI Viewer")
    app_cfg = load_settings()
    win = MainWindow(app_cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
