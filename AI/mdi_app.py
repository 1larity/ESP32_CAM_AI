# mdi_app.py
# Adds Tools → Discover ESP32-CAM…, wires dialog, no other logic changed.
from __future__ import annotations
import sys
from typing import Optional
from PyQt6 import QtCore, QtGui, QtWidgets

from detectors import DetectorThread, DetectorConfig, DetectionPacket
from overlays import OverlayFlags, draw_overlays
from recorder import PrebufferRecorder
from presence import PresenceBus
from settings import AppSettings, CameraSettings, load_settings, save_settings
from ptz import PTZClient
from utils import qimage_from_bgr, open_folder_or_warn
from stream import StreamCapture
from enrollment import EnrollDialog
from image_manager import ImageManagerDialog
from models import ModelManager
from enrollment_service import EnrollmentService
from events_pane import EventsPane
from discovery_dialog import DiscoveryDialog   # NEW

# ... [GraphicsView, CameraWidget unchanged from prior version] ...

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_cfg: AppSettings):
        super().__init__()
        self.app_cfg = app_cfg
        ModelManager.ensure_models(self, self.app_cfg)
        self.mdi = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi)
        self._build_menus()
        self._load_initial_cameras()
        self.events = EventsPane(self.app_cfg.logs_dir, parent=self)
        dock = QtWidgets.QDockWidget("Events", self); dock.setWidget(self.events); dock.setObjectName("EventsDock")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.statusBar().showMessage("Enrollment progress and ESP32-CAM discovery ready.")

    def _build_menus(self):
        menubar = self.menuBar()
        m_file = menubar.addMenu("File")
        act_add_ip = m_file.addAction("Add Camera by IP…")
        act_add_url = m_file.addAction("Add Camera by URL…")
        m_file.addSeparator(); act_save = m_file.addAction("Save Settings")
        m_file.addSeparator(); act_exit = m_file.addAction("Exit")

        m_tools = menubar.addMenu("Tools")
        act_enroll = m_tools.addAction("Enrollment…")
        act_images = m_tools.addAction("Image Manager…")
        m_tools.addSeparator()
        act_models = m_tools.addAction("Open Models Folder")
        act_record = m_tools.addAction("Open Recordings Folder")
        act_logs = m_tools.addAction("Open Logs Folder")
        m_tools.addSeparator()
        act_fetch_models = m_tools.addAction("Download Default Models…")
        act_discover = m_tools.addAction("Discover ESP32-CAM…")  # NEW

        m_view = menubar.addMenu("View")
        act_tile = m_view.addAction("Tile")
        act_cascade = m_view.addAction("Cascade")
        act_fit_all = m_view.addAction("Fit All")
        act_100_all = m_view.addAction("100% All")

        act_add_ip.triggered.connect(self._add_camera_ip_dialog)
        act_add_url.triggered.connect(self._add_camera_url_dialog)
        act_save.triggered.connect(lambda: save_settings(self.app_cfg))
        act_exit.triggered.connect(self.close)

        act_enroll.triggered.connect(self._open_enrollment)
        act_images.triggered.connect(self._open_image_manager)
        act_models.triggered.connect(lambda: open_folder_or_warn(self, self.app_cfg.models_dir))
        act_record.triggered.connect(lambda: open_folder_or_warn(self, self.app_cfg.output_dir))
        act_logs.triggered.connect(lambda: open_folder_or_warn(self, self.app_cfg.logs_dir))
        act_fetch_models.triggered.connect(lambda: ModelManager.fetch_defaults(self, self.app_cfg))
        act_discover.triggered.connect(self._discover_esp32)  # NEW

        tb = self.addToolBar("Main")
        tb.addAction("Add IP").triggered.connect(self._add_camera_ip_dialog)
        tb.addAction("Add URL").triggered.connect(self._add_camera_url_dialog)
        tb.addSeparator()
        tb.addAction("Tile").triggered.connect(self.mdi.tileSubWindows)
        tb.addAction("Cascade").triggered.connect(self.mdi.cascadeSubWindows)

        act_tile.triggered.connect(self.mdi.tileSubWindows)
        act_cascade.triggered.connect(self.mdi.cascadeSubWindows)
        act_fit_all.triggered.connect(self._fit_all)
        act_100_all.triggered.connect(self._100_all)

    # ... [rest unchanged] ...

    def _open_enrollment(self):
        EnrollDialog(self.app_cfg, self).exec()

    def _open_image_manager(self):
        ImageManagerDialog(self.app_cfg, self).exec()

    def _discover_esp32(self):
        DiscoveryDialog(self).exec()
