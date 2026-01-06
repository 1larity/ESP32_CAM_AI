from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtGui, QtWidgets

from settings import save_settings
from utils import open_folder_or_warn
from UI.face_tuner import FaceRecTunerDialog

if TYPE_CHECKING:
    from UI.main_window import MainWindow


def build_menus(win: "MainWindow") -> None:
    menubar = win.menuBar()

    # File
    m_file = menubar.addMenu("File")
    act_add_ip = m_file.addAction("Add Camera by IP…")
    act_add_ip.triggered.connect(win._add_camera_ip_dialog)
    act_add_url = m_file.addAction("Add Camera by URL…")
    act_add_url.triggered.connect(win._add_camera_url_dialog)
    m_file.addSeparator()
    act_save = m_file.addAction("Save Settings")
    act_save.triggered.connect(lambda: save_settings(win.app_cfg))
    m_file.addSeparator()
    act_exit = m_file.addAction("Exit")
    act_exit.triggered.connect(win.close)

    # Cameras
    m_cams = menubar.addMenu("Cameras")
    m_cams.addAction("Add Camera by IP…").triggered.connect(win._add_camera_ip_dialog)
    m_cams.addAction("Add Camera by URL…").triggered.connect(win._add_camera_url_dialog)
    m_cams.addAction("Remove Camera…").triggered.connect(win._remove_camera_dialog)
    m_cams.addAction("Rename Camera…").triggered.connect(win._rename_camera_dialog)
    m_cams.addSeparator()
    m_cams.addAction("Discover ESP32-CAMs…").triggered.connect(win._open_discovery)
    m_cams.addAction("Discover ONVIF Cameras…").triggered.connect(win._open_onvif_discovery)

    # Tools
    m_tools = menubar.addMenu("Tools")
    # Per request, removed: Open config/data/models folder
    m_tools.addAction("Open recordings folder").triggered.connect(
        lambda: open_folder_or_warn(win, win.app_cfg.output_dir)
    )
    m_tools.addAction("Open logs folder").triggered.connect(
        lambda: open_folder_or_warn(win, win.app_cfg.logs_dir)
    )
    m_tools.addAction("Purge auto-trained unknowns").triggered.connect(
        win._purge_auto_unknowns
    )
    m_tools.addSeparator()
    m_tools.addAction("MQTT Settings").triggered.connect(win._open_mqtt_settings)
    m_tools.addSeparator()
    m_tools.addAction("Enroll faces / pets…").triggered.connect(win._open_enrollment)
    m_tools.addAction("Image manager…").triggered.connect(win._open_image_manager)
    m_tools.addAction("Face recognizer tuner").triggered.connect(
        lambda: FaceRecTunerDialog(str(win.app_cfg.models_dir), win).exec()
    )
    # Unknown capture / auto-train dialog
    m_tools.addAction("Unknown capture & auto-train…").triggered.connect(
        win._open_unknown_capture_dialog
    )

    # LBPH toggle (ignore enrollment models)
    act_ignore = QtGui.QAction("Ignore enrollment models (disable LBPH)", win)
    act_ignore.setCheckable(True)
    act_ignore.setChecked(bool(getattr(win.app_cfg, "ignore_enrollment_models", False)))
    act_ignore.toggled.connect(win._on_ignore_enroll_toggled)
    m_tools.addAction(act_ignore)

    act_gpu = QtGui.QAction("Use GPU for YOLO (requires CUDA build)", win)
    act_gpu.setCheckable(True)
    act_gpu.setChecked(bool(getattr(win.app_cfg, "use_gpu", False)))
    act_gpu.toggled.connect(win._on_use_gpu_toggled)
    m_tools.addAction(act_gpu)

    act_archive = QtGui.QAction("Archive person/pet and rebuild", win)
    act_archive.triggered.connect(win._archive_person_folder)
    m_tools.addAction(act_archive)

    act_restore = QtGui.QAction("Restore person/pet from archive and rebuild", win)
    act_restore.triggered.connect(win._restore_person_folder)
    m_tools.addAction(act_restore)

    act_rebuild_faces = QtGui.QAction("Rebuild face model from disk", win)
    act_rebuild_faces.triggered.connect(
        lambda: win._start_face_rebuild("Rebuild Face Model")
    )
    m_tools.addAction(act_rebuild_faces)

    # View
    m_view = menubar.addMenu("View")
    act_events = m_view.addAction("Events pane")
    act_events.triggered.connect(win._toggle_events_pane)
    m_view.addSeparator()
    m_view.addAction("Tile Subwindows").triggered.connect(win.mdi.tileSubWindows)
    m_view.addAction("Cascade Subwindows").triggered.connect(win.mdi.cascadeSubWindows)
    m_view.addSeparator()
    m_view.addAction("Fit All to Window").triggered.connect(win._fit_all)
    m_view.addAction("100% All").triggered.connect(win._100_all)
    m_view.addAction("Resize windows to video size").triggered.connect(
        win._resize_all_to_video
    )

