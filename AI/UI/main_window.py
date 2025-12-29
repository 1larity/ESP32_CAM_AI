from __future__ import annotations
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal, Slot
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
from UI.face_tuner import FaceRecTunerDialog
from enrollment import get_enrollment_service


class _FaceRebuildWorker(QtCore.QObject):
    """
    Runs the LBPH model rebuild in a background thread.
    """

    finished = Signal(bool)

    @Slot()
    def run(self) -> None:
        svc = get_enrollment_service()
        ok = svc.rebuild_lbph_model_from_disk()
        self.finished.emit(ok)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_cfg: AppSettings, *, load_on_init: bool = True):
        super().__init__()
        self.app_cfg = app_cfg

        # Background face rebuild state
        self._face_rebuild_thread: Optional[QtCore.QThread] = None
        self._face_rebuild_worker: Optional[_FaceRebuildWorker] = None
        self._face_rebuild_dialog: Optional[QtWidgets.QProgressDialog] = None

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
        if load_on_init:
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

        # Restore per-camera geometry/state if available
        geom_rec = (self.app_cfg.window_geometries or {}).get(cam_cfg.name)
        if geom_rec and len(geom_rec) >= 5:
            x, y, w_geom, h_geom, maximized = geom_rec[:5]
            if maximized:
                sub.showMaximized()
            else:
                sub.setGeometry(x, y, w_geom, h_geom)

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
        # Per request, removed: Open config/data/models folder
        m_tools.addAction("Open recordings folder").triggered.connect(
            lambda: open_folder_or_warn(self, self.app_cfg.output_dir)
        )
        m_tools.addAction("Open logs folder").triggered.connect(
            lambda: open_folder_or_warn(self, self.app_cfg.logs_dir)
        )
        m_tools.addSeparator()
        m_tools.addAction("Fetch default models").triggered.connect(
            lambda: ModelManager.fetch_defaults(self, self.app_cfg)
        )
        m_tools.addAction("Face recognizer tuner").triggered.connect(
            lambda: FaceRecTunerDialog(str(self.app_cfg.models_dir), self).exec()
        )
        act_rebuild_faces = QtGui.QAction("Rebuild face model from disk", self)
        act_rebuild_faces.triggered.connect(
            lambda: self._start_face_rebuild("Rebuild Face Model")
        )
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
    # face model rebuild with progress
    # ------------------------------------------------------------------ #

    def _start_face_rebuild(self, title: str) -> None:
        """
        Start a background LBPH face model rebuild and show a modal
        progress dialog so the user can see that work is happening.
        """
        # Avoid re-entrancy; if a rebuild is already in progress, ignore.
        if self._face_rebuild_thread is not None:
            return

        # Progress dialog (indeterminate)
        dlg = QtWidgets.QProgressDialog(
            "Rebuilding face model from disk…", "", 0, 0, self
        )
        dlg.setWindowTitle(title)
        dlg.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()

        self._face_rebuild_dialog = dlg

        # Background thread + worker
        thread = QtCore.QThread(self)
        worker = _FaceRebuildWorker()
        worker.moveToThread(thread)

        worker.finished.connect(self._on_face_rebuild_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)

        self._face_rebuild_thread = thread
        self._face_rebuild_worker = worker

        thread.start()

    @Slot(bool)
    def _on_face_rebuild_finished(self, ok: bool) -> None:
        """
        Invoked in the GUI thread when the background rebuild finishes.
        """
        # Close the progress dialog
        if self._face_rebuild_dialog is not None:
            self._face_rebuild_dialog.close()
            self._face_rebuild_dialog = None

        # Clean up worker/thread handles
        self._face_rebuild_worker = None
        self._face_rebuild_thread = None

        # Inform the user of the result
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
