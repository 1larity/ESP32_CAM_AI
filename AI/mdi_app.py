# mdi_app.py
# ESP32-CAM AI Viewer / Controller
# Multi-camera MDI interface with YOLO + Face Recognition overlays,
# enrollment, events dock, and discovery.
# All PyQt6.

from __future__ import annotations
import sys
from typing import Optional
from PyQt6 import QtCore, QtGui, QtWidgets

from detectors import DetectorThread, DetectorConfig, DetectionPacket
from overlays import OverlayFlags, draw_overlays
from recorder import PrebufferRecorder
from presence import PresenceBus
from settings import AppSettings, CameraSettings, load_settings, save_settings
from utils import qimage_from_bgr, open_folder_or_warn
from stream import StreamCapture
from enrollment import EnrollDialog
from image_manager import ImageManagerDialog
from models import ModelManager
from enrollment_service import EnrollmentService
from events_pane import EventsPane
from discovery_dialog import DiscoveryDialog


# -----------------------------------------------------------------------------
# GraphicsView with zoom + pan (used inside each CameraWidget)
# -----------------------------------------------------------------------------
class GraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = QtCore.pyqtSignal(float)

    def __init__(self, scene: QtWidgets.QGraphicsScene, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
            | QtGui.QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self._scale = 1.0
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setMouseTracking(True)

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.0 + (0.0015 * e.angleDelta().y())
            self._scale = float(max(0.1, min(8.0, self._scale * factor)))
            target = self.mapToScene(e.position().toPoint())
            self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
            self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
            self.setTransform(QtGui.QTransform())
            self.scale(self._scale, self._scale)
            self.centerOn(target)
            self.zoomChanged.emit(self._scale)
        else:
            super().wheelEvent(e)


# -----------------------------------------------------------------------------
# CameraWidget for each camera window
# -----------------------------------------------------------------------------
class CameraWidget(QtWidgets.QWidget):
    def __init__(self, cam_cfg: CameraSettings, app_cfg: AppSettings, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.app_cfg = app_cfg

        # Graphics scene + view for zoom/pan
        self._scene = QtWidgets.QGraphicsScene(self)
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.view = GraphicsView(self._scene, self)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)

        self._overlays = OverlayFlags()
        self._last_bgr = None
        self._last_ts = 0

        # Prebuffer recorder: per-camera
        self._recorder = PrebufferRecorder(
            cam_name=self.cam_cfg.name,
            out_dir=self.app_cfg.output_dir,
            fps=25,
            pre_ms=self.app_cfg.prebuffer_ms,
        )

        # Presence logging bus
        self._presence = PresenceBus(self.cam_cfg.name, self.app_cfg.logs_dir)

        # Detector thread
        det_cfg = DetectorConfig.from_app(self.app_cfg)
        self._detector = DetectorThread(det_cfg, self.cam_cfg.name)
        self._detector.resultsReady.connect(self._on_detections)

        # Stream capture backend
        self._capture = StreamCapture(self.cam_cfg)

        # Poll frames from StreamCapture
        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setInterval(30)
        self._frame_timer.timeout.connect(self._poll_frame)

        self._detector.start()

    # ---- lifecycle ----
    def start(self):
        self._capture.start()
        self._frame_timer.start()

    def stop(self):
        self._frame_timer.stop()
        self._capture.stop()
        self._detector.stop()
        self._recorder.close()

    # ---- frame handling ----
    def _poll_frame(self):
        ok, frame, ts_ms = self._capture.read()
        if not ok or frame is None:
            return
        self._last_bgr = frame
        self._last_ts = ts_ms

        # Feed recorder and detector
        self._recorder.on_frame(frame, ts_ms)
        self._detector.submit_frame(self.cam_cfg.name, frame, ts_ms)

        # Show raw frame (detector callback will redraw with overlays)
        self._update_pixmap(frame, None)

    def _update_pixmap(self, bgr, pkt: Optional[DetectionPacket]):
        qimg = qimage_from_bgr(bgr)
        pixmap = QtGui.QPixmap.fromImage(qimg)

        if pkt is not None:
            painter = QtGui.QPainter(pixmap)
            try:
                draw_overlays(painter, pkt, self._overlays)
            finally:
                painter.end()

        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))

    @QtCore.pyqtSlot(object)
    def _on_detections(self, pkt_obj):
        pkt = pkt_obj
        if not isinstance(pkt, DetectionPacket):
            return
        if pkt.name != self.cam_cfg.name:
            return

        # **DEBUG** show that GUI is receiving packets
        print(
            f"[GUI:{self.cam_cfg.name}] recv pkt ts={pkt.ts_ms} "
            f"yolo={len(pkt.yolo)} faces={len(pkt.faces)} pets={len(pkt.pets)}"
        )

        self._presence.update(pkt)
        if self._last_bgr is not None:
            self._update_pixmap(self._last_bgr, pkt)

    # ---- view helpers for MainWindow ----
    def fit_to_window(self):
        self.view.fitInView(self._scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.view._scale = 1.0

    def zoom_100(self):
        self.view.resetTransform()
        self.view._scale = 1.0


# -----------------------------------------------------------------------------
# MainWindow
# -----------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_cfg: AppSettings):
        super().__init__()
        self.app_cfg = app_cfg
        self.setWindowTitle("ESP32-CAM AI Viewer")
        self.resize(1280, 800)

        self.mdi = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi)

        # Events pane dock
        self.events_pane = EventsPane(self.app_cfg.logs_dir, parent=self)
        self.dock_events = QtWidgets.QDockWidget("Events", self)
        self.dock_events.setWidget(self.events_pane)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock_events)
        self.dock_events.hide()

        self._build_menus()
        self._load_initial_cameras()

    def _load_initial_cameras(self):
        for cam in self.app_cfg.cameras:
            self._add_camera_window(cam)

    def _add_camera_window(self, cam_cfg: CameraSettings):
        w = CameraWidget(cam_cfg, self.app_cfg, self)
        sub = QtWidgets.QMdiSubWindow()
        sub.setWidget(w)
        sub.setWindowTitle(cam_cfg.name)
        self.mdi.addSubWindow(sub)
        w.start()
        sub.show()

    # ---- camera adding ----
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

    def _add_camera_ip_dialog(self):
        dlg = DiscoveryDialog(self)
        dlg.exec()

    # ---- view / tools ----
    def _toggle_events_pane(self):
        if self.dock_events.isVisible():
            self.dock_events.hide()
        else:
            self.dock_events.show()

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

    def closeEvent(self, event: QtGui.QCloseEvent):
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.stop()
        save_settings(self.app_cfg)
        super().closeEvent(event)

    def _build_menus(self):
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

        # Tools
        m_tools = menubar.addMenu("Tools")
        act_enroll = m_tools.addAction("Enrollment…")
        act_enroll.triggered.connect(self._open_enrollment)
        act_img_mgr = m_tools.addAction("Image Manager…")
        act_img_mgr.triggered.connect(self._open_image_manager)
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

        # Rebuild faces menu option
        act_rebuild_faces = QtGui.QAction("Rebuild face model from disk…", self)
        act_rebuild_faces.triggered.connect(self._rebuild_faces)
        m_tools.addAction(act_rebuild_faces)

        # View
        m_view = menubar.addMenu("View")
        act_events = m_view.addAction("Events pane")
        act_events.triggered.connect(self._toggle_events_pane)
        m_view.addSeparator()
        m_view.addAction("Tile Subwindows").triggered.connect(self.mdi.tileSubWindows)
        m_view.addAction("Cascade Subwindows").triggered.connect(self.mdi.cascadeSubWindows)
        m_view.addSeparator()
        m_view.addAction("Fit All").triggered.connect(self._fit_all)
        m_view.addAction("100% All").triggered.connect(self._100_all)

    def _open_enrollment(self):
        dlg = EnrollDialog(self.app_cfg, self)
        dlg.exec()

    def _open_image_manager(self):
        dlg = ImageManagerDialog(self.app_cfg, self)
        dlg.exec()

    def _discover_esp32(self):
        dlg = DiscoveryDialog(self)
        dlg.exec()

    def _rebuild_faces(self):
        svc = EnrollmentService.instance()
        ok = False
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


# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP32-CAM AI Viewer")
    app_cfg = load_settings()
    win = MainWindow(app_cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
