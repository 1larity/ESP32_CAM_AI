from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from settings import save_settings
from UI.camera import CameraWidget


def closeEvent(self, event: QtGui.QCloseEvent) -> None:
    for sub in self.mdi.subWindowList():
        w = sub.widget()
        if isinstance(w, CameraWidget):
            w.stop()
    if self._mqtt is not None:
        try:
            self._mqtt.stop()
        except Exception:
            pass
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
    QtWidgets.QMainWindow.closeEvent(self, event)

