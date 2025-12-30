from __future__ import annotations

import sys
from PySide6  import QtWidgets
from settings import load_settings
from UI.main_window import MainWindow
from UI.startup import StartupDialog
import utils
from utils import DebugMode

# Enable debug (prints + logs to AI/logs/debug.log)
#Only print: utils.DEBUG_MODE = DebugMode.PRINT
#Only log to file: utils.DEBUG_MODE = DebugMode.LOG
#Disable: utils.DEBUG_MODE = DebugMode.OFF
utils.DEBUG_MODE = DebugMode.BOTH

# Application version shown on the startup screen.
APP_VERSION = "0.1.8"


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP32-CAM AI Viewer")
    app_cfg = load_settings()
    win = MainWindow(app_cfg, load_on_init=False)

    # Show a short loader while wiring camera windows; loader kicks off internally.
    if app_cfg.cameras:
        try:
            dlg = StartupDialog(
                app_cfg.cameras,
                loader=win._add_camera_window,
                parent=win,
                version=APP_VERSION,
            )
            dlg.exec()
        except Exception as e:
            print(f"[Startup] dialog failed, falling back to direct load: {e}")
            for cam in app_cfg.cameras:
                try:
                    win._add_camera_window(cam)
                except Exception as e_add:
                    print(f"[Startup] failed to add {getattr(cam, 'name', '?')}: {e_add}")

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
