from __future__ import annotations

import sys
from PySide6  import QtWidgets
from settings import load_settings
from UI.main_window import MainWindow
from UI.startup import StartupDialog

# Application version shown on the startup screen.
APP_VERSION = "0.1.1"


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP32-CAM AI Viewer")
    app_cfg = load_settings()
    win = MainWindow(app_cfg, load_on_init=False)

    # Show a short loader while wiring camera windows; loader kicks off internally.
    if app_cfg.cameras:
        dlg = StartupDialog(
            app_cfg.cameras,
            loader=win._add_camera_window,
            parent=win,
            version=APP_VERSION,
        )
        dlg.exec()

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
