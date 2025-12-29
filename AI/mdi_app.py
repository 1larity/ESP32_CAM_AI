from __future__ import annotations

import sys
from PySide6  import QtWidgets
from settings import load_settings
from UI.main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP32-CAM AI Viewer")
    app_cfg = load_settings()
    win = MainWindow(app_cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
