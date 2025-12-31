
from __future__ import annotations
import cv2_dll_fix
cv2_dll_fix.enable_opencv_cuda_dll_search()
import sys
from PySide6  import QtWidgets, QtCore
from settings import load_settings
from UI.main_window import MainWindow
from UI.startup import StartupDialog
from models import ModelManager
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

    def _cuda_status_probe(dlg: StartupDialog) -> None:
        """Run a quick CUDA availability check and show it on the loader."""
        dlg.update_status("Checking CUDA...")
        cuda_msg = "Checking CUDA..."
        try:
            import cv2

            cuda_ok = False
            detail = ""
            if hasattr(cv2, "cuda"):
                try:
                    cnt = cv2.cuda.getCudaEnabledDeviceCount()
                    cuda_ok = bool(cnt and cnt > 0)
                    detail = f"devices={cnt}"
                except Exception as e:
                    detail = str(e)
            else:
                detail = "cv2.cuda missing"

            msg = "CUDA detected; GPU acceleration enabled"
            if not cuda_ok:
                msg = "CUDA not available; using CPU"
                if detail:
                    msg += f" ({detail})"
            cuda_msg = msg
            dlg.update_status(cuda_msg)
        except Exception as e:
            cuda_msg = f"CUDA check failed; using CPU ({e})"
            dlg.update_status(cuda_msg)

        # Ensure required models are present (auto-download if missing)
        def _status_cb(msg: str) -> None:
            try:
                dlg.update_status(msg)
            except Exception:
                pass

        try:
            ModelManager.ensure_models(app_cfg, status_cb=_status_cb)
            # Restore CUDA status after model downloads so the user sees it.
            dlg.update_status(cuda_msg)
        except Exception as e:
            dlg.update_status(f"Model check failed: {e}")

    # Show a short loader while wiring camera windows; loader kicks off internally.
    if app_cfg.cameras:
        try:
            dlg = StartupDialog(
                app_cfg.cameras,
                loader=win._add_camera_window,
                parent=win,
                version=APP_VERSION,
                preflight=_cuda_status_probe,
                initial_status="Starting...",
                preflight_delay_ms=1000,
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
