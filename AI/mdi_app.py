
from __future__ import annotations
import sys
import argparse
import json
import time
import base64
from urllib import request, error
from urllib.parse import urlparse
from settings import load_settings

# Application version shown on the startup screen (GUI path).
APP_VERSION = "0.1.13"


def _fetch_json(url: str, headers: dict) -> dict:
    req = request.Request(url, headers=headers)
    with request.urlopen(req, timeout=5) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8"))


def _fetch_text(url: str, headers: dict) -> str:
    req = request.Request(url, headers=headers)
    with request.urlopen(req, timeout=5) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def run_profile_probe(base_url: str, iterations: int, delay: float, stream_dwell: float, logfile: str, headers: dict, token: str | None) -> None:
    """
    Hit the camera API, UI, snapshot, and stream to mimic real usage.
    Logs JSON lines to the given logfile for later analysis.
    """
    base = base_url.rstrip("/")
    token_q = f"?token={token}" if token else ""
    status_url = f"{base}/api/status{token_q}"
    mem_url = f"{base}/api/mem{token_q}"
    snap_url = (base.replace(":80", "") + ":81/snap") if base.startswith("http://") else (base + ":81/snap")
    if token:
        snap_url += token_q
    stream_url = (base.replace(":80", "") + ":81/stream") if base.startswith("http://") else (base + ":81/stream")
    if token:
        stream_url += token_q
    cam_page = f"{base}/cam{token_q}"

    print(f"[profile] Target: {base}  log={logfile}")
    with open(logfile, "a", encoding="utf-8") as log:
        for i in range(iterations):
            entry = {
                "ts": time.time(),
                "iter": i + 1,
                "status": None,
                "mem": None,
                "snap_ok": False,
                "stream_bytes": 0,
                "cam_page_ok": False,
                "errors": [],
            }
            print(f"[profile] Iteration {i+1}/{iterations}")
            try:
                entry["status"] = _fetch_json(status_url, headers)
                print(f"  status: {entry['status']}")
            except Exception as e:
                entry["errors"].append(f"status:{e}")
                print(f"  status error: {e}")
            try:
                entry["mem"] = _fetch_json(mem_url, headers)
                print(f"  mem: {entry['mem']}")
            except Exception as e:
                entry["errors"].append(f"mem:{e}")
                print(f"  mem error: {e}")
            try:
                _ = _fetch_text(cam_page, headers)
                entry["cam_page_ok"] = True
                print("  cam page: ok")
            except Exception as e:
                entry["errors"].append(f"cam:{e}")
                print(f"  cam page error: {e}")
            try:
                snap_req = request.Request(snap_url, headers=headers)
                with request.urlopen(snap_req, timeout=5) as resp:
                    _ = resp.read(2048)
                entry["snap_ok"] = True
                print("  snap: ok")
            except Exception as e:
                entry["errors"].append(f"snap:{e}")
                print(f"  snap error: {e}")
            if stream_dwell > 0:
                try:
                    start = time.time()
                    total = 0
                    stream_req = request.Request(stream_url, headers=headers)
                    with request.urlopen(stream_req, timeout=5) as resp:
                        while time.time() - start < stream_dwell:
                            chunk = resp.read(4096)
                            if not chunk:
                                break
                            total += len(chunk)
                    entry["stream_bytes"] = total
                    print(f"  stream: {total} bytes over {stream_dwell}s")
                except Exception as e:
                    entry["errors"].append(f"stream:{e}")
                    print(f"  stream error: {e}")

            log.write(json.dumps(entry) + "\n")
            log.flush()
            time.sleep(delay)


def maybe_run_profile_cli() -> None:
    """
    If CLI args include --profile-url, run the probe and exit (no GUI).
    """
    if any(arg.startswith("--profile-url") for arg in sys.argv):
        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument("--profile-url", required=True, help="Base http://camera-ip")
        parser.add_argument("--iterations", type=int, default=5)
        parser.add_argument("--delay", type=float, default=1.0, help="Seconds between samples")
        parser.add_argument("--stream-dwell", type=float, default=5.0, help="Seconds to read from /stream each iteration")
        parser.add_argument("--logfile", default="profile_log.jsonl", help="Path to write JSON lines log")
        parser.add_argument("--auth-user", help="Basic auth username")
        parser.add_argument("--auth-pass", help="Basic auth password")
        parser.add_argument("--auth-token", help="Token query param (e.g., base64 user:pass)")
        args, _ = parser.parse_known_args()
        # Resolve auth from CLI overrides or app config (matching host)
        headers = {}
        token = args.auth_token
        user = args.auth_user
        pwd = args.auth_pass
        if user is None or pwd is None or token is None:
            try:
                cfg = load_settings()
                host = urlparse(args.profile_url).hostname
                if host:
                    for cam in cfg.cameras:
                        cam_host = urlparse(cam.stream_url).hostname
                        if cam_host == host:
                            if user is None and cam.user:
                                user = cam.user
                            if pwd is None and cam.password:
                                pwd = cam.password
                            if token is None and cam.token:
                                token = cam.token
                            break
            except Exception as e:
                print(f"[profile] warning: config lookup failed ({e})")
        if user is not None and pwd is not None:
            raw = f"{user}:{pwd}".encode("utf-8")
            b64 = base64.b64encode(raw).decode("ascii")
            headers["Authorization"] = f"Basic {b64}"
        run_profile_probe(args.profile_url, args.iterations, args.delay, args.stream_dwell, args.logfile, headers, token)
        sys.exit(0)


def main():
    # GUI-only imports are delayed so profiler mode has minimal deps.
    import cv2_dll_fix
    cv2_dll_fix.enable_opencv_cuda_dll_search()
    from PySide6 import QtWidgets
    from settings import load_settings
    from UI.main_window import MainWindow
    import utils
    from utils import DebugMode
    from mqtt_client import MqttService

    # Enable debug (prints + logs to AI/logs/debug.log)
    #Only print: utils.DEBUG_MODE = DebugMode.PRINT
    #Only log to file: utils.DEBUG_MODE = DebugMode.LOG
    #Disable: utils.DEBUG_MODE = DebugMode.OFF
    utils.DEBUG_MODE = DebugMode.BOTH

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP32-CAM AI Viewer")
    app_cfg = load_settings()
    mqtt = MqttService(app_cfg)
    mqtt.start()

    win = MainWindow(app_cfg, load_on_init=True, mqtt_service=mqtt)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    maybe_run_profile_cli()
    main()
