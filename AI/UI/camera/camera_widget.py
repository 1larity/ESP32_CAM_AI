# camera/camera_widget.py
from __future__ import annotations

import queue
import re
import threading
import time
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse
import requests

from PySide6 import QtCore, QtGui, QtWidgets

from settings import AppSettings, CameraSettings, save_settings
from utils import debug_ptz
from .camera_settings_dialog import CameraSettingsDialog

# Helper initialiser / attach functions
from .camera_widget_init import init_camera_widget
from .camera_widget_video import attach_video_handlers
from .camera_widget_overlays import attach_overlay_handlers
from .camera_widget_view import attach_view_handlers

# Map framesize enum -> (label, width, height) per esp32-camera sensor.h
FRAME_SIZES = {
    0: ("96x96", 96, 96),
    1: ("QQVGA", 160, 120),
    2: ("QCIF", 176, 144),
    3: ("HQVGA", 240, 176),
    4: ("240x240", 240, 240),
    5: ("QVGA", 320, 240),
    6: ("CIF", 352, 288),
    7: ("HVGA", 480, 320),
    8: ("VGA", 640, 480),
    9: ("SVGA", 800, 600),
    10: ("XGA", 1024, 768),
    11: ("SXGA", 1280, 1024),
    12: ("UXGA", 1600, 1200),
    13: ("QXGA", 2048, 1536),
}


class CameraWidget(QtWidgets.QWidget):
    """
    One camera widget.

    Responsibilities are split into helper modules:
      - init_camera_widget(self)               → build UI, state, wiring
      - attach_video_handlers(CameraWidget)    → frame polling, recorder, HUD, detections handler
      - attach_overlay_handlers(CameraWidget)  → AI / overlay toggles
      - attach_view_handlers(CameraWidget)     → fit / zoom helpers
    """

    # Class-level guard: ensures injected handlers exist before init wiring connects signals.
    _handlers_attached: bool = False

    def __init__(
        self,
        cam_cfg: CameraSettings,
        app_cfg: AppSettings,
        parent: Optional[QtWidgets.QWidget] = None,
        mqtt_service=None,
    ) -> None:
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.app_cfg = app_cfg
        self._mqtt = mqtt_service

        # IMPORTANT:
        # Ensure injected methods (including _on_detections) exist BEFORE init_camera_widget()
        # connects signals to them. This avoids startup crashes if module import ordering causes
        # attach_* not to have run yet.
        if not self.__class__._handlers_attached:
            attach_video_handlers(self.__class__)
            attach_overlay_handlers(self.__class__)
            attach_view_handlers(self.__class__)
            self.__class__._handlers_attached = True

        # Delegate all heavy init work
        init_camera_widget(self)

    # Lifecycle entry points used by MainWindow
    def start(self) -> None:
        debug_ptz(
            f"{getattr(self.cam_cfg, 'name', '')}: start onvif={bool(getattr(self.cam_cfg, 'is_onvif', False))} "
            f"url={self._redact_url_for_display(self.cam_cfg.effective_url())}"
        )
        self._capture.start()
        self._detector.start()
        self._frame_timer.start()
        self._publish_mqtt_snapshot()
        self._start_ptz_detection()

    def stop(self) -> None:
        debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: stop")
        try:
            if hasattr(self, "_ptz_repeat_timer"):
                self._ptz_repeat_timer.stop()
        except Exception:
            pass
        self._ptz_stop()
        self._stop_ptz_worker()
        self._frame_timer.stop()
        self._capture.stop()
        # Graceful stop for the detector thread, but never hang the app on exit.
        try:
            self._detector.stop(wait_ms=2000)
            if getattr(self._detector, "isRunning", lambda: False)():
                print(
                    f"[CameraWidget:{getattr(self.cam_cfg, 'name', '')}] detector thread still running; terminating."
                )
                try:
                    self._detector.terminate()
                    self._detector.wait(1500)
                except Exception:
                    pass
        except Exception:
            pass
        self._recorder.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self.stop()
        event.accept()

    # ------------------------------------------------------------------ #
    # Info dialog
    # ------------------------------------------------------------------ #

    def _extract_url_credentials(self) -> tuple[str | None, str | None]:
        user = getattr(self.cam_cfg, "user", None) or None
        pwd = getattr(self.cam_cfg, "password", None) or None
        if user and pwd:
            return user, pwd
        parsed = urlparse(self.cam_cfg.effective_url())
        if not user:
            user = parsed.username or None
        if not pwd:
            pwd = parsed.password or None
        return user, pwd

    @staticmethod
    def _redact_url_for_display(url: str) -> str:
        """
        Strip embedded credentials and redact sensitive query params (e.g., token)
        for display in dialogs.
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return url
        if not parsed.scheme or not parsed.netloc:
            return url

        host = parsed.hostname or ""
        netloc = host
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"

        query = parsed.query or ""
        if query:
            qs = parse_qs(query, keep_blank_values=True)
            redacted: dict[str, list[str]] = {}
            for k, vals in qs.items():
                if k.lower() in {"token", "password", "pwd", "pass"}:
                    redacted[k] = ["***" for _ in vals]
                else:
                    redacted[k] = vals
            query = urlencode(redacted, doseq=True)

        out = f"{parsed.scheme}://{netloc}{parsed.path or ''}"
        if query:
            out = f"{out}?{query}"
        return out

    def _looks_like_esp32_cam_stream(self) -> bool:
        """
        Heuristic: ESP32-CAM default stream is http://<host>:81/stream.
        """
        parsed = urlparse(self.cam_cfg.effective_url())
        if parsed.scheme not in ("http", "https"):
            return False
        if parsed.port != 81:
            return False
        return (parsed.path or "").rstrip("/") == "/stream"

    def _api_status_url(self) -> str | None:
        """
        Build the API status URL for this camera based on its stream URL.
        Defaults to http://<host>:80/api/status.
        """
        if not self._looks_like_esp32_cam_stream():
            return None
        parsed = urlparse(self.cam_cfg.effective_url())
        host = parsed.hostname
        if not host:
            return None
        params: dict[str, str] = {}
        token = getattr(self.cam_cfg, "token", None) or None
        if not token:
            qs = parse_qs(parsed.query or "")
            token_vals = qs.get("token") or []
            token = token_vals[0] if token_vals else None
        if token:
            params["token"] = token
        url = f"http://{host}:80/api/status"
        if params:
            url = f"{url}?{urlencode(params)}"
        return url

    def _show_info(self) -> None:
        if bool(getattr(self.cam_cfg, "is_onvif", False)):
            self._show_info_onvif()
            return
        url = self._api_status_url()
        if url:
            self._show_info_esp32(url)
            return
        self._show_info_generic()

    def _show_info_generic(self) -> None:
        parsed = urlparse(self.cam_cfg.effective_url())
        user, pwd = self._extract_url_credentials()
        alt = getattr(self.cam_cfg, "alt_streams", None) or []
        typ = "ONVIF" if bool(getattr(self.cam_cfg, "is_onvif", False)) else "Stream"
        text = (
            f"Name: {self.cam_cfg.name}\n"
            f"Type: {typ}\n"
            f"URL: {self._redact_url_for_display(self.cam_cfg.effective_url())}\n"
            f"Host: {parsed.hostname or 'n/a'}\n"
            f"Port: {parsed.port or 'n/a'}\n"
            f"User: {user or '<none>'}\n"
            f"Password: {'<set>' if pwd else '<none>'}\n"
            f"Alt streams: {len(alt)}"
        )
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle(f"Camera Info - {self.cam_cfg.name}")
        dlg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        dlg.setText(text)
        dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dlg.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        dlg.exec()

    def _show_info_esp32(self, url: str) -> None:
        try:
            auth = None
            user, pwd = self._extract_url_credentials()
            if user and pwd:
                auth = requests.auth.HTTPBasicAuth(user, pwd)
            resp = requests.get(url, auth=auth, timeout=3)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Camera Info",
                f"Failed to fetch camera info from {url}:\n{e}",
            )
            return

        ip = data.get("ip") or urlparse(url).hostname or "n/a"
        fs_code = data.get("framesize")
        fs_name, fs_w, fs_h = FRAME_SIZES.get(fs_code, (str(fs_code), None, None))
        ptz = data.get("ptz") or {}
        pan = ptz.get("pan", "n/a")
        tilt = ptz.get("tilt", "n/a")

        text = (
            f"IP: {ip}\n"
            f"Framesize: {fs_name} (code {fs_code})\n"
            f"Pixels: {fs_w if fs_w else 'n/a'} x {fs_h if fs_h else 'n/a'}\n"
            f"PTZ: pan={pan}, tilt={tilt}"
        )

        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle(f"Camera Info - {self.cam_cfg.name}")
        dlg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        dlg.setText(text)
        dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dlg.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        dlg.exec()

    def _show_info_onvif(self) -> None:
        from onvif import OnvifDiscoveryResult, discover_onvif
        from onvif.enrichment import enrich_onvif_device
        from UI.onvif_dialog_format import render_onvif_details

        parsed = urlparse(self.cam_cfg.effective_url())
        host = parsed.hostname
        if not host:
            QtWidgets.QMessageBox.warning(
                self, "Camera Info", "Cannot determine host for this camera."
            )
            return

        user, pwd = self._extract_url_credentials()

        res: OnvifDiscoveryResult | None = None
        info: dict | None = None
        discovery_error: str | None = None
        try:
            hits = discover_onvif(timeout=1.5, retries=1)
            res = next((h for h in hits if h.ip == host or h.host == host), None)
        except Exception as e:
            discovery_error = str(e)

        if res is None:
            # Fallback: common device-service path if WS-Discovery is blocked.
            res = OnvifDiscoveryResult(
                xaddr=f"http://{host}/onvif/device_service",
                epr=None,
                scopes=[],
                ip=host,
            )

        try:
            info = enrich_onvif_device(res, user, pwd)
        except Exception as e:
            info = {
                "ip": host,
                "xaddr": getattr(res, "xaddr", None),
                "media_xaddr": None,
                "name": getattr(self.cam_cfg, "name", host),
                "model": None,
                "firmware": None,
                "profiles": [],
                "stream_uri": None,
                "auth_required": False,
                "errors": [f"onvif: {e}"],
                "fallback_urls": [],
            }

        text = (
            f"Configured stream URL: {self._redact_url_for_display(self.cam_cfg.effective_url())}\n\n"
            f"{render_onvif_details(info, show_errors=False, show_fallbacks=False)}"
        )

        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle(f"Camera Info - {self.cam_cfg.name}")
        dlg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        dlg.setText(text)
        dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dlg.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        dlg.exec()

    def _sync_flash_from_camera(self) -> None:
        """
        Query camera /api/status for flash state and align local mode/level so we
        don't override the hardware state on startup.
        """
        url = self._api_status_url()
        if not url:
            return
        try:
            auth = None
            if self.cam_cfg.user and self.cam_cfg.password:
                auth = requests.auth.HTTPBasicAuth(self.cam_cfg.user, self.cam_cfg.password)
            resp = requests.get(url, auth=auth, timeout=2.0)
            resp.raise_for_status()
            data = resp.json()
            flash_on = bool(data.get("flash", False))
            level = int(data.get("flash_level", 0) or 0)
            level = max(0, min(255, level))
            self._flash_level = level
            self.cam_cfg.flash_level = level
            self._flash_mode = "on" if (flash_on and level > 0) else "off"
            self.cam_cfg.flash_mode = self._flash_mode
        except Exception:
            # Best-effort; keep existing config if camera unreachable.
            return

    # ------------------------------------------------------------------ #
    # PTZ (ONVIF)
    # ------------------------------------------------------------------ #

    def _start_ptz_detection(self) -> None:
        if not bool(getattr(self.cam_cfg, "is_onvif", False)):
            debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ detection skipped (not ONVIF)")
            return
        if bool(getattr(self, "_ptz_detection_started", False)):
            debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ detection already started")
            return
        self._ptz_detection_started = True

        debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ detection thread starting")
        t = threading.Thread(target=self._ptz_detect_worker, daemon=True)
        self._ptz_detect_thread = t
        t.start()

    @staticmethod
    def _guess_onvif_profile_token(stream_url: str) -> str | None:
        m = re.search(r"/Streaming/Channels/(\d+)", stream_url or "")
        if m:
            return m.group(1)
        return None

    def _ptz_detect_worker(self) -> None:
        parsed = urlparse(self.cam_cfg.effective_url())
        host = parsed.hostname
        if not host:
            debug_ptz(
                f"{getattr(self.cam_cfg, 'name', '')}: PTZ detect failed (no host) "
                f"url={self._redact_url_for_display(self.cam_cfg.effective_url())}"
            )
            return

        user, pwd = self._extract_url_credentials()
        debug_ptz(
            f"{getattr(self.cam_cfg, 'name', '')}: PTZ detect begin host={host} "
            f"user={'<set>' if user else '<none>'} pwd={'<set>' if pwd else '<none>'}"
        )

        device_xaddr = f"http://{host}/onvif/device_service"
        try:
            from onvif import discover_onvif

            hits = discover_onvif(timeout=1.0, retries=1)
            debug_ptz(
                f"{getattr(self.cam_cfg, 'name', '')}: WS-Discovery hits={len(hits)}"
            )
            for h in hits:
                if h.ip == host or h.host == host:
                    device_xaddr = h.xaddr
                    break
        except Exception:
            debug_ptz(
                f"{getattr(self.cam_cfg, 'name', '')}: WS-Discovery failed; using fallback xaddr={device_xaddr}"
            )
            pass
        debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ device_xaddr={device_xaddr}")

        ptz_xaddr: str | None = None
        profile_token: str | None = None
        profile_tokens: list[str] = []

        try:
            from onvif import OnvifClient

            cli = OnvifClient(device_xaddr, username=user, password=pwd)
            caps = cli.get_capabilities()
            debug_ptz(
                f"{getattr(self.cam_cfg, 'name', '')}: capabilities media_xaddr={getattr(caps, 'media_xaddr', None)} "
                f"ptz_xaddr={getattr(caps, 'ptz_xaddr', None)}"
            )
            ptz_xaddr = caps.ptz_xaddr
            if not ptz_xaddr:
                debug_ptz(
                    f"{getattr(self.cam_cfg, 'name', '')}: PTZ unavailable (no ptz_xaddr)"
                )
                return

            guess = self._guess_onvif_profile_token(self.cam_cfg.effective_url())
            try:
                profiles = cli.get_profiles(media_xaddr=caps.media_xaddr or device_xaddr)
            except Exception:
                profiles = []
            try:
                prof_tokens = [getattr(p, "token", None) for p in profiles]
                prof_names = [getattr(p, "name", None) for p in profiles]
                debug_ptz(
                    f"{getattr(self.cam_cfg, 'name', '')}: profiles count={len(profiles)} guess={guess} "
                    f"tokens={prof_tokens[:6]} names={prof_names[:3]}"
                )
            except Exception:
                pass

            profile_tokens = [p.token for p in profiles if getattr(p, "token", None)]

            if guess and any(getattr(p, "token", None) == guess for p in profiles):
                profile_token = guess
            elif profiles:
                # Prefer a non-audio looking profile name if possible.
                for p in profiles:
                    name = (getattr(p, "name", "") or "").lower()
                    if "audio" in name:
                        continue
                    tok = getattr(p, "token", None)
                    if tok:
                        profile_token = tok
                        break
                if not profile_token:
                    profile_token = getattr(profiles[0], "token", None)
            else:
                profile_token = guess
        except Exception as e:
            # Auth or connectivity failure: don't show PTZ UI.
            debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ detect failed: {e}")
            try:
                from onvif import OnvifAuthError

                if isinstance(e, OnvifAuthError):
                    return
            except Exception:
                pass
            return

        if not ptz_xaddr or not profile_token:
            debug_ptz(
                f"{getattr(self.cam_cfg, 'name', '')}: PTZ unavailable (ptz_xaddr={bool(ptz_xaddr)} "
                f"profile_token={profile_token})"
            )
            return

        has_zoom = False
        supports_presets = False
        try:
            has_zoom = bool(cli.ptz_supports_zoom(profile_token, ptz_xaddr=ptz_xaddr))
        except Exception as e:
            debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ zoom probe failed: {e}")
        try:
            presets = cli.ptz_get_presets(profile_token, ptz_xaddr=ptz_xaddr)
            supports_presets = bool(presets)
            if presets:
                debug_ptz(
                    f"{getattr(self.cam_cfg, 'name', '')}: PTZ presets supported (count={len(presets)})"
                )
            else:
                debug_ptz(
                    f"{getattr(self.cam_cfg, 'name', '')}: PTZ presets supported but none configured"
                )
        except Exception as e:
            debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ presets unavailable: {e}")

        self._ptz_device_xaddr = device_xaddr
        self._ptz_xaddr = ptz_xaddr
        self._ptz_profile_token = profile_token
        self._ptz_profile_tokens = profile_tokens
        self._ptz_detected_has_zoom = bool(has_zoom)
        self._ptz_detected_has_presets = bool(supports_presets)
        self._ptz_has_zoom = bool(has_zoom) and (not bool(getattr(self.cam_cfg, "ptz_disable_zoom", False)))
        self._ptz_supports_presets = bool(supports_presets) and (
            not bool(getattr(self.cam_cfg, "ptz_disable_presets", False))
        )
        self._ptz_available = True
        self._ptz_move_mode = "continuous"
        debug_ptz(
            f"{getattr(self.cam_cfg, 'name', '')}: PTZ enabled token={profile_token} "
            f"zoom={bool(getattr(self, '_ptz_has_zoom', False))} "
            f"presets={bool(getattr(self, '_ptz_supports_presets', False))} "
            f"(disabled_zoom={bool(getattr(self.cam_cfg, 'ptz_disable_zoom', False))} "
            f"disabled_presets={bool(getattr(self.cam_cfg, 'ptz_disable_presets', False))}) "
            f"ptz_xaddr={ptz_xaddr}"
        )
        # Trigger an overlay redraw so the PTZ control appears.
        self._overlay_cache_dirty = True

    def _ensure_ptz_worker(self) -> None:
        if not bool(getattr(self, "_ptz_available", False)):
            return
        t = getattr(self, "_ptz_worker_thread", None)
        if t is not None and getattr(t, "is_alive", lambda: False)():
            return

        self._ptz_worker_stop = threading.Event()
        self._ptz_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=2)

        debug_ptz(
            f"{getattr(self.cam_cfg, 'name', '')}: PTZ worker starting "
            f"ptz_xaddr={getattr(self, '_ptz_xaddr', None)} token={getattr(self, '_ptz_profile_token', None)}"
        )
        t = threading.Thread(target=self._ptz_worker_loop, daemon=True)
        self._ptz_worker_thread = t
        t.start()

    def _stop_ptz_worker(self) -> None:
        debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ worker stopping")
        try:
            getattr(self, "_ptz_worker_stop", None) and self._ptz_worker_stop.set()
        except Exception:
            pass
        t = getattr(self, "_ptz_worker_thread", None)
        if t is not None:
            try:
                t.join(timeout=0.5)
            except Exception:
                pass

    def _ptz_enqueue_move(self, pan: float, tilt: float, zoom: float) -> None:
        self._ensure_ptz_worker()
        q = getattr(self, "_ptz_queue", None)
        if q is None:
            return
        cmd = ("move", float(pan), float(tilt), float(zoom))
        try:
            q.put_nowait(cmd)
        except queue.Full:
            debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ queue full; dropping stale move")
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(cmd)
            except Exception:
                pass

    def _ptz_enqueue_stop(self) -> None:
        self._ensure_ptz_worker()
        q = getattr(self, "_ptz_queue", None)
        if q is None:
            return
        debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ enqueue stop")
        try:
            while True:
                _ = q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(("stop",))
        except Exception:
            pass

    def _ptz_enqueue_home(self) -> None:
        self._ensure_ptz_worker()
        q = getattr(self, "_ptz_queue", None)
        if q is None:
            return
        debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ enqueue home")
        try:
            while True:
                _ = q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(("stop",))
            q.put_nowait(("home",))
        except Exception:
            pass

    def _ptz_enqueue_preset(self, preset_token: str) -> None:
        self._ensure_ptz_worker()
        q = getattr(self, "_ptz_queue", None)
        if q is None:
            return
        preset_token = str(preset_token or "").strip()
        if not preset_token:
            return
        debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ enqueue preset {preset_token}")
        try:
            while True:
                _ = q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(("stop",))
            q.put_nowait(("preset", preset_token))
        except Exception:
            pass

    def _ptz_worker_loop(self) -> None:
        device_xaddr = getattr(self, "_ptz_device_xaddr", None)
        ptz_xaddr = getattr(self, "_ptz_xaddr", None)
        token = getattr(self, "_ptz_profile_token", None)
        if not device_xaddr or not ptz_xaddr or not token:
            return

        user, pwd = self._extract_url_credentials()
        debug_ptz(
            f"{getattr(self.cam_cfg, 'name', '')}: PTZ worker ready device_xaddr={device_xaddr} "
            f"ptz_xaddr={ptz_xaddr} token={token} user={'<set>' if user else '<none>'} pwd={'<set>' if pwd else '<none>'}"
        )
        try:
            from onvif import OnvifClient, OnvifHttpError

            cli = OnvifClient(device_xaddr, username=user, password=pwd)
        except Exception:
            debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ worker failed to init OnvifClient")
            return

        stop_evt = getattr(self, "_ptz_worker_stop", None)
        q = getattr(self, "_ptz_queue", None)
        if stop_evt is None or q is None:
            return

        last_move_logged: tuple[float, float, float] | None = None
        last_move_log_ts = 0.0
        last_err: str | None = None
        last_err_ts = 0.0

        while not stop_evt.is_set():
            try:
                cmd = q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if not isinstance(cmd, tuple) or not cmd:
                    continue
                if cmd[0] == "stop":
                    debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ send stop")
                    cli.ptz_stop(
                        token,
                        ptz_xaddr=ptz_xaddr,
                        zoom=bool(getattr(self, "_ptz_has_zoom", False)),
                    )
                elif cmd[0] == "home":
                    debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ send home")
                    cli.ptz_goto_home(token, ptz_xaddr=ptz_xaddr)
                elif cmd[0] == "preset" and len(cmd) >= 2:
                    preset_token = str(cmd[1] or "").strip()
                    if not preset_token:
                        continue
                    debug_ptz(
                        f"{getattr(self.cam_cfg, 'name', '')}: PTZ send preset {preset_token}"
                    )
                    cli.ptz_goto_preset(token, preset_token, ptz_xaddr=ptz_xaddr)
                elif cmd[0] == "move" and len(cmd) >= 4:
                    pan, tilt, zoom = float(cmd[1]), float(cmd[2]), float(cmd[3])
                    now = time.time()
                    move = (pan, tilt, zoom)
                    if (
                        last_move_logged is None
                        or move != last_move_logged
                        or (now - last_move_log_ts) > 2.0
                    ):
                        debug_ptz(
                            f"{getattr(self.cam_cfg, 'name', '')}: PTZ send move pan={pan:.2f} tilt={tilt:.2f} zoom={zoom:.2f}"
                        )
                        last_move_logged = move
                        last_move_log_ts = now

                    mode = (getattr(self, "_ptz_move_mode", "continuous") or "continuous").lower()
                    if mode == "relative":
                        step = 0.08
                        cli.ptz_relative_move(
                            token,
                            ptz_xaddr=ptz_xaddr,
                            pan=pan * step,
                            tilt=tilt * step,
                            zoom=zoom * step,
                        )
                    else:
                        try:
                            cli.ptz_continuous_move(
                                token,
                                ptz_xaddr=ptz_xaddr,
                                pan=pan,
                                tilt=tilt,
                                zoom=zoom,
                                timeout_s=0.6,
                            )
                        except OnvifHttpError as e:
                            if e.status_code == 400:
                                debug_ptz(
                                    f"{getattr(self.cam_cfg, 'name', '')}: PTZ ContinuousMove rejected; "
                                    f"switching to RelativeMove ({e.detail})"
                                )
                                self._ptz_move_mode = "relative"
                                step = 0.08
                                cli.ptz_relative_move(
                                    token,
                                    ptz_xaddr=ptz_xaddr,
                                    pan=pan * step,
                                    tilt=tilt * step,
                                    zoom=zoom * step,
                                )
                            else:
                                raise
            except Exception as e:
                now = time.time()
                msg = str(e)
                if msg != last_err or (now - last_err_ts) > 2.0:
                    debug_ptz(
                        f"{getattr(self.cam_cfg, 'name', '')}: PTZ command failed ({cmd[0]}): {e}"
                    )
                    last_err = msg
                    last_err_ts = now
                continue

    def _ptz_repeat_tick(self) -> None:
        pan, tilt, zoom = getattr(self, "_ptz_active_vel", (0.0, 0.0, 0.0))
        if pan == 0.0 and tilt == 0.0 and zoom == 0.0:
            return
        self._ptz_enqueue_move(pan, tilt, zoom)

    def _ptz_map_pan_tilt_for_view(self, pan: float, tilt: float) -> tuple[float, float]:
        """
        Map PTZ directions to match the displayed stream orientation.

        The video frame is rotated/flipped for display via `_apply_orientation()`. If a
        camera is mounted upside-down or mirrored, the PTZ axes can feel reversed. This
        helper compensates so "left/right/up/down" controls match what the user sees.

        Input/Output use screen-style coordinates:
          - `pan` > 0 means "move right"
          - `tilt` > 0 means "move up"
        """
        try:
            rot = int(getattr(self.cam_cfg, "rotation_deg", 0) or 0) % 360
        except Exception:
            rot = 0
        flip_h = bool(getattr(self.cam_cfg, "flip_horizontal", False))
        flip_v = bool(getattr(self.cam_cfg, "flip_vertical", False))

        # Work in image-vector space (x right, y down) to match cv2 rotation semantics.
        x = float(pan)
        y = float(-tilt)

        # Undo flips (display applies flips after rotation).
        if flip_v:
            y = -y
        if flip_h:
            x = -x

        # Undo rotation (display rotates the image).
        if rot == 90:
            # Inverse of ROTATE_90_CLOCKWISE is 90 CCW.
            x, y = y, -x
        elif rot == 180:
            x, y = -x, -y
        elif rot == 270:
            # Inverse of ROTATE_90_COUNTERCLOCKWISE is 90 CW.
            x, y = -y, x

        return x, -y

    def _ptz_set_velocity(self, pan: float, tilt: float, zoom: float) -> None:
        if not bool(getattr(self, "_ptz_available", False)):
            return
        pan_in, tilt_in, zoom_in = float(pan), float(tilt), float(zoom)

        # Zoom: clamp to zero if the camera doesn't expose a zoom axis.
        if not bool(getattr(self, "_ptz_has_zoom", False)):
            zoom_in = 0.0

        pan_m, tilt_m = self._ptz_map_pan_tilt_for_view(pan_in, tilt_in)

        pan = float(max(-1.0, min(1.0, pan_m)))
        tilt = float(max(-1.0, min(1.0, tilt_m)))
        zoom = float(max(-1.0, min(1.0, zoom_in)))

        prev = getattr(self, "_ptz_active_vel", (0.0, 0.0, 0.0))
        if (pan, tilt, zoom) != prev:
            if (pan_in, tilt_in) != (pan, tilt) and (pan_in != 0.0 or tilt_in != 0.0):
                debug_ptz(
                    f"{getattr(self.cam_cfg, 'name', '')}: PTZ mapped view->cam "
                    f"({pan_in:.2f}, {tilt_in:.2f}) -> ({pan:.2f}, {tilt:.2f}) "
                    f"rot={int(getattr(self.cam_cfg, 'rotation_deg', 0) or 0)} "
                    f"flip_h={bool(getattr(self.cam_cfg, 'flip_horizontal', False))} "
                    f"flip_v={bool(getattr(self.cam_cfg, 'flip_vertical', False))}"
                )
            debug_ptz(
                f"{getattr(self.cam_cfg, 'name', '')}: PTZ velocity {prev} -> ({pan:.2f}, {tilt:.2f}, {zoom:.2f})"
            )
        self._ptz_active_vel = (pan, tilt, zoom)
        if pan == 0.0 and tilt == 0.0 and zoom == 0.0:
            try:
                self._ptz_repeat_timer.stop()
            except Exception:
                pass
            self._ptz_enqueue_stop()
            return

        try:
            if not self._ptz_repeat_timer.isActive():
                self._ptz_repeat_timer.start()
        except Exception:
            pass
        self._ptz_enqueue_move(pan, tilt, zoom)

    def _ptz_stop(self) -> None:
        debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ stop")
        try:
            self._ptz_mouse_action = None
        except Exception:
            pass
        try:
            self._ptz_keys_down = set()
        except Exception:
            pass
        self._ptz_set_velocity(0.0, 0.0, 0.0)

    def _ptz_confirm_home(self) -> bool:
        if bool(getattr(self.cam_cfg, "ptz_home_confirmed", False)):
            return True

        if bool(getattr(self, "_ptz_home_warned", False)):
            return True

        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("PTZ Home")
        dlg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dlg.setText("Go to PTZ home position?")
        dlg.setInformativeText(
            "Some cheaper PTZ cameras may run motors against endstops for several seconds when going home.\n\n"
            "Continue?"
        )
        dlg.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        dlg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)

        cb = QtWidgets.QCheckBox("Don't show this warning again for this camera")
        try:
            dlg.setCheckBox(cb)
        except Exception:
            pass

        if dlg.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
            return False

        self._ptz_home_warned = True
        if bool(getattr(cb, "isChecked", lambda: False)()):
            try:
                self.cam_cfg.ptz_home_confirmed = True
                save_settings(self.app_cfg)
            except Exception:
                pass
        return True

    def _ptz_goto_home(self) -> None:
        if not bool(getattr(self, "_ptz_available", False)):
            return
        if not self._ptz_confirm_home():
            return
        self._ptz_stop()
        self._ptz_enqueue_home()

    def _ptz_open_presets_dialog(self) -> None:
        if not bool(getattr(self, "_ptz_available", False)):
            return
        if bool(getattr(self, "_ptz_presets_loading", False)):
            return
        device_xaddr = getattr(self, "_ptz_device_xaddr", None)
        ptz_xaddr = getattr(self, "_ptz_xaddr", None)
        token = getattr(self, "_ptz_profile_token", None)
        if not device_xaddr or not ptz_xaddr or not token:
            return

        self._ptz_presets_loading = True
        debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ presets load start")

        user, pwd = self._extract_url_credentials()

        def worker() -> None:
            presets = []
            err: str | None = None
            try:
                from onvif import OnvifClient

                cli = OnvifClient(device_xaddr, username=user, password=pwd)
                presets = cli.ptz_get_presets(token, ptz_xaddr=ptz_xaddr) or []
            except Exception as e:
                err = str(e)

            def show() -> None:
                self._ptz_presets_loading = False
                if err:
                    QtWidgets.QMessageBox.warning(
                        self, "PTZ Presets", f"Failed to load presets:\n{err}"
                    )
                    return
                if not presets:
                    QtWidgets.QMessageBox.information(
                        self, "PTZ Presets", "No presets reported by this camera."
                    )
                    return

                items: list[str] = []
                mapping: dict[str, str] = {}
                used: dict[str, int] = {}
                for p in presets:
                    tok = (getattr(p, "token", None) or "").strip()
                    if not tok:
                        continue
                    name = (getattr(p, "name", None) or "").strip()
                    label = f"{name} ({tok})" if (name and name != tok) else tok
                    n = used.get(label, 0) + 1
                    used[label] = n
                    if n > 1:
                        label = f"{label} [{n}]"
                    items.append(label)
                    mapping[label] = tok

                if not items:
                    QtWidgets.QMessageBox.information(
                        self, "PTZ Presets", "No usable presets reported by this camera."
                    )
                    return

                choice, ok = QtWidgets.QInputDialog.getItem(
                    self,
                    f"PTZ Presets - {self.cam_cfg.name}",
                    "Go to preset:",
                    items,
                    0,
                    False,
                )
                if ok and choice:
                    preset_token = mapping.get(choice)
                    if preset_token:
                        self._ptz_stop()
                        self._ptz_enqueue_preset(preset_token)

            try:
                QtCore.QTimer.singleShot(0, self, show)
            except Exception:
                self._ptz_presets_loading = False

        threading.Thread(target=worker, daemon=True).start()

    def _ptz_velocity_from_keys(self) -> tuple[float, float, float]:
        keys = getattr(self, "_ptz_keys_down", set()) or set()
        speed = 0.5
        pan = 0.0
        tilt = 0.0
        zoom = 0.0
        if QtCore.Qt.Key.Key_Left in keys:
            pan -= speed
        if QtCore.Qt.Key.Key_Right in keys:
            pan += speed
        if QtCore.Qt.Key.Key_Up in keys:
            tilt += speed
        if QtCore.Qt.Key.Key_Down in keys:
            tilt -= speed
        if bool(getattr(self, "_ptz_has_zoom", False)):
            if QtCore.Qt.Key.Key_PageUp in keys:
                zoom += speed
            if QtCore.Qt.Key.Key_PageDown in keys:
                zoom -= speed
        return pan, tilt, zoom

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        # Key events: delivered to QGraphicsView widget.
        if obj is getattr(self, "view", None):
            if event.type() == QtCore.QEvent.Type.FocusOut:
                self._ptz_stop()
                return False
            if event.type() in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease):
                if not bool(getattr(self, "_ptz_available", False)):
                    return False
                key = getattr(event, "key", lambda: None)()
                allowed = {
                    QtCore.Qt.Key.Key_Left,
                    QtCore.Qt.Key.Key_Right,
                    QtCore.Qt.Key.Key_Up,
                    QtCore.Qt.Key.Key_Down,
                }
                if bool(getattr(self, "_ptz_has_zoom", False)):
                    allowed.add(QtCore.Qt.Key.Key_PageUp)
                    allowed.add(QtCore.Qt.Key.Key_PageDown)
                if key not in allowed:
                    return False
                if getattr(event, "isAutoRepeat", lambda: False)():
                    return True
                if event.type() == QtCore.QEvent.Type.KeyPress:
                    debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ key down {key}")
                    self._ptz_keys_down.add(key)
                else:
                    debug_ptz(f"{getattr(self.cam_cfg, 'name', '')}: PTZ key up {key}")
                    try:
                        self._ptz_keys_down.discard(key)
                    except Exception:
                        pass
                if self._ptz_mouse_action is None:
                    pan, tilt, zoom = self._ptz_velocity_from_keys()
                    self._ptz_set_velocity(pan, tilt, zoom)
                return True

        # Mouse events: delivered to the viewport widget in QGraphicsView.
        try:
            viewport = self.view.viewport()  # type: ignore[attr-defined]
        except Exception:
            viewport = None
        if viewport is not None and obj is viewport:
            if event.type() in (
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.MouseButtonRelease,
            ):
                if not bool(getattr(self, "_ptz_available", False)):
                    return False
                if getattr(event, "button", lambda: None)() != QtCore.Qt.MouseButton.LeftButton:
                    return False
                regions = getattr(self, "_ptz_hit_regions", None) or {}
                if not isinstance(regions, dict) or not regions:
                    debug_ptz(
                        f"{getattr(self.cam_cfg, 'name', '')}: PTZ mouse event but no hit regions yet"
                    )
                    return False

                try:
                    try:
                        pos = event.position().toPoint()
                    except Exception:
                        pos = event.pos()
                    pt = self.view.mapToScene(pos)
                except Exception:
                    return False

                if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                    for name, rect in regions.items():
                        try:
                            if rect.contains(pt):
                                self.view.setFocus()
                                name = str(name)
                                if name == "home":
                                    debug_ptz(
                                        f"{getattr(self.cam_cfg, 'name', '')}: PTZ mouse press action=home "
                                        f"pt=({pt.x():.0f},{pt.y():.0f})"
                                    )
                                    self._ptz_goto_home()
                                    return True
                                if name == "presets":
                                    debug_ptz(
                                        f"{getattr(self.cam_cfg, 'name', '')}: PTZ mouse press action=presets "
                                        f"pt=({pt.x():.0f},{pt.y():.0f})"
                                    )
                                    self._ptz_open_presets_dialog()
                                    return True

                                self._ptz_mouse_action = name
                                debug_ptz(
                                    f"{getattr(self.cam_cfg, 'name', '')}: PTZ mouse press action={name} "
                                    f"pt=({pt.x():.0f},{pt.y():.0f})"
                                )
                                speed = 0.5
                                if name == "up":
                                    self._ptz_set_velocity(0.0, speed, 0.0)
                                elif name == "down":
                                    self._ptz_set_velocity(0.0, -speed, 0.0)
                                elif name == "left":
                                    self._ptz_set_velocity(-speed, 0.0, 0.0)
                                elif name == "right":
                                    self._ptz_set_velocity(speed, 0.0, 0.0)
                                elif name == "zoom_in":
                                    self._ptz_set_velocity(0.0, 0.0, speed)
                                elif name == "zoom_out":
                                    self._ptz_set_velocity(0.0, 0.0, -speed)
                                return True
                        except Exception:
                            continue
                    return False

                # Release
                if self._ptz_mouse_action is not None:
                    debug_ptz(
                        f"{getattr(self.cam_cfg, 'name', '')}: PTZ mouse release action={self._ptz_mouse_action}"
                    )
                    self._ptz_mouse_action = None
                    pan, tilt, zoom = self._ptz_velocity_from_keys()
                    self._ptz_set_velocity(pan, tilt, zoom)
                    return True

        return super().eventFilter(obj, event)

    def _open_camera_settings(self) -> None:
        dlg = CameraSettingsDialog(self.cam_cfg, self.app_cfg, self, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            dlg.apply()
            save_settings(self.app_cfg)
            # make sure any motion toggle/sensitivity changes apply immediately
            self._overlay_cache_dirty = True

    def _mqtt_topic_base(self) -> str:
        return getattr(self, "_mqtt_topic", None) or (self.cam_cfg.name or "cam").replace(" ", "_")

    def _publish_mqtt_cleared_state(self) -> None:
        if not getattr(self, "_mqtt", None):
            return
        if not getattr(self._mqtt, "connected", False):
            return
        topic_base = self._mqtt_topic_base()
        try:
            self._mqtt.publish(f"{topic_base}/presence/person", "OFF", retain=True)
            self._mqtt.publish(f"{topic_base}/presence/pet", "OFF", retain=True)
            self._mqtt.publish(f"{topic_base}/counts/person", "0", retain=True)
            self._mqtt.publish(f"{topic_base}/counts/pet", "0", retain=True)
            self._mqtt.publish(f"{topic_base}/recognized", "", retain=True)

            pres = getattr(self, "_presence", None)
            for k in list(getattr(pres, "present", []) or []):
                if isinstance(k, str) and k.startswith("person:"):
                    label = k.split(":", 1)[1].strip()
                    if label:
                        self._mqtt.publish(
                            f"{topic_base}/presence/person/{label}", "OFF", retain=True
                        )
                elif k in ("dog", "cat"):
                    self._mqtt.publish(f"{topic_base}/presence/{k}", "OFF", retain=True)
        except Exception:
            pass

    def _publish_mqtt_snapshot(self) -> None:
        if not getattr(self, "_mqtt", None):
            return
        if not getattr(self._mqtt, "connected", False):
            return
        if not bool(getattr(self.cam_cfg, "mqtt_publish", True)):
            return

        # Publish current presence state based on the PresenceBus set.
        pres = getattr(self, "_presence", None)
        if pres is not None:
            try:
                pres._mqtt = self._mqtt  # ensure publish is enabled
                for k in list(getattr(pres, "present", []) or []):
                    pres._publish_state(k, True)
                pres._publish_aggregate_states()
            except Exception:
                pass

        # Publish counts/recognised based on last packet (fallback to zeros).
        pkt = getattr(self, "_last_pkt", None)
        if pkt is not None:
            try:
                self._publish_mqtt_state(pkt)
                return
            except Exception:
                pass

        topic_base = self._mqtt_topic_base()
        try:
            self._mqtt.publish(f"{topic_base}/counts/person", "0", retain=True)
            self._mqtt.publish(f"{topic_base}/counts/pet", "0", retain=True)
            self._mqtt.publish(f"{topic_base}/recognized", "", retain=True)
        except Exception:
            pass

    def _apply_mqtt_publish(self, enabled: bool) -> None:
        enabled = bool(enabled)
        prev = bool(getattr(self.cam_cfg, "mqtt_publish", True))
        if prev == enabled:
            return

        if not enabled:
            # Publish a final retained "cleared" state so HA doesn't stay stuck ON.
            self._publish_mqtt_cleared_state()

        self.cam_cfg.mqtt_publish = enabled
        if hasattr(self, "_presence"):
            try:
                self._presence._mqtt = self._mqtt if enabled else None
            except Exception:
                pass

        if enabled:
            self._publish_mqtt_snapshot()


# Keep module-level attachment too (harmless with the class guard above).
attach_video_handlers(CameraWidget)
attach_overlay_handlers(CameraWidget)
attach_view_handlers(CameraWidget)
