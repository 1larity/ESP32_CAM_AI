# camera/camera_widget.py
from __future__ import annotations

from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse
import requests

from PySide6 import QtCore, QtGui, QtWidgets

from settings import AppSettings, CameraSettings, save_settings
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
      - attach_view_handlers(CameraWidget)     → fit / lock helpers
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
        self._capture.start()
        self._detector.start()
        self._frame_timer.start()
        self._publish_mqtt_snapshot()

    def stop(self) -> None:
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
