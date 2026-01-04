"""
Simple PySide6 GUI for discovering ONVIF/WS-Discovery cameras on the local network.
"""

import base64
import contextlib
import io
import select
import socket
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urlparse, urlunparse
from urllib import request as urlrequest

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

try:
    # onvif-zeep provides ONVIFCamera
    from onvif import ONVIFCamera  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ONVIFCamera = None

try:
    import keyring  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    keyring = None

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


MULTICAST_ADDRESS = ("239.255.255.250", 3702)
PROBE_TIMEOUT = 4.0  # seconds per round
MAX_DATAGRAM = 65507  # safe UDP payload


@dataclass
class DiscoveredDevice:
    address: str
    xaddrs: str
    scopes: str
    hardware: str


class WSDiscoveryClient:
    """Minimal WS-Discovery probe helper for ONVIF devices."""

    def __init__(self, timeout: float = PROBE_TIMEOUT) -> None:
        self.timeout = timeout
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def probe(self, retries: int = 2) -> Iterable[DiscoveredDevice]:
        """Send discovery probe and yield responses."""
        message_id = f"uuid:{uuid.uuid4()}"
        probe = f"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>{message_id}</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04/discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>
"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 0)
        sock.bind(("", 0))
        sock.setblocking(False)

        try:
            payload = probe.encode("utf-8")
            for _ in range(retries):
                if self._stop.is_set():
                    break
                sock.sendto(payload, MULTICAST_ADDRESS)
                deadline = time.time() + self.timeout
                while time.time() < deadline and not self._stop.is_set():
                    remaining = max(0.0, deadline - time.time())
                    readable, _, _ = select.select([sock], [], [], remaining)
                    if not readable:
                        continue
                    try:
                        data, (addr, _) = sock.recvfrom(MAX_DATAGRAM)
                    except OSError:
                        continue
                    device = self._parse_response(data, addr)
                    if device:
                        yield device
        finally:
            sock.close()

    def _parse_response(self, payload: bytes, addr: str) -> Optional[DiscoveredDevice]:
        """Parse a ProbeMatch response into a device record."""
        try:
            root = ET.fromstring(payload.decode("utf-8", errors="ignore"))
        except ET.ParseError:
            return None

        namespaces = {
            "d": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
            "a": "http://schemas.xmlsoap.org/ws/2004/08/addressing",
        }
        scopes_text = []
        hardware = ""
        xaddrs = ""

        for match in root.findall(".//d:ProbeMatch", namespaces):
            scope_el = match.find("d:Scopes", namespaces)
            xaddr_el = match.find("d:XAddrs", namespaces)
            if scope_el is not None:
                scopes_text.append(scope_el.text or "")
                for scope in (scope_el.text or "").split():
                    if scope.startswith("onvif://www.onvif.org/hardware/"):
                        hardware = scope.split("/")[-1]
            if xaddr_el is not None:
                xaddrs = xaddr_el.text or ""

        scopes = " ".join(scopes_text).strip()
        if not xaddrs and not scopes:
            return None

        return DiscoveredDevice(
            address=addr,
            xaddrs=xaddrs,
            scopes=scopes,
            hardware=hardware,
        )


class DiscoveryWorker(QtCore.QThread):
    deviceFound = QtCore.Signal(DiscoveredDevice)
    scanFinished = QtCore.Signal()

    def __init__(self, timeout: float = PROBE_TIMEOUT, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.timeout = timeout
        self._client = WSDiscoveryClient(timeout=timeout)

    def run(self) -> None:
        for device in self._client.probe():
            self.deviceFound.emit(device)
        self.scanFinished.emit()

    def stop(self) -> None:
        self._client.stop()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


class OnvifSession:
    """Thin wrapper around ONVIF camera services (media + PTZ)."""

    def __init__(self, host: str, port: int, user: str, password: str) -> None:
        if ONVIFCamera is None:
            raise RuntimeError("onvif package not installed. Install with: python3 -m pip install onvif-zeep")
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.camera = ONVIFCamera(host, port, user, password)
        # onvif-zeep uses create_devicemgmt_service; keep fallback for variants.
        create_dev = getattr(self.camera, "create_devicemgmt_service", None) or getattr(
            self.camera, "create_device_service", None
        )
        if not create_dev:
            raise RuntimeError("ONVIFCamera missing device management service creator")
        self.device = create_dev()
        self.media = self.camera.create_media_service()
        self.ptz = self.camera.create_ptz_service()
        self.deviceio = None
        try:
            self.deviceio = self.camera.create_deviceio_service()
        except Exception:
            self.deviceio = None

        profiles = self.media.GetProfiles()
        if not profiles:
            raise RuntimeError("Camera returned no media profiles")
        self.profile = profiles[0]
        self.profile_token = self.profile.token
        self.stream_uri = self._fetch_stream_uri()
        self.capabilities = self._fetch_capabilities()
        self.services = self._fetch_services()
        self.relays = self._fetch_relays()
        self.imaging_settings = self._fetch_imaging_settings()
        self.imaging_options = self._fetch_imaging_options()
        self.vendor_probe_targets = self._build_vendor_probes()

    def _fetch_stream_uri(self) -> str:
        # Try TCP (interleaved) first to avoid UDP dropouts; fall back to RTSP default.
        for protocol in ("TCP", "RTSP"):
            request = {
                "StreamSetup": {"Stream": "RTP-Unicast", "Transport": {"Protocol": protocol}},
                "ProfileToken": self.profile_token,
            }
            try:
                uri = self.media.GetStreamUri(request)
                if uri and getattr(uri, "Uri", None):
                    return self._add_credentials(uri.Uri)
            except Exception as exc:
                _log(f"GetStreamUri with protocol {protocol} failed: {exc}")
                continue
        # Last resort: empty string
        return ""

    def _add_credentials(self, uri: str) -> str:
        """Insert username/password into RTSP URI if provided and not already present."""
        if not self.user:
            return uri
        parsed = urlparse(uri)
        if parsed.username:
            return uri  # camera already embedded credentials
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        netloc = f"{self.user}:{self.password}@{netloc}"
        return urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    def continuous_move(self, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0) -> None:
        if not self.ptz:
            return
        request = self.ptz.create_type("ContinuousMove")
        request.ProfileToken = self.profile_token
        request.Velocity = {}
        if pan or tilt:
            request.Velocity["PanTilt"] = {"x": pan, "y": tilt}
        if zoom:
            request.Velocity["Zoom"] = {"x": zoom}
        self.ptz.ContinuousMove(request)

    def stop(self) -> None:
        if not self.ptz:
            return
        try:
            self.ptz.Stop({"ProfileToken": self.profile_token})
        except Exception:
            # Some cameras reject Stop when not moving; ignore.
            pass

    def set_ir_mode(self, mode: str) -> None:
        """Set IR cut filter mode: 'ON', 'OFF', or 'AUTO'."""
        imaging = self.camera.create_imaging_service()
        token_obj = self.profile.VideoSourceConfiguration.SourceToken
        token = self._normalize_token(token_obj)
        settings = imaging.GetImagingSettings({"VideoSourceToken": token})
        settings.ImageStabilization = None  # reduce payload size
        settings.Extension = None
        settings.IrCutFilter = mode
        imaging.SetImagingSettings({"VideoSourceToken": token, "ImagingSettings": settings, "ForcePersistence": True})

    def _fetch_capabilities(self) -> dict:
        try:
            caps = self.device.GetCapabilities({"Category": "All"})
            return {
                "Analytics": bool(getattr(caps, "Analytics", None)),
                "Events": bool(getattr(caps, "Events", None)),
                "Imaging": bool(getattr(caps, "Imaging", None)),
                "Media": bool(getattr(caps, "Media", None)),
                "PTZ": bool(getattr(caps, "PTZ", None)),
                "DeviceIO": bool(getattr(caps, "DeviceIO", None)),
            }
        except Exception:
            return {}

    def _fetch_services(self) -> list:
        try:
            return self.device.GetServices({"IncludeCapability": False})
        except Exception:
            return []

    def _fetch_relays(self) -> list:
        if not self.deviceio:
            return []
        try:
            relays = self.deviceio.GetRelayOutputs()
            return relays
        except Exception:
            return []

    def set_relay_state(self, token: str, state: str) -> None:
        if not self.deviceio:
            raise RuntimeError("DeviceIO service not available")
        self.deviceio.SetRelayOutputState({"RelayOutputToken": token, "LogicalState": state})

    def _normalize_token(self, token_obj) -> str:
        # Some implementations return zeep.AnyType or collections; pick string value.
        if isinstance(token_obj, (list, tuple)) and token_obj:
            token_obj = token_obj[0]
        return str(getattr(token_obj, "_value_1", token_obj))

    def _fetch_imaging_settings(self):
        try:
            imaging = self.camera.create_imaging_service()
            token = self._normalize_token(self.profile.VideoSourceConfiguration.SourceToken)
            return imaging.GetImagingSettings({"VideoSourceToken": token})
        except Exception:
            return None

    def _fetch_imaging_options(self):
        try:
            imaging = self.camera.create_imaging_service()
            token = self._normalize_token(self.profile.VideoSourceConfiguration.SourceToken)
            return imaging.GetOptions({"VideoSourceToken": token})
        except Exception:
            return None

    def dump_wsdl(self) -> str:
        """Dump available WSDL operations for key services to help debugging."""
        out = io.StringIO()
        services = {
            "device": self.device,
            "media": self.media,
            "ptz": self.ptz,
        }
        try:
            services["imaging"] = self.camera.create_imaging_service()
        except Exception:
            pass
        try:
            if self.deviceio:
                services["deviceio"] = self.deviceio
        except Exception:
            pass
        for name, svc in services.items():
            wsdl = getattr(svc, "wsdl", None)
            if wsdl:
                out.write(f"== {name} operations ==\n")
                with contextlib.redirect_stdout(out):
                    try:
                        wsdl.dump()
                    except Exception as exc:  # pragma: no cover
                        out.write(f"(failed to dump {name}: {exc})\n")
            else:
                out.write(f"== {name}: no wsdl available ==\n")
        return out.getvalue()

    def _build_vendor_probes(self) -> list[tuple[str, str]]:
        """Return candidate vendor-specific HTTP endpoints for IR/light toggles (best-effort)."""
        host = self.host
        port = self.port or 80
        base = f"http://{host}:{port}"
        probes = []
        for mode in ["open", "close", "auto"]:
            probes.append((f"hi3510 setinfrared {mode}", f"{base}/web/cgi-bin/hi3510/param.cgi?cmd=setinfrared&-infraredstat={mode}"))
            probes.append((f"hi3510 setinfraredlightattr {mode}", f"{base}/web/cgi-bin/hi3510/param.cgi?cmd=setinfraredlightattr&-mode={mode}"))
            probes.append((f"hi3510 setswitch led_stat {mode}", f"{base}/web/cgi-bin/hi3510/param.cgi?cmd=setswitch&-led_stat={mode}"))
        return probes


class CredentialStore:
    """Store credentials in the OS keyring (if available)."""

    def __init__(self, service_name: str = "onvif_discover") -> None:
        self.service_name = service_name

    def load(self, key: str) -> tuple[str, str]:
        if keyring is None:
            return ("", "")
        username = keyring.get_password(self.service_name, f"{key}:user") or ""
        password = keyring.get_password(self.service_name, f"{key}:pass") or ""
        return (username, password)

    def save(self, key: str, username: str, password: str) -> None:
        if keyring is None:
            return
        keyring.set_password(self.service_name, f"{key}:user", username)
        keyring.set_password(self.service_name, f"{key}:pass", password)


class ConnectWorker(QtCore.QThread):
    connected = QtCore.Signal(OnvifSession)
    error = QtCore.Signal(str)

    def __init__(self, host: str, port: int, user: str, password: str, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    def run(self) -> None:
        try:
            session = OnvifSession(self.host, self.port, self.user, self.password)
            self.connected.emit(session)
        except Exception as exc:  # pragma: no cover - depends on network/camera
            self.error.emit(str(exc))


class OpenCVStreamWorker(QtCore.QThread):
    frameReady = QtCore.Signal(QtGui.QImage)
    error = QtCore.Signal(str)

    def __init__(self, url: str, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.url = url
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        if cv2 is None:
            self.error.emit("OpenCV not available")
            return
        urls_to_try = [self.url]
        if "tcp" not in self.url.lower():
            sep = "&" if "?" in self.url else "?"
            urls_to_try.append(f"{self.url}{sep}tcp")
        for candidate in urls_to_try:
            cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
            # Try forcing TCP transport where supported.
            if hasattr(cv2, "CAP_PROP_RTSP_TRANSPORT") and hasattr(cv2, "RTSP_TRANSPORT_TCP"):
                cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.RTSP_TRANSPORT_TCP)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                continue
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error.emit("Failed to read frame (retrying next URL if any)")
                    break
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
                self.frameReady.emit(img)
                self.msleep(10)
            cap.release()
            if not self._stop.is_set():
                # try next candidate URL
                continue
            else:
                break
        else:
            self.error.emit("Failed to open RTSP via OpenCV")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ONVIF Discovery + Control")
        self.setMinimumSize(900, 600)

        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["IP", "XAddrs", "Scopes", "Hardware"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._table_selection_changed)

        self._scan_button = QtWidgets.QPushButton("Scan Network")
        self._scan_button.clicked.connect(self.start_scan)

        self._status_label = QtWidgets.QLabel("Ready")
        self._status_label.setStyleSheet("color: gray;")

        self._host_edit = QtWidgets.QLineEdit()
        self._port_edit = QtWidgets.QLineEdit("80")
        self._user_edit = QtWidgets.QLineEdit()
        self._password_edit = QtWidgets.QLineEdit()
        self._password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self._remember_check = QtWidgets.QCheckBox("Remember (keyring)")
        self._connect_button = QtWidgets.QPushButton("Connect")
        self._connect_button.clicked.connect(self._start_connect)
        self._connect_status = QtWidgets.QLabel("Not connected")
        self._connect_status.setStyleSheet("color: gray;")
        self._services_view = QtWidgets.QTreeWidget()
        self._services_view.setHeaderLabels(["Service", "Namespace", "Address"])
        self._services_view.setRootIsDecorated(False)
        self._capabilities_label = QtWidgets.QLabel("Capabilities: -")
        self._dump_button = QtWidgets.QPushButton("Dump details to console")
        self._dump_button.clicked.connect(self._dump_details)
        self._probe_vendor_button = QtWidgets.QPushButton("Probe common vendor IR/light endpoints")
        self._probe_vendor_button.clicked.connect(self._probe_vendor_endpoints)
        self._details_text = QtWidgets.QTextEdit()
        self._details_text.setReadOnly(True)

        self._video_widget = QVideoWidget()
        self._image_label = QtWidgets.QLabel("Alt player idle")
        self._image_label.setAlignment(QtCore.Qt.AlignCenter)
        self._image_label.setStyleSheet("background: #111; color: #888;")
        self._image_label.setMinimumHeight(240)
        self._video_stack = QtWidgets.QStackedWidget()
        self._video_stack.addWidget(self._video_widget)  # index 0
        self._video_stack.addWidget(self._image_label)  # index 1
        self._audio_output = QAudioOutput()
        self._player = QMediaPlayer()
        self._player.setVideoOutput(self._video_widget)
        self._player.setAudioOutput(self._audio_output)
        self._player.errorOccurred.connect(self._player_error)
        self._stream_url_edit = QtWidgets.QLineEdit()
        self._tcp_variant_button = QtWidgets.QPushButton("Try TCP param")
        self._tcp_variant_button.setEnabled(True)
        self._udp_note = QtWidgets.QLabel("")  # updated when stream changes
        self._udp_note.setStyleSheet("color: #888; font-size: 11px;")
        self._substream_button = QtWidgets.QPushButton("Try substream (102)")
        self._base_url_button = QtWidgets.QPushButton("Use base URL")
        self._alt_start_button = QtWidgets.QPushButton("Start alt player (OpenCV)")
        self._alt_stop_button = QtWidgets.QPushButton("Stop alt player")
        self._alt_start_button.clicked.connect(self._start_alt_player)
        self._alt_stop_button.clicked.connect(self._stop_alt_player)
        self._alt_stop_button.setEnabled(False)
        self._tcp_variant_button.clicked.connect(self._apply_tcp_variant)
        self._substream_button.clicked.connect(self._apply_substream_variant)
        self._base_url_button.clicked.connect(self._apply_base_url)

        self._btn_up = QtWidgets.QPushButton("▲")
        self._btn_down = QtWidgets.QPushButton("▼")
        self._btn_left = QtWidgets.QPushButton("◀")
        self._btn_right = QtWidgets.QPushButton("▶")
        self._btn_zoom_in = QtWidgets.QPushButton("Zoom +")
        self._btn_zoom_out = QtWidgets.QPushButton("Zoom -")
        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_ir_auto = QtWidgets.QPushButton("IR Auto")
        self._btn_ir_on = QtWidgets.QPushButton("IR On")
        self._btn_ir_off = QtWidgets.QPushButton("IR Off")
        self._relay_combo = QtWidgets.QComboBox()
        self._relay_on = QtWidgets.QPushButton("Relay On")
        self._relay_off = QtWidgets.QPushButton("Relay Off")
        for btn in [self._btn_up, self._btn_down, self._btn_left, self._btn_right, self._btn_zoom_in, self._btn_zoom_out, self._btn_stop]:
            btn.setEnabled(False)
        for btn in [self._btn_ir_auto, self._btn_ir_on, self._btn_ir_off, self._relay_on, self._relay_off]:
            btn.setEnabled(False)

        self._btn_up.pressed.connect(lambda: self._ptz_move(0.0, 0.4, 0.0))
        self._btn_down.pressed.connect(lambda: self._ptz_move(0.0, -0.4, 0.0))
        self._btn_left.pressed.connect(lambda: self._ptz_move(-0.4, 0.0, 0.0))
        self._btn_right.pressed.connect(lambda: self._ptz_move(0.4, 0.0, 0.0))
        self._btn_zoom_in.pressed.connect(lambda: self._ptz_move(0.0, 0.0, 0.4))
        self._btn_zoom_out.pressed.connect(lambda: self._ptz_move(0.0, 0.0, -0.4))
        for btn in [self._btn_up, self._btn_down, self._btn_left, self._btn_right, self._btn_zoom_in, self._btn_zoom_out, self._btn_stop]:
            btn.released.connect(self._ptz_stop)
        self._btn_ir_auto.clicked.connect(lambda: self._set_ir("AUTO"))
        self._btn_ir_on.clicked.connect(lambda: self._set_ir("ON"))
        self._btn_ir_off.clicked.connect(lambda: self._set_ir("OFF"))
        self._relay_on.clicked.connect(lambda: self._set_relay("active"))
        self._relay_off.clicked.connect(lambda: self._set_relay("inactive"))

        top = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(top)

        # Left column: discovery table, connection, services
        left = QtWidgets.QVBoxLayout()
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._scan_button, 0, QtCore.Qt.AlignLeft)
        controls.addStretch(1)
        controls.addWidget(self._status_label, 0, QtCore.Qt.AlignRight)
        left.addLayout(controls)
        left.addWidget(self._table)

        connect_box = QtWidgets.QGroupBox("Connect to camera")
        form = QtWidgets.QGridLayout(connect_box)
        form.addWidget(QtWidgets.QLabel("Host/IP"), 0, 0)
        form.addWidget(self._host_edit, 0, 1)
        form.addWidget(QtWidgets.QLabel("Port"), 0, 2)
        form.addWidget(self._port_edit, 0, 3)
        form.addWidget(QtWidgets.QLabel("User"), 1, 0)
        form.addWidget(self._user_edit, 1, 1)
        form.addWidget(QtWidgets.QLabel("Password"), 1, 2)
        form.addWidget(self._password_edit, 1, 3)
        form.addWidget(self._remember_check, 2, 0, 1, 2)
        form.addWidget(self._connect_button, 0, 4, 2, 1)
        form.addWidget(self._connect_status, 3, 0, 1, 5)
        left.addWidget(connect_box)

        services_box = QtWidgets.QGroupBox("Services and capabilities (for analytics/tracking/audio/light availability)")
        services_layout = QtWidgets.QVBoxLayout(services_box)
        services_layout.addWidget(self._capabilities_label)
        services_layout.addWidget(self._services_view)
        services_layout.addWidget(self._dump_button)
        services_layout.addWidget(self._probe_vendor_button)
        services_layout.addWidget(self._details_text)
        left.addWidget(services_box)

        # Right column: video + PTZ/IR/relay
        stream_box = QtWidgets.QGroupBox("Live video / PTZ")
        stream_layout = QtWidgets.QVBoxLayout(stream_box)
        stream_layout.addWidget(self._video_stack, stretch=1)
        stream_layout.addWidget(QtWidgets.QLabel("RTSP URL"))
        url_row = QtWidgets.QHBoxLayout()
        url_row.addWidget(self._stream_url_edit, 1)
        url_row.addWidget(self._tcp_variant_button)
        url_row.addWidget(self._substream_button)
        url_row.addWidget(self._base_url_button)
        url_row.addWidget(self._alt_start_button)
        url_row.addWidget(self._alt_stop_button)
        stream_layout.addLayout(url_row)
        stream_layout.addWidget(self._udp_note)

        ptz_layout = QtWidgets.QHBoxLayout()
        ptz_layout.addWidget(self._btn_zoom_out)
        ptz_layout.addWidget(self._btn_up)
        ptz_layout.addWidget(self._btn_zoom_in)
        ptz_layout.addSpacing(8)
        ptz_layout.addWidget(self._btn_left)
        ptz_layout.addWidget(self._btn_stop)
        ptz_layout.addWidget(self._btn_right)
        ptz_layout.addSpacing(8)
        ptz_layout.addWidget(self._btn_down)
        stream_layout.addLayout(ptz_layout)

        ir_layout = QtWidgets.QHBoxLayout()
        ir_layout.addWidget(self._btn_ir_auto)
        ir_layout.addWidget(self._btn_ir_on)
        ir_layout.addWidget(self._btn_ir_off)
        stream_layout.addLayout(ir_layout)

        relay_layout = QtWidgets.QHBoxLayout()
        relay_layout.addWidget(QtWidgets.QLabel("Relay"))
        relay_layout.addWidget(self._relay_combo, 1)
        relay_layout.addWidget(self._relay_on)
        relay_layout.addWidget(self._relay_off)
        stream_layout.addLayout(relay_layout)

        root_layout.addLayout(left, 1)
        root_layout.addWidget(stream_box, 1)

        self.setCentralWidget(top)

        self._worker: Optional[DiscoveryWorker] = None
        self._connect_worker: Optional[ConnectWorker] = None
        self._session: Optional[OnvifSession] = None
        self._creds = CredentialStore()
        self._cv_worker: Optional["OpenCVStreamWorker"] = None
        self._base_stream_url: str = ""

    @QtCore.Slot()
    def start_scan(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(100)
        self._table.setRowCount(0)
        self._status_label.setText("Scanning...")
        self._status_label.setStyleSheet("color: #0066cc;")
        self._scan_button.setEnabled(False)

        self._worker = DiscoveryWorker()
        self._worker.deviceFound.connect(self._add_device)
        self._worker.scanFinished.connect(self._scan_done)
        self._worker.start()

    @QtCore.Slot(DiscoveredDevice)
    def _add_device(self, device: DiscoveredDevice) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col, value in enumerate([device.address, device.xaddrs, device.scopes, device.hardware]):
            item = QtWidgets.QTableWidgetItem(value)
            item.setData(QtCore.Qt.UserRole, value)
            item.setToolTip(value)
            self._table.setItem(row, col, item)

    @QtCore.Slot()
    def _scan_done(self) -> None:
        self._status_label.setText("Done")
        self._status_label.setStyleSheet("color: green;")
        self._scan_button.setEnabled(True)

    @QtCore.Slot()
    def _table_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        ip_item = self._table.item(row, 0)
        xaddr_item = self._table.item(row, 1)
        if ip_item:
            self._host_edit.setText(ip_item.text())
        if xaddr_item:
            parsed = urlparse(xaddr_item.text())
            if parsed.port:
                self._port_edit.setText(str(parsed.port))
        self._load_saved_credentials()

    @QtCore.Slot()
    def _start_connect(self) -> None:
        host = self._host_edit.text().strip()
        port_text = self._port_edit.text().strip() or "80"
        user = self._user_edit.text().strip()
        password = self._password_edit.text()
        try:
            port = int(port_text)
        except ValueError:
            self._connect_status.setText("Port must be a number")
            self._connect_status.setStyleSheet("color: red;")
            _log("Connection error: Port must be a number")
            return
        if not host:
            self._connect_status.setText("Enter host/IP")
            self._connect_status.setStyleSheet("color: red;")
            _log("Connection error: host/IP missing")
            return
        if self._connect_worker and self._connect_worker.isRunning():
            return

        self._connect_status.setText("Connecting...")
        self._connect_status.setStyleSheet("color: #0066cc;")
        self._connect_button.setEnabled(False)
        self._connect_worker = ConnectWorker(host, port, user, password)
        self._connect_worker.connected.connect(self._on_connected)
        self._connect_worker.error.connect(self._on_connect_error)
        self._connect_worker.start()

    @QtCore.Slot(OnvifSession)
    def _on_connected(self, session: OnvifSession) -> None:
        self._session = session
        self._connect_status.setText(f"Connected to {session.host}:{session.port}")
        self._connect_status.setStyleSheet("color: green;")
        self._connect_button.setEnabled(True)
        self._base_stream_url = session.stream_uri or ""
        self._stream_url_edit.setText(session.stream_uri)
        self._udp_note.setText("Tip: camera may be UDP-only; if TCP fails, use base URL or substream.")
        for btn in [self._btn_up, self._btn_down, self._btn_left, self._btn_right, self._btn_zoom_in, self._btn_zoom_out, self._btn_stop]:
            btn.setEnabled(True)
        ir_modes = []
        if session.imaging_options and getattr(session.imaging_options, "IrCutFilterModes", None):
            ir_modes = list(session.imaging_options.IrCutFilterModes)
        allow_ir = bool(ir_modes)
        for btn in [self._btn_ir_auto, self._btn_ir_on, self._btn_ir_off]:
            btn.setEnabled(allow_ir)
        self._play_stream(session.stream_uri)
        caps_text = ", ".join([name for name, available in session.capabilities.items() if available]) or "-"
        self._capabilities_label.setText(f"Capabilities: {caps_text}")
        self._services_view.clear()
        for svc in session.services:
            namespace = getattr(svc, "Namespace", "")
            address = getattr(svc, "XAddr", "")
            item = QtWidgets.QTreeWidgetItem([getattr(svc, "Name", namespace.split('/')[-1] or "Service"), namespace, address])
            self._services_view.addTopLevelItem(item)
        self._services_view.resizeColumnToContents(0)
        self._services_view.resizeColumnToContents(1)
        self._services_view.resizeColumnToContents(2)
        self._relay_combo.clear()
        if session.relays:
            for relay in session.relays:
                token = getattr(relay, "token", "")
                mode = getattr(relay, "Properties", None)
                mode_text = getattr(mode, "Mode", "") if mode else ""
                self._relay_combo.addItem(f"{token} ({mode_text})", token)
            self._relay_on.setEnabled(True)
            self._relay_off.setEnabled(True)
        else:
            self._relay_on.setEnabled(False)
            self._relay_off.setEnabled(False)
        if self._remember_check.isChecked():
            self._creds.save(session.host, self._user_edit.text().strip(), self._password_edit.text())

    @QtCore.Slot(str)
    def _on_connect_error(self, message: str) -> None:
        self._connect_status.setText(message)
        self._connect_status.setStyleSheet("color: red;")
        self._connect_button.setEnabled(True)
        _log(f"Connection error: {message}")
        self._session = None
        for btn in [self._btn_up, self._btn_down, self._btn_left, self._btn_right, self._btn_zoom_in, self._btn_zoom_out, self._btn_stop]:
            btn.setEnabled(False)
        for btn in [self._btn_ir_auto, self._btn_ir_on, self._btn_ir_off, self._relay_on, self._relay_off]:
            btn.setEnabled(False)
        self._player.stop()
        self._services_view.clear()
        self._capabilities_label.setText("Capabilities: -")
        self._details_text.clear()
        self._stream_url_edit.clear()
        self._stop_alt_player()
        self._base_stream_url = ""

    def _load_saved_credentials(self) -> None:
        host = self._host_edit.text().strip()
        if not host:
            return
        user, password = self._creds.load(host)
        if user:
            self._user_edit.setText(user)
        if password:
            self._password_edit.setText(password)

    @QtCore.Slot()
    def _dump_details(self) -> None:
        if not self._session:
            self._connect_status.setText("Connect first to dump details")
            self._connect_status.setStyleSheet("color: red;")
            return
        details = []
        details.append(f"Stream URI: {self._session.stream_uri}")
        details.append(f"Profile token: {self._session.profile_token}")
        if self._session.imaging_settings:
            details.append(f"Imaging settings: {self._session.imaging_settings}")
        if self._session.imaging_options:
            details.append(f"Imaging options: {self._session.imaging_options}")
        details.append(f"Capabilities: {self._session.capabilities}")
        details.append("Services:")
        for svc in self._session.services:
            details.append(f" - {getattr(svc, 'Name', '')} ns={getattr(svc, 'Namespace', '')} addr={getattr(svc, 'XAddr', '')}")
        wsdl_dump = self._session.dump_wsdl()
        details.append("WSDL operations:")
        details.append(wsdl_dump)
        text = "\n".join(details)
        self._details_text.setPlainText(text)
        _log(text)

    @QtCore.Slot()
    def _probe_vendor_endpoints(self) -> None:
        if not self._session:
            self._connect_status.setText("Connect first to probe vendor endpoints")
            self._connect_status.setStyleSheet("color: red;")
            return
        user = self._user_edit.text().strip()
        password = self._password_edit.text()
        results = []
        opener = urlrequest.build_opener()
        for name, url in self._session.vendor_probe_targets:
            try:
                req = urlrequest.Request(url)
                if user:
                    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
                    req.add_header("Authorization", f"Basic {token}")
                with opener.open(req, timeout=3) as resp:
                    status = resp.status
                    body = resp.read(120).decode(errors="ignore")
                    results.append(f"{name}: HTTP {status} body={body!r}")
            except Exception as exc:
                results.append(f"{name}: FAILED {exc}")
        text = "\n".join(results) if results else "No vendor probes defined"
        if self._details_text.toPlainText():
            self._details_text.append("\n=== Vendor probe ===")
        self._details_text.append(text)
        _log(text)

    def _set_ir(self, mode: str) -> None:
        if not self._session:
            return
        modes = []
        if self._session.imaging_options and getattr(self._session.imaging_options, "IrCutFilterModes", None):
            modes = list(self._session.imaging_options.IrCutFilterModes)
        if modes and mode not in modes:
            self._connect_status.setText(f"IR mode {mode} not supported; available: {modes}")
            self._connect_status.setStyleSheet("color: red;")
            _log(f"IR mode {mode} not supported; available: {modes}")
            return
        if not modes:
            self._connect_status.setText("Camera did not advertise IR modes; control disabled")
            self._connect_status.setStyleSheet("color: red;")
            _log("Camera did not advertise IR modes; skipping IR command")
            return
        try:
            self._session.set_ir_mode(mode)
            self._connect_status.setText(f"IR set to {mode}")
            self._connect_status.setStyleSheet("color: green;")
        except Exception as exc:
            self._connect_status.setText(f"IR error: {exc}")
            self._connect_status.setStyleSheet("color: red;")
            _log(f"IR error: {exc}")

    def _set_relay(self, state: str) -> None:
        if not self._session:
            return
        token = self._relay_combo.currentData()
        if not token:
            self._connect_status.setText("No relay selected")
            self._connect_status.setStyleSheet("color: red;")
            return
        try:
            self._session.set_relay_state(token, state)
            self._connect_status.setText(f"Relay {token} -> {state}")
            self._connect_status.setStyleSheet("color: green;")
        except Exception as exc:
            self._connect_status.setText(f"Relay error: {exc}")
            self._connect_status.setStyleSheet("color: red;")
            _log(f"Relay error: {exc}")

    def _ptz_move(self, pan: float, tilt: float, zoom: float) -> None:
        if not self._session:
            return
        try:
            self._session.continuous_move(pan=pan, tilt=tilt, zoom=zoom)
        except Exception as exc:
            self._connect_status.setText(f"PTZ error: {exc}")
            self._connect_status.setStyleSheet("color: red;")
            _log(f"PTZ error: {exc}")

    def _ptz_stop(self) -> None:
        if not self._session:
            return
        try:
            self._session.stop()
        except Exception:
            pass

    def _start_alt_player(self) -> None:
        if cv2 is None:
            msg = "OpenCV not installed. Install with: python3 -m pip install opencv-python"
            self._connect_status.setText(msg)
            self._connect_status.setStyleSheet("color: red;")
            _log(msg)
            return
        url = self._stream_url_edit.text().strip()
        if not url:
            self._connect_status.setText("No stream URI to play")
            self._connect_status.setStyleSheet("color: red;")
            return
        self._player.stop()
        self._stop_alt_player()
        self._cv_worker = OpenCVStreamWorker(url)
        self._cv_worker.frameReady.connect(self._update_frame)
        self._cv_worker.error.connect(self._on_alt_error)
        self._cv_worker.finished.connect(self._on_alt_finished)
        self._cv_worker.start()
        self._alt_start_button.setEnabled(False)
        self._alt_stop_button.setEnabled(True)
        self._connect_status.setText("Alt player running (OpenCV)")
        self._connect_status.setStyleSheet("color: #0066cc;")
        self._video_stack.setCurrentWidget(self._image_label)

    def _stop_alt_player(self) -> None:
        if self._cv_worker:
            self._cv_worker.stop()
            self._cv_worker.wait(200)
            self._cv_worker = None
        self._alt_start_button.setEnabled(True)
        self._alt_stop_button.setEnabled(False)
        self._video_stack.setCurrentWidget(self._video_widget)

    @QtCore.Slot(QtGui.QImage)
    def _update_frame(self, img: QtGui.QImage) -> None:
        pix = QtGui.QPixmap.fromImage(img)
        self._image_label.setPixmap(pix.scaled(self._image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    @QtCore.Slot(str)
    def _on_alt_error(self, message: str) -> None:
        self._connect_status.setText(message)
        self._connect_status.setStyleSheet("color: red;")
        _log(f"Alt player error: {message}")

    @QtCore.Slot()
    def _on_alt_finished(self) -> None:
        self._alt_start_button.setEnabled(True)
        self._alt_stop_button.setEnabled(False)
    def _play_stream(self, url: str) -> None:
        if not url:
            self._connect_status.setText("No stream URI")
            self._connect_status.setStyleSheet("color: red;")
            _log("No stream URI to play")
            return
        self._player.setSource(QUrl(url))
        self._player.play()
        _log(f"Playing stream: {url}")
        self._video_stack.setCurrentWidget(self._video_widget)

    @QtCore.Slot()
    def _apply_tcp_variant(self) -> None:
        url = self._stream_url_edit.text().strip()
        if not url:
            return
        if "tcp" in url.lower():
            self._play_stream(url)
            return
        sep = "&" if "?" in url else "?"
        url_tcp = f"{url}{sep}tcp"
        self._stream_url_edit.setText(url_tcp)
        self._play_stream(url_tcp)
        self._udp_note.setText("If TCP fails with 454/timeout, revert to base URL or substream (UDP).")

    @QtCore.Slot()
    def _apply_substream_variant(self) -> None:
        url = self._stream_url_edit.text().strip()
        if not url:
            return
        url_sub = url
        if "/101" in url:
            url_sub = url.replace("/101", "/102", 1)
        elif "/Streaming/Channels/1" in url:
            url_sub = url.replace("/Streaming/Channels/1", "/Streaming/Channels/2", 1)
        self._stream_url_edit.setText(url_sub)
        self._play_stream(url_sub)

    @QtCore.Slot()
    def _apply_base_url(self) -> None:
        if not self._base_stream_url:
            return
        self._stream_url_edit.setText(self._base_stream_url)
        self._play_stream(self._base_stream_url)

    @QtCore.Slot(QMediaPlayer.Error)
    def _player_error(self, err: QMediaPlayer.Error) -> None:
        msg = f"Player error ({err}): {self._player.errorString()}"
        self._connect_status.setText(msg)
        self._connect_status.setStyleSheet("color: red;")
        _log(msg)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(200)
        if self._connect_worker and self._connect_worker.isRunning():
            self._connect_worker.wait(200)
        self._stop_alt_player()
        if self._player:
            self._player.stop()
        event.accept()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
