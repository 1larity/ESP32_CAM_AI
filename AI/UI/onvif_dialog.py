from __future__ import annotations

import threading
from typing import Dict, Optional

from PySide6 import QtCore, QtWidgets

from onvif import (
    discover_onvif,
    OnvifDiscoveryResult,
    OnvifClient,
    OnvifAuthError,
    OnvifError,
    try_onvif_zeep_stream,
)
from settings import CameraSettings


class OnvifDiscoveryDialog(QtWidgets.QDialog):
    addItemSignal = QtCore.Signal(object)      # info dict
    statusSignal = QtCore.Signal(str)
    finishedSignal = QtCore.Signal()
    applyItemSignal = QtCore.Signal(object, object)
    authFetchDone = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Discover ONVIF Cameras")

        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_fetch_auth = QtWidgets.QPushButton("Fetch with credentials…")
        self.btn_fetch_auth.setEnabled(False)
        self.btn_add = QtWidgets.QPushButton("Add Selected")
        self.btn_add.setEnabled(False)

        self.list = QtWidgets.QListWidget()
        self.list.itemSelectionChanged.connect(self._on_selection)
        self.list.itemDoubleClicked.connect(self._on_add_selected)

        self.lbl_status = QtWidgets.QLabel("Idle")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximum(0)
        self.progress.setValue(0)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(500)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_scan)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_fetch_auth)
        btns.addWidget(self.btn_add)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("WS-Discovery will broadcast on the local network and query camera capabilities."))
        lay.addLayout(btns)
        lay.addWidget(self.lbl_status)
        lay.addWidget(self.progress)
        lay.addWidget(self.list)
        lay.addWidget(QtWidgets.QLabel("Camera details"))
        lay.addWidget(self.details)

        self.btn_scan.clicked.connect(self._start_scan)
        self.btn_stop.clicked.connect(self._stop_scan)
        self.btn_fetch_auth.clicked.connect(self._fetch_with_credentials)
        self.btn_add.clicked.connect(self._on_add_selected)

        self.addItemSignal.connect(self._add_item)
        self.statusSignal.connect(self._set_status)
        self.finishedSignal.connect(self._done_scan)
        self.applyItemSignal.connect(self._apply_enriched)
        self.authFetchDone.connect(self._auth_fetch_done)

        self._stop_evt = threading.Event()
        self._scan_thread: Optional[threading.Thread] = None
        self._selected_cam: Optional[CameraSettings] = None

    def selected_camera(self) -> Optional[CameraSettings]:
        return self._selected_cam

    # ----------------- scan ----------------- #
    def _start_scan(self) -> None:
        self.list.clear()
        self._stop_evt.clear()
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_fetch_auth.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.progress.setMaximum(0)
        self.progress.setValue(0)
        self.lbl_status.setText("Scanning...")

        def worker() -> None:
            try:
                hits = discover_onvif(timeout=2.0, retries=2, stop_event=self._stop_evt)
                for res in hits:
                    if self._stop_evt.is_set():
                        break
                    info = self._enrich(res, None, None)
                    self.addItemSignal.emit(info)
            finally:
                self.finishedSignal.emit()

        t = threading.Thread(target=worker, daemon=True)
        self._scan_thread = t
        t.start()

    def _stop_scan(self) -> None:
        self._stop_evt.set()
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopping...")

    def _done_scan(self) -> None:
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setMaximum(1)
        self.progress.setValue(1)
        if not self.list.count():
            self.lbl_status.setText("No ONVIF cameras found.")
        else:
            self.lbl_status.setText("Scan complete.")

    # ----------------- enrichment ----------------- #
    def _enrich(self, res: OnvifDiscoveryResult, user: Optional[str], pwd: Optional[str]) -> dict:
        info = {
            "xaddr": res.xaddr,
            "ip": res.ip,
            "auth_required": False,
            "name": res.ip,
            "model": None,
            "firmware": None,
            "profiles": [],
            "media_xaddr": None,
            "stream_uri": None,
            "user": user,
            "password": pwd,
            "errors": [],
            "fallback_urls": [],
            "zeep_used": False,
        }
        # Try zeep (onvif-zeep) first if available; helps picky cameras.
        profiles_zeep, stream_zeep, errs_zeep = try_onvif_zeep_stream(res.xaddr, user, pwd)
        if profiles_zeep or stream_zeep:
            info["profiles"] = profiles_zeep
            info["stream_uri"] = stream_zeep
            info["zeep_used"] = True
        if errs_zeep:
            info["errors"].extend(errs_zeep)

        client = OnvifClient(res.xaddr, username=user, password=pwd)
        try:
            dev = client.get_device_information()
            if dev:
                info["name"] = dev.manufacturer or dev.model or res.ip
                info["model"] = dev.model
                info["firmware"] = dev.firmware
        except OnvifAuthError:
            info["auth_required"] = True
        except OnvifError as e:
            info["errors"].append(f"device_info: {e}")

        media_xaddr = None
        try:
            caps = client.get_capabilities()
            media_xaddr = caps.media_xaddr
            info["media_xaddr"] = media_xaddr
        except OnvifAuthError:
            info["auth_required"] = True
        except OnvifError as e:
            info["errors"].append(f"capabilities: {e}")

        try:
            profiles = client.get_profiles(media_xaddr=media_xaddr or res.xaddr)
            info["profiles"] = [{"token": p.token, "name": p.name} for p in profiles]
            stream = None
            if profiles:
                stream = client.get_stream_uri(profiles[0].token, media_xaddr=media_xaddr or res.xaddr)
            info["stream_uri"] = stream
            info["fallback_urls"] = self._guess_fallback_urls(res.ip)
        except OnvifAuthError:
            info["auth_required"] = True
        except OnvifError as e:
            info["errors"].append(f"media: {e}")
        return info

    # ----------------- UI wiring ----------------- #
    @QtCore.Slot(object)
    def _add_item(self, info: object) -> None:
        if not isinstance(info, dict):
            return
        label = self._format_label(info)
        item = QtWidgets.QListWidgetItem(label)
        self.list.addItem(item)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, info)
        self._update_buttons()

    @QtCore.Slot(str)
    def _set_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _update_item_label(self, item: QtWidgets.QListWidgetItem, info: dict) -> None:
        item.setText(self._format_label(info))

    def _format_label(self, info: dict) -> str:
        name = info.get("name") or info.get("model") or info.get("ip")
        stream = info.get("stream_uri") or "<no stream>"
        auth = "auth" if info.get("auth_required") else "open"
        profile_count = len(info.get("profiles") or [])
        return f"{name} | {stream} | profiles:{profile_count} | {auth}"

    def _on_selection(self) -> None:
        self._update_buttons()
        sel = self.list.selectedItems()
        if not sel:
            self.details.clear()
            return
        info = sel[0].data(QtCore.Qt.ItemDataRole.UserRole) or {}
        self.details.setPlainText(self._render_details(info))

    def _update_buttons(self) -> None:
        has_sel = len(self.list.selectedItems()) > 0
        self.btn_add.setEnabled(has_sel)
        self.btn_fetch_auth.setEnabled(has_sel)

    def _fetch_with_credentials(self) -> None:
        sel = self.list.selectedItems()
        if not sel:
            return
        item = sel[0]
        info = item.data(QtCore.Qt.ItemDataRole.UserRole) or {}
        user, ok = QtWidgets.QInputDialog.getText(self, "Camera credentials", "Username:", QtWidgets.QLineEdit.EchoMode.Normal)
        if not ok:
            return
        pwd, ok = QtWidgets.QInputDialog.getText(self, "Camera credentials", "Password:", QtWidgets.QLineEdit.EchoMode.Password)
        if not ok:
            return
        user = user.strip()
        self.lbl_status.setText("Fetching profiles/stream with credentials...")
        self.btn_fetch_auth.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_scan.setEnabled(False)

        def worker() -> None:
            try:
                enriched = self._enrich(
                    OnvifDiscoveryResult(
                        xaddr=info.get("xaddr"),
                        epr=None,
                        scopes=[],
                        ip=info.get("ip"),
                    ),
                    user or None,
                    pwd,
                )
                self.applyItemSignal.emit(item, enriched)
            except Exception as e:
                self.statusSignal.emit(f"Fetch failed: {e}")
            finally:
                self.authFetchDone.emit()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    @QtCore.Slot(object, object)
    def _apply_enriched(self, item: object, enriched: object) -> None:
        if not isinstance(item, QtWidgets.QListWidgetItem) or not isinstance(enriched, dict):
            return
        item.setData(QtCore.Qt.ItemDataRole.UserRole, enriched)
        self._update_item_label(item, enriched)
        self.lbl_status.setText("Updated with credentials.")
        self._update_buttons()

    @QtCore.Slot()
    def _auth_fetch_done(self) -> None:
        self.btn_scan.setEnabled(True)
        self.btn_fetch_auth.setEnabled(True)
        self.btn_add.setEnabled(True)

    def _on_add_selected(self) -> None:
        sel = self.list.selectedItems()
        if not sel:
            return
        item = sel[0]
        info = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not info:
            return
        stream = info.get("stream_uri")
        user = info.get("user")
        pwd = info.get("password")
        variants: list[str] = []
        def _add_variant(u: str) -> None:
            if u and u not in variants:
                variants.append(u)
        _add_variant(stream)
        for u in info.get("fallback_urls") or []:
            _add_variant(u)
        _add_variant(self._guess_fallback_urls(info.get("ip") or "")[0] if self._guess_fallback_urls(info.get("ip") or "") else None)
        if (not stream) or info.get("auth_required"):
            # Prompt for creds and attempt once
            user, ok = QtWidgets.QInputDialog.getText(self, "Camera credentials", "Username:", QtWidgets.QLineEdit.EchoMode.Normal)
            if not ok:
                return
            pwd, ok = QtWidgets.QInputDialog.getText(self, "Camera credentials", "Password:", QtWidgets.QLineEdit.EchoMode.Password)
            if not ok:
                return
            user = user.strip()
            try:
                enriched = self._enrich(
                    OnvifDiscoveryResult(
                        xaddr=info.get("xaddr"),
                        epr=None,
                        scopes=[],
                        ip=info.get("ip"),
                    ),
                    user or None,
                    pwd,
                )
                info.update(enriched)
                stream = info.get("stream_uri") or stream
                if info.get("stream_uri"):
                    _add_variant(info.get("stream_uri"))
                for u in info.get("fallback_urls") or []:
                    _add_variant(u)
                self._update_item_label(item, info)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Add Camera", f"Failed to fetch stream URI: {e}")
                return

        if not stream:
            # fall back to common RTSP paths
            candidates = info.get("fallback_urls") or self._guess_fallback_urls(info.get("ip") or "")
            if candidates:
                stream = candidates[0]
        # Embed credentials into URL if provided and not already present
        stream = self._inject_auth(stream, user, pwd)
        alt_streams = [u for u in variants if u and u != stream]
        name = info.get("name") or info.get("model") or info.get("ip") or "ONVIF-Camera"
        cam_cfg = CameraSettings(
            name=name,
            stream_url=stream,
            alt_streams=alt_streams,
            user=user or None,
            password=pwd or None,
        )
        self._selected_cam = cam_cfg
        self.accept()

    def _render_details(self, info: dict) -> str:
        parts = []
        parts.append(f"IP: {info.get('ip')}")
        parts.append(f"XAddr: {info.get('xaddr')}")
        if info.get("media_xaddr"):
            parts.append(f"Media XAddr: {info.get('media_xaddr')}")
        parts.append(f"Name/Model: {info.get('name')} / {info.get('model')}")
        if info.get("firmware"):
            parts.append(f"Firmware: {info.get('firmware')}")
        profiles = info.get("profiles") or []
        if profiles:
            prof_lines = []
            for p in profiles:
                if isinstance(p, dict):
                    prof_lines.append(f"- {p.get('name') or p.get('token')} (token={p.get('token')})")
                else:
                    prof_lines.append(f"- {p}")
            parts.append("Profiles:\n" + "\n".join(prof_lines))
        parts.append(f"Stream URI: {info.get('stream_uri') or '<none>'}")
        if info.get("error"):
            parts.append(f"Error: {info.get('error')}")
        errs = info.get("errors") or []
        if errs:
            parts.append("Errors:")
            parts.extend(f"- {e}" for e in errs)
        if info.get("auth_required"):
            parts.append("Auth required: yes")
        fallbacks = info.get("fallback_urls") or []
        if fallbacks:
            parts.append("Fallback RTSP guesses:")
            parts.extend(f"- {u}" for u in fallbacks)
        return "\n".join(parts)

    @staticmethod
    def _guess_fallback_urls(ip: str) -> list[str]:
        if not ip:
            return []
        return [
            f"rtsp://{ip}:554/Streaming/Channels/101",
            f"rtsp://{ip}:554/Streaming/Channels/102",
            f"rtsp://{ip}:554/live/ch0",
            f"rtsp://{ip}:554/live/ch1",
        ]

    @staticmethod
    def _inject_auth(url: str, user: Optional[str], pwd: Optional[str]) -> str:
        if not url or not user or not pwd:
            return url
        try:
            parsed = QtCore.QUrl(url)
            if parsed.userName():
                return url  # already has creds
            parsed.setUserName(user)
            parsed.setPassword(pwd)
            return parsed.toString()
        except Exception:
            return url
