from __future__ import annotations

import threading
from typing import Optional

from PySide6 import QtCore, QtWidgets

from onvif import discover_onvif, OnvifDiscoveryResult
from onvif.enrichment import enrich_onvif_device
from onvif.rtsp import guess_fallback_urls, inject_auth
from settings import CameraSettings
from UI.onvif_dialog_ui import build_onvif_discovery_dialog_ui


class OnvifDiscoveryDialog(QtWidgets.QDialog):
    addItemSignal = QtCore.Signal(object)  # info dict
    statusSignal = QtCore.Signal(str)
    finishedSignal = QtCore.Signal()
    applyItemSignal = QtCore.Signal(object, object)
    authFetchDone = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        ui = build_onvif_discovery_dialog_ui(self)
        self.btn_scan = ui.btn_scan
        self.btn_stop = ui.btn_stop
        self.btn_fetch_auth = ui.btn_fetch_auth
        self.btn_add = ui.btn_add
        self.list = ui.list
        self.lbl_status = ui.lbl_status
        self.progress = ui.progress
        self.details = ui.details

        self.list.itemSelectionChanged.connect(self._on_selection)
        self.list.itemDoubleClicked.connect(self._on_add_selected)

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
                    info = enrich_onvif_device(res, None, None)
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
        user, ok = QtWidgets.QInputDialog.getText(
            self,
            "Camera credentials",
            "Username:",
            QtWidgets.QLineEdit.EchoMode.Normal,
        )
        if not ok:
            return
        pwd, ok = QtWidgets.QInputDialog.getText(
            self,
            "Camera credentials",
            "Password:",
            QtWidgets.QLineEdit.EchoMode.Password,
        )
        if not ok:
            return
        user = user.strip()
        self.lbl_status.setText("Fetching profiles/stream with credentials...")
        self.btn_fetch_auth.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_scan.setEnabled(False)

        def worker() -> None:
            try:
                enriched = enrich_onvif_device(
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
        if not isinstance(item, QtWidgets.QListWidgetItem) or not isinstance(
            enriched, dict
        ):
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

        def _add_variant(u: str | None) -> None:
            if u and u not in variants:
                variants.append(u)

        _add_variant(stream)
        for u in info.get("fallback_urls") or []:
            _add_variant(u)
        guesses = guess_fallback_urls(info.get("ip") or "")
        _add_variant(guesses[0] if guesses else None)
        if (not stream) or info.get("auth_required"):
            # Prompt for creds and attempt once
            user, ok = QtWidgets.QInputDialog.getText(
                self,
                "Camera credentials",
                "Username:",
                QtWidgets.QLineEdit.EchoMode.Normal,
            )
            if not ok:
                return
            pwd, ok = QtWidgets.QInputDialog.getText(
                self,
                "Camera credentials",
                "Password:",
                QtWidgets.QLineEdit.EchoMode.Password,
            )
            if not ok:
                return
            user = user.strip()
            try:
                enriched = enrich_onvif_device(
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
                QtWidgets.QMessageBox.warning(
                    self, "Add Camera", f"Failed to fetch stream URI: {e}"
                )
                return

        if not stream:
            # fall back to common RTSP paths
            candidates = info.get("fallback_urls") or guess_fallback_urls(
                info.get("ip") or ""
            )
            if candidates:
                stream = candidates[0]
        # Embed credentials into URL if provided and not already present
        stream = inject_auth(stream, user, pwd)
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
                    prof_lines.append(
                        f"- {p.get('name') or p.get('token')} (token={p.get('token')})"
                    )
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

