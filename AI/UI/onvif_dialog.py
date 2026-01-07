from __future__ import annotations

import threading
from typing import Optional

from PySide6 import QtCore, QtWidgets

from UI.onvif_dialog_add_camera import build_camera_settings_from_selection
from UI.onvif_dialog_format import format_onvif_label, render_onvif_details
from UI.onvif_dialog_ui import build_onvif_discovery_dialog_ui
from UI.onvif_dialog_workers import (
    enrich_onvif_info,
    prompt_for_credentials,
    run_scan_worker,
)


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
        self._selected_cam: Optional[object] = None

    def selected_camera(self) -> Optional[object]:
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

        t = threading.Thread(
            target=run_scan_worker,
            kwargs={
                "stop_event": self._stop_evt,
                "on_info": self.addItemSignal.emit,
                "on_finished": self.finishedSignal.emit,
                "timeout": 2.0,
                "retries": 2,
            },
            daemon=True,
        )
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
        item = QtWidgets.QListWidgetItem(format_onvif_label(info))
        self.list.addItem(item)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, info)
        self._update_buttons()

    @QtCore.Slot(str)
    def _set_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _update_item_label(self, item: QtWidgets.QListWidgetItem, info: dict) -> None:
        item.setText(format_onvif_label(info))

    def _on_selection(self) -> None:
        self._update_buttons()
        sel = self.list.selectedItems()
        if not sel:
            self.details.clear()
            return
        info = sel[0].data(QtCore.Qt.ItemDataRole.UserRole) or {}
        self.details.setPlainText(render_onvif_details(info))

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
        creds = prompt_for_credentials(self)
        if creds is None:
            return
        user, pwd = creds
        self.lbl_status.setText("Fetching profiles/stream with credentials...")
        self.btn_fetch_auth.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_scan.setEnabled(False)

        def worker() -> None:
            try:
                enriched = enrich_onvif_info(info, user or None, pwd)
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
        if not isinstance(info, dict):
            return
        cam_cfg = build_camera_settings_from_selection(self, item, info)
        if cam_cfg is None:
            return
        self._selected_cam = cam_cfg
        self.accept()
