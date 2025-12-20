# discovery_dialog.py
# Local subnet scanner for ESP32-CAM.
# Fast concurrent scan; detects cameras by /api/status (200 OK or 401 Unauthorized),
# with fallbacks to /ping, /, and :81/stream.

from __future__ import annotations

import socket
import threading
from typing import Set, Optional, Dict, Any

from concurrent.futures import ThreadPoolExecutor, as_completed

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Signal, Slot

import requests


def _guess_primary_ipv4() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _default_subnet() -> str:
    ip = _guess_primary_ipv4()
    if not ip:
        return "192.168.1."
    parts = ip.split(".")
    if len(parts) != 4:
        return "192.168.1."
    return ".".join(parts[:3]) + "."


def _probe_ip(ip: str, timeout: float = 0.35) -> Dict[str, Any] | None:
    """
    Probe an IP for ESP32-CAM endpoints without knowing credentials.

    Hit criteria:
      - /api/status returns 200 OR 401  -> HIT (via=status)
      - /ping returns "pong"           -> HIT (via=ping)

    Returns:
      {
        "ip": str,
        "hit": bool,
        "hit_via": "status" | "ping",
        "auth": bool,
        "status": "-" | "ok" | "401",
        "ping": None | True | False,        # None = not tried
        "stream": "-" | "ok" | "401",
        "notes": str,
      }
    or None if nothing detected.
    """
    base = f"http://{ip}"
    session = requests.Session()
    session.headers.update({"User-Agent": "ESP32-CAM-Discovery/2.0"})

    info: Dict[str, Any] = {
        "ip": ip,
        "hit": False,
        "hit_via": "",
        "auth": False,
        "status": "-",      # "-", "ok", "401"
        "ping": None,       # None=not tried, True/False=tried
        "stream": "-",      # "-", "ok", "401"
        "notes": "",
    }

    # 1) Prefer /api/status (your discovery endpoint)
    try:
        r = session.get(f"{base}/api/status", timeout=timeout)
        if r.status_code == 200:
            info["hit"] = True
            info["hit_via"] = "status"
            info["status"] = "ok"
            try:
                js = r.json()
                name = js.get("name") or js.get("cam") or js.get("id")
                info["notes"] = f"status OK ({name})" if name else "status OK"
            except Exception:
                info["notes"] = "status OK"
        elif r.status_code == 401:
            # Unknown creds, but still a discovered camera
            info["hit"] = True
            info["hit_via"] = "status"
            info["status"] = "401"
            info["auth"] = True
            info["notes"] = "status auth required"
    except Exception:
        pass

    # 2) Fallback: /ping
    if not info["hit"]:
        info["ping"] = False
        try:
            r = session.get(f"{base}/ping", timeout=timeout)
            text = (r.text or "").strip().lower()
            if r.status_code == 200 and text.startswith("pong"):
                info["hit"] = True
                info["hit_via"] = "ping"
                info["ping"] = True
                info["notes"] = "pong"
        except Exception:
            pass

    if not info["hit"]:
        return None

    # 3) Enrich: check / (may 401)
    try:
        r2 = session.get(f"{base}/", timeout=timeout)
        if r2.status_code == 401:
            info["auth"] = True
            if not info["notes"]:
                info["notes"] = "auth required"
        elif r2.status_code == 200 and not info["notes"]:
            info["notes"] = "HTTP OK"
    except Exception:
        pass

    # 4) Enrich: try :81/stream (may 200 or 401)
    try:
        r3 = session.get(f"{base}:81/stream", timeout=timeout, stream=True)
        if r3.status_code == 200:
            info["stream"] = "ok"
        elif r3.status_code == 401:
            info["stream"] = "401"
            info["auth"] = True
    except Exception:
        pass

    return info


class DiscoveryDialog(QtWidgets.QDialog):
    # idx, total, ip
    progress = Signal(int, int, str)
    # label
    addItemSignal = Signal(str)
    # scan finished
    scanFinished = Signal()

    def __init__(self, app_cfg, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_cfg = app_cfg

        self.setWindowTitle("Discover ESP32-CAMs")
        self.resize(720, 420)

        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_ips: Set[str] = set()

        self._build_ui()
        self._wire_signals()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()

        self.subnet_edit = QtWidgets.QLineEdit(self)
        self.subnet_edit.setText(_default_subnet())
        form.addRow("Subnet (x.x.x.)", self.subnet_edit)

        rng_row = QtWidgets.QHBoxLayout()
        self.start_spin = QtWidgets.QSpinBox(self)
        self.start_spin.setRange(1, 254)
        self.start_spin.setValue(1)
        rng_row.addWidget(QtWidgets.QLabel("Start host", self))
        rng_row.addWidget(self.start_spin)

        self.end_spin = QtWidgets.QSpinBox(self)
        self.end_spin.setRange(1, 254)
        self.end_spin.setValue(254)
        rng_row.addWidget(QtWidgets.QLabel("End host", self))
        rng_row.addWidget(self.end_spin)

        rng_wrap = QtWidgets.QWidget(self)
        rng_wrap.setLayout(rng_row)
        form.addRow("Range", rng_wrap)

        perf_row = QtWidgets.QHBoxLayout()
        self.workers_spin = QtWidgets.QSpinBox(self)
        self.workers_spin.setRange(1, 256)
        self.workers_spin.setValue(64)
        perf_row.addWidget(QtWidgets.QLabel("Workers", self))
        perf_row.addWidget(self.workers_spin)

        self.timeout_ms_spin = QtWidgets.QSpinBox(self)
        self.timeout_ms_spin.setRange(50, 5000)
        self.timeout_ms_spin.setValue(350)
        perf_row.addWidget(QtWidgets.QLabel("Timeout (ms)", self))
        perf_row.addWidget(self.timeout_ms_spin)

        perf_wrap = QtWidgets.QWidget(self)
        perf_wrap.setLayout(perf_row)
        form.addRow("Performance", perf_wrap)

        layout.addLayout(form)

        self.scan_button = QtWidgets.QPushButton("Scan", self)
        self.scan_button.clicked.connect(self._on_scan_clicked)
        layout.addWidget(self.scan_button)

        self.list_widget = QtWidgets.QListWidget(self)
        layout.addWidget(self.list_widget)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_selected = QtWidgets.QPushButton("Add selected camera", self)
        self.btn_close = QtWidgets.QPushButton("Close", self)
        btn_row.addWidget(self.btn_add_selected)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.btn_add_selected.clicked.connect(self._on_add_selected)
        self.btn_close.clicked.connect(self.reject)

    def _wire_signals(self) -> None:
        self.progress.connect(self._on_progress)
        self.addItemSignal.connect(self._on_add_item)
        self.scanFinished.connect(self._on_scan_finished)

    # ------------------------------------------------------------------ scanning logic

    @Slot()
    def _on_scan_clicked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._stop_flag.set()
            self.scan_button.setEnabled(False)
            return

        subnet = self.subnet_edit.text().strip()
        if not subnet.endswith("."):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid subnet",
                "Subnet must end with a dot, e.g. 192.168.1.",
            )
            return

        start_h = int(self.start_spin.value())
        end_h = int(self.end_spin.value())
        if end_h < start_h:
            QtWidgets.QMessageBox.warning(
                self, "Invalid range", "End host must be >= start host."
            )
            return

        workers = int(self.workers_spin.value())
        timeout_s = float(self.timeout_ms_spin.value()) / 1000.0

        self._stop_flag.clear()
        self._seen_ips.clear()
        self.list_widget.clear()

        self.scan_button.setText("Stop")
        self.scan_button.setEnabled(True)

        self._thread = threading.Thread(
            target=self._scan_worker, args=(subnet, start_h, end_h, workers, timeout_s)
        )
        self._thread.daemon = True
        self._thread.start()

    def _scan_worker(
        self, subnet: str, start_h: int, end_h: int, workers: int, timeout_s: float
    ) -> None:
        hosts = list(range(start_h, end_h + 1))
        total = len(hosts)

        submitted = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_ip = {}
            for host in hosts:
                if self._stop_flag.is_set():
                    break
                ip = f"{subnet}{host}"
                fut = ex.submit(_probe_ip, ip, timeout_s)
                fut_to_ip[fut] = ip
                submitted += 1
                self.progress.emit(submitted, total, ip)

            for fut in as_completed(list(fut_to_ip.keys())):
                if self._stop_flag.is_set():
                    break
                ip = fut_to_ip[fut]
                completed += 1
                self.progress.emit(completed, total, ip)

                try:
                    info = fut.result()
                except Exception:
                    continue
                if info is None:
                    continue

                ip2 = info["ip"]
                if ip2 in self._seen_ips:
                    continue
                self._seen_ips.add(ip2)

                self.addItemSignal.emit(
                    f"{ip2}  via={info['hit_via']}  status={info['status']}  "
                    f"auth={info['auth']}  stream={info['stream']}  {info['notes']}"
                )

        self.scanFinished.emit()

    @Slot(int, int, str)
    def _on_progress(self, idx: int, total: int, ip: str) -> None:
        self.setWindowTitle(f"Discover ESP32-CAMs – {idx}/{total} ({ip})")

    @Slot(str)
    def _on_add_item(self, label: str) -> None:
        self.list_widget.addItem(label)

    @Slot()
    def _on_scan_finished(self) -> None:
        self.scan_button.setText("Scan")
        self.scan_button.setEnabled(True)
        self.setWindowTitle("Discover ESP32-CAMs")

    # ------------------------------------------------------------------ selection

    @Slot()
    def _on_add_selected(self) -> None:
        item = self.list_widget.currentItem()
        if not item:
            QtWidgets.QMessageBox.information(
                self, "No selection", "Please select a discovered camera."
            )
            return

        label = item.text()
        ip = label.split()[0]
        self.selected_ip = ip
        self.accept()

    # ------------------------------------------------------------------ close

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self._stop_flag.set()
        super().closeEvent(event)
