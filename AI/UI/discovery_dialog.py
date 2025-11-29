# discovery_dialog.py
# Local subnet scanner for ESP32-CAM. Looks for /api/status (preferred),
# then /status, /stream, / on ports 80 and 81.

from __future__ import annotations
import socket
import threading
import queue
from typing import Set
from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Signal, Slot
import requests


def _guess_primary_ipv4() -> str | None:
    """
    Try to get the primary IPv4 by opening a UDP socket to a public IP.
    This avoids platform-specific APIs.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        # We don't actually connect, just use routing table
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _default_subnet() -> str:
    """
    Try to guess the local /24 from the primary IPv4; fall back to 192.168.1.x.
    """
    ip = _guess_primary_ipv4()
    if not ip:
        return "192.168.1."

    parts = ip.split(".")
    if len(parts) != 4:
        return "192.168.1."

    # assume /24
    return ".".join(parts[:3]) + "."


def _probe_ip(ip: str, timeout: float = 0.5) -> dict | None:
    """
    Probe an IP for ESP32-CAM HTTP endpoints.

    Returns a dict with:
      {
        "ip": str,
        "auth": bool,
        "stream_ok": bool | None,
        "ping": bool,
        "notes": str,
      }
    or None if nothing useful was detected.
    """
    base = f"http://{ip}"
    session = requests.Session()
    session.headers.update({"User-Agent": "ESP32-CAM-Discovery/1.0"})
    # 1) /ping
    try:
        r = session.get(f"{base}/ping", timeout=timeout)
        text = (r.text or "").strip().lower()
        if r.status_code == 200 and text.startswith("pong"):
            info = {
                "ip": ip,
                "ping": True,
                "auth": False,
                "stream_ok": None,
                "notes": "",
            }
            # try / (may 401 if auth on)
            try:
                r2 = session.get(base + "/", timeout=timeout)
                if r2.status_code == 401:
                    info["auth"] = True
                    info["notes"] = "Auth required"
                elif r2.status_code == 200:
                    info["notes"] = "HTTP OK"
            except Exception:
                pass
            # try :81/stream
            try:
                r3 = session.get(f"{base}:81/stream", timeout=timeout, stream=True)
                if r3.status_code == 200:
                    info["stream_ok"] = True
                elif r3.status_code == 401:
                    info["auth"] = True
            except Exception:
                pass
            return info
    except Exception:
        pass

    return None


class DiscoveryDialog(QtWidgets.QDialog):
    # idx, total, ip
    progress = Signal(int, int, str)
    # label
    addItemSignal = Signal(str)
    # scan finished (renamed earlier to avoid clashing with QDialog.done())
    scanFinished = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Discover ESP32-CAMs")
        self.resize(600, 400)

        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue: "queue.Queue[str]" = queue.Queue()
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
            # Stop existing scan
            self._stop_flag.set()
            return

        subnet = self.subnet_edit.text().strip()
        if not subnet.endswith("."):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid subnet",
                "Subnet must end with a dot, e.g. 192.168.1.",
            )
            return

        self._stop_flag.clear()
        self._seen_ips.clear()
        self.list_widget.clear()

        self.scan_button.setText("Stop")
        self.scan_button.setEnabled(True)

        self._thread = threading.Thread(target=self._scan_worker, args=(subnet,))
        self._thread.daemon = True
        self._thread.start()

    def _scan_worker(self, subnet: str) -> None:
        """
        Run in a background thread; probes IPs.
        """
        total = 254
        for idx, host in enumerate(range(1, 255), start=1):
            if self._stop_flag.is_set():
                break

            ip = f"{subnet}{host}"
            self.progress.emit(idx, total, ip)

            info = _probe_ip(ip)
            if info is not None:
                self.addItemSignal.emit(
                    f"{info['ip']}  ping={info['ping']}  auth={info['auth']}  stream={info['stream_ok']}  {info['notes']}"
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

        # The label starts with the IP
        label = item.text()
        ip = label.split()[0]

        # Just close with Accepted and stash IP in a property so the caller
        # can read it (or extend later with more data).
        self.selected_ip = ip
        self.accept()

    # ------------------------------------------------------------------ close

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self._stop_flag.set()
        super().closeEvent(event)
