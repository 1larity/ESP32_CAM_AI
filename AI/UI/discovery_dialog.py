# UI/discovery_dialog.py
# Local subnet scanner for ESP32-CAM. Checks ONLY /api/status on port 80.

from __future__ import annotations

import socket
import threading

from PySide6 import QtCore, QtWidgets

from discovery.scanner import RangeScanner, ScanConfig
from discovery.probe import ProbeResult


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
    return f"{parts[0]}.{parts[1]}.{parts[2]}."


class DiscoveryDialog(QtWidgets.QDialog):
    progress = QtCore.Signal(int, int, str)   # idx, total, ip
    addItemSignal = QtCore.Signal(str)        # label
    scanFinished = QtCore.Signal()            # scan finished

    def __init__(self, app_cfg, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_cfg = app_cfg

        self.setWindowTitle("Discover ESP32-CAM")

        self.edit_subnet = QtWidgets.QLineEdit(_default_subnet())

        self.edit_range_from = QtWidgets.QSpinBox()
        self.edit_range_from.setRange(1, 254)
        self.edit_range_from.setValue(1)

        self.edit_range_to = QtWidgets.QSpinBox()
        self.edit_range_to.setRange(1, 254)
        self.edit_range_to.setValue(254)

        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.list = QtWidgets.QListWidget()

        self.lbl_help = QtWidgets.QLabel(
            "Checks only /api/status on port 80.\n"
            "401 Unauthorized is treated as a hit (device is present but locked).\n"
            "Concurrent worker pool scan for speed."
        )
        self.lbl_progress = QtWidgets.QLabel("Idle")

        self.pb = QtWidgets.QProgressBar()
        self.pb.setMinimum(0)
        self.pb.setMaximum(0)
        self.pb.setValue(0)

        form = QtWidgets.QFormLayout()
        form.addRow("Subnet prefix", self.edit_subnet)

        range_row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(range_row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.edit_range_from)
        h.addWidget(QtWidgets.QLabel("to"))
        h.addWidget(self.edit_range_to)
        h.addStretch(1)
        form.addRow("Range", range_row)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_scan)
        btns.addWidget(self.btn_stop)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(btns)
        lay.addWidget(self.lbl_help)
        lay.addWidget(self.lbl_progress)
        lay.addWidget(self.pb)
        lay.addWidget(self.list)

        self._stop = threading.Event()
        self._scanner: RangeScanner | None = None

        self.btn_scan.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._cancel)

        self.progress.connect(self._on_progress)
        self.addItemSignal.connect(self._add_item)
        self.scanFinished.connect(self._done)

    def _start(self) -> None:
        self.list.clear()
        self._stop.clear()

        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)

        subnet = self.edit_subnet.text().strip()
        a = int(self.edit_range_from.value())
        b = int(self.edit_range_to.value())
        total = max(0, b - a + 1)

        self.pb.setMinimum(0)
        if total > 0:
            self.pb.setMaximum(total)
            self.pb.setValue(0)
            self.lbl_progress.setText(f"Scanning {subnet}{a} to {subnet}{b} (0/{total})")
        else:
            self.pb.setMaximum(0)
            self.lbl_progress.setText("Scanning...")

        t = threading.Thread(target=self._scan_range, args=(subnet, a, b), daemon=True)
        t.start()

    def _cancel(self) -> None:
        self._stop.set()
        if self._scanner is not None:
            self._scanner.stop()
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_progress.setText("Scan cancelled.")

    def _scan_range(self, subnet: str, a: int, b: int) -> None:
        cfg = ScanConfig(
            max_workers=64,
            timeout=0.6,
            port=80,
            path="/api/status",
        )
        self._scanner = RangeScanner(config=cfg, stop_event=self._stop)

        def on_progress(idx: int, total: int, ip: str) -> None:
            self.progress.emit(idx, total, ip)

        def on_result(res: ProbeResult) -> None:
            kind = "AUTH" if res.auth_required else "OK"
            extra = f"{kind} {res.status_code}  {res.elapsed_ms}ms"
            if res.name:
                label = f"{res.ip}:{res.port}  {res.name}  ({res.path})  [{extra}]"
            else:
                label = f"{res.ip}:{res.port}  {res.path}  [{extra}]"
            self.addItemSignal.emit(label)

        self._scanner.scan_range(subnet, a, b, on_progress=on_progress, on_result=on_result)
        self.scanFinished.emit()

    @QtCore.Slot()
    def _done(self) -> None:
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.pb.setMaximum(1)
        self.pb.setValue(1)
        self.lbl_progress.setText("Scan complete.")

    @QtCore.Slot(int, int, str)
    def _on_progress(self, idx: int, total: int, ip: str) -> None:
        if total > 0:
            self.pb.setMaximum(total)
            self.pb.setValue(min(idx, total))
            self.lbl_progress.setText(f"Scanning {ip} ({idx}/{total})")
        else:
            self.pb.setMaximum(0)
            self.lbl_progress.setText(f"Scanning {ip}")

    @QtCore.Slot(str)
    def _add_item(self, label: str) -> None:
        self.list.addItem(label)
