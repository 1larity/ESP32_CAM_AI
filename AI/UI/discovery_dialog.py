# discovery_dialog.py
# Local subnet scanner for ESP32-CAM. Looks for /api/status (preferred),
# then /status, /stream, / on ports 80 and 81.

from __future__ import annotations
import socket
import threading
import queue
from typing import Set

from PyQt6 import QtWidgets, QtCore
import requests


def _guess_primary_ipv4() -> str | None:
    """
    Try to get the primary IPv4 by opening a UDP socket to a public IP.
    This avoids cases where gethostname() resolves to 127.0.x.x.
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
    if ip:
        parts = ip.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3]) + "."
    # Fallback
    return "192.168.1."


class DiscoveryDialog(QtWidgets.QDialog):
    # idx, total, ip
    progress = QtCore.pyqtSignal(int, int, str)
    # label
    addItemSignal = QtCore.pyqtSignal(str)
    # scan finished (renamed earlier to avoid clashing with QDialog.done())
    scanFinished = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Discover ESP32-CAM")

        # --- UI ---
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
        self.lbl = QtWidgets.QLabel(
            "Finds devices responding on /api/status (preferred), "
            "/status, /stream, or / on ports 80/81.\n"
            "401 Unauthorized is treated as a hit (auth-only /api/status).\n"
            "Scanning uses a concurrent worker pool for speed."
        )
        self.lbl_progress = QtWidgets.QLabel("Idle")
        self.pb = QtWidgets.QProgressBar()
        self.pb.setMinimum(0)
        self.pb.setMaximum(0)
        self.pb.setValue(0)

        form = QtWidgets.QFormLayout()
        form.addRow("Subnet prefix", self.edit_subnet)
        form.addRow("Range", self._range_row())

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_scan)
        btns.addWidget(self.btn_stop)
        btns.addStretch(1)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(btns)
        lay.addWidget(self.lbl_progress)
        lay.addWidget(self.pb)
        lay.addWidget(self.list)
        lay.addWidget(self.lbl)

        # --- state ---
        self._stop = threading.Event()
        self._seen_keys: Set[str] = set()
        self._lock = threading.Lock()  # protects _seen_keys and progress counter

        # --- wire ---
        self.btn_scan.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._cancel)

        self.progress.connect(self._on_progress)
        self.addItemSignal.connect(self._add_item)
        self.scanFinished.connect(self._done)

    # ------------------------------------------------------------------ UI helpers

    def _range_row(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.edit_range_from)
        h.addWidget(QtWidgets.QLabel("to"))
        h.addWidget(self.edit_range_to)
        h.addStretch(1)
        return w

    # ------------------------------------------------------------------ control

    def _start(self) -> None:
        self.list.clear()
        self._seen_keys.clear()
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

        # Run the IP scanning in a single manager thread which spawns a worker pool.
        t = threading.Thread(
            target=self._scan_range,
            args=(subnet, a, b, total),
            daemon=True,
        )
        t.start()

    def _cancel(self) -> None:
        self._stop.set()
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_progress.setText("Scan cancelled.")

    # ------------------------------------------------------------------ scanning (concurrent workers over IPs)

    def _scan_range(self, subnet: str, a: int, b: int, total: int) -> None:
        # Prepare queue of IPs to scan
        ips = [f"{subnet}{i}" for i in range(a, b + 1)]
        q: queue.Queue[str] = queue.Queue()
        for ip in ips:
            q.put(ip)

        total = len(ips)
        if total == 0:
            self.scanFinished.emit()
            return

        # Shared progress counter
        idx = 0

        # Prioritise /api/status on port 80, then other combinations.
        paths = ("/api/status", "/status", "/stream", "/")
        ports = (80, 81)

        def worker() -> None:
            nonlocal idx
            sess = requests.Session()
            sess.headers.update({"User-Agent": "ESP32-CAM-Discovery/1.0"})
            while not self._stop.is_set():
                try:
                    ip = q.get_nowait()
                except queue.Empty:
                    break

                found_for_ip = False

                for port in ports:
                    if self._stop.is_set():
                        break

                    for path in paths:
                        if self._stop.is_set():
                            break

                        key = f"{ip}:{port}"
                        # If we've already got this IP:port, skip further paths
                        with self._lock:
                            if key in self._seen_keys:
                                found_for_ip = True
                                break

                        url = f"http://{ip}:{port}{path}"
                        try:
                            r = sess.get(url, timeout=0.6)
                        except Exception:
                            continue

                        # Accept 2xx and 401 as hits
                        if (200 <= r.status_code < 300) or (r.status_code == 401):
                            with self._lock:
                                if key in self._seen_keys:
                                    # Another worker already recorded it
                                    found_for_ip = True
                                    break
                                self._seen_keys.add(key)

                            label = f"{ip}:{port}  {path}"

                            # If this is /api/status and NOT 401, try to pull a name from JSON
                            if path == "/api/status" and (200 <= r.status_code < 300):
                                try:
                                    data = r.json()
                                    nm = data.get("name") or data.get("camera")
                                    if nm:
                                        label = f"{ip}:{port}  {nm} (/api/status)"
                                except Exception:
                                    pass

                            # emit to GUI thread
                            self.addItemSignal.emit(label)
                            found_for_ip = True
                            break

                    if found_for_ip:
                        break

                # progress update
                with self._lock:
                    idx += 1
                    cur_idx = idx

                self.progress.emit(cur_idx, total, ip)
                q.task_done()

        # Spawn worker pool
        num_workers = min(32, total)
        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        # Wait for workers to finish
        for t in threads:
            t.join()

        self.scanFinished.emit()

    # ------------------------------------------------------------------ slots (GUI thread)

    @QtCore.pyqtSlot()
    def _done(self) -> None:
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_progress.setText("Scan complete.")

    @QtCore.pyqtSlot(int, int, str)
    def _on_progress(self, idx: int, total: int, ip: str) -> None:
        if total > 0:
            self.pb.setMaximum(total)
            self.pb.setValue(min(idx, total))
            self.lbl_progress.setText(f"Scanning {ip} ({idx}/{total})")
        else:
            self.pb.setMaximum(0)
            self.lbl_progress.setText(f"Scanning {ip}")

    @QtCore.pyqtSlot(str)
    def _add_item(self, label: str) -> None:
        self.list.addItem(label)
