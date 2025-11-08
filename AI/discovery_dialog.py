# discovery_dialog.py
# Local subnet scanner for ESP32-CAM. Looks for /status or /stream on :80 and :81.
from __future__ import annotations
import socket
import threading
from PyQt6 import QtWidgets, QtCore
import requests

def _default_subnet() -> str:
    # Try to guess local /24
    try:
        host = socket.gethostbyname(socket.gethostname())
        parts = host.split(".")
        if len(parts) == 4:
            return ".".join(parts[:3]) + "."
    except Exception:
        pass
    return "192.168.1."

class DiscoveryDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Discover ESP32-CAM")
        self.edit_subnet = QtWidgets.QLineEdit(_default_subnet())
        self.edit_range_from = QtWidgets.QSpinBox(); self.edit_range_from.setRange(1,254); self.edit_range_from.setValue(1)
        self.edit_range_to   = QtWidgets.QSpinBox(); self.edit_range_to.setRange(1,254); self.edit_range_to.setValue(254)
        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_stop = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.list = QtWidgets.QListWidget()
        self.lbl = QtWidgets.QLabel("Finds devices responding on port 80/81 with /status or /stream.")
        form = QtWidgets.QFormLayout()
        form.addRow("Subnet prefix", self.edit_subnet)
        form.addRow("Range", self._range_row())
        btns = QtWidgets.QHBoxLayout(); btns.addWidget(self.btn_scan); btns.addWidget(self.btn_stop); btns.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self); lay.addLayout(form); lay.addLayout(btns); lay.addWidget(self.list); lay.addWidget(self.lbl)
        self._stop = threading.Event()
        self.btn_scan.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._cancel)

    def _range_row(self):
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0,0,0,0)
        h.addWidget(self.edit_range_from); h.addWidget(QtWidgets.QLabel("to")); h.addWidget(self.edit_range_to); h.addStretch(1)
        return w

    def _start(self):
        self.list.clear()
        self._stop.clear()
        self.btn_scan.setEnabled(False); self.btn_stop.setEnabled(True)
        subnet = self.edit_subnet.text().strip()
        a = int(self.edit_range_from.value()); b = int(self.edit_range_to.value())
        t = threading.Thread(target=self._scan_range, args=(subnet, a, b), daemon=True)
        t.start()

    def _cancel(self):
        self._stop.set()
        self.btn_scan.setEnabled(True); self.btn_stop.setEnabled(False)

    def _scan_range(self, subnet: str, a: int, b: int):
        sess = requests.Session()
        sess.headers.update({"User-Agent":"ESP32-CAM-AI-Discovery/1.0"})
        for i in range(a, b+1):
            if self._stop.is_set(): break
            ip = f"{subnet}{i}"
            for port in (81, 80):
                for path in ("/status", "/api/status", "/stream", "/"):
                    url = f"http://{ip}:{port}{path}"
                    try:
                        r = sess.get(url, timeout=0.6)
                        if r.status_code == 200:
                            self._add_item(ip, port, path)
                            raise StopIteration  # found something on this IP
                    except StopIteration:
                        break
                    except Exception:
                        pass

        self._done()

    @QtCore.pyqtSlot()
    def _done(self):
        self.btn_scan.setEnabled(True); self.btn_stop.setEnabled(False)

    def _add_item(self, ip: str, port: int, path: str):
        QtCore.QMetaObject.invokeMethod(self.list, "addItem", QtCore.Qt.ConnectionType.QueuedConnection,
                                        QtCore.Q_ARG(str, f"{ip}:{port}  {path}"))
