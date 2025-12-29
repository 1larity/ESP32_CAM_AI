# events_pane.py
# Dockable pane that tails JSONL event logs and shows a live list.
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
from PySide6 import QtWidgets, QtCore

class EventsPane(QtWidgets.QWidget):
    def __init__(self, logs_dir: Path, parent=None):
        super().__init__(parent)
        self.logs_dir = Path(logs_dir)
        self.list = QtWidgets.QListWidget()
        self.btn_open = QtWidgets.QPushButton("Open Logs Folder")
        self.btn_clear = QtWidgets.QPushButton("Clear View")
        btns = QtWidgets.QHBoxLayout(); btns.addWidget(self.btn_open); btns.addStretch(1); btns.addWidget(self.btn_clear)
        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(self.list); lay.addLayout(btns)
        self.btn_open.clicked.connect(self._open_logs)
        self.btn_clear.clicked.connect(self.list.clear)
        self._pos: Dict[Path, int] = {}
        self._timer = QtCore.QTimer(self); self._timer.timeout.connect(self._poll); self._timer.start(500)

    def _open_logs(self):
        from utils import open_folder_or_warn
        open_folder_or_warn(self, self.logs_dir)

    def _poll(self):
        if not self.logs_dir.exists(): return
        for p in sorted(self.logs_dir.glob("*.jsonl")):
            last = self._pos.get(p, 0)
            try:
                with p.open("rb") as fp:
                    fp.seek(last)
                    for line in fp:
                        try:
                            rec = json.loads(line.decode("utf-8", "ignore"))
                        except Exception:
                            continue
                        ts = rec.get("ts")
                        cam = rec.get("camera")
                        ev = rec.get("event")
                        typ = rec.get("type")
                        self.list.addItem(f"{cam} - {ev} {typ} @ {ts}")
                    self._pos[p] = fp.tell()
            except FileNotFoundError:
                self._pos.pop(p, None)
