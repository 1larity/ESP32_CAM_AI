# presence.py
# Uses YOLO or faces to generate events; prevents “no events” when YOLO is absent.
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Set
from detectors import DetectionPacket
from utils import ensure_dir

class PresenceBus:
    def __init__(self, cam_name: str, logs_dir: Path, ttl_ms: int = 2500):
        self.cam = cam_name
        self.logs_dir = Path(logs_dir)
        self.ttl = ttl_ms
        self.last_seen: Dict[str, int] = {}
        self.present: Set[str] = set()

    def update(self, pkt: DetectionPacket):
        now = pkt.ts_ms
        seen = set()
        # Prefer YOLO
        for b in pkt.yolo:
            if b.cls in ("person", "dog", "cat"):
                seen.add(b.cls)
        # Fallback: any face counts as a person presence
        if not seen and pkt.faces:
            seen.add("person")

        # update timestamps
        for k in seen:
            self.last_seen[k] = now

        # exit events
        for k in list(self.present):
            if now - self.last_seen.get(k, 0) > self.ttl:
                self.present.remove(k)
                self._write({"ts": now, "camera": self.cam, "event": "exit", "type": k})

        # enter events
        for k in seen:
            if k not in self.present:
                self.present.add(k)
                self._write({"ts": now, "camera": self.cam, "event": "enter", "type": k})

    def _write(self, rec: Dict):
        ensure_dir(self.logs_dir)
        f = self.logs_dir / f"{self.cam}.jsonl"
        with f.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec) + "\n")
