# presence.py
# Uses YOLO or faces to generate events; prevents “no events” when YOLO is absent.
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Set, Tuple
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
        seen: Set[str] = set()

        # Faces: treat recognised names as distinct identities
        for b in getattr(pkt, "faces", []) or []:
            label = b.cls or "person"
            if label in ("face", "unknown", "person"):
                key = "person"
                rec_label = None
            else:
                key = f"person:{label}"
                rec_label = label
            seen.add(key)
            self.last_seen[key] = now

        # YOLO detections for pets/persons (without names)
        for b in getattr(pkt, "yolo", []) or []:
            if b.cls in ("person", "dog", "cat"):
                key = b.cls
                seen.add(key)
                self.last_seen[key] = now

        # exit events
        for k in list(self.present):
            if now - self.last_seen.get(k, 0) > self.ttl:
                self.present.remove(k)
                self._write(self._rec_payload(now, k, "exit"))

        # enter events
        for k in seen:
            if k not in self.present:
                self.present.add(k)
                self._write(self._rec_payload(now, k, "enter"))

    def _write(self, rec: Dict):
        ensure_dir(self.logs_dir)
        f = self.logs_dir / f"{self.cam}.jsonl"
        with f.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(rec) + "\n")

    def _rec_payload(self, ts: int, key: str, event: str) -> Dict:
        """
        key may be "person", "dog", "cat", or "person:<name>".
        Returns record with optional label field.
        """
        ts_wall = int(time.time() * 1000)
        label = None
        base_type = key
        if key.startswith("person:"):
            base_type = "person"
            label = key.split(":", 1)[1]
        return {"ts": ts_wall, "camera": self.cam, "event": event, "type": base_type, "label": label}
