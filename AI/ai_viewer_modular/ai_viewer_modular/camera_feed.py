import threading
import urllib.request
import time
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from constants import CONFIDENCE_THRESHOLD, IGNORE_CLASSES
from overlay import render_overlay_canvas

class CameraFeed:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.frame = None
        self.overlay = None
        self.lock = threading.Lock()
        self.running = True
        self.model = YOLO('yolov8s.pt')
        self.tracks = defaultdict(list)
        self.last_seen = {}
        self.speeds = {}
        self.show_trails = True
        self.show_ids = True
        self.show_speed = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _mjpeg_stream(self):
        stream = urllib.request.urlopen(self.url)
        bytes_ = b''
        while self.running:
            bytes_ += stream.read(1024)
            a = bytes_.find(b'\xff\xd8')
            b = bytes_.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_[a:b+2]
                bytes_ = bytes_[b+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                yield img

    def _run(self):
        for frame in self._mjpeg_stream():
            frame = cv2.resize(frame, (320, 240))
            results = self.model.track(frame, persist=True, verbose=False)[0]
            now = time.time()
            labels = []
            trails_data = []
            for box in results.boxes:
                if box.conf < CONFIDENCE_THRESHOLD or box.id is None:
                    continue
                cls = self.model.names[int(box.cls)]
                if cls in IGNORE_CLASSES:
                    continue
                track_id = int(box.id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.tracks[track_id].append((cx, cy))
                if len(self.tracks[track_id]) > 50:
                    self.tracks[track_id] = self.tracks[track_id][-50:]
                if track_id in self.last_seen:
                    last_cx, last_cy, last_time = self.last_seen[track_id]
                    dt = now - last_time
                    if dt > 0:
                        vx = (cx - last_cx) / dt
                        vy = (cy - last_cy) / dt
                        speed = (vx**2 + vy**2)**0.5
                        self.speeds[track_id] = (vx, vy, speed)
                self.last_seen[track_id] = (cx, cy, now)
                label = f"{cls}"
                if self.show_ids:
                    label += f" ID:{track_id}"
                if self.show_speed and track_id in self.speeds:
                    label += f" {self.speeds[track_id][2]:.1f}px/s"
                labels.append((label, (x1, y1 - 5)))
                if self.show_trails:
                    trails_data.append(self.tracks[track_id])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            overlay = render_overlay_canvas((320, 240), labels, trails_data, font_scale=0.4)
            with self.lock:
                self.frame = frame.copy()
                self.overlay = overlay.copy()

    def get_composite(self):
        with self.lock:
            if self.frame is None:
                return None
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            if self.overlay is not None:
                overlay_rgb = cv2.cvtColor(self.overlay, cv2.COLOR_BGRA2RGBA)
                return frame_rgb, overlay_rgb
            return frame_rgb, None
