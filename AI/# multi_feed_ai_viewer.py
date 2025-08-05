# Multi-Feed AI Viewer with dropdown menu, resizable windows, and layout control

import threading
import urllib.request
import cv2
import numpy as np
import pygame
import time
import json
import tkinter as tk
from tkinter import simpledialog, filedialog
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import defaultdict
import requests
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)

WINDOW_SIZE = (1280, 720)
CONFIDENCE_THRESHOLD = 0.3
IGNORE_CLASSES = ['kangaroo']
PTZ_COMMANDS = ['up', 'down', 'left', 'right']
LAYOUT_FILE = 'layout.json'

class CameraFeed:
    def __init__(self, name, url):
        self.name = name
        self.url = url
        self.frame = None
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
            annotator = Annotator(frame)
            now = time.time()

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
                        speed = (vx**2 + vy**2) ** 0.5
                        self.speeds[track_id] = (vx, vy, speed)
                self.last_seen[track_id] = (cx, cy, now)

                label = f"{cls}"
                if self.show_ids:
                    label += f" ID:{track_id}"
                if self.show_speed and track_id in self.speeds:
                    label += f" {self.speeds[track_id][2]:.1f}px/s"

                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                pts = self.tracks[track_id]
                if self.show_trails:
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i-1], pts[i], (255, 255, 0), 1)

            with self.lock:
                self.frame = annotator.result()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

class FeedWindow:
    def __init__(self, name, camera_feed, x, y, width=320, height=240):
        self.name = name
        self.feed = camera_feed
        self.rect = pygame.Rect(x, y, width, height)
        self.dragging = False
        self.resizing = False
        self.selected = False
        self.resize_margin = 10

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and self.rect.collidepoint(event.pos):
            # Right-click context menu
            root = tk.Tk()
            root.withdraw()
            selection = simpledialog.askstring("Overlay Options", "Options: trails, ids, speed")
            if selection:
                if "trail" in selection.lower():
                    self.feed.show_trails = not self.feed.show_trails
                if "id" in selection.lower():
                    self.feed.show_ids = not self.feed.show_ids
                if "speed" in selection.lower():
                    self.feed.show_speed = not self.feed.show_speed

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self._in_resize_zone(event.pos):
                self.resizing = True
                self.selected = True
            elif self.rect.collidepoint(event.pos):
                self.dragging = True
                self.mouse_offset = (self.rect.x - event.pos[0], self.rect.y - event.pos[1])
                self.selected = True
            else:
                self.selected = False

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            self.resizing = False

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.rect.x = event.pos[0] + self.mouse_offset[0]
                self.rect.y = event.pos[1] + self.mouse_offset[1]
            elif self.resizing:
                self.rect.width = max(100, event.pos[0] - self.rect.x)
                self.rect.height = max(80, event.pos[1] - self.rect.y)

    def _in_resize_zone(self, pos):
        x, y = pos
        return self.rect.collidepoint(x, y) and (
            abs(x - (self.rect.x + self.rect.width)) < self.resize_margin or
            abs(y - (self.rect.y + self.rect.height)) < self.resize_margin)

    def draw(self, surface, font):
        frame = self.feed.get_frame()
        if frame is None:
            return
        scaled = cv2.resize(frame, (self.rect.width, self.rect.height))
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surface.blit(surf, (self.rect.x, self.rect.y))

        pygame.draw.rect(surface, (0, 255, 0) if self.selected else (255, 255, 0), self.rect, 2)
        surface.blit(font.render(self.name, True, (255, 255, 255)), (self.rect.x + 5, self.rect.y + 5))

        for i, cmd in enumerate(PTZ_COMMANDS):
            btn = pygame.Rect(self.rect.x + 5 + i*75, self.rect.y + self.rect.height - 30, 70, 25)
            pygame.draw.rect(surface, (50, 50, 50), btn)
            surface.blit(font.render(cmd.upper(), True, (255, 255, 255)), (btn.x + 5, btn.y + 5))
            if pygame.mouse.get_pressed()[0] and btn.collidepoint(pygame.mouse.get_pos()):
                requests.get(self.feed.url.replace('/stream', f'/action?go={cmd}'))

def save_layout(windows):
    layout = [{"name": w.name, "x": w.rect.x, "y": w.rect.y, "w": w.rect.width, "h": w.rect.height, "url": w.feed.url} for w in windows]
    with open(LAYOUT_FILE, 'w') as f:
        json.dump(layout, f)

def load_layout():
    try:
        with open(LAYOUT_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def prompt_for_url():
    root = tk.Tk()
    root.withdraw()
    return simpledialog.askstring("New Feed", "Enter MJPEG stream URL:")

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("Multi-Feed AI Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)

    feeds, windows = {}, []
    for item in load_layout():
        feeds[item['name']] = feeds.get(item['name']) or CameraFeed(item['name'], item['url'])
        windows.append(FeedWindow(item['name'], feeds[item['name']], item['x'], item['y'], item.get('w', 320), item.get('h', 240)))

    menu_rects = {
        'New': pygame.Rect(10, 5, 60, 25),
        'Save': pygame.Rect(80, 5, 60, 25),
        'Load': pygame.Rect(150, 5, 60, 25),
        'Remove': pygame.Rect(220, 5, 90, 25),
        'Quit': pygame.Rect(320, 5, 60, 25)
    }

    running = True
    while running:
        screen.fill((30, 30, 30))

        for event in pygame.event.get():
            for w in windows:
                w.handle_event(event)
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                for label, rect in menu_rects.items():
                    if rect.collidepoint(event.pos):
                        if label == 'New':
                            url = prompt_for_url()
                            if url:
                                name_base = "Cam"
                                index = 1
                                while f"{name_base}{index}" in feeds:
                                    index += 1
                                name = f"{name_base}{index}"
                                feed = CameraFeed(name, url)
                                feeds[name] = feed
                                start_x = 30 + (index * 40) % (WINDOW_SIZE[0] - 350)
                                start_y = 50 + (index * 40) % (WINDOW_SIZE[1] - 250)
                                windows.append(FeedWindow(name, feed, start_x, start_y))
                        elif label == 'Save':
                            save_layout(windows)
                        elif label == 'Load':
                            windows.clear()
                            for item in load_layout():
                                feeds[item['name']] = feeds.get(item['name']) or CameraFeed(item['name'], item['url'])
                                windows.append(FeedWindow(item['name'], feeds[item['name']], item['x'], item['y'], item.get('w', 320), item.get('h', 240)))
                        elif label == 'Remove':
                            windows = [w for w in windows if not w.selected]
                        elif label == 'Quit':
                            running = False

        for w in windows:
            w.draw(screen, font)

        for label, rect in menu_rects.items():
            pygame.draw.rect(screen, (70, 70, 70), rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
            screen.blit(font.render(label, True, (255, 255, 255)), (rect.x + 5, rect.y + 5))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()
