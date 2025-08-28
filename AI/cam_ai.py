#!/usr/bin/env python3
"""
ESP32-CAM AI Viewer

Features
- Connects to ESP32-CAM MJPEG stream (port 81) using Basic Auth or a token query param
- Pygame UI for visualization and keyboard control
- Face detection + LBPH face recognition with on-the-fly enrollment
- Object detection (person, dog, cat) via YOLOv8n ONNX (optional)
- Simple target tracking with PT (pan/tilt) commands sent back to the camera

Folders
- ai/data/faces/<person_name>/*.jpg  # stored face samples
- ai/models/{MobileNetSSD_deploy.prototxt, MobileNetSSD_deploy.caffemodel}

Controls
- q: quit
- e: enroll currently detected face (prompts for name)
- r: retrain face recognizer from saved samples
- t: cycle target tracking mode (off -> person -> face-known -> dog -> cat)

Dependencies
- python -m pip install pygame opencv-contrib-python numpy requests

Usage
- python ai/cam_ai.py --host 192.168.1.50 --user admin --password YOURPASS
  or: python ai/cam_ai.py --host 192.168.1.50 --token BASE64_USER_COLON_PASS
"""

import os
import time
import base64
import threading
import argparse
import json
from typing import Optional, Tuple, List

import numpy as np
import requests
import pygame
import cv2
import math


# ---------- Config ----------
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


def ensure_dirs():
    os.makedirs(os.path.join('ai', 'data', 'faces'), exist_ok=True)
    os.makedirs(os.path.join('ai', 'data', 'pets', 'dogs'), exist_ok=True)
    os.makedirs(os.path.join('ai', 'data', 'pets', 'cats'), exist_ok=True)
    os.makedirs(os.path.join('ai', 'models'), exist_ok=True)


# (Removed MobileNet-SSD code)


class MJPEGStream:
    def __init__(self, url: str, auth_header: Optional[str] = None):
        self.url = url
        self.auth_header = auth_header
        self.session = requests.Session()
        self.resp = None
        self.lock = threading.Lock()
        self.buf = b""
        self.running = False
        self.frame = None

    def start(self):
        headers = {}
        if self.auth_header:
            headers['Authorization'] = self.auth_header
        self.resp = self.session.get(self.url, headers=headers, stream=True, timeout=10)
        self.resp.raise_for_status()
        self.running = True
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        return self

    def _reader(self):
        boundary = b"--frame"
        try:
            for chunk in self.resp.iter_content(chunk_size=8192):
                if not self.running:
                    break
                if not chunk:
                    continue
                self.buf += chunk
                while True:
                    # Look for a Content-Length header in buffer
                    hdr_end = self.buf.find(b"\r\n\r\n")
                    if hdr_end == -1:
                        break
                    headers = self.buf[:hdr_end].decode('latin1', errors='ignore')
                    cl_idx = headers.lower().find('content-length:')
                    if cl_idx == -1:
                        # drop until next boundary
                        bidx = self.buf.find(boundary)
                        if bidx == -1:
                            self.buf = self.buf[hdr_end+4:]
                        else:
                            self.buf = self.buf[bidx:]
                        continue
                    try:
                        cl_line = headers[cl_idx:].split('\r\n', 1)[0]
                        length = int(cl_line.split(':', 1)[1].strip())
                    except Exception:
                        # Unable to parse length; resync
                        bidx = self.buf.find(boundary)
                        if bidx == -1:
                            self.buf = self.buf[hdr_end+4:]
                        else:
                            self.buf = self.buf[bidx:]
                        continue
                    start = hdr_end + 4
                    if len(self.buf) < start + length:
                        # wait for more data
                        break
                    jpg = self.buf[start:start+length]
                    # move past this part + boundary
                    tail = self.buf[start+length:]
                    bmark = b"\r\n--frame\r\n"
                    if tail.startswith(bmark):
                        self.buf = tail[len(bmark):]
                    else:
                        bpos = tail.find(bmark)
                        self.buf = tail[bpos+len(bmark):] if bpos >= 0 else b""
                    # decode
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    with self.lock:
                        self.frame = img
        except Exception:
            # Stream ended or network error; mark as stopped gracefully
            pass
        finally:
            self.running = False

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        try:
            if self.resp is not None:
                self.resp.close()
        except Exception:
            pass


class FaceDB:
    def __init__(self, base_dir='ai/data/faces'):
        self.base = base_dir
        self.labels: List[str] = []
        self.model = None
        self.size = (160, 160)
        # Threshold for LBPH: lower distance means better match
        # Typical good matches are often < 80–100 depending on lighting/data
        self.threshold = 95.0
        # Haar cascade for face detection
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # ORB fallback (for environments without cv2.face)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.db_descs = {}

    def load_and_train(self):
        people = []
        samples = []
        labels = []
        label_map = {}
        next_id = 0
        for name in sorted(os.listdir(self.base)):
            p = os.path.join(self.base, name)
            if not os.path.isdir(p):
                continue
            label_map[name] = next_id
            for fname in os.listdir(p):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img = cv2.imread(os.path.join(p, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.size)
                samples.append(img)
                labels.append(next_id)
            next_id += 1
        self.labels = [None] * next_id
        for name, i in label_map.items():
            self.labels[i] = name
        if samples and hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            self.model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
            self.model.train(samples, np.array(labels))
        else:
            self.model = None
            # Build ORB descriptor DB fallback
            self.db_descs = {}
            for name in sorted(os.listdir(self.base)):
                p = os.path.join(self.base, name)
                if not os.path.isdir(p):
                    continue
                descs = []
                for fname in os.listdir(p):
                    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    img = cv2.imread(os.path.join(p, fname), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, self.size)
                    try:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        img = clahe.apply(img)
                    except Exception:
                        img = cv2.equalizeHist(img)
                    kp, d = self.orb.detectAndCompute(img, None)
                    if d is not None:
                        descs.append(d)
                if descs:
                    self.db_descs[name] = descs

    def debug_dump(self, report_path='ai/face_dump.txt'):
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        lines = []
        total_imgs = 0
        persons = []
        for name in sorted(os.listdir(self.base)):
            p = os.path.join(self.base, name)
            if not os.path.isdir(p):
                continue
            files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            total = len(files)
            total_imgs += total
            preview = ', '.join(files[:5])
            persons.append((name, total))
            lines.append(f"[{name}] samples: {total}\n  preview: {preview}\n")
        lines.insert(0, f"Faces base: {self.base}\nPersons: {len(persons)}  Total images: {total_imgs}\n\n")
        with open(report_path,'w',encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return report_path

    def debug_verify_training(self):
        """Evaluate recognizer on training images. Returns (per_person, overall)."""
        per_person = {}
        total = 0
        correct = 0
        # Evaluate using whichever recognizer is available (LBPH or ORB fallback)
        for name in sorted(os.listdir(self.base)):
            p = os.path.join(self.base, name)
            if not os.path.isdir(p):
                continue
            files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            c = 0
            t = 0
            for fn in files:
                img = cv2.imread(os.path.join(p, fn), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                pred, score = self.recognize(img, img)
                t += 1
                total += 1
                if pred == name:
                    c += 1
                    correct += 1
            per_person[name] = {'total': t, 'correct': c, 'acc': (c / t * 100.0) if t else 0.0}
        overall = {'total': total, 'correct': correct, 'acc': (correct / total * 100.0) if total else 0.0}
        return per_person, overall

    def detect_faces(self, frame_gray):
        # Improve robustness with CLAHE and tuned parameters
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            eq = clahe.apply(frame_gray)
        except Exception:
            eq = cv2.equalizeHist(frame_gray)
        min_size = max(40, int(0.12 * min(frame_gray.shape[:2])))
        faces = self.cascade.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=4, minSize=(min_size, min_size))
        # Fallback: try slightly different params if none found
        if len(faces) == 0:
            faces = self.cascade.detectMultiScale(eq, scaleFactor=1.05, minNeighbors=3, minSize=(min_size, min_size))
        return faces

    def recognize(self, frame_gray, face_roi) -> Tuple[str, float]:
        roi = cv2.resize(face_roi, self.size)
        if self.model is not None:
            try:
                pred_id, conf = self.model.predict(roi)
                name = self.labels[pred_id] if 0 <= pred_id < len(self.labels) else 'unknown'
                if conf <= self.threshold:
                    score = max(0.0, min(1.0, 1.0 - (conf / self.threshold)))
                    return (name, float(score))
                else:
                    return ('unknown', 0.0)
            except Exception:
                return ('unknown', 0.0)
        # ORB fallback
        if not self.db_descs:
            return ('unknown', 0.0)
        try:
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                roi_p = clahe.apply(roi)
            except Exception:
                roi_p = cv2.equalizeHist(roi)
            kp, desc = self.orb.detectAndCompute(roi_p, None)
            if desc is None:
                return ('unknown', 0.0)
            best_name = 'unknown'
            best_score = 0.0
            for name, descs in self.db_descs.items():
                matches_total = 0
                good = 0
                for d in descs:
                    matches = self.bf.match(desc, d)
                    matches_total += len(matches)
                    good += sum(1 for m in matches if m.distance < 40)
                if matches_total == 0:
                    continue
                score = good / float(matches_total)
                if score > best_score:
                    best_score = score
                    best_name = name
            return (best_name if best_score > 0.12 else 'unknown', float(best_score))
        except Exception:
            return ('unknown', 0.0)

    def enroll(self, frame_gray, x, y, w, h, name: str):
        name = ''.join(c for c in name.strip() if c.isalnum() or c in ('_', '-'))
        if not name:
            return False
        p = os.path.join(self.base, name)
        os.makedirs(p, exist_ok=True)
        roi = frame_gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, self.size)
        ts = int(time.time() * 1000)
        # Save original and a mirrored version to help robustness
        cv2.imwrite(os.path.join(p, f'{ts}.jpg'), roi)
        try:
            cv2.imwrite(os.path.join(p, f'{ts}_flip.jpg'), cv2.flip(roi, 1))
        except Exception:
            pass
        return True


# (Removed SSDDetector)


class YOLODetector:
    """YOLOv8n ONNX detector using OpenCV DNN. Returns (cls, conf, x, y, w, h)."""
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    def __init__(self, model_path='ai/models/yolov8n.onnx', input_size=640, conf_thresh=0.25, iou_thresh=0.45):
        self.model_path = model_path
        self.net = None
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception:
                    pass
            except Exception:
                self.net = None

    def available(self):
        return self.net is not None

    def _letterbox(self, image, new_shape=640, color=(114, 114, 114)):
        h, w = image.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / h, new_shape[1] / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
        top = (new_shape[0] - nh) // 2
        left = (new_shape[1] - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, (left, top)

    def detect(self, frame_bgr):
        if self.net is None:
            return []
        img, r, (dx, dy) = self._letterbox(frame_bgr, self.input_size)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        # Support outputs in shapes (1, 84, N) or (1, N, 84)
        out = np.squeeze(out)
        if out.ndim == 2:
            if out.shape[0] in (84, 85):
                out = out.T
            # now (N, 84)
        elif out.ndim == 3:
            # pick the first batch and try to orient to (N, 84)
            o = out[0]
            if o.shape[0] in (84, 85):
                out = o.T
            else:
                out = o
        else:
            return []

        boxes = []
        scores = []
        classes = []
        H, W = frame_bgr.shape[:2]
        for det in out:
            cx, cy, w, h = det[:4]
            # YOLOv5 has objectness at det[4] and class scores at det[5:]
            # YOLOv8 commonly has class scores directly at det[4:]
            if det.shape[0] >= 85:
                obj = float(det[4])
                class_scores = det[5:]
                c = int(np.argmax(class_scores))
                conf = obj * float(class_scores[c])
            else:
                class_scores = det[4:]
                c = int(np.argmax(class_scores))
                conf = float(class_scores[c])
            if conf < self.conf_thresh:
                continue
            # map box back to original image coordinates
            x1 = (cx - w/2 - dx) / r
            y1 = (cy - h/2 - dy) / r
            x2 = (cx + w/2 - dx) / r
            y2 = (cy + h/2 - dy) / r
            x1 = max(0, min(W-1, x1))
            y1 = max(0, min(H-1, y1))
            x2 = max(0, min(W-1, x2))
            y2 = max(0, min(H-1, y2))
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(conf)
            classes.append(c)

        if not boxes:
            return []
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.iou_thresh)
        if isinstance(idxs, tuple) or isinstance(idxs, list):
            idxs = np.array(idxs).reshape(-1)
        else:
            idxs = np.array(idxs).reshape(-1)
        out_dets = []
        for i in idxs:
            cls_name = self.COCO_CLASSES[classes[i]] if 0 <= classes[i] < len(self.COCO_CLASSES) else str(classes[i])
            x, y, w, h = boxes[i]
            out_dets.append((cls_name, float(scores[i]), x, y, w, h))
        return out_dets


def download_yolo_model(dst_path='ai/models/yolo.onnx'):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        print(f"[YOLO] Exists: {dst_path}")
        return True
    mirrors = [
        # Ultralytics YOLOv5 (COCO 80 classes) – reliable release asset
        'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
        # Other official v5 releases to try as fallback
        'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.onnx',
        # YOLOv8 tiny/nano variants (may work if available)
        'https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.onnx',
        'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx',
    ]
    headers = {'User-Agent': 'ESP32-CAM-AI-Downloader/1.0'}
    last_err = None
    for url in mirrors:
        try:
            print(f"[YOLO] Downloading from {url} ...")
            with requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', '0'))
                done = 0
                with open(dst_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1<<15):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"  {done*100//total}%\r", end='')
            print(f"[YOLO] Saved to {dst_path}")
            return True
        except Exception as e:
            last_err = e
            continue
    print(f"[YOLO] Failed to download: {last_err}")
    return False


class PetsDB:
    """Simple pet enrollment and recognition using ORB feature matching.
    Directory layout:
      ai/data/pets/dogs/<name>/*.jpg
      ai/data/pets/cats/<name>/*.jpg
    """
    def __init__(self, base_dir='ai/data/pets'):
        self.base = base_dir
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.db = { 'dogs': {}, 'cats': {} }  # species -> name -> list of desc

    def load(self):
        self.db = { 'dogs': {}, 'cats': {} }
        for species in ('dogs', 'cats'):
            root = os.path.join(self.base, species)
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if not os.path.isdir(p):
                    continue
                descs = []
                for fn in os.listdir(p):
                    if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    img = cv2.imread(os.path.join(p, fn), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (320, 320))
                    kp, desc = self.orb.detectAndCompute(img, None)
                    if desc is not None:
                        descs.append(desc)
                if descs:
                    self.db[species][name] = descs

    def enroll(self, roi_bgr, name: str, species: str) -> bool:
        species = 'dogs' if species.lower().startswith('dog') else 'cats'
        name = ''.join(c for c in name.strip() if c.isalnum() or c in ('_', '-'))
        if not name:
            return False
        p = os.path.join(self.base, species, name)
        os.makedirs(p, exist_ok=True)
        img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (320, 320))
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(p, f'{ts}.jpg'), img)
        return True

    def recognize(self, roi_bgr, species: str) -> Tuple[str, float]:
        """Return (name, score[0..1]) or ('unknown', 0)"""
        species = 'dogs' if species == 'dog' else 'cats'
        if not self.db[species]:
            return ('unknown', 0.0)
        img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (320, 320))
        kp, desc = self.orb.detectAndCompute(img, None)
        if desc is None:
            return ('unknown', 0.0)
        best_name = 'unknown'
        best_score = 0.0
        for name, descs in self.db[species].items():
            matches_total = 0
            good = 0
            for d in descs:
                matches = self.bf.match(desc, d)
                matches_total += len(matches)
                # Count good matches (distance threshold)
                good += sum(1 for m in matches if m.distance < 40)
            if matches_total == 0:
                continue
            score = good / float(matches_total)
            if score > best_score:
                best_score = score
                best_name = name
        return (best_name if best_score > 0.12 else 'unknown', float(best_score))


# ---------- Simple Pygame Config UI ----------
class InputField:
    def __init__(self, rect, text='', masked=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.focus = False
        self.masked = masked

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.focus = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.focus:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_TAB:
                # lose focus; external manager will change focus
                self.focus = False
            elif event.key == pygame.K_RETURN:
                self.focus = False
            else:
                ch = event.unicode
                if ch and 32 <= ord(ch) < 127:
                    self.text += ch

    def draw(self, surf, font):
        bg = (220, 220, 220)
        bd = (60, 120, 200) if self.focus else (120, 120, 120)
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, bd, self.rect, 2, border_radius=6)
        disp = self.text if not self.masked else ('*' * len(self.text))
        label = font.render(disp, True, (20, 20, 20))
        ty = self.rect.y + (self.rect.h - label.get_height()) // 2
        tx = self.rect.x + 8
        surf.blit(label, (tx, ty))
        # caret (blinking)
        if self.focus:
            try:
                w = font.size(disp)[0]
            except Exception:
                w = label.get_width()
            blink = int(time.time() * 2) % 2 == 0
            if blink:
                cx = tx + w + 1
                cy1 = self.rect.y + 6
                cy2 = self.rect.y + self.rect.height - 6
                pygame.draw.line(surf, (30, 30, 30), (cx, cy1), (cx, cy2), 2)


def config_ui(defaults):
    pygame.init()
    W, H = 700, 420
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('ESP32-CAM AI Config')
    font = pygame.font.SysFont('segoeui', 18)
    small = pygame.font.SysFont('consolas', 14)

    host = InputField((140, 40, 380, 36), defaults.get('host', ''))
    user = InputField((140, 90, 180, 36), defaults.get('user', ''))
    password = InputField((340, 90, 180, 36), defaults.get('password', ''), masked=True)
    token = InputField((140, 140, 380, 36), defaults.get('token', ''))
    width = InputField((140, 190, 100, 36), str(defaults.get('width', DEFAULT_WIDTH)))
    height = InputField((260, 190, 100, 36), str(defaults.get('height', DEFAULT_HEIGHT)))

    buttons = {
        'start': pygame.Rect(140, 260, 120, 40),
        'download_yolo': pygame.Rect(270, 260, 180, 40),
        'quit': pygame.Rect(460, 260, 120, 40),
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            for f in (host, user, password, token, width, height):
                f.handle(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if buttons['start'].collidepoint(event.pos):
                    try:
                        w = int(width.text or DEFAULT_WIDTH)
                        h = int(height.text or DEFAULT_HEIGHT)
                    except ValueError:
                        w, h = DEFAULT_WIDTH, DEFAULT_HEIGHT
                    cfg = {
                        'host': host.text.strip(),
                        'user': user.text.strip(),
                        'password': password.text,
                        'token': token.text.strip(),
                        'width': w,
                        'height': h,
                    }
                    if not cfg['host']:
                        continue
                    return cfg
                if buttons['download_yolo'].collidepoint(event.pos):
                    # Download YOLOv8n ONNX (blocking)
                    try:
                        ok = download_yolo_model('ai/models/yolo.onnx')
                        print('YOLO download:', 'OK' if ok else 'FAILED')
                    except Exception as e:
                        print('Download YOLO error:', e)
                if buttons['quit'].collidepoint(event.pos):
                    return None

        screen.fill((240, 245, 255))
        # Labels
        screen.blit(font.render('ESP32-CAM AI Configuration', True, (20, 40, 90)), (20, 10))
        screen.blit(font.render('Host (ip[:port])', True, (0, 0, 0)), (20, 45))
        screen.blit(font.render('User / Password (Basic)', True, (0, 0, 0)), (20, 95))
        screen.blit(font.render('Token (Base64 user:pass)', True, (0, 0, 0)), (20, 145))
        screen.blit(font.render('Width / Height', True, (0, 0, 0)), (20, 195))
        screen.blit(small.render('Note: If token is provided, user/password are ignored.', True, (60, 60, 60)), (20, 230))

        # Inputs
        for f in (host, user, password, token, width, height):
            f.draw(screen, font)

        # Buttons
        for key, rect in buttons.items():
            pygame.draw.rect(screen, (60, 120, 200), rect, border_radius=8)
            label = 'Start' if key == 'start' else ('Download YOLO' if key == 'download_yolo' else 'Quit')
            txt = font.render(label, True, (255, 255, 255))
            screen.blit(txt, (rect.x + (rect.w - txt.get_width()) // 2, rect.y + (rect.h - txt.get_height()) // 2))

        pygame.display.flip()


# ---------- Overlay UI in viewer ----------
def draw_button(surf, rect, text, font, bg=(60,120,200), fg=(255,255,255)):
    pygame.draw.rect(surf, bg, rect, border_radius=8)
    label = font.render(text, True, fg)
    surf.blit(label, (rect.x + (rect.w - label.get_width()) // 2, rect.y + (rect.h - label.get_height()) // 2))


def modal_text_input(screen, title, fields, confirm_label='Save'):
    """Simple blocking modal for text input.
    fields: list of dicts [{'label': 'Name', 'value': '', 'masked': False}]
    Returns list of strings or None if cancelled.
    """
    W, H = screen.get_size()
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    box = pygame.Rect(W//2-260, H//2-160, 520, 240)
    font = pygame.font.SysFont('segoeui', 20)
    small = pygame.font.SysFont('segoeui', 16)

    inputs = []
    y = box.y + 60
    for f in fields:
        inp = InputField((box.x + 140, y, 320, 34), f.get('value',''), masked=f.get('masked', False))
        inputs.append((f['label'], inp))
        y += 46
    btn_ok = pygame.Rect(box.x + 220, box.y + box.h - 50, 100, 34)
    btn_cancel = pygame.Rect(box.x + 340, box.y + box.h - 50, 100, 34)

    # focus first
    if inputs:
        inputs[0][1].focus = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            for _, inp in inputs:
                inp.handle(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_ok.collidepoint(event.pos):
                    return [inp.text for _, inp in inputs]
                if btn_cancel.collidepoint(event.pos):
                    return None

        screen.blit(overlay, (0, 0))
        pygame.draw.rect(screen, (245, 248, 255), box, border_radius=10)
        pygame.draw.rect(screen, (100, 120, 170), box, 2, border_radius=10)
        screen.blit(font.render(title, True, (20, 40, 90)), (box.x + 16, box.y + 16))
        yy = box.y + 60
        for label, inp in inputs:
            screen.blit(small.render(label, True, (0,0,0)), (box.x + 16, yy+6))
            inp.draw(screen, small)
            yy += 46
        draw_button(screen, btn_ok, confirm_label, small)
        draw_button(screen, btn_cancel, 'Cancel', small, bg=(160,160,160))
        pygame.display.flip()


def gallery_delete(screen, dir_path, title='Manage Images'):
    """Overlay gallery to select and delete specific images in a directory.
    - Left click toggles selection
    - Buttons: Prev, Next, Delete Selected, Close
    """
    if not os.path.isdir(dir_path):
        return False
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if not files:
        return False

    W, H = screen.get_size()
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    box = pygame.Rect(40, 40, W - 80, H - 80)
    title_font = pygame.font.SysFont('segoeui', 20)
    small = pygame.font.SysFont('segoeui', 14)

    # Grid layout
    pad = 10
    thumb_w = 120
    thumb_h = 90
    cols = max(1, (box.w - 2*pad) // (thumb_w + pad))
    rows = max(1, (box.h - 100 - 2*pad) // (thumb_h + pad))  # leave space for buttons
    per_page = cols * rows
    page = 0
    selected = set()

    # Cache of thumbnails
    thumb_cache = {
        'paths': [],
        'surfs': [],
    }

    def load_page(p):
        start = p * per_page
        end = min(len(files), start + per_page)
        thumb_cache['paths'] = files[start:end]
        thumb_cache['surfs'] = []
        for fp in thumb_cache['paths']:
            try:
                img = pygame.image.load(fp)
                img = pygame.transform.smoothscale(img, (thumb_w, thumb_h))
                thumb_cache['surfs'].append(img.convert())
            except Exception:
                # placeholder
                surf = pygame.Surface((thumb_w, thumb_h))
                surf.fill((60,60,60))
                thumb_cache['surfs'].append(surf)

    load_page(page)

    # Buttons
    btn_h = 36
    btn_w = 140
    def buttons_rects():
        y = box.bottom - btn_h - 12
        x = box.left + 12
        bprev = pygame.Rect(x, y, 80, btn_h); x += 90
        bnext = pygame.Rect(x, y, 80, btn_h); x += 90
        bdel  = pygame.Rect(x, y, btn_w, btn_h); x += btn_w + 10
        bclose= pygame.Rect(x, y, 100, btn_h)
        return bprev, bnext, bdel, bclose

    def draw():
        screen.blit(overlay, (0, 0))
        pygame.draw.rect(screen, (245,248,255), box, border_radius=10)
        pygame.draw.rect(screen, (100,120,170), box, 2, border_radius=10)
        screen.blit(title_font.render(title, True, (20,40,90)), (box.x + 16, box.y + 12))
        # grid
        gx = box.x + pad
        gy = box.y + pad + 32
        idx = 0
        for r in range(rows):
            x = gx
            for c in range(cols):
                gidx = page*per_page + idx
                if gidx >= len(files):
                    break
                surf = thumb_cache['surfs'][idx]
                rect = pygame.Rect(x, gy, thumb_w, thumb_h)
                screen.blit(surf, rect.topleft)
                fp = thumb_cache['paths'][idx]
                if fp in selected:
                    pygame.draw.rect(screen, (220,60,60), rect, 3)
                else:
                    pygame.draw.rect(screen, (180,180,180), rect, 1)
                # filename
                name = os.path.basename(fp)
                label = small.render(name[:16], True, (30,30,30))
                screen.blit(label, (x, rect.bottom + 2))
                x += thumb_w + pad
                idx += 1
            gy += thumb_h + pad + 20
        # buttons
        bprev, bnext, bdel, bclose = buttons_rects()
        draw_button(screen, bprev, 'Prev', small)
        draw_button(screen, bnext, 'Next', small)
        draw_button(screen, bdel, 'Delete Selected', small, bg=(200,60,60))
        draw_button(screen, bclose, 'Close', small, bg=(160,160,160))
        # page info
        info = small.render(f"Page {page+1}/{max(1, math.ceil(len(files)/per_page))}  Selected: {len(selected)}", True, (30,30,30))
        screen.blit(info, (box.x + 16, box.bottom - btn_h - 48))
        pygame.display.flip()

    while True:
        draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                # click grid
                gx = box.x + pad
                gy = box.y + pad + 32
                idx = 0
                clicked = False
                for r in range(rows):
                    x = gx
                    for c in range(cols):
                        gidx = page*per_page + idx
                        if gidx >= len(files):
                            break
                        rect = pygame.Rect(x, gy, thumb_w, thumb_h)
                        if rect.collidepoint(pos):
                            fp = thumb_cache['paths'][idx]
                            if fp in selected:
                                selected.remove(fp)
                            else:
                                selected.add(fp)
                            clicked = True
                            break
                        x += thumb_w + pad
                        idx += 1
                    if clicked:
                        break
                    gy += thumb_h + pad + 20
                if clicked:
                    continue
                # buttons
                bprev, bnext, bdel, bclose = buttons_rects()
                if bprev.collidepoint(pos):
                    if page > 0:
                        page -= 1
                        load_page(page)
                elif bnext.collidepoint(pos):
                    if (page+1)*per_page < len(files):
                        page += 1
                        load_page(page)
                elif bdel.collidepoint(pos):
                    # delete selected
                    for fp in list(selected):
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
                    # refresh files
                    files[:] = [p for p in files if os.path.exists(p)]
                    selected.clear()
                    # reload current page (clamp page)
                    if page*per_page >= len(files) and page > 0:
                        page -= 1
                    load_page(page)
                elif bclose.collidepoint(pos):
                    return True


def build_auth_header(user: Optional[str], password: Optional[str], token: Optional[str]) -> Tuple[Optional[str], str]:
    """Return (Authorization header value or None, stream_url_suffix)"""
    if token:
        return None, ("?token=" + token)
    if user and password:
        up = f"{user}:{password}".encode('utf-8')
        return "Basic " + base64.b64encode(up).decode('ascii'), ""
    return None, ""


def send_ptz(base_http: str, auth_header: Optional[str], token_suffix: str, direction: str):
    try:
        url = f"{base_http}/action?go={direction}{('&token='+token_suffix.split('=')[1]) if token_suffix else ''}"
        headers = {'Authorization': auth_header} if auth_header else {}
        requests.get(url, headers=headers, timeout=2)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--host', help='ESP32-CAM host or host:port (port 80)')
    parser.add_argument('--user', help='Basic auth username')
    parser.add_argument('--password', help='Basic auth password')
    parser.add_argument('--token', help='Token (Base64 of user:pass) for query param auth')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH)
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
    # (removed --download-ssd)
    cli_args, _ = parser.parse_known_args()

    ensure_dirs()

    # Load saved config if present
    cfg_path = os.path.join('ai', 'config.json')
    saved = {
        'host': cli_args.host or '',
        'user': cli_args.user or '',
        'password': cli_args.password or '',
        'token': cli_args.token or '',
        'width': cli_args.width,
        'height': cli_args.height,
    }
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                disk = json.load(f)
            saved.update({k: disk.get(k, saved[k]) for k in saved})
    except Exception:
        pass

    # (no CLI download actions)

    # If missing host (or user input desired), show UI config screen
    if not saved['host']:
        saved = config_ui(defaults=saved)
        if saved is None:
            return
    # Persist config
    try:
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(saved, f, indent=2)
    except Exception:
        pass

    host = saved['host']
    width = int(saved.get('width') or DEFAULT_WIDTH)
    height = int(saved.get('height') or DEFAULT_HEIGHT)
    user = saved.get('user') or None
    password = saved.get('password') or None
    token = saved.get('token') or None

    base_http = f"http://{host}"
    auth_header, token_suffix = build_auth_header(user, password, token)
    stream_url = f"http://{host.split(':')[0]}:81/stream{token_suffix}"

    # Start stream
    stream = MJPEGStream(stream_url, auth_header=auth_header).start()

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ESP32-CAM AI Viewer')
    font = pygame.font.SysFont('consolas', 16)
    ui_font = pygame.font.SysFont('segoeui', 16)

    # Init detectors
    facedb = FaceDB()
    facedb.load_and_train()
    # Ensure YOLO model exists (attempt auto-download once if missing)
    yolo_model_path = os.path.join('ai','models','yolo.onnx')
    detector_yolo = YOLODetector(yolo_model_path)
    det_mode = 'yolo' if detector_yolo.available() else 'off'
    if det_mode == 'off':
        try:
            print('[YOLO] Model missing, attempting auto-download...')
            if download_yolo_model(yolo_model_path):
                detector_yolo = YOLODetector(yolo_model_path)
                det_mode = 'yolo' if detector_yolo.available() else 'off'
                print('[YOLO] Auto-download complete:', 'OK' if det_mode=='yolo' else 'FAILED')
        except Exception as e:
            print('[YOLO] Auto-download error:', e)
    pets = PetsDB()
    pets.load()

    # Tracking
    tracker = None
    track_bbox = None  # x,y,w,h
    track_mode = 0  # 0=off, 1=person, 2=face-known, 3=dog, 4=cat
    last_ptz = 0.0

    def start_tracker(frame, bbox):
        nonlocal tracker, track_bbox
        try:
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                tracker = cv2.legacy.TrackerCSRT_create()
            elif hasattr(cv2, 'TrackerCSRT_create'):
                tracker = cv2.TrackerCSRT_create()
            elif hasattr(cv2, 'TrackerKCF_create'):
                tracker = cv2.TrackerKCF_create()
            else:
                tracker = None
        except Exception:
            tracker = None
        if tracker is not None:
            tracker.init(frame, tuple(bbox))
            track_bbox = bbox

    running = True
    enroll_name_input = ''
    enroll_bbox = None
    last_dog_bbox = None
    last_cat_bbox = None
    # Face sample collection state
    collect = None  # {'name': str, 'n': int, 'collected': int, 'interval': float, 'last': float}
    # Pet sample collection state
    pet_collect = None  # {'name': str, 'species': 'dog'|'cat', 'n': int, 'collected': int, 'interval': float, 'last': float}

    # UI visibility toggle (toolbar)
    show_ui = True

    # Overlay UI helpers
    def layout_buttons():
        W, H = screen.get_size()
        pad = 8
        btn_w = 130
        btn_h = 32
        y = H - btn_h - pad
        x = pad
        buttons = {}
        def add(name, label):
            nonlocal x, y
            # wrap to new row if exceeding width
            if x + btn_w > W - pad:
                x = pad
                y -= (btn_h + pad)
            rect = pygame.Rect(x, y, btn_w, btn_h)
            buttons[name] = (rect, label)
            x += btn_w + pad
        add('ui', 'UI')
        if show_ui:
            add('settings', 'Settings')
            add('mode', f"Mode:{track_mode}")
            add('detector', f"Det:{det_mode.upper()}")
            add('enroll_face', 'Enroll Face')
            add('collect', 'Collect Samples')
        add('enroll_pet', 'Enroll Pet')
        add('collect_pet', 'Collect Pet')
        add('manage_faces', 'Manage Faces')
        add('manage_pets', 'Manage Pets')
        add('retrain', 'Retrain Faces')
        add('download_yolo', 'Download YOLO')
        add('debug', 'Debug Dump')
        add('quit', 'Quit')
        return buttons

    toast_msg = ''
    toast_until = 0
    def toast(msg, dur=2.0):
        nonlocal toast_msg, toast_until
        toast_msg = msg
        toast_until = time.time() + dur

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_t:
                    track_mode = (track_mode + 1) % 5
                elif event.key == pygame.K_r:
                    facedb.load_and_train()
                elif event.key == pygame.K_e:
                    # Trigger enroll on the last detected face bbox
                    if enroll_bbox is not None and enroll_name_input:
                        # Will be handled after frame is processed
                        pass
                elif event.key == pygame.K_p:
                    pass
            elif event.type == pygame.MOUSEBUTTONDOWN:
                buttons = layout_buttons()
                for name, (rect, label) in buttons.items():
                    if rect.collidepoint(event.pos):
                        if name == 'quit':
                            running = False
                        elif name == 'ui':
                            show_ui = not show_ui
                            toast('UI: %s' % ('shown' if show_ui else 'hidden'))
                        elif name == 'retrain':
                            facedb.load_and_train()
                            toast('Faces retrained')
                        elif name == 'download_yolo':
                            if download_yolo_model(os.path.join('ai','models','yolo.onnx')):
                                detector_yolo.__init__(os.path.join('ai','models','yolo.onnx'))
                                det_mode = 'yolo'
                                toast('YOLO downloaded')
                            else:
                                toast('YOLO download failed')
                        elif name == 'mode':
                            track_mode = (track_mode + 1) % 5
                        elif name == 'detector':
                            det_mode = 'off' if det_mode == 'yolo' else 'yolo'
                        elif name == 'debug':
                            # Rebuild model, dump enrollment, and verify
                            facedb.load_and_train()
                            report = facedb.debug_dump('ai/face_dump.txt')
                            per, overall = facedb.debug_verify_training()
                            print('[DEBUG] Enrollment report saved to:', report)
                            print('[DEBUG] Overall training accuracy: %.2f%% (%d/%d)' % (overall['acc'], overall['correct'], overall['total']))
                            for k,v in per.items():
                                print(' - %-16s: %5.1f%% (%d/%d)' % (k, v['acc'], v['correct'], v['total']))
                            toast('Debug dump saved (console)')
                        elif name == 'settings':
                            cfg = config_ui({'host': host, 'user': user or '', 'password': password or '', 'token': token or '', 'width': width, 'height': height})
                            if cfg:
                                try:
                                    with open(os.path.join('ai','config.json'),'w',encoding='utf-8') as f:
                                        json.dump(cfg, f, indent=2)
                                except Exception:
                                    pass
                                host = cfg['host']; user = cfg.get('user') or None; password = cfg.get('password') or None; token = cfg.get('token') or None
                                width = int(cfg.get('width') or width); height = int(cfg.get('height') or height)
                                base_http = f"http://{host}"
                                auth_header, token_suffix = build_auth_header(user, password, token)
                                stream.stop()
                                stream = MJPEGStream(f"http://{host.split(':')[0]}:81/stream{token_suffix}", auth_header=auth_header).start()
                                screen = pygame.display.set_mode((width, height))
                                toast('Settings applied')
                        elif name == 'enroll_face':
                            if enroll_bbox is None:
                                toast('No face detected')
                            else:
                                ret = modal_text_input(screen, 'Enroll Face', [{'label':'Name','value':'','masked':False}], confirm_label='Enroll')
                                if ret:
                                    (x, y, w, h) = enroll_bbox
                                    gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    if facedb.enroll(gray_now, x, y, w, h, ret[0]):
                                        facedb.load_and_train()
                                        toast(f"Enrolled {ret[0]}")
                        elif name == 'collect':
                            if enroll_bbox is None:
                                toast('No face detected')
                            else:
                                # If already collecting, cancel
                                if collect is not None:
                                    collect = None
                                    toast('Collection cancelled')
                                else:
                                    ret = modal_text_input(screen, 'Collect N Samples', [
                                        {'label':'Name','value':'','masked':False},
                                        {'label':'Samples (e.g., 20)','value':'20','masked':False}
                                    ], confirm_label='Start')
                                    if ret:
                                        try:
                                            n = max(1, int(ret[1]))
                                        except Exception:
                                            n = 20
                                        collect = {'name': ret[0].strip() or 'person', 'n': n, 'collected': 0, 'interval': 0.2, 'last': 0.0}
                                        toast(f"Collecting {n} samples for {collect['name']}")
                        elif name == 'enroll_pet':
                            bb = last_dog_bbox or last_cat_bbox
                            if bb is None:
                                toast('No pet detected')
                            else:
                                ret = modal_text_input(screen, 'Enroll Pet', [
                                    {'label':'Name','value':'','masked':False},
                                    {'label':'Species (dog/cat)','value':'dog','masked':False}
                                ], confirm_label='Enroll')
                                if ret:
                                    (pname, pspecies) = ret
                                    (x, y, w, h) = bb
                                    roi = frame[max(0,y):y+h, max(0,x):x+w]
                                    if pets.enroll(roi, pname, pspecies):
                                        pets.load()
                                        toast(f"Enrolled {pspecies}:{pname}")
                        elif name == 'collect_pet':
                            # Toggle pet collection
                            if pet_collect is not None:
                                pet_collect = None
                                toast('Pet collection cancelled')
                            else:
                                ret = modal_text_input(screen, 'Collect Pet Samples', [
                                    {'label':'Name','value':'','masked':False},
                                    {'label':'Species (dog/cat)','value':'dog','masked':False},
                                    {'label':'Samples (e.g., 40)','value':'40','masked':False}
                                ], confirm_label='Start')
                                if ret:
                                    try:
                                        n = max(1, int(ret[2]))
                                    except Exception:
                                        n = 40
                                    sp = 'dog' if ret[1].lower().startswith('dog') else 'cat'
                                    pet_collect = {'name': ret[0].strip() or 'pet', 'species': sp, 'n': n, 'collected': 0, 'interval': 0.25, 'last': 0.0}
                                    toast(f"Collecting {n} {sp} samples for {pet_collect['name']}")
                        elif name == 'manage_faces':
                            ret = modal_text_input(screen, 'Manage Faces', [
                                {'label':'Name','value':'','masked':False}
                            ], confirm_label='Open')
                            if ret:
                                fname = ret[0].strip()
                                target = os.path.join('ai','data','faces', fname)
                                if gallery_delete(screen, target, title=f'Faces: {fname}'):
                                    facedb.load_and_train()
                        elif name == 'manage_pets':
                            ret = modal_text_input(screen, 'Manage Pets', [
                                {'label':'Species (dog/cat)','value':'dog','masked':False},
                                {'label':'Name','value':'','masked':False}
                            ], confirm_label='Open')
                            if ret:
                                sp = 'dogs' if ret[0].lower().startswith('dog') else 'cats'
                                pname = ret[1].strip()
                                base = os.path.join('ai','data','pets', sp, pname)
                                if gallery_delete(screen, base, title=f'{sp[:-1].title()}: {pname}'):
                                    pets.load()

        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = cv2.resize(frame, (width, height))
        out = frame.copy()

        # Object detection (if model available)
        if det_mode == 'yolo' and detector_yolo.available():
            dets = detector_yolo.detect(frame)
        else:
            dets = []

        # Face detection/recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedb.detect_faces(gray)
        enroll_bbox = None

        known_face_bboxes = []
        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (160, 160))
            name, score = facedb.recognize(gray, roi)
            color = (0, 255, 0) if name != 'unknown' else (0, 165, 255)
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
            cv2.putText(out, f"{name} {score:.2f}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if name != 'unknown':
                known_face_bboxes.append((x, y, w, h))
            enroll_bbox = (x, y, w, h)

        # Handle collection of multiple samples
        if collect is not None and enroll_bbox is not None:
            now_ts = time.time()
            if now_ts - collect['last'] >= collect['interval']:
                (x, y, w, h) = enroll_bbox
                gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if facedb.enroll(gray_now, x, y, w, h, collect['name']):
                    collect['collected'] += 1
                    collect['last'] = now_ts
                if collect['collected'] >= collect['n']:
                    facedb.load_and_train()
                    toast(f"Collected {collect['collected']} samples for {collect['name']}")
                    collect = None

        # Handle pet collection of multiple samples
        if pet_collect is not None:
            now_ts = time.time()
            if now_ts - pet_collect['last'] >= pet_collect['interval']:
                # pick bbox for requested species from current dets
                species = pet_collect['species']
                candidates = [(x,y,w,h) for (cls,conf,x,y,w,h) in dets if cls == species]
                if candidates:
                    # choose largest
                    bx = max(candidates, key=lambda b: b[2]*b[3])
                    (x,y,w,h) = bx
                    roi = frame[max(0,y):y+h, max(0,x):x+w]
                    if pets.enroll(roi, pet_collect['name'], species):
                        pet_collect['collected'] += 1
                        pet_collect['last'] = now_ts
                if pet_collect['collected'] >= pet_collect['n']:
                    pets.load()
                    toast(f"Collected {pet_collect['collected']} for {species}:{pet_collect['name']}")
                    pet_collect = None

        # Draw object detections
        dog_name = 'unknown'
        for (cls, conf, x, y, w, h) in dets:
            if cls not in ('person', 'dog', 'cat'):
                continue
            color = (255, 0, 0) if cls == 'person' else (0, 0, 255) if cls == 'dog' else (255, 0, 255)
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
            label = f"{cls} {conf:.2f}"
            if cls in ('dog', 'cat'):
                roi = frame[max(0,y):max(0,y)+max(1,h), max(0,x):max(0,x)+max(1,w)]
                name, score = pets.recognize(roi, cls)
                if cls == 'dog':
                    dog_name = name
                    last_dog_bbox = (x, y, w, h)
                else:
                    last_cat_bbox = (x, y, w, h)
                if name != 'unknown':
                    label = f"{cls}:{name} {conf:.2f}/{score:.2f}"
            cv2.putText(out, label, (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Tracking logic: pick target based on mode
        target_bbox = None
        if track_mode == 1:  # person
            persons = [(x, y, w, h) for (cls, conf, x, y, w, h) in dets if cls == 'person']
            if persons:
                target_bbox = max(persons, key=lambda b: b[2]*b[3])
        elif track_mode == 2:  # known face
            if known_face_bboxes:
                target_bbox = max(known_face_bboxes, key=lambda b: b[2]*b[3])
        elif track_mode == 3:  # dog
            dogs = [(x, y, w, h) for (cls, conf, x, y, w, h) in dets if cls == 'dog']
            if dogs:
                target_bbox = max(dogs, key=lambda b: b[2]*b[3])
        elif track_mode == 4:  # cat
            cats = [(x, y, w, h) for (cls, conf, x, y, w, h) in dets if cls == 'cat']
            if cats:
                target_bbox = max(cats, key=lambda b: b[2]*b[3])

        # Update or start tracker
        if target_bbox is not None:
            if tracker is None:
                start_tracker(frame, target_bbox)
            else:
                ok, bbox = tracker.update(frame)
                if ok:
                    track_bbox = tuple(map(int, bbox))
                else:
                    tracker = None
                    start_tracker(frame, target_bbox)

        # PTZ command to follow tracked bbox
        if track_bbox is not None:
            (x, y, w, h) = track_bbox
            cx = x + w/2
            cy = y + h/2
            fx = frame.shape[1] / 2
            fy = frame.shape[0] / 2
            dx = cx - fx
            dy = cy - fy
            cv2.circle(out, (int(cx), int(cy)), 4, (0, 255, 255), -1)
            # deadzone
            thresh_x = frame.shape[1] * 0.05
            thresh_y = frame.shape[0] * 0.05
            now = time.time()
            if now - last_ptz > 0.2:  # rate-limit
                if dx > thresh_x:
                    send_ptz(base_http, auth_header, token_suffix, 'left')
                    last_ptz = now
                elif dx < -thresh_x:
                    send_ptz(base_http, auth_header, token_suffix, 'right')
                    last_ptz = now
                if dy > thresh_y:
                    send_ptz(base_http, auth_header, token_suffix, 'down')
                    last_ptz = now
                elif dy < -thresh_y:
                    send_ptz(base_http, auth_header, token_suffix, 'up')
                    last_ptz = now

        # Old console-based enrollment retained as fallback via hotkeys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_e] and enroll_bbox is not None:
            ret = modal_text_input(screen, 'Enroll Face', [{'label':'Name','value':'','masked':False}], confirm_label='Enroll')
            if ret:
                (x, y, w, h) = enroll_bbox
                gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if facedb.enroll(gray_now, x, y, w, h, ret[0]):
                    facedb.load_and_train()
                    toast(f"Enrolled {ret[0]}")
        if keys[pygame.K_p] and (last_dog_bbox is not None or last_cat_bbox is not None):
            ret = modal_text_input(screen, 'Enroll Pet', [
                {'label':'Name','value':'','masked':False},
                {'label':'Species (dog/cat)','value':'dog','masked':False}
            ], confirm_label='Enroll')
            if ret:
                (pname, pspecies) = ret
                bb = last_dog_bbox if pspecies.lower().startswith('dog') or last_cat_bbox is None else last_cat_bbox
                (x, y, w, h) = bb
                roi = frame[max(0,y):y+h, max(0,x):x+w]
                if pets.enroll(roi, pname, pspecies):
                    pets.load()
                    toast(f"Enrolled {pspecies}:{pname}")

        # Pygame blit
        surf = pygame.image.frombuffer(out.tobytes(), out.shape[1::-1], 'BGR')
        screen.blit(surf, (0, 0))
        # HUD
        det_label = 'YOLO' if det_mode=='yolo' and detector_yolo.available() else 'noDet'
        text = f"mode:{track_mode} {det_label} faces:{len(faces)} dog:{dog_name}"
        screen.blit(font.render(text, True, (255, 255, 255)), (8, 8))

        # Collection progress HUD
        if collect is not None:
            prog = f"Collecting {collect['collected']}/{collect['n']} for {collect['name']} (click Collect to cancel)"
            msg = ui_font.render(prog, True, (255,255,255))
            bg_rect = pygame.Rect(8, 24, msg.get_width()+14, msg.get_height()+10)
            pygame.draw.rect(screen, (0,0,0), bg_rect, border_radius=8)
            screen.blit(msg, (bg_rect.x + 7, bg_rect.y + 5))

        # Draw overlay buttons
        buttons = layout_buttons()
        for name, (rect, label) in buttons.items():
            draw_button(screen, rect, label, ui_font)

        # Toast message and collection progress
        if time.time() < toast_until:
            msg = ui_font.render(toast_msg, True, (255,255,255))
            bg_rect = pygame.Rect(8, 40, msg.get_width()+14, msg.get_height()+10)
            pygame.draw.rect(screen, (0,0,0), bg_rect, border_radius=8)
            screen.blit(msg, (bg_rect.x + 7, bg_rect.y + 5))
        if collect is not None:
            prog = ui_font.render(f"Face: {collect['collected']}/{collect['n']} {collect['name']}", True, (255,255,255))
            screen.blit(prog, (8, 64))
        if pet_collect is not None:
            prog2 = ui_font.render(f"{pet_collect['species']}: {pet_collect['collected']}/{pet_collect['n']} {pet_collect['name']}", True, (255,255,255))
            screen.blit(prog2, (8, 84))

        pygame.display.flip()

    stream.stop()
    pygame.quit()


if __name__ == '__main__':
    main()
