#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools

# ------------------------------
# Simple AI helpers (YOLO, FaceDB, PetsDB)
# ------------------------------

class YOLODetector:
    """Lightweight YOLO (ONNX) wrapper for COCO classes.
    Uses OpenCV DNN and letterbox preprocessing.
    """
    COCO = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant',
        'bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
        'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
        'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli',
        'carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet',
        'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
        'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]
    def __init__(self, model_path='ai/models/yolo.onnx', input_size=640, conf=0.35, iou=0.45):
        self.net = None
        self.size = input_size
        self.conf = conf
        self.iou = iou
        if os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception as e:
                print('[YOLO] load failed:', e)

    def available(self):
        return self.net is not None

    def _letterbox(self, img, new=640):
        h,w = img.shape[:2]
        r = min(new/h, new/w)
        nh, nw = int(h*r), int(w*r)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((new,new,3), 114, np.uint8)
        top = (new-nh)//2; left=(new-nw)//2
        canvas[top:top+nh, left:left+nw] = resized
        return canvas, r, left, top

    def detect(self, frame_bgr):
        if self.net is None:
            return []
        img, r, dx, dy = self._letterbox(frame_bgr, self.size)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (self.size,self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        out = np.squeeze(out)
        if out.ndim == 2 and out.shape[0] in (84,85):
            out = out.T
        elif out.ndim == 3:
            o = out[0]
            out = o.T if o.shape[0] in (84,85) else o
        H,W = frame_bgr.shape[:2]
        boxes=[]; scores=[]; classes=[]
        for det in out:
            cx,cy,w,h = det[:4]
            if det.shape[0] >= 85:
                obj = float(det[4]); cls_scores = det[5:]
                c = int(np.argmax(cls_scores)); conf = obj*float(cls_scores[c])
            else:
                cls_scores = det[4:]; c = int(np.argmax(cls_scores)); conf=float(cls_scores[c])
            if conf < self.conf: continue
            x1 = int(max(0, min(W-1, (cx-w/2 - dx)/r)))
            y1 = int(max(0, min(H-1, (cy-h/2 - dy)/r)))
            x2 = int(max(0, min(W-1, (cx+w/2 - dx)/r)))
            y2 = int(max(0, min(H-1, (cy+h/2 - dy)/r)))
            boxes.append([x1,y1,x2-x1,y2-y1]); scores.append(conf); classes.append(c)
        if not boxes: return []
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
        idxs = np.array(idxs).reshape(-1) if isinstance(idxs,(list,tuple,np.ndarray)) else np.array([])
        dets=[]
        for i in idxs:
            name = self.COCO[classes[i]] if 0<=classes[i]<len(self.COCO) else str(classes[i])
            x,y,w,h = boxes[i]
            dets.append((name, float(scores[i]), x,y,w,h))
        return dets


class FaceDB:
    """Face recognition store.
    - Trains LBPH if contrib available; falls back to ORB matching.
    - Persists samples as images on disk under ai/data/faces/<name>.
    """
    def __init__(self, base='ai/data/faces'):
        self.base = base
        os.makedirs(self.base, exist_ok=True)
        self.size = (160,160)
        self.model = None
        try:
            self.model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        except Exception:
            self.model = None
        self.labels=[]
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.db_descs={}
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load(self):
        samples=[]; labels=[]; label_map={}; lid=0
        for name in sorted(os.listdir(self.base)):
            p=os.path.join(self.base,name)
            if not os.path.isdir(p): continue
            label_map[name]=lid
            for fn in os.listdir(p):
                if not fn.lower().endswith(('.jpg','.jpeg','.png')): continue
                img=cv2.imread(os.path.join(p,fn),cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                samples.append(cv2.resize(img,self.size)); labels.append(lid)
            lid+=1
        self.labels=[None]*lid
        for n,i in label_map.items(): self.labels[i]=n
        # LBPH
        if samples and self.model is not None:
            try:
                self.model.train(samples, np.array(labels))
            except Exception:
                self.model=None
        # ORB fallback DB
        self.db_descs={}
        for name in sorted(os.listdir(self.base)):
            p=os.path.join(self.base,name)
            if not os.path.isdir(p): continue
            ds=[]
            for fn in os.listdir(p):
                if not fn.lower().endswith(('.jpg','.jpeg','.png')): continue
                img=cv2.imread(os.path.join(p,fn),cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img=cv2.resize(img,self.size)
                kp,d=self.orb.detectAndCompute(img,None)
                if d is not None: ds.append(d)
            if ds: self.db_descs[name]=ds

    def detect_faces(self, gray):
        try:
            eq=cv2.createCLAHE(2.0,(8,8)).apply(gray)
        except Exception:
            eq=cv2.equalizeHist(gray)
        minsz=max(40,int(0.12*min(gray.shape[:2])))
        faces=self.cascade.detectMultiScale(eq,1.1,4,minSize=(minsz,minsz))
        if len(faces)==0:
            faces=self.cascade.detectMultiScale(eq,1.05,3,minSize=(minsz,minsz))
        return faces

    def recognize_roi(self, gray_roi):
        roi=cv2.resize(gray_roi,self.size)
        if self.model is not None:
            try:
                pred,dist=self.model.predict(roi)
                if 0<=pred<len(self.labels) and dist<=95.0:
                    return self.labels[pred], max(0.0,min(1.0,1.0-(dist/95.0)))
            except Exception:
                pass
        # ORB fallback
        kp,d=self.orb.detectAndCompute(roi,None)
        if d is None or not self.db_descs: return 'unknown',0.0
        best='unknown'; bs=0.0
        for name,ds in self.db_descs.items():
            tot=0; good=0
            for dbd in ds:
                m=self.bf.match(d,dbd); tot+=len(m); good+=sum(1 for mm in m if mm.distance<40)
            if tot==0: continue
            sc=good/float(tot)
            if sc>bs: bs=sc; best=name
        return (best if bs>0.12 else 'unknown'), bs

    def enroll(self, frame_gray, x,y,w,h, name:str):
        name=''.join(c for c in name.strip() if c.isalnum() or c in ('_','-')) or 'person'
        p=os.path.join(self.base,name); os.makedirs(p,exist_ok=True)
        roi=cv2.resize(frame_gray[y:y+h,x:x+w], self.size)
        now=time.time(); ts=int((now - int(now))*1000); ds=time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
        base=f'{ds}_{ts:03d}'
        cv2.imwrite(os.path.join(p,f'{base}.jpg'), roi)
        try: cv2.imwrite(os.path.join(p,f'{base}_flip.jpg'), cv2.flip(roi,1))
        except Exception: pass
        return True


class PetsDB:
    """Pet recognition store (dogs/cats) using ORB descriptors.
    - Persists samples as grayscale images under ai/data/pets/{dogs,cats}/<name>.
    """
    def __init__(self, base='ai/data/pets'):
        self.base=base; os.makedirs(os.path.join(base,'dogs'),exist_ok=True); os.makedirs(os.path.join(base,'cats'),exist_ok=True)
        self.orb=cv2.ORB_create(nfeatures=1000); self.bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.db={'dogs':{},'cats':{}}

    def load(self):
        self.db={'dogs':{},'cats':{}}
        for sp in ('dogs','cats'):
            root=os.path.join(self.base,sp)
            if not os.path.isdir(root): continue
            for name in os.listdir(root):
                p=os.path.join(root,name)
                if not os.path.isdir(p): continue
                ds=[]
                for fn in os.listdir(p):
                    if not fn.lower().endswith(('.jpg','.jpeg','.png')): continue
                    img=cv2.imread(os.path.join(p,fn),cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    img=cv2.resize(img,(320,320))
                    kp,d=self.orb.detectAndCompute(img,None)
                    if d is not None: ds.append(d)
                if ds: self.db[sp][name]=ds

    def enroll(self, roi_bgr, name:str, species:str):
        sp='dogs' if species.lower().startswith('dog') else 'cats'
        name=''.join(c for c in name.strip() if c.isalnum() or c in ('_','-')) or 'pet'
        p=os.path.join(self.base,sp,name); os.makedirs(p,exist_ok=True)
        gray=cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY); gray=cv2.resize(gray,(320,320))
        now=time.time(); ts=int((now - int(now))*1000); ds=time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
        fn=f'{ds}_{ts:03d}.jpg'
        cv2.imwrite(os.path.join(p,fn), gray); return True

    def recognize(self, roi_bgr, species:str):
        sp='dogs' if species=='dog' else 'cats'
        if not self.db[sp]: return 'unknown',0.0
        gray=cv2.cvtColor(roi_bgr,cv2.COLOR_BGR2GRAY); gray=cv2.resize(gray,(320,320))
        kp,d=self.orb.detectAndCompute(gray,None)
        if d is None: return 'unknown',0.0
        best='unknown'; bs=0.0
        for name,ds in self.db[sp].items():
            tot=0; good=0
            for dbd in ds:
                m=self.bf.match(d,dbd); tot+=len(m); good+=sum(1 for mm in m if mm.distance<40)
            if tot==0: continue
            sc=good/float(tot)
            if sc>bs: bs=sc; best=name
        return (best if bs>0.12 else 'unknown'), bs


@dataclass
class CameraConfig:
    name: str
    host: str                 # ip[:port] for port 80
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None  # Base64 of user:pass

    def stream_url(self) -> str:
        base = self.host.split(':')[0]
        suffix = f"?token={self.token}" if self.token else ""
        return f"http://{base}:81/stream{suffix}"

    def stream_url_with_creds(self) -> Optional[str]:
        """Fallback URL form that embeds user:pass in the authority.
        Some ESP32 firmwares only honor this style for the :81 stream.
        """
        if not (self.user and self.password):
            return None
        base = self.host.split(':')[0]
        return f"http://{self.user}:{self.password}@{base}:81/stream"

    def auth_header(self) -> Optional[str]:
        if self.token:
            return None
        if self.user and self.password:
            import base64
            up = f"{self.user}:{self.password}".encode('utf-8')
            return "Basic " + base64.b64encode(up).decode('ascii')
        return None


class CameraStreamThread(QtCore.QThread):
    frameReady = QtCore.Signal(np.ndarray, float)  # (bgr_frame, timestamp)

    def __init__(self, cfg: CameraConfig, parent=None, prebuffer_seconds: float = 5.0):
        super().__init__(parent)
        self.cfg = cfg
        self._stop = threading.Event()
        self._session = None
        self._resp = None
        self._buf = bytearray()
        self._boundary = b"--frame"
        self.prebuffer: Deque[Tuple[np.ndarray, float]] = deque(maxlen=int(prebuffer_seconds * 20))  # assume ~20fps cap

    def stop(self):
        self._stop.set()
        try:
            if self._resp is not None:
                self._resp.close()
        except Exception:
            pass

    def run(self):
        headers = {}
        # Prefer requests' BasicAuth when user/pass provided
        req_auth = None
        if not self.cfg.token and self.cfg.user and self.cfg.password:
            try:
                from requests.auth import HTTPBasicAuth
                req_auth = HTTPBasicAuth(self.cfg.user, self.cfg.password)
            except Exception:
                # fallback to manual header
                ah = self.cfg.auth_header()
                if ah:
                    headers['Authorization'] = ah
        self._session = requests.Session()
        # Attempt 1: standard URL (+token if provided) with auth/header
        try:
            self._resp = self._session.get(self.cfg.stream_url(), headers=headers, auth=req_auth, stream=True, timeout=10)
            self._resp.raise_for_status()
        except Exception as e1:
            # Attempt 2: URL-embedded credentials (user:pass@host)
            tried2 = False
            url2 = self.cfg.stream_url_with_creds()
            if url2:
                try:
                    self._resp = self._session.get(url2, stream=True, timeout=10)
                    self._resp.raise_for_status()
                    tried2 = True
                except Exception:
                    tried2 = True
            # Attempt 3: user/password as query params (some forks)
            tried3 = False
            if not (self.cfg.token) and (self.cfg.user and self.cfg.password):
                base = self.cfg.host.split(':')[0]
                url3 = f"http://{base}:81/stream?user={self.cfg.user}&password={self.cfg.password}"
                try:
                    self._resp = self._session.get(url3, stream=True, timeout=10)
                    self._resp.raise_for_status()
                    tried3 = True
                except Exception:
                    tried3 = True
            if not (tried2 or tried3):
                # No other forms attempted; report original error
                print(f"[Stream] Failed to connect {self.cfg.name}: {e1}")
                return
            # If both fallbacks failed, report combined failure
            if (self._resp is None) or (getattr(self._resp, 'status_code', 0) >= 400):
                print(f"[Stream] Failed to connect {self.cfg.name}: {e1}")
                return

        for chunk in self._resp.iter_content(chunk_size=8192):
            if self._stop.is_set():
                break
            if not chunk:
                continue
            self._buf += chunk
            while True:
                hdr_end = self._buf.find(b"\r\n\r\n")
                if hdr_end == -1:
                    break
                headers_blob = self._buf[:hdr_end].decode('latin1', errors='ignore').lower()
                cl_idx = headers_blob.find('content-length:')
                if cl_idx == -1:
                    # resync to boundary
                    bidx = self._buf.find(self._boundary)
                    self._buf = self._buf[bidx:] if bidx != -1 else self._buf[hdr_end+4:]
                    continue
                try:
                    cl_line = headers_blob[cl_idx:].split('\r\n', 1)[0]
                    length = int(cl_line.split(':', 1)[1].strip())
                except Exception:
                    bidx = self._buf.find(self._boundary)
                    self._buf = self._buf[bidx:] if bidx != -1 else self._buf[hdr_end+4:]
                    continue
                start = hdr_end + 4
                if len(self._buf) < start + length:
                    break
                jpg = self._buf[start:start+length]
                tail = self._buf[start+length:]
                bmark = b"\r\n--frame\r\n"
                if tail.startswith(bmark):
                    self._buf = bytearray(tail[len(bmark):])
                else:
                    bpos = tail.find(bmark)
                    self._buf = bytearray(tail[bpos+len(bmark):] if bpos >= 0 else b"")

                # decode
                arr = np.frombuffer(jpg, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                ts = time.time()
                self.prebuffer.append((frame, ts))
                self.frameReady.emit(frame, ts)


class CameraWidget(QtWidgets.QWidget):
    closed = QtCore.Signal(dict)
    def __init__(self, cfg: CameraConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle(cfg.name)
        self.label = QtWidgets.QLabel('Connecting…')
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(320, 240)
        self.label.setStyleSheet('background:#000; color:#9cf;')

        # Recording state
        self.recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.out_dir = os.path.join('ai', 'recordings')
        os.makedirs(self.out_dir, exist_ok=True)
        self.target_fps = 20.0

        # Controls (local toolbar)
        btns = QtWidgets.QToolBar()
        act_start = btns.addAction('Start')
        act_stop = btns.addAction('Stop')
        btns.addSeparator()
        act_rec = btns.addAction('Start Rec')
        act_stoprec = btns.addAction('Stop Rec')

        act_start.triggered.connect(self.start_stream)
        act_stop.triggered.connect(self.stop_stream)
        act_rec.triggered.connect(self.start_recording)
        act_stoprec.triggered.connect(self.stop_recording)

        # Toolbar toggles
        self.chk_yolo = QtWidgets.QCheckBox('YOLO'); self.chk_yolo.setChecked(True)
        self.chk_face = QtWidgets.QCheckBox('Face'); self.chk_face.setChecked(True)
        btns.addWidget(self.chk_yolo)
        btns.addWidget(self.chk_face)

        self.lbl_status = QtWidgets.QLabel('Ready')

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4)
        layout.addWidget(btns)
        layout.addWidget(self.label, 1)

        # Stream thread
        self.thr = CameraStreamThread(cfg)
        self.thr.frameReady.connect(self.on_frame)
        self._last_ts = None
        # AI components per camera
        self.yolo = YOLODetector()
        self.facedb = FaceDB(); self.facedb.load()
        self.pets = PetsDB(); self.pets.load()
        self.last_face_bbox=None; self.last_pet_bbox=None
        self.collect_face=None  # {'name':str,'n':int,'col':int,'last':float}
        self.collect_pet=None   # {'name':str,'sp':str,'n':int,'col':int,'last':float}

        # Shared frame and results (thread-safe)
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._last_dets = []      # [(cls,conf,x,y,w,h)] in original coords
        self._last_faces = []     # [(name,score,x,y,w,h)] in original coords
        self._last_overlay_ts = 0

        # Simple tracker for object permanence + aiming
        self.tracker = SimpleTracker(ttl=1.0)  # seconds before track expires
        # Detection worker thread (decouples heavy CV from UI), interval overridable by settings
        self.det_thr = DetectionThread(self)
        self.det_thr.resultsReady.connect(self.on_results)
        self.det_thr.start()
        # PTZ aiming timer
        self._aim_timer = QtCore.QTimer(self)
        self._aim_timer.setInterval(300)
        self._aim_timer.timeout.connect(self.aim_at_target)
        self._aim_timer.start()

        # hidden inputs used by MainWindow to pass parameters
        self.ed_name = QtWidgets.QLineEdit()
        self.cmb_species = QtWidgets.QComboBox(); self.cmb_species.addItems(['dog','cat'])

    def start_stream(self):
        if not self.thr.isRunning():
            self.thr._stop.clear()
            self.thr.start()

    def stop_stream(self):
        if self.thr.isRunning():
            self.thr.stop()
            # Give the worker time to unwind network loop
            if not self.thr.wait(3000):
                try:
                    # As a last resort on shutdown
                    self.thr.terminate()
                    self.thr.wait(500)
                except Exception:
                    pass

    def apply_config(self, new_cfg: CameraConfig):
        # Stop current stream thread and swap config
        try:
            self.stop_stream()
        except Exception:
            pass
        self.cfg = new_cfg
        self.setWindowTitle(new_cfg.name)
        # Rebuild stream thread to ensure fresh session/auth
        try:
            if hasattr(self.thr, 'frameReady'):
                try:
                    self.thr.frameReady.disconnect(self.on_frame)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.thr = CameraStreamThread(new_cfg)
            self.thr.frameReady.connect(self.on_frame)
        except Exception:
            # If anything goes wrong, keep old thread but update its cfg
            try:
                self.thr.cfg = new_cfg
            except Exception:
                pass
        # Clear prebuffer and frames
        with self._frame_lock:
            self._latest_frame = None
        try:
            self.thr._buf = bytearray()
            if hasattr(self.thr, 'prebuffer') and hasattr(self.thr.prebuffer, 'clear'):
                self.thr.prebuffer.clear()
        except Exception:
            pass
        # Restart
        try:
            self.start_stream()
        except Exception:
            pass

    def start_recording(self):
        if self.recording:
            return
        # estimate FPS from prebuffer timing
        fps = self.target_fps
        if len(self.thr.prebuffer) >= 5:
            tspan = self.thr.prebuffer[-1][1] - self.thr.prebuffer[0][1]
            frames = len(self.thr.prebuffer)
            if tspan > 0:
                fps = max(5.0, min(30.0, frames / tspan))
        # open writer
        ts_str = time.strftime('%Y%m%d_%H%M%S')
        outfile = os.path.join(self.out_dir, f"{self.cfg.name}_{ts_str}.mp4")
        h, w = self.current_size()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(outfile, fourcc, fps, (w, h))
        # dump prebuffer first
        for frm, _ in list(self.thr.prebuffer):
            if frm.shape[1] != w or frm.shape[0] != h:
                frm = cv2.resize(frm, (w, h))
            self.writer.write(frm)
        self.recording = True
        self.setWindowTitle(f"{self.cfg.name} (REC)")

    def stop_recording(self):
        if self.recording and self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        self.recording = False
        self.writer = None
        self.setWindowTitle(self.cfg.name)

    def current_size(self) -> Tuple[int,int]:
        # return H, W for writer
        pix = self.label.pixmap()
        if pix and not pix.isNull():
            return pix.height(), pix.width()
        return 480, 640

    @QtCore.Slot(np.ndarray, float)
    def on_frame(self, bgr: np.ndarray, ts: float):
        # Save latest frame for worker
        with self._frame_lock:
            self._latest_frame = bgr.copy()

        # Draw last results as overlays
        # (Rendering stale by <= detection interval, keeps UI responsive)
        dets = self._last_dets
        faces = self._last_faces
        for (cls,conf,x,y,w,h) in dets:
            color = (255,0,0) if cls=='person' else (0,0,255) if cls=='dog' else (255,0,255) if cls=='cat' else (0,255,255)
            cv2.rectangle(bgr,(x,y),(x+w,y+h),color,2)
            cv2.putText(bgr,f"{cls} {conf:.2f}", (x,max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
            if cls in ('dog','cat'):
                self.last_pet_bbox=(x,y,w,h,cls)
        self.last_face_bbox=None
        for (name,score,x,y,w,h) in faces:
            color=(0,255,0) if name!='unknown' else (0,165,255)
            cv2.rectangle(bgr,(x,y),(x+w,y+h),color,2)
            cv2.putText(bgr,f"{name} {score:.2f}", (x,max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
            self.last_face_bbox=(x,y,w,h)

        # handle collections
        now=time.time()
        if self.collect_face and self.last_face_bbox:
            if now - self.collect_face['last'] >= 0.2:
                x,y,w,h=self.last_face_bbox
                gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                self.facedb.enroll(gray,x,y,w,h,self.collect_face['name'])
                self.collect_face['col']+=1; self.collect_face['last']=now
                if self.collect_face['col']>=self.collect_face['n']:
                    self.facedb.load(); self.collect_face=None; self.lbl_status.setText('Face collection done')
        if self.collect_pet:
            if now - self.collect_pet['last'] >= 0.25:
                # choose largest of desired species
                species=self.collect_pet['sp']
                candidates=[(x,y,w,h) for (cls,conf,x,y,w,h) in dets if cls==species]
                if candidates:
                    bx=max(candidates,key=lambda b:b[2]*b[3])
                    x,y,w,h=bx
                    roi=bgr[max(0,y):y+h, max(0,x):x+w]
                    self.pets.enroll(roi,self.collect_pet['name'],species)
                    self.collect_pet['col']+=1; self.collect_pet['last']=now
                    if self.collect_pet['col']>=self.collect_pet['n']:
                        self.pets.load(); self.collect_pet=None; self.lbl_status.setText('Pet collection done')

        # show
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        # write
        if self.recording and self.writer is not None:
            # ensure writer size consistency
            W = int(self.writer.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(self.writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if bgr.shape[1] != W or bgr.shape[0] != H:
                bgr = cv2.resize(bgr, (W, H))
            self.writer.write(bgr)

    def shutdown(self):
        # Stop timers first to prevent new actions
        try:
            if hasattr(self, '_aim_timer'):
                self._aim_timer.stop()
        except Exception:
            pass
        # Stop recording and threads
        self.stop_recording()
        self.stop_stream()
        try:
            if self.det_thr.isRunning():
                self.det_thr.stop()
                if not self.det_thr.wait(2000):
                    self.det_thr.terminate()
                    self.det_thr.wait(500)
        except Exception:
            pass

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self.shutdown()
        try:
            self.closed.emit(self.cfg.__dict__)
        except Exception:
            pass
        super().closeEvent(e)

    # ----- actions
    def do_enroll_face(self, name: str | None = None):
        if not self.last_face_bbox:
            self.lbl_status.setText('No face detected')
            return
        if name is None:
            name=self.ed_name.text().strip() or 'person'
        x,y,w,h=self.last_face_bbox
        pix=self.label.pixmap()
        if pix is None or pix.isNull():
            self.lbl_status.setText('No frame')
            return
        # Best-effort enroll from last frame already handled in on_frame
        self.facedb.load(); self.lbl_status.setText(f'Enrolled {name}')

    def do_collect_face(self, name: str | None = None, n: int = 20):
        if name is None:
            name=self.ed_name.text().strip() or 'person'
        self.collect_face={'name':name,'n':20,'col':0,'last':0.0}
        self.lbl_status.setText(f'Collecting face: {name}')

    def do_enroll_pet(self, name: str | None = None, species: str | None = None):
        sp= species or self.cmb_species.currentText()
        name= (name or self.ed_name.text().strip()) or 'pet'
        if self.last_pet_bbox:
            x,y,w,h,cls=self.last_pet_bbox
            if cls!=sp:
                self.lbl_status.setText(f'Last bbox is {cls}')
                return
            # capture current displayed frame area
            # For simplicity, rely on on_frame collection path for consistency
            self.pets.load(); self.lbl_status.setText(f'Enrolled {sp}:{name}')
        else:
            self.lbl_status.setText('No pet detected')

    def do_collect_pet(self, name: str | None = None, species: str | None = None, n: int = 40):
        sp= species or self.cmb_species.currentText()
        name= (name or self.ed_name.text().strip()) or 'pet'
        self.collect_pet={'name':name,'sp':sp,'n':n,'col':0,'last':0.0}
        self.lbl_status.setText(f'Collecting {sp}: {name}')

    # ----- management helpers
    def _select_and_delete_images(self, root_dir: str, title: str):
        if not os.path.isdir(root_dir):
            QtWidgets.QMessageBox.warning(self, 'Manage', f'Path not found:\n{root_dir}')
            return
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, title, root_dir, 'Images (*.jpg *.jpeg *.png)')
        if not files:
            return
        if QtWidgets.QMessageBox.question(self, 'Delete', f'Delete {len(files)} files?') != QtWidgets.QMessageBox.Yes:
            return
        cnt=0
        for fp in files:
            try:
                os.remove(fp); cnt+=1
            except Exception:
                pass
        QtWidgets.QMessageBox.information(self, 'Manage', f'Deleted {cnt} files')

    def do_manage_faces(self):
        name=self.ed_name.text().strip()
        if not name:
            name, ok = QtWidgets.QInputDialog.getText(self, 'Manage Faces', 'Name:')
            if not ok or not name: return
        path=os.path.join('ai','data','faces',name)
        self._select_and_delete_images(path, f'Faces: {name}')
        self.facedb.load()

    def do_manage_pets(self):
        name=self.ed_name.text().strip()
        sp=self.cmb_species.currentText()
        if not name:
            name, ok = QtWidgets.QInputDialog.getText(self, 'Manage Pets', f'{sp} name:')
            if not ok or not name: return
        path=os.path.join('ai','data','pets','dogs' if sp=='dog' else 'cats', name)
        self._select_and_delete_images(path, f'{sp.title()}: {name}')
        self.pets.load()

    @QtCore.Slot(list, list)
    def on_results(self, dets, faces):
        # Called from worker thread via Qt signal (thread-safe)
        self._last_dets = dets
        self._last_faces = faces
        # Update tracker using detections
        now = time.time()
        self.tracker.update(dets, now)

    def aim_at_target(self):
        # Aim at cats only; use simple proportional step toward center
        target = self.tracker.primary_target(prefer='cat')
        if not target:
            return
        x,y,w,h,cls,last_ts = target
        # compute from last displayed pixmap size
        pix = self.label.pixmap()
        if not pix or pix.isNull():
            return
        W = pix.width(); H = pix.height()
        cx = x + w/2; cy = y + h/2
        dx = cx - W/2; dy = cy - H/2
        deadzone_px = max(8, int(0.05*min(W,H)))
        base = f"http://{self.cfg.host}"
        token = ("?token="+self.cfg.token) if self.cfg.token else ""
        headers = {}
        ah = self.cfg.auth_header()
        if ah: headers['Authorization']=ah
        req_auth = None
        if not self.cfg.token and self.cfg.user and self.cfg.password:
            try:
                from requests.auth import HTTPBasicAuth
                req_auth = HTTPBasicAuth(self.cfg.user, self.cfg.password)
            except Exception:
                req_auth = None
        try:
            if dx > deadzone_px:
                requests.get(base+"/action?go=left"+token, headers=headers, auth=req_auth, timeout=0.5)
            elif dx < -deadzone_px:
                requests.get(base+"/action?go=right"+token, headers=headers, auth=req_auth, timeout=0.5)
            if dy > deadzone_px:
                requests.get(base+"/action?go=down"+token, headers=headers, auth=req_auth, timeout=0.5)
            elif dy < -deadzone_px:
                requests.get(base+"/action?go=up"+token, headers=headers, auth=req_auth, timeout=0.5)
        except Exception:
            pass


class DetectionThread(QtCore.QThread):
    resultsReady = QtCore.Signal(list, list)  # dets, faces
    def __init__(self, widget: CameraWidget, interval_ms: int = 200):
        super().__init__(widget)
        self.w = widget
        self.interval = interval_ms
        self._stop = threading.Event()
        self._skip_cycles = 0
        self.max_skip_cycles = 1  # skip this many cycles if tracks are active

    def stop(self):
        self._stop.set()

    def run(self):
        last = 0
        while not self._stop.is_set():
            now = time.time()
            if (now - last)*1000.0 < self.interval:
                self.msleep(10)
                continue
            last = now
            # fetch latest frame
            with self.w._frame_lock:
                frame = None if self.w._latest_frame is None else self.w._latest_frame.copy()
            if frame is None:
                self.msleep(10)
                continue
            dets=[]; faces=[]
            try:
                # Skip YOLO on some cycles if we already have active tracks (object permanence)
                do_yolo = True
                if self.w.tracker.has_active():
                    if self._skip_cycles < self.max_skip_cycles:
                        do_yolo = False; self._skip_cycles += 1
                    else:
                        self._skip_cycles = 0
                if do_yolo and self.w.chk_yolo.isChecked() and self.w.yolo.available():
                    dets = self.w.yolo.detect(frame)
                # Face detection/recognition
                if self.w.chk_face.isChecked():
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    for (x,y,w,h) in self.w.facedb.detect_faces(gray):
                        name,score=self.w.facedb.recognize_roi(gray[y:y+h,x:x+w])
                        faces.append((name,score,x,y,w,h))
            except Exception as e:
                # Avoid crashing the thread on sporadic errors
                pass
            self.resultsReady.emit(dets, faces)


class SimpleTracker:
    """Very simple IOU-based tracker with TTL for object permanence.
    Tracks: list of dicts with bbox, cls, last_ts.
    """
    def __init__(self, ttl: float = 1.0, iou_thresh: float = 0.3):
        self.ttl = ttl
        self.iou_thresh = iou_thresh
        self.tracks = []

    def update(self, dets, now: float):
        # Match new detections to existing tracks via IOU
        def iou(a,b):
            ax,ay,aw,ah=a; bx,by,bw,bh=b
            x1=max(ax,bx); y1=max(ay,by); x2=min(ax+aw,bx+bw); y2=min(ay+ah,by+bh)
            inter=max(0,x2-x1)*max(0,y2-y1)
            ua=aw*ah + bw*bh - inter
            return inter/ua if ua>0 else 0.0
        # decay tracks
        self.tracks = [t for t in self.tracks if now - t['last_ts'] <= self.ttl]
        for cls,conf,x,y,w,h in dets:
            bb=(x,y,w,h)
            best=None; best_iou=0
            for t in self.tracks:
                ii=iou(bb,t['bbox'])
                if ii>best_iou: best_iou=ii; best=t
            if best and best_iou>=self.iou_thresh and best['cls']==cls:
                best['bbox']=bb; best['last_ts']=now
            else:
                self.tracks.append({'cls':cls,'bbox':bb,'last_ts':now})

    def has_active(self) -> bool:
        return len(self.tracks)>0

    def primary_target(self, prefer: str = 'cat'):
        # choose preferred class if present, otherwise largest box
        if not self.tracks:
            return None
        cats=[t for t in self.tracks if t['cls']==prefer]
        chosen = cats or self.tracks
        # Largest area
        t=max(chosen, key=lambda t: t['bbox'][2]*t['bbox'][3])
        x,y,w,h = t['bbox']
        return (x,y,w,h,t['cls'], t['last_ts'])


class AddCameraDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial: Optional[CameraConfig]=None, title: str='Add Camera'):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        form = QtWidgets.QFormLayout(self)
        self.ed_name = QtWidgets.QLineEdit(initial.name if initial else 'Camera')
        self.ed_host = QtWidgets.QLineEdit(initial.host if initial else '192.168.1.100')
        self.ed_user = QtWidgets.QLineEdit(initial.user or '' if initial else '')
        self.ed_pass = QtWidgets.QLineEdit(initial.password or '' if initial else '')
        self.ed_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ed_token = QtWidgets.QLineEdit(initial.token or '' if initial else '')
        form.addRow('Name', self.ed_name)
        form.addRow('Host (ip[:port])', self.ed_host)
        form.addRow('User', self.ed_user)
        form.addRow('Password', self.ed_pass)
        form.addRow('Token (Base64 user:pass)', self.ed_token)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_config(self) -> Optional[CameraConfig]:
        if self.exec() == QtWidgets.QDialog.Accepted:
            return CameraConfig(
                name=self.ed_name.text().strip() or 'Camera',
                host=self.ed_host.text().strip(),
                user=(self.ed_user.text().strip() or None),
                password=self.ed_pass.text(),
                token=(self.ed_token.text().strip() or None),
            )
        return None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ESP32-CAM MDI')
        self.resize(1200, 800)
        self.mdi = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi)

        # Menu bar
        mb = self.menuBar()
        menu_file = mb.addMenu('&File')
        menu_tools = mb.addMenu('&Tools')
        act_dl_yolo = menu_tools.addAction('Download YOLO Model')
        act_dl_yolo.triggered.connect(self.download_yolo_model)
        # Import and Cull submenus
        menu_import = menu_tools.addMenu('Import')
        act_imp_faces = menu_import.addAction('Import Faces...')
        act_imp_pets = menu_import.addAction('Import Pets...')
        act_imp_faces.triggered.connect(self.import_faces)
        act_imp_pets.triggered.connect(self.import_pets)
        menu_cull = menu_tools.addMenu('Cull Similar')
        act_cull_faces = menu_cull.addAction('Cull Faces...')
        act_cull_pets = menu_cull.addAction('Cull Pets...')
        act_cull_faces.triggered.connect(self.cull_faces)
        act_cull_pets.triggered.connect(self.cull_pets)
        # Settings dialog
        act_settings = menu_tools.addAction('Settings...')
        act_settings.triggered.connect(self.open_settings)
        # File set actions
        self.act_new = menu_file.addAction('New Set')
        self.act_load = menu_file.addAction('Load Set...')
        self.act_save = menu_file.addAction('Save Set')
        self.act_save_as = menu_file.addAction('Save Set As...')
        menu_file.addSeparator()
        self.act_exit = menu_file.addAction('Exit')

        tb = QtWidgets.QToolBar('Main')
        tb.setIconSize(QtCore.QSize(16,16))
        self.addToolBar(tb)

        act_add = tb.addAction('Add Camera')
        act_tile = tb.addAction('Tile')
        act_cascade = tb.addAction('Cascade')
        tb.addSeparator()
        act_edit = tb.addAction('Edit Camera')
        tb.addSeparator()
        act_rec_all = tb.addAction('Start Rec All')
        act_stop_all = tb.addAction('Stop Rec All')
        tb.addSeparator()
        # Global enrollment/management controls
        tb.addWidget(QtWidgets.QLabel('Name:'))
        self.ed_name_global = QtWidgets.QLineEdit(); self.ed_name_global.setMaximumWidth(160)
        tb.addWidget(self.ed_name_global)
        tb.addWidget(QtWidgets.QLabel('Species:'))
        self.cmb_species_global = QtWidgets.QComboBox(); self.cmb_species_global.addItems(['dog','cat'])
        tb.addWidget(self.cmb_species_global)
        # Enrollment actions moved to Tools menu (below)
        act_manage_faces_g = tb.addAction('Manage Faces')
        act_manage_pets_g = tb.addAction('Manage Pets')

        act_add.triggered.connect(self.add_camera)
        act_tile.triggered.connect(self.mdi.tileSubWindows)
        act_cascade.triggered.connect(self.mdi.cascadeSubWindows)
        act_rec_all.triggered.connect(self.start_rec_all)
        act_stop_all.triggered.connect(self.stop_rec_all)
        act_edit.triggered.connect(self.edit_camera)
        # File set wiring
        self.act_new.triggered.connect(self.new_set)
        self.act_load.triggered.connect(self.load_set_dialog)
        self.act_save.triggered.connect(self.save_set)
        self.act_save_as.triggered.connect(self.save_set_as)
        self.act_exit.triggered.connect(self.close)

        act_manage_faces_g.triggered.connect(self.manage_faces_active)
        act_manage_pets_g.triggered.connect(self.manage_pets_active)

        os.makedirs(os.path.join('ai','recordings'), exist_ok=True)

        # Load saved cameras
        self._prefs_path = os.path.join('ai','cameras.json')
        self._cam_defs = []
        self.load_cameras()
        for d in self._cam_defs:
            self.add_camera_from_cfg(CameraConfig(**d))

        # Add enrollment to Tools menu
        enroll_menu = menu_tools.addMenu('Enrollment')
        act_enroll_face = enroll_menu.addAction('Enroll Face')
        act_collect_face = enroll_menu.addAction('Collect Face (20)')
        act_enroll_pet = enroll_menu.addAction('Enroll Pet')
        act_collect_pet = enroll_menu.addAction('Collect Pet (40)')
        act_enroll_face.triggered.connect(self.enroll_face_active)
        act_collect_face.triggered.connect(self.collect_face_active)
        act_enroll_pet.triggered.connect(self.enroll_pet_active)
        act_collect_pet.triggered.connect(self.collect_pet_active)

    # -------- Tools --------
    def download_yolo_model(self):
        """Download YOLO ONNX to ai/models/yolo.onnx with progress."""
        urls = [
            'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
            'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.onnx',
            'https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.onnx',
        ]
        dst = os.path.join('ai','models','yolo.onnx')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for url in urls:
            try:
                dlg = QtWidgets.QProgressDialog('Downloading YOLO model...', 'Cancel', 0, 100, self)
                dlg.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
                dlg.show()
                with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
                    r.raise_for_status()
                    total = int(r.headers.get('content-length', '0'))
                    done = 0
                    with open(dst, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1<<15):
                            if dlg.wasCanceled():
                                raise RuntimeError('Canceled')
                            if not chunk:
                                continue
                            f.write(chunk)
                            done += len(chunk)
                            if total:
                                dlg.setValue(int(done*100/total))
                                QtWidgets.QApplication.processEvents()
                dlg.setValue(100)
                QtWidgets.QMessageBox.information(self, 'Download', f'Saved model to {dst}')
                return
            except Exception as e:
                # try next URL
                pass
        QtWidgets.QMessageBox.warning(self, 'Download', 'Failed to download YOLO model from known mirrors')

    def add_camera(self):
        dlg = AddCameraDialog(self)
        cfg = dlg.get_config()
        if not cfg:
            return
        # save to prefs
        self._cam_defs.append(cfg.__dict__)
        self.save_cameras()
        self.add_camera_from_cfg(cfg)

    def add_camera_from_cfg(self, cfg: CameraConfig):
        w = CameraWidget(cfg)
        # listen for close to update prefs
        try:
            w.closed.connect(self.on_camera_closed)
        except Exception:
            pass
        sub = QtWidgets.QMdiSubWindow()
        sub.setWidget(w)
        sub.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        sub.setWindowTitle(cfg.name)
        self.mdi.addSubWindow(sub)
        sub.resize(500, 420)
        sub.show()
        w.start_stream()

    def edit_camera(self):
        w = self._active_camera()
        if not w:
            QtWidgets.QMessageBox.information(self, 'Edit Camera', 'Select a camera window to edit its settings.')
            return
        old_cfg = w.cfg
        dlg = AddCameraDialog(self, initial=old_cfg, title='Edit Camera')
        new_cfg = dlg.get_config()
        if not new_cfg:
            return
        # Update prefs: replace matching entry on name+host (fallback: first matching host)
        try:
            replaced = False
            for i, d in enumerate(self._cam_defs):
                if (d.get('name') == old_cfg.name and d.get('host') == old_cfg.host) or (d.get('host') == old_cfg.host):
                    self._cam_defs[i] = new_cfg.__dict__
                    replaced = True
                    break
            if not replaced:
                self._cam_defs.append(new_cfg.__dict__)
            self.save_cameras()
        except Exception:
            pass
        # Apply to widget: stop worker, swap config, start again
        try:
            w.apply_config(new_cfg)
        except Exception:
            # Fallback: close and re-add
            sub = self.mdi.activeSubWindow()
            if sub:
                sub.close()
            self.add_camera_from_cfg(new_cfg)

    def start_rec_all(self):
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.start_recording()

    def stop_rec_all(self):
        for sub in self.mdi.subWindowList():
            w = sub.widget()
            if isinstance(w, CameraWidget):
                w.stop_recording()

    # ----- Prefs (ensure defined as class methods)
    def load_cameras(self):
        try:
            import json
            self._cam_defs = []
            if hasattr(self, '_prefs_path') and os.path.exists(self._prefs_path):
                with open(self._prefs_path,'r',encoding='utf-8') as f:
                    self._cam_defs = json.load(f)
        except Exception:
            self._cam_defs = []

    def save_cameras(self):
        try:
            import json
            if not hasattr(self, '_prefs_path'):
                self._prefs_path = os.path.join('ai','cameras.json')
            os.makedirs(os.path.dirname(self._prefs_path), exist_ok=True)
            with open(self._prefs_path,'w',encoding='utf-8') as f:
                json.dump(self._cam_defs, f, indent=2)
        except Exception:
            pass

    @QtCore.Slot(object)
    def on_camera_closed(self, cfg: CameraConfig):
        try:
            self._cam_defs = [d for d in self._cam_defs if not (d.get('name')==cfg.name and d.get('host')==cfg.host)]
            self.save_cameras()
        except Exception:
            pass

    # ----- Active camera helpers
    def _active_camera(self) -> CameraWidget | None:
        sub = self.mdi.activeSubWindow()
        if not sub: return None
        w = sub.widget()
        return w if isinstance(w, CameraWidget) else None

    def enroll_face_active(self):
        w=self._active_camera();
        if not w: return
        name=self.ed_name_global.text().strip()
        # Prompt to choose/confirm a specific person
        if not name or name.lower()=='person':
            # Offer existing names as hint
            faces_dir=os.path.join('ai','data','faces')
            names=[d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir,d))] if os.path.isdir(faces_dir) else []
            name, ok = QtWidgets.QInputDialog.getText(self, 'Enroll Face', f'Enter person name to enroll{" (existing: "+", ".join(names)+")" if names else ""}:')
            if not ok or not name: return
        w.do_enroll_face(name)

    def collect_face_active(self):
        w=self._active_camera();
        if not w: return
        name=self.ed_name_global.text().strip()
        if not name or name.lower()=='person':
            faces_dir=os.path.join('ai','data','faces')
            names=[d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir,d))] if os.path.isdir(faces_dir) else []
            name, ok = QtWidgets.QInputDialog.getText(self, 'Collect Face Samples', f'Collect samples for which person?{" (existing: "+", ".join(names)+")" if names else ""}:')
            if not ok or not name: return
        w.do_collect_face(name, 20)

    def enroll_pet_active(self):
        w=self._active_camera();
        if not w: return
        name=self.ed_name_global.text().strip() or None
        sp=self.cmb_species_global.currentText()
        # Policy: cats are treated generically for prevention; skip cat enrollment
        if sp=='cat':
            QtWidgets.QMessageBox.information(self, 'Pet Enrollment', 'Cat prevention mode: specific cat enrollment is disabled. Any cat will be treated generically.')
            return
        # For dogs, ensure a named identity
        if not name:
            dogs_dir=os.path.join('ai','data','pets','dogs')
            names=[d for d in os.listdir(dogs_dir) if os.path.isdir(os.path.join(dogs_dir,d))] if os.path.isdir(dogs_dir) else []
            name, ok = QtWidgets.QInputDialog.getText(self, 'Enroll Dog', f'Enter dog name to enroll{" (existing: "+", ".join(names)+")" if names else ""}:')
            if not ok or not name: return
        w.do_enroll_pet(name, sp)

    def collect_pet_active(self):
        w=self._active_camera();
        if not w: return
        name=self.ed_name_global.text().strip() or None
        sp=self.cmb_species_global.currentText()
        if sp=='cat':
            QtWidgets.QMessageBox.information(self, 'Collect Pet', 'Cat prevention mode: specific cat datasets are not required. Any detected cat triggers prevention.')
            return
        if not name:
            dogs_dir=os.path.join('ai','data','pets','dogs')
            names=[d for d in os.listdir(dogs_dir) if os.path.isdir(os.path.join(dogs_dir,d))] if os.path.isdir(dogs_dir) else []
            name, ok = QtWidgets.QInputDialog.getText(self, 'Collect Dog Samples', f'Collect samples for which dog?{" (existing: "+", ".join(names)+")" if names else ""}:')
            if not ok or not name: return
        w.do_collect_pet(name, sp, 40)

    def manage_faces_active(self):
        w=self._active_camera();
        if not w: return
        name=self.ed_name_global.text().strip()
        if not name:
            name, ok = QtWidgets.QInputDialog.getText(self, 'Manage Faces', 'Name:')
            if not ok or not name: return
        path=os.path.join('ai','data','faces',name)
        # Open built-in gallery dialog
        dlg = GalleryDialog(path, f'Faces: {name}', self)
        dlg.exec()
        w.facedb.load()

    def manage_pets_active(self):
        w=self._active_camera();
        if not w: return
        sp=self.cmb_species_global.currentText()
        name=self.ed_name_global.text().strip()
        if not name:
            name, ok = QtWidgets.QInputDialog.getText(self, 'Manage Pets', f'{sp} name:')
            if not ok or not name: return
        path=os.path.join('ai','data','pets','dogs' if sp=='dog' else 'cats', name)
        dlg = GalleryDialog(path, f'{sp.title()}: {name}', self)
        dlg.exec()
        w.pets.load()

    # -------- Import helpers (menu actions) --------
    def import_faces(self):
        """Import a directory of images into faces/<name>."""
        name, ok = QtWidgets.QInputDialog.getText(self, 'Import Faces', 'Person name:')
        if not ok or not name:
            return
        src_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Source Directory')
        if not src_dir:
            return
        safe = ''.join(c for c in name if c.isalnum() or c in ('_','-'))
        dest = os.path.join('ai','data','faces', safe)
        os.makedirs(dest, exist_ok=True)
        count = tools.ingest_images(src_dir, dest, (160,160), gray=True)
        QtWidgets.QMessageBox.information(self, 'Import Faces', f'Imported {count} images to {dest}')

    def import_pets(self):
        """Import a directory of images into pets/<species>/<name>."""
        species, ok = QtWidgets.QInputDialog.getItem(self, 'Import Pets', 'Species:', ['dog','cat'], 0, False)
        if not ok:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, 'Import Pets', f'{species} name:')
        if not ok or not name:
            return
        src_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Source Directory')
        if not src_dir:
            return
        sp_dir = 'dogs' if species=='dog' else 'cats'
        safe = ''.join(c for c in name if c.isalnum() or c in ('_','-'))
        dest = os.path.join('ai','data','pets', sp_dir, safe)
        os.makedirs(dest, exist_ok=True)
        count = tools.ingest_images(src_dir, dest, (320,320), gray=True)
        QtWidgets.QMessageBox.information(self, 'Import Pets', f'Imported {count} images to {dest}')

    # -------- Cull helpers (menu actions) --------
    def cull_faces(self):
        name, ok = QtWidgets.QInputDialog.getText(self, 'Cull Faces', 'Person name:')
        if not ok or not name:
            return
        target = os.path.join('ai','data','faces', name)
        files, remove = tools.find_similar_in_dir(target)
        dlg = CullDialog(target, files, remove, f'Cull Faces: {name}', self)
        dlg.exec()

    def cull_pets(self):
        species, ok = QtWidgets.QInputDialog.getItem(self, 'Cull Pets', 'Species:', ['dog','cat'], 0, False)
        if not ok:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, 'Cull Pets', f'{species} name:')
        if not ok or not name:
            return
        target = os.path.join('ai','data','pets', 'dogs' if species=='dog' else 'cats', name)
        files, remove = tools.find_similar_in_dir(target)
        dlg = CullDialog(target, files, remove, f'Cull Pets: {species} {name}', self)
        dlg.exec()

    # -------- Camera Set management --------
    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        # Proactively stop all camera workers before app quits
        try:
            for sub in list(self.mdi.subWindowList()):
                w = sub.widget()
                if hasattr(w, 'shutdown'):
                    try:
                        w.shutdown()
                    except Exception:
                        pass
                try:
                    sub.close()
                except Exception:
                    pass
        except Exception:
            pass
        super().closeEvent(e)
    def new_set(self):
        """Close all camera windows and clear current set."""
        for sub in list(self.mdi.subWindowList()):
            sub.close()
        self._cam_defs = []
        self.save_cameras()

    def load_set_dialog(self):
        """Load camera definitions from JSON and recreate subwindows."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load Camera Set', 'ai', 'JSON (*.json)')
        if not path:
            return
        try:
            import json
            with open(path,'r',encoding='utf-8') as f:
                defs = json.load(f)
            for sub in list(self.mdi.subWindowList()):
                sub.close()
            self._cam_defs = defs if isinstance(defs, list) else []
            self._prefs_path = path
            self.save_cameras()
            for d in self._cam_defs:
                self.add_camera_from_cfg(CameraConfig(**d))
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, 'Load', f'Failed to load set:\n{ex}')

    def save_set(self):
        """Save to current prefs path."""
        self.save_cameras()
        QtWidgets.QMessageBox.information(self, 'Save', f'Saved: {self._prefs_path}')

    def save_set_as(self):
        """Save camera set to a new JSON path."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Camera Set As', 'ai/cameras.json', 'JSON (*.json)')
        if not path:
            return
        self._prefs_path = path
        self.save_cameras()
        QtWidgets.QMessageBox.information(self, 'Save As', f'Saved: {self._prefs_path}')
    # -------- Settings dialog --------
    def open_settings(self):
        dlg = SettingsDialog(self)
        # prefill from current state
        dlg.s_det_interval.setValue(getattr(self, '_det_interval', 200))
        dlg.s_hash_size.setValue(getattr(self, '_hash_size', 8))
        dlg.s_hamming.setValue(getattr(self, '_hamming', 4))
        dlg.s_ptz_interval.setValue(getattr(self, '_ptz_interval', 300))
        dlg.s_deadzone.setValue(getattr(self, '_deadzone_pct', 5))
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._det_interval = dlg.s_det_interval.value()
            self._hash_size = dlg.s_hash_size.value()
            self._hamming = dlg.s_hamming.value()
            self._ptz_interval = dlg.s_ptz_interval.value()
            self._deadzone_pct = dlg.s_deadzone.value()
            # apply to open cameras
            for sub in self.mdi.subWindowList():
                w=sub.widget()
                if isinstance(w, CameraWidget):
                    w.det_thr.interval = self._det_interval
                    w.det_thr.max_skip_cycles = 1
                    w._aim_timer.setInterval(self._ptz_interval)
            QtWidgets.QMessageBox.information(self, 'Settings', 'Settings applied')


class SettingsDialog(QtWidgets.QDialog):
    """App settings: detector interval, aHash grid size + Hamming, PTZ aim timing, deadzone."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.resize(420, 240)
        form = QtWidgets.QFormLayout(self)
        self.s_det_interval = QtWidgets.QSpinBox(); self.s_det_interval.setRange(50, 2000); self.s_det_interval.setSingleStep(50); self.s_det_interval.setSuffix(' ms')
        self.s_hash_size = QtWidgets.QSpinBox(); self.s_hash_size.setRange(4, 16)
        self.s_hamming = QtWidgets.QSpinBox(); self.s_hamming.setRange(0, 32)
        self.s_ptz_interval = QtWidgets.QSpinBox(); self.s_ptz_interval.setRange(100, 2000); self.s_ptz_interval.setSingleStep(50); self.s_ptz_interval.setSuffix(' ms')
        self.s_deadzone = QtWidgets.QSpinBox(); self.s_deadzone.setRange(2, 20); self.s_deadzone.setSuffix(' %')
        form.addRow('Detector interval', self.s_det_interval)
        form.addRow('aHash grid size', self.s_hash_size)
        form.addRow('Hamming threshold', self.s_hamming)
        form.addRow('PTZ aim interval', self.s_ptz_interval)
        form.addRow('PTZ deadzone', self.s_deadzone)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

class CullDialog(QtWidgets.QDialog):
    """Preview duplicates with highlight before deletion.
    Includes a tolerance slider to adjust the near-duplicate matching threshold.
    Lower values are stricter; higher tolerate more difference.
    """
    def __init__(self, dir_path: str, files: list[str], remove: set[int], title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 700)
        self.dir_path = dir_path
        self.files = files or []
        self.remove = set(remove or set())
        # Matching parameters (aHash on 8x8 grid → Hamming distance 0..64).
        # Practical duplicate tolerance range ~0..16; start at 4 by default.
        self.hash_size = 8
        self.thresh = 4
        v = QtWidgets.QVBoxLayout(self)
        self.view = QtWidgets.QListWidget()
        self.view.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.view.setIconSize(QtCore.QSize(160, 120))
        self.view.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self.view, 1)
        # Controls
        h = QtWidgets.QHBoxLayout()
        # Tolerance slider block
        tol_box = QtWidgets.QHBoxLayout()
        self.lbl_tol = QtWidgets.QLabel('Tolerance:')
        self.sld_tol = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld_tol.setMinimum(0)
        self.sld_tol.setMaximum(16)
        self.sld_tol.setValue(self.thresh)
        self.sld_tol.setTickInterval(1)
        self.sld_tol.setSingleStep(1)
        self.lbl_tol_val = QtWidgets.QLabel(str(self.thresh))
        tol_box.addWidget(self.lbl_tol)
        tol_box.addWidget(self.sld_tol)
        tol_box.addWidget(self.lbl_tol_val)
        tol_box_w = QtWidgets.QWidget(); tol_box_w.setLayout(tol_box)
        h.addWidget(tol_box_w, 2)
        # Info + actions
        self.lbl_info = QtWidgets.QLabel()
        h.addWidget(self.lbl_info, 1)
        self.btn_confirm = QtWidgets.QPushButton('Confirm Delete')
        self.btn_cancel = QtWidgets.QPushButton('Cancel')
        h.addWidget(self.btn_confirm)
        h.addWidget(self.btn_cancel)
        v.addLayout(h)
        self.btn_confirm.clicked.connect(self.on_confirm)
        self.btn_cancel.clicked.connect(self.reject)
        self.sld_tol.valueChanged.connect(self.on_tol_changed)
        self.populate()

    def populate(self):
        self.view.clear()
        removed = 0
        for idx, fp in enumerate(self.files):
            if not os.path.exists(fp):
                continue
            item = QtWidgets.QListWidgetItem(os.path.basename(fp))
            pix = QtGui.QPixmap(fp)
            if not pix.isNull():
                item.setIcon(QtGui.QIcon(pix.scaled(160,120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)))
            if idx in self.remove:
                item.setBackground(QtGui.QBrush(QtGui.QColor(255, 220, 220)))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, (idx, fp))
            self.view.addItem(item)
        self.lbl_info.setText(f"Marked for removal: {len(self.remove)} / {len(self.files)}")

    def on_confirm(self):
        # Optionally allow manual deselect: unmark by selecting those to keep
        for it in self.view.selectedItems():
            idx, _ = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if idx in self.remove:
                self.remove.remove(idx)
        cnt=0
        for idx in sorted(self.remove):
            fp = self.files[idx]
            try:
                os.remove(fp); cnt+=1
            except Exception:
                pass
        QtWidgets.QMessageBox.information(self, 'Cull', f'Deleted {cnt} files')
        self.accept()

    def on_tol_changed(self, val: int):
        """Recompute suggested deletions when the tolerance slider changes."""
        self.thresh = int(val)
        self.lbl_tol_val.setText(str(self.thresh))
        # Recompute using tools.find_similar_in_dir with new threshold
        files, remove = tools.find_similar_in_dir(self.dir_path, hash_size=self.hash_size, hamming_thresh=self.thresh)
        if files:
            self.files = files
            self.remove = remove
            self.populate()


def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()





