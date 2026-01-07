from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

import numpy as np

from utils import monotonic_ms
from .config import DetectorConfig
from .core import run_yolo
from .lbph import load_lbph, run_faces, run_faces_dnn
from .opencv_backend import (
    cuda_supported,
    limit_opencv_threads,
    load_face_detectors,
    load_yolo_net,
    backend_label,
    target_label,
)
from .packet import DetectionPacket


# Limit concurrent YOLO runs to reduce CPU contention across cameras.
YOLO_SEMAPHORE = threading.Semaphore(1)

# Gate to avoid spamming CUDA status messages across many cameras.
_CUDA_STATUS_PRINTED = False

limit_opencv_threads()


class DetectorEngine:
    def __init__(self, cfg: DetectorConfig, name: str):
        self.cfg = cfg
        self.name = name

        self._profile_next_ms = 0
        self._last_yolo: Optional[Tuple[int, list, list]] = None  # (ts_ms, yolo_boxes, pet_boxes)

        self._face_mode = "none"  # dnn | haar | none
        self._face_backend = None
        self._face_target = None

        self.models_dir = os.path.dirname(self.cfg.yolo_model)

        cuda_ok = False
        if self.cfg.use_gpu:
            cuda_ok = cuda_supported()
            global _CUDA_STATUS_PRINTED
            if not _CUDA_STATUS_PRINTED:
                msg = (
                    "CUDA detected; will use GPU for YOLO where possible"
                    if cuda_ok
                    else "CUDA not available; using CPU for YOLO"
                )
                print(f"[Detector] {msg}")
                _CUDA_STATUS_PRINTED = True

        self._net = load_yolo_net(
            self.cfg.yolo_model,
            name=self.name,
            use_gpu=self.cfg.use_gpu,
            cuda_ok=cuda_ok,
        )

        (
            self._face,
            self._face_dnn,
            self._face_mode,
            self._face_backend,
            self._face_target,
        ) = load_face_detectors(
            face_model=self.cfg.face_model,
            face_cascade=self.cfg.face_cascade,
            name=self.name,
            use_gpu=self.cfg.use_gpu,
            cuda_ok=cuda_ok,
        )

        if self.cfg.use_lbph:
            self._rec, self._labels = load_lbph(self.models_dir)
            self._lbph_mtime = self._lbph_models_mtime()
        else:
            self._rec, self._labels = None, {}
            self._lbph_mtime = 0

        print(
            f"[Detector:{self.name}] init: net={'OK' if self._net is not None else 'NONE'}, "
            f"face={'DNN' if self._face_dnn is not None else ('OK' if self._face is not None else 'NONE')}, "
            f"lbph={'OK' if self._rec is not None else 'NONE'}, "
            f"yolo_conf={self.cfg.yolo_conf} "
            f"use_gpu={self.cfg.use_gpu}"
        )

    def face_mode_label(self) -> str:
        if self._face_mode == "dnn":
            return f"dnn({backend_label(self._face_backend)}/{target_label(self._face_target)})"
        return self._face_mode

    def _lbph_models_mtime(self) -> int:
        """
        Return a high-resolution mtime marker for the LBPH model+labels files.

        Note: on some platforms, float mtimes can lose precision and cause hot-reload
        to miss quick successive retrains, so we use nanosecond mtimes.
        """
        try:
            xml = os.path.join(self.models_dir, "lbph_faces.xml")
            labels = os.path.join(self.models_dir, "labels_faces.json")
            return max(os.stat(xml).st_mtime_ns, os.stat(labels).st_mtime_ns)
        except Exception:
            return 0

    def _maybe_reload_lbph(self) -> None:
        try:
            mtime = self._lbph_models_mtime()
            if mtime > int(getattr(self, "_lbph_mtime", 0) or 0):
                self._rec, self._labels = load_lbph(self.models_dir)
                self._lbph_mtime = mtime
                print(f"[Detector:{self.name}] reloaded LBPH models (labels={len(self._labels)})")
        except Exception:
            pass

    def process_frame(self, bgr: np.ndarray, ts_ms: int) -> tuple[DetectionPacket, bool, int]:
        """
        Process a single frame snapshot.

        Returns (packet, yolo_skipped, end_ms) where end_ms is the monotonic time at completion.
        """
        H, W = bgr.shape[:2]
        pkt = DetectionPacket(self.name, ts_ms, (W, H))
        t_start = monotonic_ms()
        t0 = t_start

        yolo_skipped = False

        if self._net is not None:
            # Try to acquire YOLO slot; wait briefly before skipping.
            acquired = YOLO_SEMAPHORE.acquire(timeout=0.05)
            if acquired:
                try:
                    yolo_boxes, pet_boxes, t_yolo = run_yolo(
                        self._net, bgr, self.cfg.yolo_conf, self.cfg.yolo_nms
                    )
                    pkt.yolo.extend(yolo_boxes)
                    pkt.pets.extend(pet_boxes)
                    pkt.timing_ms["yolo_core"] = t_yolo
                    self._last_yolo = (ts_ms, yolo_boxes, pet_boxes)
                except Exception as e:
                    print(f"[Detector:{self.name}] YOLO error: {e}")
                finally:
                    YOLO_SEMAPHORE.release()
            else:
                pkt.timing_ms["yolo_core"] = 0  # skipped to avoid contention
                yolo_skipped = True
                if self._last_yolo:
                    last_ts, last_yolo_boxes, last_pet_boxes = self._last_yolo
                    if ts_ms - last_ts <= 2000:
                        pkt.yolo.extend(last_yolo_boxes)
                        pkt.pets.extend(last_pet_boxes)

        t1 = monotonic_ms()

        if self._face_dnn is not None:
            # Hot-reload LBPH if models changed on disk (e.g., purge / image manager / auto-train).
            if self.cfg.use_lbph:
                self._maybe_reload_lbph()
            face_boxes, t_faces = run_faces_dnn(
                bgr,
                self._face_dnn,
                self._rec if self.cfg.use_lbph else None,
                self._labels if self.cfg.use_lbph else {},
            )
            pkt.faces.extend(face_boxes)
            pkt.timing_ms["faces_core"] = t_faces
        elif self._face is not None:
            # Hot-reload LBPH if models changed on disk (e.g., auto-train/unknowns)
            if self.cfg.use_lbph:
                self._maybe_reload_lbph()
            face_boxes, t_faces = run_faces(bgr, self._face, self._rec, self._labels)
            pkt.faces.extend(face_boxes)
            pkt.timing_ms["faces_core"] = t_faces

        t2 = monotonic_ms()
        pkt.timing_ms["yolo"] = int(t1 - t0)
        pkt.timing_ms["faces"] = int(t2 - t1)

        return pkt, yolo_skipped, t2

