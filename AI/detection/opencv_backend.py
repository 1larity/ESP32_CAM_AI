from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2


def limit_opencv_threads() -> None:
    # Keep OpenCV from spinning up many worker threads; helps reduce contention.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def cuda_supported() -> bool:
    try:
        info = cv2.getBuildInformation()
        if "CUDA" not in info.upper():
            return False
        if not hasattr(cv2, "cuda"):
            return False
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            return False
        return bool(count and count > 0)
    except Exception:
        return False


def backend_label(val) -> str:
    if val == cv2.dnn.DNN_BACKEND_CUDA:
        return "CUDA"
    if val == cv2.dnn.DNN_BACKEND_OPENCV:
        return "CPU"
    return str(val)


def target_label(val) -> str:
    if val == cv2.dnn.DNN_TARGET_CUDA:
        return "CUDA"
    if val == cv2.dnn.DNN_TARGET_CPU:
        return "CPU"
    return str(val)


def load_yolo_net(
    model_path: str, *, name: str, use_gpu: bool, cuda_ok: bool
) -> Optional[cv2.dnn_Net]:
    if not os.path.exists(model_path):
        print(f"[Detector:{name}] YOLO model not found at {model_path}")
        return None

    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        backend = cv2.dnn.DNN_BACKEND_OPENCV
        target = cv2.dnn.DNN_TARGET_CPU
        if use_gpu and cuda_ok:
            try:
                backend = cv2.dnn.DNN_BACKEND_CUDA
                target = cv2.dnn.DNN_TARGET_CUDA
                net.setPreferableBackend(backend)
                net.setPreferableTarget(target)
                print(f"[Detector:{name}] using CUDA backend for YOLO")
            except Exception as e:
                print(f"[Detector:{name}] CUDA not available, falling back to CPU: {e}")
                backend = cv2.dnn.DNN_BACKEND_OPENCV
                target = cv2.dnn.DNN_TARGET_CPU
        net.setPreferableBackend(backend)
        net.setPreferableTarget(target)
        return net
    except Exception as e:
        print(f"[Detector:{name}] YOLO load failed: {e}")
        return None


def load_face_detectors(
    *,
    face_model: Optional[str],
    face_cascade: Optional[str],
    name: str,
    use_gpu: bool,
    cuda_ok: bool,
) -> Tuple[Optional[cv2.CascadeClassifier], Optional[object], str, object, object]:
    face = None
    face_dnn = None
    face_mode = "none"  # dnn | haar | none
    face_backend = None
    face_target = None

    if face_model and os.path.exists(face_model):
        try:
            backend = cv2.dnn.DNN_BACKEND_OPENCV
            target = cv2.dnn.DNN_TARGET_CPU
            if use_gpu and cuda_ok:
                backend = cv2.dnn.DNN_BACKEND_CUDA
                target = cv2.dnn.DNN_TARGET_CUDA
            face_dnn = cv2.FaceDetectorYN.create(  # type: ignore[attr-defined]
                face_model,
                "",
                (320, 320),  # updated per frame to match actual size
                0.85,
                0.3,
                5000,
                backend,
                target,
            )
            print(
                f"[Detector:{name}] YuNet face detector loaded "
                f"(backend={backend_label(backend)}, target={target_label(target)})"
            )
            face_mode = "dnn"
            face_backend = backend
            face_target = target
        except Exception as e:
            print(f"[Detector:{name}] YuNet load failed, falling back to Haar: {e}")

    if face_dnn is None:
        if face_cascade and os.path.exists(face_cascade):
            try:
                face = cv2.CascadeClassifier(face_cascade)
                face_mode = "haar"
            except Exception as e:
                print(f"[Detector:{name}] Haar load failed: {e}")
                face = None
        else:
            print(f"[Detector:{name}] Haar cascade not found at {face_cascade}")

    return face, face_dnn, face_mode, face_backend, face_target

