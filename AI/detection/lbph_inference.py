from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import time

import cv2
import numpy as np

from .packet import DetBox

# Default distance threshold for LBPH recognition. Larger values are more lenient.
LBPH_DEFAULT_THRESHOLD = 140.0


def run_faces(
    bgr: np.ndarray,
    cascade: Optional[cv2.CascadeClassifier],
    rec: Optional[object],
    labels: Dict[int, str],
) -> Tuple[List[DetBox], int]:
    """
    Run Haar cascade + optional LBPH recognition.
    Returns (face_boxes, elapsed_ms).
    """
    start = time.monotonic()
    faces_out: List[DetBox] = []

    if cascade is None:
        return faces_out, 0

    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        try:
            eq = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        except Exception:
            eq = cv2.equalizeHist(gray)

        minsz = max(40, int(0.12 * min(gray.shape[:2])))
        faces = cascade.detectMultiScale(eq, 1.1, 4, minSize=(minsz, minsz))

        for (fx, fy, fw, fh) in faces:
            name = "face"
            score = 0.6
            if rec is not None and labels:
                try:
                    roi = gray[fy : fy + fh, fx : fx + fw]
                    roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
                    pred, dist = rec.predict(roi)
                    threshold = LBPH_DEFAULT_THRESHOLD
                    if 0 <= pred and dist <= threshold:
                        label_name = labels.get(int(pred), "face")
                        name = label_name
                        # map distance ƒÅ' [0.3, 1.0]
                        score = max(0.3, min(1.0, (threshold - dist) / threshold + 0.3))
                    else:
                        name = "unknown"
                        score = 0.4
                    print(
                        f"[LBPH] pred={pred} name={labels.get(int(pred), '?')} "
                        f"dist={dist:.1f} -> {name}"
                    )
                except Exception as e:
                    print(f"[LBPH] predict error: {e}")
                    name = "face"
                    score = 0.6

            x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
            faces_out.append(DetBox(name, float(score), (x1, y1, x2, y2)))
    except Exception as e:
        print(f"[Faces] error: {e}")

    elapsed_ms = int((time.monotonic() - start) * 1000.0)
    return faces_out, elapsed_ms


def run_faces_dnn(
    bgr: np.ndarray,
    detector: Optional[object],
    rec: Optional[object],
    labels: Dict[int, str],
    score_threshold: float = 0.85,
) -> Tuple[List[DetBox], int]:
    """
    Run YuNet (FaceDetectorYN) + optional LBPH recognition.
    Returns (face_boxes, elapsed_ms).
    """
    start = time.monotonic()
    faces_out: List[DetBox] = []
    if detector is None:
        return faces_out, 0

    try:
        # YuNet requires the input size to match the current frame.
        H, W = bgr.shape[:2]
        detector.setInputSize((W, H))
        _, faces = detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return faces_out, int((time.monotonic() - start) * 1000.0)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for row in faces:
            x, y, w, h, score = row[:5]
            if score < score_threshold:
                continue
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            name = "face"
            conf = float(score)

            if rec is not None and labels:
                try:
                    roi = gray[y1:y2, x1:x2]
                    if roi.size > 0:
                        roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
                        pred, dist = rec.predict(roi)
                        threshold = LBPH_DEFAULT_THRESHOLD
                        if 0 <= pred and dist <= threshold:
                            label_name = labels.get(int(pred), "face")
                            name = label_name
                            conf = max(0.3, min(1.0, (threshold - dist) / threshold + 0.3))
                        else:
                            name = "unknown"
                            conf = 0.4
                except Exception as e:
                    print(f"[LBPH] predict error (DNN faces): {e}")

            faces_out.append(DetBox(name, conf, (x1, y1, x2, y2)))
    except Exception as e:
        print(f"[Faces][DNN] error: {e}")

    elapsed_ms = int((time.monotonic() - start) * 1000.0)
    return faces_out, elapsed_ms


__all__ = ["LBPH_DEFAULT_THRESHOLD", "run_faces", "run_faces_dnn"]

