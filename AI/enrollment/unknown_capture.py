from __future__ import annotations

from pathlib import Path

import numpy as np


def count_unknown(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.jpg")))


def maybe_save_unknowns(service, cam_name: str, bgr: np.ndarray, pkt, now_ms: int) -> None:
    """Persist unknown faces/pets into queue folders for later training."""
    if getattr(service, "collect_unknown_faces", False):
        for face in getattr(pkt, "faces", []) or []:
            label = (face.cls or "").lower()
            if label in ("", "face", "unknown", "person"):
                last = getattr(service, "_last_unknown_face", {}).get(cam_name, 0)
                if now_ms - last < 800:
                    continue
                # enforce per-cam limit
                if (
                    count_unknown(getattr(service, "unknown_face_dir") / cam_name)
                    >= getattr(service, "unknown_capture_limit", 50)
                ):
                    continue
                getattr(service, "_last_unknown_face")[cam_name] = now_ms
                try:
                    x1, y1, x2, y2 = map(int, face.xyxy)
                    roi = bgr[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                    if roi.size == 0:
                        continue
                    out_dir = getattr(service, "unknown_face_dir") / cam_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{cam_name}_{now_ms}.jpg"
                    from cv2 import imwrite

                    imwrite(str(out_dir / fname), roi)
                    if getattr(service, "auto_train_unknowns", False):
                        service._promote_unknown(roi, cam_name, is_pet=False)
                except Exception:
                    continue

    if getattr(service, "collect_unknown_pets", False):
        for pet in getattr(pkt, "pets", []) or []:
            last = getattr(service, "_last_unknown_pet", {}).get(cam_name, 0)
            if now_ms - last < 800:
                continue
            if (
                count_unknown(getattr(service, "unknown_pet_dir") / cam_name)
                >= getattr(service, "unknown_capture_limit", 50)
            ):
                continue
            getattr(service, "_last_unknown_pet")[cam_name] = now_ms
            try:
                x1, y1, x2, y2 = map(int, pet.xyxy)
                roi = bgr[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if roi.size == 0:
                    continue
                out_dir = getattr(service, "unknown_pet_dir") / cam_name
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{cam_name}_{now_ms}_{pet.cls}.jpg"
                from cv2 import imwrite

                imwrite(str(out_dir / fname), roi)
                if getattr(service, "auto_train_unknowns", False):
                    service._promote_unknown(roi, cam_name, is_pet=True)
            except Exception:
                continue

