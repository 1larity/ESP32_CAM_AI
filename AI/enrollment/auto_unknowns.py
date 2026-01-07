from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore


def promote_unknown(
    service,
    roi,
    cam_name: str,
    *,
    is_pet: bool,
    train_now: Callable[[], None],
) -> None:
    """
    Save unknown into auto_* folder for provisional training and kick rebuild.
    """
    try:
        from cv2 import imwrite

        label_prefix = "auto_pet" if is_pet else "auto_person"
        base_dir = getattr(service, "pet_dir") if is_pet else getattr(service, "face_dir")
        out_dir = base_dir / f"{label_prefix}_{cam_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{label_prefix}_{getattr(service, '_auto_label_idx', 1):05d}.jpg"
        service._auto_label_idx += 1
        imwrite(str(out_dir / fname), roi)

        # Kick rebuild asynchronously (faces only); pets are stored for the Pets manager.
        if not is_pet:
            QtCore.QTimer.singleShot(0, train_now)
    except Exception:
        pass


def bootstrap_auto_unknowns(service, *, train_now: Callable[[], None]) -> None:
    """
    If auto-train is enabled, seed auto_person/auto_pet folders from any existing
    unknown capture queues so recognition can start without waiting for new saves.
    """
    try:
        import cv2
    except Exception:
        return

    promoted = False

    # Faces
    for cam_dir in getattr(service, "unknown_face_dir").glob("*"):
        if not cam_dir.is_dir():
            continue
        dest = getattr(service, "face_dir") / f"auto_person_{cam_dir.name}"
        if dest.exists() and any(dest.glob("*.jpg")):
            continue
        files = sorted(cam_dir.glob("*.jpg"))
        dest.mkdir(parents=True, exist_ok=True)
        for f in files[: min(20, len(files))]:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            fname = f"{dest.name}_{getattr(service, '_auto_label_idx', 1):05d}.jpg"
            service._auto_label_idx += 1
            cv2.imwrite(str(dest / fname), img)
            promoted = True

    # Pets
    for cam_dir in getattr(service, "unknown_pet_dir").glob("*"):
        if not cam_dir.is_dir():
            continue
        dest = getattr(service, "pet_dir") / f"auto_pet_{cam_dir.name}"
        if dest.exists() and any(dest.glob("*.jpg")):
            continue
        files = sorted(cam_dir.glob("*.jpg"))
        dest.mkdir(parents=True, exist_ok=True)
        for f in files[: min(20, len(files))]:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            fname = f"{dest.name}_{getattr(service, '_auto_label_idx', 1):05d}.jpg"
            service._auto_label_idx += 1
            cv2.imwrite(str(dest / fname), img)
            promoted = True

    if promoted:
        QtCore.QTimer.singleShot(0, train_now)

