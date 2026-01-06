from __future__ import annotations

import datetime
import shutil
from typing import Callable

from PySide6 import QtWidgets

from settings import BASE_DIR


def archive_person_folder(
    parent: QtWidgets.QWidget, start_face_rebuild: Callable[[str], None]
) -> None:
    """
    Move a named person/pet folder out of data/faces into an archive,
    then rebuild LBPH without that data.
    """
    name, ok = QtWidgets.QInputDialog.getText(
        parent,
        "Archive person/pet",
        "Folder name under data/faces to archive:",
    )
    if not ok:
        return
    name = name.strip()
    if not name:
        return

    src = BASE_DIR / "data" / "faces" / name
    if not src.exists() or not src.is_dir():
        QtWidgets.QMessageBox.warning(
            parent,
            "Archive person/pet",
            f"Folder not found: {src}",
        )
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = BASE_DIR / "data" / "archive" / "faces"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{name}_{ts}"
    try:
        shutil.move(str(src), str(dest))
    except Exception as e:
        QtWidgets.QMessageBox.warning(
            parent,
            "Archive person/pet",
            f"Failed to archive {name}:\n{e}",
        )
        return

    QtWidgets.QMessageBox.information(
        parent,
        "Archive person/pet",
        f"Archived {name} to:\n{dest}\n\nRebuilding face model without this data...",
    )

    start_face_rebuild(f"Rebuilding face model without {name}")


def restore_person_folder(
    parent: QtWidgets.QWidget, start_face_rebuild: Callable[[str], None]
) -> None:
    """
    Restore a previously archived person/pet folder back into data/faces and rebuild.
    """
    archive_root = BASE_DIR / "data" / "archive" / "faces"
    if not archive_root.exists():
        QtWidgets.QMessageBox.warning(
            parent,
            "Restore person/pet",
            f"No archive folder found at:\n{archive_root}",
        )
        return

    # List available archives
    archives = sorted([p.name for p in archive_root.iterdir() if p.is_dir()])
    if not archives:
        QtWidgets.QMessageBox.information(
            parent,
            "Restore person/pet",
            "No archived folders found.",
        )
        return

    name, ok = QtWidgets.QInputDialog.getItem(
        parent,
        "Restore person/pet",
        "Select archive to restore:",
        archives,
        editable=False,
    )
    if not ok or not name:
        return

    src = archive_root / name
    target_name = name.split("_")[0] if "_" in name else name
    dest = BASE_DIR / "data" / "faces" / target_name

    if dest.exists():
        QtWidgets.QMessageBox.warning(
            parent,
            "Restore person/pet",
            f"Destination already exists:\n{dest}\nRemove/rename it first.",
        )
        return

    try:
        shutil.move(str(src), str(dest))
    except Exception as e:
        QtWidgets.QMessageBox.warning(
            parent,
            "Restore person/pet",
            f"Failed to restore {name}:\n{e}",
        )
        return

    QtWidgets.QMessageBox.information(
        parent,
        "Restore person/pet",
        f"Restored to:\n{dest}\n\nRebuilding face model...",
    )
    start_face_rebuild(f"Rebuilding face model with {target_name}")


def purge_auto_unknowns(
    parent: QtWidgets.QWidget, start_face_rebuild: Callable[[str], None]
) -> None:
    """
    Remove auto-trained unknowns (auto_person_*/auto_pet_*) and rebuild LBPH.
    """
    face_root = BASE_DIR / "data" / "faces"
    pet_root = BASE_DIR / "data" / "pets"
    removed = []
    for child in face_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("auto_person_") or name.startswith("auto_pet_"):
            try:
                shutil.rmtree(child)
                removed.append(name)
            except Exception:
                continue
    # Auto-trained pets belong under data/pets; also remove them there.
    try:
        for child in pet_root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("auto_pet_"):
                try:
                    shutil.rmtree(child)
                    removed.append(name)
                except Exception:
                    continue
    except Exception:
        pass
    msg = (
        "No auto-trained folders found."
        if not removed
        else f"Removed: {', '.join(removed)}"
    )
    QtWidgets.QMessageBox.information(
        parent,
        "Purge auto-trained unknowns",
        f"{msg}\nRebuilding face model...",
    )
    start_face_rebuild("Rebuilding face model without auto-trained unknowns")

