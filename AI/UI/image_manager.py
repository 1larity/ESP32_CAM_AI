# image_manager.py
# Faces and Pets manager with rename, delete, open folder, and Gallery launcher.
from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable, Optional

from PySide6 import QtWidgets

from utils import open_folder_or_warn
from settings import BASE_DIR
from UI.gallery import GalleryDialog


class ImageManagerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        app_cfg,
        parent=None,
        *,
        start_face_rebuild: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Image Manager")
        self.faces_root = BASE_DIR / "data" / "faces"
        self.pets_root = BASE_DIR / "data" / "pets"
        self._start_face_rebuild = start_face_rebuild

        self.tabs = QtWidgets.QTabWidget()
        self.faces_tab = self._build_tab(self.faces_root, is_pets=False)
        self.pets_tab = self._build_tab(self.pets_root, is_pets=True)
        self.tabs.addTab(self.faces_tab, "Faces")
        self.tabs.addTab(self.pets_tab, "Pets")

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.tabs)

    def _maybe_start_rebuild(self, title: str) -> None:
        cb = getattr(self, "_start_face_rebuild", None)
        if cb is None:
            return
        try:
            cb(title)
        except Exception:
            pass

    def _build_tab(self, root: Path, is_pets: bool) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lst = QtWidgets.QListWidget()
        lst.setObjectName("list")
        btn_open = QtWidgets.QPushButton("Open Folder")
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_rename = QtWidgets.QPushButton("Renameƒ?İ")
        btn_delete = QtWidgets.QPushButton("Deleteƒ?İ")
        btn_gallery = QtWidgets.QPushButton("Open Galleryƒ?İ")

        buttons = QtWidgets.QHBoxLayout()
        for b in (btn_open, btn_refresh, btn_rename, btn_delete, btn_gallery):
            buttons.addWidget(b)
        buttons.addStretch(1)

        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(lst)
        lay.addLayout(buttons)

        def _refresh():
            lst.clear()
            root.mkdir(parents=True, exist_ok=True)
            for d in sorted(root.glob("*")):
                if d.is_dir():
                    lst.addItem(d.name)

        def _sel_path():
            it = lst.currentItem()
            if not it:
                return None
            return root / it.text()

        def _open():
            open_folder_or_warn(self, root)

        def _rename():
            p = _sel_path()
            if not p:
                return
            new, ok = QtWidgets.QInputDialog.getText(
                self, "Rename", f"New name for '{p.name}':", text=p.name
            )
            if not ok or not new.strip():
                return
            dest = root / new.strip()
            try:
                dest.mkdir(parents=True, exist_ok=True)
                for f in p.glob("*"):
                    f.rename(dest / f.name)
                try:
                    p.rmdir()
                except Exception:
                    pass
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Rename", f"Failed to rename '{p.name}':\n{e}"
                )
                return
            _refresh()
            self._maybe_start_rebuild("Rebuilding face model after rename")

        def _delete():
            p = _sel_path()
            if not p:
                return

            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg.setWindowTitle("Delete enrollment data")
            msg.setText(f"Delete '{p.name}' and all samples?")
            msg.setInformativeText(
                "The LBPH recognition model will be retrained from the remaining data.\n\n"
                "This may take some time if a large number of people/pets have been enrolled."
            )
            btn_delete_confirm = msg.addButton(
                "Delete and retrain",
                QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
            )
            btn_cancel = msg.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            msg.setDefaultButton(btn_cancel)
            msg.exec()
            if msg.clickedButton() != btn_delete_confirm:
                return

            try:
                shutil.rmtree(p)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Delete", f"Failed to delete '{p.name}':\n{e}"
                )
                return
            _refresh()
            self._maybe_start_rebuild("Rebuilding face model after deletion")

        def _gallery():
            p = _sel_path()
            if not p:
                return

            def _snapshot(folder: Path) -> set[str]:
                pats = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
                out: set[str] = set()
                for pat in pats:
                    out.update([f.name for f in folder.glob(pat)])
                return out

            before = _snapshot(p)
            GalleryDialog(p, parent=self).exec()
            after = _snapshot(p)
            if before != after:
                self._maybe_start_rebuild("Rebuilding face model after image changes")

        btn_open.clicked.connect(_open)
        btn_refresh.clicked.connect(_refresh)
        btn_rename.clicked.connect(_rename)
        btn_delete.clicked.connect(_delete)
        btn_gallery.clicked.connect(_gallery)

        _refresh()
        return w
