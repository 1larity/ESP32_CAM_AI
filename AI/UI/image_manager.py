# image_manager.py
# Faces and Pets manager with rename, delete, open folder, and Gallery launcher.
from __future__ import annotations
from pathlib import Path
from PyQt6 import QtWidgets
from utils import open_folder_or_warn
from settings import BASE_DIR
from UI.gallery import GalleryDialog

class ImageManagerDialog(QtWidgets.QDialog):
    def __init__(self, app_cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Manager")
        self.faces_root = BASE_DIR / "data" / "faces"
        self.pets_root  = BASE_DIR / "data" / "pets"

        self.tabs = QtWidgets.QTabWidget()
        self.faces_tab = self._build_tab(self.faces_root, is_pets=False)
        self.pets_tab  = self._build_tab(self.pets_root,  is_pets=True)
        self.tabs.addTab(self.faces_tab, "Faces")
        self.tabs.addTab(self.pets_tab,  "Pets")

        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(self.tabs)

    def _build_tab(self, root: Path, is_pets: bool) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lst = QtWidgets.QListWidget(); lst.setObjectName("list")
        btn_open = QtWidgets.QPushButton("Open Folder"); btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_rename = QtWidgets.QPushButton("Rename…"); btn_delete = QtWidgets.QPushButton("Delete…")
        btn_gallery = QtWidgets.QPushButton("Open Gallery…")

        buttons = QtWidgets.QHBoxLayout()
        for b in (btn_open, btn_refresh, btn_rename, btn_delete, btn_gallery): buttons.addWidget(b)
        buttons.addStretch(1)

        lay = QtWidgets.QVBoxLayout(w); lay.addWidget(lst); lay.addLayout(buttons)

        def _refresh():
            lst.clear(); root.mkdir(parents=True, exist_ok=True)
            for d in sorted(root.glob("*")):
                if d.is_dir(): lst.addItem(d.name)
        def _sel_path():
            it = lst.currentItem()
            if not it: return None
            return root / it.text()
        def _open():
            open_folder_or_warn(self, root)
        def _rename():
            p = _sel_path()
            if not p: return
            new, ok = QtWidgets.QInputDialog.getText(self, "Rename", f"New name for '{p.name}':", text=p.name)
            if not ok or not new.strip(): return
            (root / new.strip()).mkdir(parents=True, exist_ok=True)
            for f in p.glob("*"): f.rename(root / new.strip() / f.name)
            try: p.rmdir()
            except Exception: pass
            _refresh()
        def _delete():
            p = _sel_path()
            if not p: return
            if QtWidgets.QMessageBox.question(self, "Delete", f"Delete '{p.name}' and all samples?") != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            for f in p.glob("**/*"): f.unlink(missing_ok=True)
            try: p.rmdir()
            except Exception: pass
            _refresh()
        def _gallery():
            p = _sel_path()
            if not p: return
            GalleryDialog(p, parent=self).exec()

        btn_open.clicked.connect(_open)
        btn_refresh.clicked.connect(_refresh)
        btn_rename.clicked.connect(_rename)
        btn_delete.clicked.connect(_delete)
        btn_gallery.clicked.connect(_gallery)

        _refresh()
        return w
