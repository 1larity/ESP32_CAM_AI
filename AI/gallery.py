# gallery.py
# Loads PNG/JPG; robust thumbnails for grayscale; fixes empty view issues.
from __future__ import annotations
from pathlib import Path
from typing import List
import cv2 as cv, numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore

def _thumb(path: Path, max_size: int = 160) -> QtGui.QPixmap:
    im = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if im is None:
        return QtGui.QPixmap()
    if im.ndim == 2:
        im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    elif im.shape[2] == 4:
        im = cv.cvtColor(im, cv.COLOR_BGRA2RGB)
    else:
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    s = max_size / float(max(h, w)) if max(h, w) else 1.0
    im = cv.resize(im, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv.INTER_AREA)
    qimg = QtGui.QImage(im.data, im.shape[1], im.shape[0], int(im.strides[0]), QtGui.QImage.Format.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)

class GalleryDialog(QtWidgets.QDialog):
    def __init__(self, folder: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Gallery — {Path(folder).name}")
        self.folder = Path(folder)
        self.view = QtWidgets.QListWidget()
        self.view.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.view.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.view.setIconSize(QtCore.QSize(160, 160))
        self.view.setMovement(QtWidgets.QListView.Movement.Static)
        self.view.setSpacing(8)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.btn_del = QtWidgets.QPushButton("Delete Selected")
        self.btn_prune = QtWidgets.QPushButton("Self-Prune Near Duplicates")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_refresh); btns.addStretch(1); btns.addWidget(self.btn_prune); btns.addWidget(self.btn_del)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.view); lay.addLayout(btns)

        self.btn_refresh.clicked.connect(self._load)
        self.btn_del.clicked.connect(self._delete_selected)
        self.btn_prune.clicked.connect(self._self_prune)
        self._load()

    def _load(self):
        self.view.clear()
        pats = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        files: List[Path] = []
        for p in pats:
            files += sorted(self.folder.glob(p))
        for f in files:
            pm = _thumb(f)
            it = QtWidgets.QListWidgetItem(QtGui.QIcon(pm), f.name)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, str(f))
            self.view.addItem(it)

    def _delete_selected(self):
        sel = self.view.selectedItems()
        for it in sel:
            Path(it.data(QtCore.Qt.ItemDataRole.UserRole)).unlink(missing_ok=True)
            self.view.takeItem(self.view.row(it))

    def _self_prune(self):
        # Fast ORB similarity prune
        files = [Path(self.view.item(i).data(QtCore.Qt.ItemDataRole.UserRole)) for i in range(self.view.count())]
        imgs = []
        for f in files:
            im = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
            if im is None: continue
            imgs.append((f, cv.resize(im, (160, 160), interpolation=cv.INTER_AREA)))
        if len(imgs) < 2:
            QtWidgets.QMessageBox.information(self, "Self-Prune", "Not enough images to compare.")
            return
        orb = cv.ORB_create()
        kept = []
        pruned = 0
        for f, im in imgs:
            k, d = orb.detectAndCompute(im, None)
            if d is None or len(k) < 10:
                kept.append((f, k, d)); continue
            drop = False
            for fk, kk, dk in kept:
                if dk is None: continue
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
                m = bf.match(d, dk)
                if not m: continue
                dmean = float(np.mean([mm.distance for mm in m]))
                sim = 1.0 - (dmean / 100.0)
                if sim >= 0.82:
                    try: f.unlink()
                    except Exception: pass
                    pruned += 1
                    drop = True
                    break
            if not drop:
                kept.append((f, k, d))
        self._load()
        QtWidgets.QMessageBox.information(self, "Self-Prune", f"Removed {pruned} near-duplicates.")
