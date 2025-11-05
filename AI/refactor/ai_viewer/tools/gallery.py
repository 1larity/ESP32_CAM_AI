"""Qt thumbnail gallery dialog for selecting and deleting images."""
from __future__ import annotations
import os
from PySide6 import QtCore, QtGui, QtWidgets


class GalleryDialog(QtWidgets.QDialog):
    def __init__(self, dir_path: str, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        self.dir_path = dir_path
        v = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        self.list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list.setIconSize(QtCore.QSize(160, 120))
        self.list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self.list, 1)
        btns = QtWidgets.QDialogButtonBox()
        self.btn_del = btns.addButton('Delete Selected', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.btn_close = btns.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        v.addWidget(btns)
        self.btn_del.clicked.connect(self.do_delete)
        self.btn_close.clicked.connect(self.accept)
        self.populate()

    def populate(self):
        self.list.clear()
        if not os.path.isdir(self.dir_path):
            return
        files = [os.path.join(self.dir_path,f) for f in os.listdir(self.dir_path)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for fp in files:
            item = QtWidgets.QListWidgetItem(os.path.basename(fp))
            try:
                pix = QtGui.QPixmap(fp)
                if not pix.isNull():
                    item.setIcon(QtGui.QIcon(pix.scaled(160,120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)))
            except Exception:
                pass
            item.setData(QtCore.Qt.ItemDataRole.UserRole, fp)
            self.list.addItem(item)

    def do_delete(self):
        items = self.list.selectedItems()
        if not items:
            return
        if QtWidgets.QMessageBox.question(self, 'Delete', f'Delete {len(items)} images?') != QtWidgets.QMessageBox.Yes:
            return
        cnt=0
        for it in items:
            fp = it.data(QtCore.Qt.ItemDataRole.UserRole)
            try:
                os.remove(fp)
                cnt+=1
            except Exception:
                pass
        self.populate()
        QtWidgets.QMessageBox.information(self, 'Manage', f'Deleted {cnt} files')

