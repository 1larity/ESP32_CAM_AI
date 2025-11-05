# ui/dialogs.py

from PySide6 import QtWidgets
from ..core.config import CameraConfig

class AddCameraDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial: CameraConfig | None = None, title: str = 'Add Camera'):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        form = QtWidgets.QFormLayout(self)

        self.ed_name   = QtWidgets.QLineEdit(initial.name if initial else 'Camera')
        self.ed_host   = QtWidgets.QLineEdit(initial.host if initial else '192.168.1.100')
        self.ed_user   = QtWidgets.QLineEdit(initial.user or '')
        self.ed_pass   = QtWidgets.QLineEdit(initial.password or '')
        self.ed_pass.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ed_token  = QtWidgets.QLineEdit(initial.token or '')

        form.addRow('Name',   self.ed_name)
        form.addRow('Host',   self.ed_host)
        form.addRow('User',   self.ed_user)
        form.addRow('Pass',   self.ed_pass)
        form.addRow('Token',  self.ed_token)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_config(self) -> CameraConfig | None:
        if self.exec() == QtWidgets.QDialog.Accepted:
            return CameraConfig(
                name=self.ed_name.text().strip() or 'Camera',
                host=self.ed_host.text().strip(),
                user=self.ed_user.text().strip() or None,
                password=self.ed_pass.text(),
                token=self.ed_token.text().strip() or None
            )
        return None