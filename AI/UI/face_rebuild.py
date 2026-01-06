from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal, Slot

from enrollment import get_enrollment_service


class _FaceRebuildWorker(QtCore.QObject):
    """
    Runs the LBPH model rebuild in a background thread.
    """

    finished = Signal(bool)

    @Slot()
    def run(self) -> None:
        svc = get_enrollment_service()
        ok = svc.rebuild_lbph_model_from_disk()
        self.finished.emit(ok)


class FaceRebuildController(QtCore.QObject):
    def __init__(self, host: QtWidgets.QWidget):
        super().__init__(host)
        self._host = host
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_FaceRebuildWorker] = None
        self._dialog: Optional[QtWidgets.QProgressDialog] = None

    def start(self, title: str) -> None:
        # Avoid re-entrancy; if a rebuild is already in progress, ignore.
        if self._thread is not None:
            return

        # Progress dialog (indeterminate)
        dlg = QtWidgets.QProgressDialog(
            "Rebuilding face model from disk…", "", 0, 0, self._host
        )
        dlg.setWindowTitle(title)
        dlg.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()

        self._dialog = dlg

        # Background thread + worker
        thread = QtCore.QThread(self._host)
        worker = _FaceRebuildWorker()
        worker.moveToThread(thread)

        worker.finished.connect(self._on_finished)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)

        self._thread = thread
        self._worker = worker

        thread.start()

    @Slot(bool)
    def _on_finished(self, ok: bool) -> None:
        # Close the progress dialog
        if self._dialog is not None:
            self._dialog.close()
            self._dialog = None

        # Clean up worker/thread handles
        self._worker = None
        self._thread = None

        # Inform the user of the result
        if ok:
            QtWidgets.QMessageBox.information(
                self._host,
                "Rebuild Face Model",
                "LBPH model rebuilt from disk samples.",
            )
        else:
            QtWidgets.QMessageBox.information(
                self._host,
                "Rebuild Face Model",
                "No face samples found to rebuild.",
            )

