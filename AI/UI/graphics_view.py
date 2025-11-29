from __future__ import annotations
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal, Slot

class GraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = QtCore.Signal(float)

    def __init__(self, scene: QtWidgets.QGraphicsScene, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(scene, parent)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
            | QtGui.QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self._scale = 1.0
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setMouseTracking(True)

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.0 + (0.0015 * e.angleDelta().y())
            self._scale = float(max(0.1, min(8.0, self._scale * factor)))
            target = self.mapToScene(e.position().toPoint())
            self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
            self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
            self.setTransform(QtGui.QTransform())
            self.scale(self._scale, self._scale)
            self.centerOn(target)
            self.zoomChanged.emit(self._scale)
        else:
            super().wheelEvent(e)
