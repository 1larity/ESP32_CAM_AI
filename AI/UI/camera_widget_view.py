from __future__ import annotations

from PyQt6 import QtCore, QtWidgets, QtGui


def attach_view_handlers(cls) -> None:
    """
    Inject fit/zoom/lock/scrollbar helpers into CameraWidget.

    Assumes CameraWidget defines:
      - self.view              : QGraphicsView
      - self._scene            : QGraphicsScene
      - self._pixmap_item      : QGraphicsPixmapItem
      - self.btn_rec / btn_snap / btn_overlay_menu / btn_ai_menu / btn_view_menu
      - self.act_ai_* / self.act_overlay_*
      - self._locked           : bool
      - self._subwindow        : QMdiSubWindow | None
      - self._locked_geometry  : QRect
    """

    def _find_mdi_subwindow(self) -> QtWidgets.QMdiSubWindow | None:
        """
        Walk up the parent chain to find the QMdiSubWindow that hosts this widget.
        """
        w: QtWidgets.QWidget | None = self.parentWidget()
        while w is not None:
            if isinstance(w, QtWidgets.QMdiSubWindow):
                return w
            w = w.parentWidget()
        return None

    def fit_window_to_video(self) -> None:
        """
        Resize the camera subwindow so that the visible video area matches
        the current frame size at the current zoom level.
        """
        if self._pixmap_item.pixmap().isNull():
            return

        pixmap = self._pixmap_item.pixmap()
        video_w = pixmap.width()
        video_h = pixmap.height()

        if video_w <= 0 or video_h <= 0:
            return

        # Current zoom factor (GraphicsView may track this on _scale)
        scale = getattr(self.view, "_scale", 1.0)
        if scale <= 0:
            scale = 1.0

        logical_w = int(video_w * scale)
        logical_h = int(video_h * scale)

        # Size of the camera widget vs the view area
        widget_size = self.size()
        view_size = self.view.size()

        overhead_w = widget_size.width() - view_size.width()
        overhead_h = widget_size.height() - view_size.height()

        desired_widget_w = logical_w + overhead_w
        desired_widget_h = logical_h + overhead_h

        # Find the QMdiSubWindow container, if any
        sub = _find_mdi_subwindow(self)
        if sub is None:
            # Not in MDI, resize the widget itself
            self.resize(desired_widget_w, desired_widget_h)
            QtCore.QTimer.singleShot(0, self._update_scrollbars)
            return

        # Include the subwindow frame/border overhead so the outer window
        # actually fits the child widget size we just computed.
        sub_size = sub.size()
        frame_w = sub_size.width() - widget_size.width()
        frame_h = sub_size.height() - widget_size.height()

        geom = sub.geometry()
        geom.setWidth(desired_widget_w + frame_w)
        geom.setHeight(desired_widget_h + frame_h)
        sub.setGeometry(geom)

        QtCore.QTimer.singleShot(0, self._update_scrollbars)

    def _on_lock_toggled(self, checked: bool) -> None:
        """
        Lock/unlock the camera subwindow.

        When locked:
          - Disable controls/menus.
          - Prevent the QMdiSubWindow from being moved/resized.
        """
        self._locked = bool(checked)

        if getattr(self, "_subwindow", None) is None:
            w = _find_mdi_subwindow(self)
            if w is not None:
                self._subwindow = w
                self._subwindow.installEventFilter(self)

        if self._subwindow is not None:
            if self._locked:
                self._locked_geometry = self._subwindow.geometry()
            else:
                self._locked_geometry = QtCore.QRect()

        self._update_lock_state()

    def _update_lock_state(self) -> None:
        """Enable/disable all interactive controls according to lock state."""
        locked = bool(getattr(self, "_locked", False))

        # Toolbar buttons
        for btn in (
            getattr(self, "btn_rec", None),
            getattr(self, "btn_snap", None),
            getattr(self, "btn_overlay_menu", None),
            getattr(self, "btn_ai_menu", None),
            getattr(self, "btn_view_menu", None),
        ):
            if btn is not None:
                btn.setEnabled(not locked)


        # AI + overlay menu actions
        for act in (
            getattr(self, "act_ai_enabled", None),
            getattr(self, "act_ai_yolo", None),
            getattr(self, "act_ai_faces", None),
            getattr(self, "act_ai_pets", None),
            getattr(self, "act_overlay_detections", None),
            getattr(self, "act_overlay_hud", None),
        ):
            if act is not None:
                act.setEnabled(not locked)

    def _update_scrollbars(self) -> None:
        """
        Hide scrollbars when the view fully contains the scene rect.

        If the content is larger than the viewport at the current zoom,
        scrollbars are enabled as needed.
        """
        view: QtWidgets.QGraphicsView = self.view
        view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        if self._pixmap_item.pixmap().isNull():
            return

        visible_rect = view.mapToScene(view.viewport().rect()).boundingRect()
        scene_rect = self._scene.sceneRect()

        if visible_rect.contains(scene_rect):
            view.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            view.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )

    def eventFilter(
        self, obj: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:  # type: ignore[override]
        """
        Prevent moving/resizing the subwindow while locked.
        """
        if obj is getattr(self, "_subwindow", None) and getattr(
            self, "_locked", False
        ):
            if event.type() in (
                QtCore.QEvent.Type.Move,
                QtCore.QEvent.Type.Resize,
            ):
                if (
                    getattr(self, "_locked_geometry", None)
                    and not self._locked_geometry.isNull()
                ):
                    self._subwindow.setGeometry(self._locked_geometry)
                return True

        # Fall back to QWidget's eventFilter
        return QtWidgets.QWidget.eventFilter(self, obj, event)

    def resizeEvent(
        self, event: QtGui.QResizeEvent
    ) -> None:  # type: ignore[override]
        """
        After any resize, update scrollbars to reflect whether they are needed.
        """
        QtWidgets.QWidget.resizeEvent(self, event)
        QtCore.QTimer.singleShot(0, self._update_scrollbars)

    def fit_to_window(self) -> None:
        """
        Fit the video into the view so the entire scene is visible.
        """
        if self._scene is None:
            return
        self.view.fitInView(
            self._scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        # Reset our logical scale tracking
        self.view._scale = 1.0
        self._update_scrollbars()

    def zoom_100(self) -> None:
        """
        Reset zoom to 100% and update scrollbars.
        """
        self.view.resetTransform()
        self.view._scale = 1.0
        self._update_scrollbars()

    # Bind helpers into the target class
    cls.fit_window_to_video = fit_window_to_video
    cls._on_lock_toggled = _on_lock_toggled
    cls._update_lock_state = _update_lock_state
    cls._update_scrollbars = _update_scrollbars
    cls.eventFilter = eventFilter
    cls.resizeEvent = resizeEvent
    cls.fit_to_window = fit_to_window
    cls.zoom_100 = zoom_100
