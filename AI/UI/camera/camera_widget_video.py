# camera/camera_widget_video.py
# Video polling + detection handling + overlay rendering (with cached overlay layer)

from __future__ import annotations

from .camera_widget_overlay_layer import attach_overlay_layer
from .camera_widget_video_loop import attach_video_loop_handlers
from .camera_widget_video_motion import attach_video_motion_handlers
from .camera_widget_video_streams import attach_video_stream_handlers


def attach_video_handlers(cls) -> None:
    """Inject frame / detector / overlay / HUD helpers into CameraWidget."""

    attach_overlay_layer(cls)
    attach_video_motion_handlers(cls)
    attach_video_loop_handlers(cls)
    attach_video_stream_handlers(cls)

