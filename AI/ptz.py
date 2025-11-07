# ptz.py
# PTZ client stub.
from __future__ import annotations
from settings import CameraSettings

class PTZClient:
    def __init__(self, cam: CameraSettings):
        self.cam = cam

    def nudge(self, dx: int, dy: int):
        return
