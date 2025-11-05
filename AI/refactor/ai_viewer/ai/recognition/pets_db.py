#!/usr/bin/env python3
"""
ESP32-CAM MDI Viewer (Qt)

Multi-document interface (MDI) master application to manage multiple
ESP32-CAM feeds as independent, floating, resizable windows inside a
single main window. Includes a standard toolbar with camera management
and recording controls. Supports basic MJPEG streaming with optional
Basic-Auth or token, plus pre-buffered video capture per camera.

Dependencies (install on your PC):
  - pip install PySide6 requests opencv-python numpy

Notes:
  - This is a first pass skeleton designed to get the MDI scaffolding,
    multi-camera streaming, and pre-buffered recording in place.
  - Face/pet recognition and the advanced UI from cam_ai.py can be
    integrated in phased steps by adding overlays and per-camera tool
    panels.
"""

from __future__ import annotations
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Tuple

import requests
import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets
import sys as _sys
_sys.path.append(os.path.dirname(__file__))  # allow local module imports
from gallery import GalleryDialog
import tools



class PetsDB:
    """Pet recognition store (dogs/cats) using ORB descriptors.
    - Persists samples as grayscale images under ai/data/pets/{dogs,cats}/<name>.
    """
    def __init__(self, base='ai/data/pets'):
        self.base=base; os.makedirs(os.path.join(base,'dogs'),exist_ok=True); os.makedirs(os.path.join(base,'cats'),exist_ok=True)
        self.orb=cv2.ORB_create(nfeatures=1000); self.bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.db={'dogs':{},'cats':{}}

