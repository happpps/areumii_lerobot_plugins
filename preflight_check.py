#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as md
import sys

print("Python:", sys.version.split()[0])

try:
    print("LeRobot:", md.version("lerobot"))
except Exception as e:
    print("LeRobot version: ERROR", e)

checks = [
    ("lerobot_robot_areumii", "Areumii robot plugin"),
    ("lerobot_teleoperator_areumii_xr", "Areumii XR teleoperator plugin"),
    ("isaac_teleop", "IsaacTeleop Python package"),
    ("isaacteleop", "IsaacTeleop retargeting package"),
]

for module, label in checks:
    try:
        __import__(module)
        print(f"[OK] {label}: {module}")
    except Exception as e:
        print(f"[FAIL] {label}: {module}: {type(e).__name__}: {e}")

try:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.cameras import Cv2Backends
    print("[OK] OpenCV camera API")
    print("     Cv2Backends.V4L2 =", int(Cv2Backends.V4L2))
except Exception as e:
    print("[FAIL] OpenCV camera API:", type(e).__name__, e)
