from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig, Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


def default_areumii_cameras() -> dict[str, CameraConfig]:
    """Default physical camera mapping for the current Areumii USB hub setup.

    head        : RealSense D435i RGB
    left_wrist  : U20 on hub port 2.2 (currently /dev/video10)
    right_wrist : U20 on hub port 2.3 (currently /dev/video12)

    Device indices 8, 10, and 12 and capture resolutions are intentionally fixed
    for the current recording setup. Device numbering may change after reboot.
    """
    return {
        "head": OpenCVCameraConfig(
            index_or_path=8,
            fps=30,
            width=640,
            height=480,
            fourcc="UYVY",
            backend=Cv2Backends.V4L2,
        ),

        "left_wrist": OpenCVCameraConfig(
            index_or_path=10,
            fps=30,
            width=1280,
            height=720,
            fourcc="MJPG",
            backend=Cv2Backends.V4L2,
        ),

        "right_wrist": OpenCVCameraConfig(
            index_or_path=12,
            fps=30,
            width=1280,
            height=720,
            fourcc="MJPG",
            backend=Cv2Backends.V4L2,
        ),
    }


@RobotConfig.register_subclass("areumii")
@dataclass
class AreumiiConfig(RobotConfig):
    """Configuration for the real bimanual Areumii follower robot."""

    cmd_host: str = "127.0.0.1"
    cmd_port: int = 5005
    feedback_host: str = "127.0.0.1"
    feedback_port: int = 5006

    connect_timeout_s: float = 5.0
    feedback_timeout_s: float = 0.20

    # Safety default: compute/record actions but do not transmit them.
    enable_command: bool = False
    require_32d_feedback: bool = True
    enforce_joint_limits: bool = True

    # Camera mapping is fixed here, so --robot.cameras=... is no longer needed.
    cameras: dict[str, CameraConfig] = field(default_factory=default_areumii_cameras)
