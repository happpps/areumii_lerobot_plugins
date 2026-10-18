from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from lerobot.cameras import CameraConfig, Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


def detect_areumii_camera_paths(
    device_dir: Path = Path("/dev/v4l/by-path"),
    sys_dir: Path = Path("/sys/class/video4linux"),
) -> dict[str, str]:
    """Discover RealSense RGB and wrists on sibling USB hub ports .2 and .3.

    Raises:
        RuntimeError: If a unique complete camera set cannot be identified.
    """
    candidates = []
    for head in sorted(device_dir.glob("*-usb-0:*:1.3-video-index0")):
        if not head.exists():
            continue
        name_file = sys_dir / head.resolve().name / "name"
        if not name_file.exists() or "realsense" not in name_file.read_text().lower():
            continue
        match = re.fullmatch(r"(.+-usb-0:)(.+)\.1:1\.3-video-index0", head.name)
        if match is None:
            continue
        prefix, hub = match.groups()
        paths = {"head": head}
        for role, port in (("left_wrist", 2), ("right_wrist", 3)):
            paths[role] = device_dir / f"{prefix}{hub}.{port}:1.0-video-index0"
        if not all(p.exists() for p in paths.values()):
            continue
        if not all(
            (sys_dir / paths[role].resolve().name / "name").exists()
            and "u20" in (sys_dir / paths[role].resolve().name / "name").read_text().lower()
            for role in ("left_wrist", "right_wrist")
        ):
            continue
        candidates.append({role: str(path) for role, path in paths.items()})
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one Areumii camera hub, found {len(candidates)}. "
            "Connect RealSense to hub port 1, left U20 to port 2, right U20 to port 3. "
            "Alternatively supply explicit --robot.cameras paths."
        )
    return candidates[0]


def default_areumii_cameras() -> dict[str, CameraConfig]:
    """Default physical camera mapping for the current Areumii USB hub setup.

    RealSense uses its RGB interface; wrists use hub ports .2 and .3.
    Paths are discovered on every connection, including after moving the hub.
    """
    return {
        "head": OpenCVCameraConfig(
            index_or_path=Path("auto:head"),
            fps=30,
            width=640,
            height=480,
            fourcc="YUYV",
            backend=Cv2Backends.V4L2,
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path=Path("auto:left_wrist"),
            fps=30,
            width=1280,
            height=720,
            fourcc="MJPG",
            backend=Cv2Backends.V4L2,
        ),

        "right_wrist": OpenCVCameraConfig(
            index_or_path=Path("auto:right_wrist"),
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

    # Discover devices at connect time; explicit camera paths also remain supported.
    cameras: dict[str, CameraConfig] = field(default_factory=default_areumii_cameras)
