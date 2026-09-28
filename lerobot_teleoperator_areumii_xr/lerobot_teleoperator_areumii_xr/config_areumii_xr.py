from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("areumii_xr")
@dataclass
class AreumiiXRConfig(TeleoperatorConfig):
    urdf: Path | None = None

    cloudxr_env: str | None = None
    auto_launch_cloudxr: bool = True
    clutch_threshold: float = 0.5

    orientation_weight: float = 0.01
    max_ee_step_m: float = 0.005
    max_joint_step_rad: float = 0.02
    max_gripper_step_rad: float = 0.05
    feedback_timeout_s: float = 0.20
    warn_period_s: float = 0.25

    # Defaults copied from the current real teleop script.
    left_gripper_open: float = -1.1
    left_gripper_close: float = 0.8
    right_gripper_open: float = -1.1
    right_gripper_close: float = 0.8
