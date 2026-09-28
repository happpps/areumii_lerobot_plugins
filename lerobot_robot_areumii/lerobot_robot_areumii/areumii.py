from __future__ import annotations

import logging
from typing import Any

import numpy as np
import cv2
import time

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from .config_areumii import AreumiiConfig
from .shared_state import (
    FEATURE_KEYS,
    LEFT_ARM_SLICE,
    LEFT_GRIPPER_INDEX,
    RIGHT_ARM_SLICE,
    RIGHT_GRIPPER_INDEX,
    action_dict_to_vector,
    get_snapshot,
    update_from_packet32,
    vector_to_action_dict,
)
from .udp_bridge import UdpBridge

logger = logging.getLogger(__name__)

JOINT_LOWER = np.asarray([-3.142, -0.262, -1.396, -0.873, -1.745, -0.524, -1.571], dtype=np.float64)
JOINT_UPPER = np.asarray([3.142, 2.967, 1.396, 2.094, 1.745, 0.524, 1.571], dtype=np.float64)

RESET_LEFT_Q = np.array(
    [0.5, 0.1, 0.1, 1.5, 0.0, 0.0, 0.7],
    dtype=np.float64,
)

RESET_RIGHT_Q = np.array(
    [0.5, 0.1, 0.1, 1.5, 0.0, 0.0, 0.7],
    dtype=np.float64,
)


class Areumii(Robot):
    """LeRobot follower wrapper around the existing Areumii UDP/CAN bridge."""

    config_class = AreumiiConfig
    name = "areumii"

    def __init__(self, config: AreumiiConfig):
        super().__init__(config)
        self.config = config
        self.bridge: UdpBridge | None = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._connected = False

    @property
    def _joint_features(self) -> dict[str, type]:
        return {key: float for key in FEATURE_KEYS}

    @property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        features = {}

        for key, cam in self.cameras.items():
            if key in ("left_wrist", "right_wrist"):
                features[key] = (180, 320, 3)
            else:
                features[key] = (cam.height, cam.width, 3)

        return features

    @property
    def observation_features(self) -> dict:
        return {**self._joint_features, **self._camera_features}

    @property
    def action_features(self) -> dict:
        return self._joint_features

    @property
    def is_connected(self) -> bool:
        return self._connected and self.bridge is not None and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return

        self.bridge = UdpBridge(
            cmd_host=self.config.cmd_host,
            cmd_port=self.config.cmd_port,
            feedback_host=self.config.feedback_host,
            feedback_port=self.config.feedback_port,
        )

        q = self.bridge.wait(self.config.connect_timeout_s)
        if q.shape != (32,):
            self.bridge.close()
            self.bridge = None
            raise RuntimeError(
                f"Areumii bridge returned {q.shape[0]}D feedback; this plugin expects the 32D arm+gripper bridge."
            )

        update_from_packet32(q)

        try:
            for cam in self.cameras.values():
                cam.connect()
        except Exception:
            self.bridge.close()
            self.bridge = None
            for cam in self.cameras.values():
                try:
                    if cam.is_connected:
                        cam.disconnect()
                except Exception:
                    pass
            raise

        self._connected = True
        mode = "COMMAND ENABLED" if self.config.enable_command else "DRY RUN (no motor command)"
        logger.info("Areumii connected: %s", mode)

    def _poll_fresh_packet(self) -> np.ndarray:
        if not self._connected or self.bridge is None:
            raise ConnectionError("Areumii is not connected.")

        q = self.bridge.poll()
        if q is None:
            raise RuntimeError("Areumii bridge has not produced feedback.")
        if q.shape != (32,):
            raise RuntimeError(f"Expected 32D feedback, got {q.shape[0]}D.")
        if self.bridge.age_s > self.config.feedback_timeout_s:
            raise TimeoutError(
                f"Areumii feedback stale: {self.bridge.age_s:.3f}s > {self.config.feedback_timeout_s:.3f}s"
            )

        update_from_packet32(q)
        return q
    
    def get_observation(self) -> dict[str, Any]:
        self._poll_fresh_packet()

        from .shared_state import get_snapshot

        snapshot = get_snapshot()
        if snapshot is None:
            raise RuntimeError("Areumii shared state was not initialized.")

        obs: dict[str, Any] = vector_to_action_dict(snapshot.measured)

        for cam_key, cam in self.cameras.items():
            frame = cam.async_read()

            if cam_key in ("left_wrist", "right_wrist"):
                # 먼저 작게 줄이고
                frame = cv2.resize(
                    frame,
                    (320, 180),
                    interpolation=cv2.INTER_AREA,
                )

                # 작은 이미지에서 회전
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            obs[cam_key] = frame

        return obs
    

    # def _validate_arm_limits(self, values: np.ndarray) -> None:
    #     for side, q7 in (("left", values[LEFT_ARM_SLICE]), ("right", values[RIGHT_ARM_SLICE])):
    #         bad = np.where((q7 < JOINT_LOWER) | (q7 > JOINT_UPPER))[0]
    #         if bad.size:
    #             raise ValueError(
    #                 f"{side} action outside hard joint limits at joints {bad.tolist()}: {np.round(q7, 4)}"
    #             )

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected or self.bridge is None:
            raise ConnectionError("Areumii is not connected.")

        values = action_dict_to_vector(action)

        values[LEFT_ARM_SLICE] = np.clip(
            values[LEFT_ARM_SLICE],
            JOINT_LOWER,
            JOINT_UPPER,
        )

        values[RIGHT_ARM_SLICE] = np.clip(
            values[RIGHT_ARM_SLICE],
            JOINT_LOWER,
            JOINT_UPPER,
        )

        # DRY RUN에서는 실제 command를 보내지 않으므로
        # feedback freshness 때문에 중단할 필요 없음.
        if not self.config.enable_command:
            return vector_to_action_dict(values)

        # 실제 모터 명령을 보낼 때만 최신 feedback을 다시 읽고 확인.
        self._poll_fresh_packet()

        self.bridge.send_arm("left", values[LEFT_ARM_SLICE])
        self.bridge.send_arm("right", values[RIGHT_ARM_SLICE])
        self.bridge.send_gripper(
            "left", values[LEFT_GRIPPER_INDEX]
        )
        self.bridge.send_gripper(
            "right", values[RIGHT_GRIPPER_INDEX]
        )

        return vector_to_action_dict(values)

    def reset_pose(
        self,
        duration_s: float = 3.0,
        hz: float = 50.0,
    ) -> None:
        """Move both arms to the dataset reset pose through the UDP bridge."""

        if not self._connected or self.bridge is None:
            raise ConnectionError("Areumii is not connected.")

        if not self.config.enable_command:
            print("[RESET] command disabled - reset skipped")
            return

        # Get the latest measured/control feedback from the existing bridge.
        self._poll_fresh_packet()

        from .shared_state import get_snapshot

        snapshot = get_snapshot()
        if snapshot is None:
            raise RuntimeError("No Areumii feedback available for reset.")

        # IMPORTANT:
        # Start from the CURRENT CONTROL target, not from encoder/measured q.
        # This preserves command continuity.
        left_start = np.asarray(
            snapshot.control[LEFT_ARM_SLICE],
            dtype=np.float64,
        ).copy()

        right_start = np.asarray(
            snapshot.control[RIGHT_ARM_SLICE],
            dtype=np.float64,
        ).copy()

        left_target = RESET_LEFT_Q.copy()
        right_target = RESET_RIGHT_Q.copy()

        steps = max(1, int(duration_s * hz))
        dt = 1.0 / hz

        print(
            "[RESET] start\n"
            f"  left : {np.round(left_start, 3)} "
            f"-> {np.round(left_target, 3)}\n"
            f"  right: {np.round(right_start, 3)} "
            f"-> {np.round(right_target, 3)}"
        )

        for i in range(1, steps + 1):
            # Keep consuming bridge feedback so stale detection remains meaningful.
            self._poll_fresh_packet()

            if self.bridge.age_s > self.config.feedback_timeout_s:
                raise TimeoutError(
                    "Areumii feedback became stale during reset."
                )

            u = i / steps

            # Smoothstep: velocity starts/ends smoothly instead of a plain linear ramp.
            s = u * u * (3.0 - 2.0 * u)

            left_q = left_start + s * (left_target - left_start)
            right_q = right_start + s * (right_target - right_start)

            # Final safety guard.
            left_q = np.clip(left_q, JOINT_LOWER, JOINT_UPPER)
            right_q = np.clip(right_q, JOINT_LOWER, JOINT_UPPER)

            self.bridge.send_arm("left", left_q)
            self.bridge.send_arm("right", right_q)

            time.sleep(dt)

        # Send exact final pose once more.
        self.bridge.send_arm("left", left_target)
        self.bridge.send_arm("right", right_target)

        # Refresh shared state after reset.
        time.sleep(0.05)
        self._poll_fresh_packet()

        print("[RESET] done")

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception:
                logger.exception("Failed to disconnect camera")

        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None

        self._connected = False
        logger.info("Areumii disconnected.")
