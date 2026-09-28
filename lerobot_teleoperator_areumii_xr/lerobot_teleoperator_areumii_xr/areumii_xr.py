from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.teleoperator import Teleoperator

from isaac_teleop.clutch import Clutch
from isaac_teleop.config_isaac_teleop import XRControllerConfig
from isaac_teleop.teleop_xr_controller import XRController
from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
from isaacteleop.retargeting_engine.tensor_types import TransformMatrix
from isaacteleop.retargeting_engine.tensor_types.indices import ControllerInputIndex

from lerobot_robot_areumii.shared_state import (
    FEATURE_KEYS,
    LEFT_ARM_SLICE,
    LEFT_GRIPPER_INDEX,
    RIGHT_ARM_SLICE,
    RIGHT_GRIPPER_INDEX,
    get_snapshot,
    vector_to_action_dict,
)

from .config_areumii_xr import AreumiiXRConfig

logger = logging.getLogger(__name__)

JOINT_NAMES = {
    "left": [
        "shoulder_pitch_joint_L",
        "shoulder_roll_joint_L",
        "shouler_yaw_joint_L",  # current URDF spelling
        "elbow_joint_L",
        "wrist_roll_joint_L",
        "wrist_yaw_joint_L",
        "wrist_pitch_joint_L",
    ],
    "right": [
        "shoulder_pitch_joint_R",
        "shoulder_roll_joint_R",
        "shoulder_yaw_joint_R",
        "elbow_joint_R",
        "wrist_roll_joint_R",
        "wrist_yaw_joint_R",
        "wrist_pitch_joint_R",
    ],
}

TARGET_FRAME = {
    "left": "gripper_hand_link_L_1",
    "right": "gripper_hand_link_R_1",
}

JOINT_LOWER = np.asarray([-3.142, -0.262, -1.396, -0.873, -1.745, -0.524, -1.571], dtype=np.float64)
JOINT_UPPER = np.asarray([3.142, 2.967, 1.396, 2.094, 1.745, 0.524, 1.571], dtype=np.float64)
JOINT_MARGIN = 0.01

# IK safety guards.
# A position-only 7-DoF IK can occasionally jump to a very different branch or
# return a boundary solution.  Small single-joint limit violations are still
# clipped (standalone behavior), but clearly unstable candidates are held.
IK_MAX_BRANCH_JUMP_RAD = 0.35
IK_REJECT_LIMIT_COUNT = 2

# Keep redundant 7-DoF IK close to the current/previous posture.
# The EE position task has weight 1.0 in LeRobot; this weak task mainly acts
# in the null-space instead of fighting the Cartesian target.
IK_POSTURE_WEIGHT = 1e-3


class BimanualXRController(XRController):
    """One IsaacTeleop/OpenXR session exposing both controllers."""

    def _build_pipeline(self):
        controllers = ControllersSource(name="controllers")
        xform = ValueInput("base_T_anchor", TransformMatrix())
        transformed = controllers.transformed(xform.output("value"))
        return OutputCombiner({
            "left": transformed.output("controller_left"),
            "right": transformed.output("controller_right"),
        })

    def get_bimanual_action(self) -> dict[str, dict[str, Any] | None]:
        result = self._step(
            execution_events=self._running_events(),
            external_inputs=self._external_inputs,
        )

        actions: dict[str, dict[str, Any] | None] = {}
        for side in ("left", "right"):
            c = result[side]
            if getattr(c, "is_none", False):
                actions[side] = None
                continue

            try:
                actions[side] = {
                    "grip_pos": np.asarray(c[ControllerInputIndex.GRIP_POSITION], dtype=np.float64),
                    "grip_quat": np.asarray(c[ControllerInputIndex.GRIP_ORIENTATION], dtype=np.float64),
                    "squeeze": float(c[ControllerInputIndex.SQUEEZE_VALUE]),
                    "trigger": float(c[ControllerInputIndex.TRIGGER_VALUE]),
                }
            except (IndexError, KeyError, TypeError, ValueError):
                actions[side] = None

        return actions


class Kinematics:
    def __init__(self, urdf: str, frame: str, names: list[str]):
        self.names = list(names)
        self.kin = RobotKinematics(
            urdf_path=urdf,
            target_frame_name=frame,
            joint_names=self.names,
        )

        # LeRobot's RobotKinematics only adds an EE frame task.  For a 7-DoF
        # arm with orientation_weight=0 this leaves four unconstrained DoFs,
        # so Placo is free to wander to a very different joint-space branch.
        # Add a weak posture task whose target is refreshed to the current IK
        # seed before every solve.  It therefore acts as null-space
        # regularization rather than commanding a fixed home pose.
        self.posture_task = self.kin.solver.add_joints_task()
        self.posture_task.configure("seed_posture", "soft", IK_POSTURE_WEIGHT)

        # Enforce the limits contained in the URDF during the solve itself,
        # instead of relying only on the post-solve clip below.
        self.kin.solver.enable_joint_limits(True)

    def fk(self, q_rad: np.ndarray) -> np.ndarray:
        return self.kin.forward_kinematics(np.rad2deg(q_rad))

    def ik(self, q_seed_rad: np.ndarray, target_T: np.ndarray, orientation_weight: float) -> np.ndarray:
        q_seed_rad = np.asarray(q_seed_rad, dtype=np.float64)

        # JointsTask expects revolute-joint targets in radians.
        self.posture_task.set_joints({
            name: float(q_seed_rad[i])
            for i, name in enumerate(self.names)
        })

        q_deg = self.kin.inverse_kinematics(
            np.rad2deg(q_seed_rad),
            target_T,
            orientation_weight=orientation_weight,
        )
        return np.deg2rad(np.asarray(q_deg, dtype=np.float64))


def clamp_ee_step(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    norm = np.linalg.norm(delta)
    if norm <= max_step or norm == 0:
        return np.asarray(target, dtype=np.float64)
    return np.asarray(current, dtype=np.float64) + delta * (max_step / norm)


class AreumiiXR(Teleoperator):
    """XR -> IK -> 16D q_target teleoperator for lerobot-record.

    The Areumii Robot owns UDP feedback port 5006 and updates a process-local
    shared snapshot. This teleoperator reads that snapshot after each
    robot.get_observation(), so Robot and Teleoperator do not compete for the
    same UDP socket.
    """

    config_class = AreumiiXRConfig
    name = "areumii_xr"

    def __init__(self, config: AreumiiXRConfig):
        super().__init__(config)
        self.config = config

        if self.config.urdf is None:
            raise ValueError("--teleop.urdf is required for areumii_xr")

        self.urdf = str(self.config.urdf)
        self.xr: BimanualXRController | None = None
        self._connected = False
        self.kin = {
            side: Kinematics(self.urdf, TARGET_FRAME[side], JOINT_NAMES[side])
            for side in ("left", "right")
        }
        self.arms: dict[str, dict[str, Any]] = {}
        self._state_initialized = False

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in FEATURE_KEYS}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

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

        self.xr = BimanualXRController(
            XRControllerConfig(
                app_name="AreumiiLeRobotRecord",
                hand_side="right",
                clutch_threshold=self.config.clutch_threshold,
                cloudxr_env_file=self.config.cloudxr_env,
                auto_launch_cloudxr=self.config.auto_launch_cloudxr,
            )
        )
        logger.info("Connecting Areumii XR teleoperator...")
        self.xr.connect()
        self._connected = True
        logger.info("Areumii XR connected. squeeze=arm clutch, trigger=gripper.")

    def _require_fresh_snapshot(self):
        snapshot = get_snapshot()
        if snapshot is None:
            raise RuntimeError(
                "No Areumii robot state is available yet. lerobot-record should call "
                "robot.get_observation() before teleop.get_action()."
            )
        if snapshot.age_s > self.config.feedback_timeout_s:
            raise TimeoutError(f"Areumii shared feedback stale: {snapshot.age_s:.3f}s")
        return snapshot

    def _initialize_from_robot(self, snapshot) -> None:
        for side, arm_slice in (("left", LEFT_ARM_SLICE), ("right", RIGHT_ARM_SLICE)):
            q_measured = snapshot.measured[arm_slice].copy()
            q_control = snapshot.control[arm_slice].copy()
            T = self.kin[side].fk(q_measured)
            self.arms[side] = {
                "clutch": Clutch(T),
                "q_seed": q_measured.copy(),
                "q_cmd": q_control.copy(),
                "last_ee": T[:3, 3].copy(),
                "enabled": False,
                # Set after XR tracking loss while squeezed.  The operator must
                # release squeeze once before this arm is allowed to re-engage.
                "needs_rearm": False,
                "last_warn": 0.0,
            }
        self._state_initialized = True

    def _warn(self, side: str, message: str) -> None:
        st = self.arms[side]
        now = time.monotonic()
        if now - st["last_warn"] >= self.config.warn_period_s:
            logger.warning("[%s] %s", side.upper(), message)
            st["last_warn"] = now

    def _gripper_endpoints(self, side: str) -> tuple[float, float]:
        if side == "left":
            return self.config.left_gripper_open, self.config.left_gripper_close
        return self.config.right_gripper_open, self.config.right_gripper_close

    @staticmethod
    def _side_indices(side: str):
        if side == "left":
            return LEFT_ARM_SLICE, LEFT_GRIPPER_INDEX
        return RIGHT_ARM_SLICE, RIGHT_GRIPPER_INDEX

    def get_action(self) -> dict[str, float]:
        if not self._connected or self.xr is None:
            raise ConnectionError("AreumiiXR is not connected.")

        snapshot = self._require_fresh_snapshot()
        if not self._state_initialized:
            self._initialize_from_robot(snapshot)

        xr_actions = self.xr.get_bimanual_action()

        # Default action is the bridge's existing hold/control target.
        output = snapshot.control.copy()
        lo = JOINT_LOWER + JOINT_MARGIN
        hi = JOINT_UPPER - JOINT_MARGIN

        for side in ("left", "right"):
            arm_slice, grip_index = self._side_indices(side)
            st = self.arms[side]
            a = xr_actions[side]

            q_measured = snapshot.measured[arm_slice].copy()
            q_control = snapshot.control[arm_slice].copy()

            pose_valid = False
            pos = None
            quat = None

            if a is not None:
                pos = np.asarray(a["grip_pos"], dtype=np.float64)
                quat = np.asarray(a["grip_quat"], dtype=np.float64)
                quat_norm = np.linalg.norm(quat)
                pose_valid = (
                    np.all(np.isfinite(pos))
                    and np.all(np.isfinite(quat))
                    and quat_norm > 1e-6
                )
                if pose_valid:
                    quat = quat / quat_norm

            # Gripper follows trigger independently of squeeze/clutch.
            if pose_valid:
                gripper_open, gripper_close = self._gripper_endpoints(side)
                trigger = float(np.clip(a["trigger"], 0.0, 1.0))
                gripper_target = gripper_open + trigger * (gripper_close - gripper_open)
                current_gripper_control = float(snapshot.control[grip_index])
                output[grip_index] = float(
                    current_gripper_control
                    + np.clip(
                        gripper_target - current_gripper_control,
                        -self.config.max_gripper_step_rad,
                        self.config.max_gripper_step_rad,
                    )
                )

            squeeze_down = (
                a is not None
                and np.isfinite(float(a["squeeze"]))
                and float(a["squeeze"]) > self.config.clutch_threshold
            )

            # After XR tracking is lost while the clutch is engaged, do not
            # automatically create a new anchor just because tracking came back
            # while squeeze is still held.  Require one physical release first.
            if st["needs_rearm"]:
                if not squeeze_down:
                    st["needs_rearm"] = False
                    logger.info("[%s] clutch RE-ARMED after squeeze release", side.upper())
                enabled = False
            else:
                enabled = pose_valid and squeeze_down

            if enabled and not st["enabled"]:
                T = self.kin[side].fk(q_measured)
                st["clutch"].engage(pos, quat, measured_base_T_ee=T)
                st["q_seed"] = q_measured.copy()
                st["q_cmd"] = q_control.copy()
                st["last_ee"] = T[:3, 3].copy()
                logger.info(
                    "[%s] clutch ENGAGED measured=%s control=%s",
                    side.upper(), np.round(q_measured, 3), np.round(q_control, 3)
                )

            if enabled:
                ee_pos, _ = st["clutch"].rebase(pos, quat)

                R_xr = Rotation.from_quat(quat).as_matrix()

                # clutch 첫 프레임에 기준 저장
                if not st["enabled"]:
                    T_now = self.kin[side].fk(q_measured)
                    st["xr_R0"] = R_xr.copy()
                    st["robot_R0"] = T_now[:3, :3].copy()

                # XR의 상대 회전만 로봇에 적용
                R_delta = R_xr @ st["xr_R0"].T
                R_target = R_delta @ st["robot_R0"]

                ee_quat = Rotation.from_matrix(R_target).as_quat()

                # # DEBUG: clutch engage 직후 FK와 target이 같은지 확인
                # if not st["enabled"]:
                #     T_now = self.kin[side].fk(st["q_seed"])
                #     print(
                #         f"[{side.upper()} FIRST TARGET]",
                #         "FK =", np.round(T_now[:3, 3], 4),
                #         "target =", np.round(ee_pos, 4),
                #         "diff mm =",
                #         round(np.linalg.norm(np.asarray(ee_pos) - T_now[:3, 3]) * 1000, 2),
                #     )

                if not st["enabled"]:
                    T_now = self.kin[side].fk(st["q_seed"])

                    R_now = T_now[:3, :3]
                    R_target = Rotation.from_quat(ee_quat).as_matrix()

                    R_err = R_now.T @ R_target
                    angle_deg = np.degrees(Rotation.from_matrix(R_err).magnitude())

                    print(
                        f"[{side.upper()} FIRST TARGET]",
                        "pos diff mm =",
                        round(np.linalg.norm(np.asarray(ee_pos) - T_now[:3, 3]) * 1000, 2),
                        "rot diff deg =",
                        round(angle_deg, 2),
                    )


                # Candidate EE target for this cycle.
                # IMPORTANT: do not advance last_ee until the IK result is accepted.
                ee_candidate = clamp_ee_step(
                    st["last_ee"],
                    np.asarray(ee_pos, dtype=np.float64),
                    self.config.max_ee_step_m,
                )

                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = Rotation.from_quat(ee_quat).as_matrix()
                T[:3, 3] = ee_candidate

                q_ik = self.kin[side].ik(
                    st["q_seed"],
                    T,
                    self.config.orientation_weight,
                )

                if not np.all(np.isfinite(q_ik)):
                    self._warn(side, f"STOP: IK non-finite {np.round(q_ik, 3)}")

                else:
                    bad = np.where((q_ik < lo) | (q_ik > hi))[0]
                    raw_ik_delta = q_ik - st["q_seed"]
                    jump = np.where(np.abs(raw_ik_delta) > IK_MAX_BRANCH_JUMP_RAD)[0]

                    reject_reason = None
                    if jump.size:
                        reject_reason = (
                            f"IK branch jump joints={jump.tolist()} "
                            f"dq={np.round(raw_ik_delta, 3)}"
                        )
                    elif bad.size >= IK_REJECT_LIMIT_COUNT:
                        reject_reason = (
                            f"IK boundary solution joints={bad.tolist()} "
                            f"ik={np.round(q_ik, 3)}"
                        )

                    if reject_reason is not None:
                        # HOLD this cycle: do not change q_seed/q_cmd/last_ee.
                        # The bridge's existing control target remains in output.
                        self._warn(side, f"STOP: {reject_reason}")
                    else:
                        # Standalone-style behavior is preserved for a small
                        # single-joint violation: only that joint is clipped, while
                        # the rest of the IK solution keeps moving.
                        if bad.size:
                            q_safe = np.clip(q_ik, lo, hi)
                            self._warn(
                                side,
                                f"LIMIT CLIP joints={bad.tolist()} "
                                f"ik={np.round(q_ik, 3)} "
                                f"safe={np.round(q_safe, 3)}",
                            )
                        else:
                            q_safe = q_ik

                        raw_delta = q_safe - st["q_seed"]
                        ik_delta = np.clip(
                            raw_delta,
                            -self.config.max_joint_step_rad,
                            self.config.max_joint_step_rad,
                        )

                        # Keep the internal seed synchronized with the command:
                        # both advance by the exact same accepted increment.
                        st["q_seed"] = st["q_seed"] + ik_delta
                        st["q_cmd"] = st["q_cmd"] + ik_delta

                        # Only advance the EE reference after an accepted step.
                        st["last_ee"] = ee_candidate.copy()

                        output[arm_slice] = st["q_cmd"].copy()
            else:
                # While unclutched, preserve the CAN/SHM hold target.
                st["q_cmd"] = q_control.copy()

            if st["enabled"] and not enabled:
                if not pose_valid:
                    # Latch the arm off until squeeze is physically released.
                    st["needs_rearm"] = True
                    logger.info(
                        "[%s STOP] XR pose/tracking invalid; release squeeze to re-arm",
                        side.upper(),
                    )
                else:
                    logger.info("[%s] clutch RELEASED", side.upper())

            st["enabled"] = enabled

        if not np.all(np.isfinite(output)):
            raise ValueError(f"Non-finite Areumii XR action: {output}")

        return vector_to_action_dict(output)

    def send_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        return {}

    def disconnect(self) -> None:
        if self.xr is not None:
            try:
                self.xr.disconnect()
            finally:
                self.xr = None
        self._connected = False
        self._state_initialized = False
        self.arms.clear()
        logger.info("Areumii XR disconnected.")