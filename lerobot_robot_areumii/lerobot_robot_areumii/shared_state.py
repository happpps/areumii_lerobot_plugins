from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np


FEATURE_KEYS = (
    "left_shoulder_pitch.pos",
    "left_shoulder_roll.pos",
    "left_shoulder_yaw.pos",
    "left_elbow.pos",
    "left_wrist_roll.pos",
    "left_wrist_yaw.pos",
    "left_wrist_pitch.pos",
    "left_gripper.pos",
    "right_shoulder_pitch.pos",
    "right_shoulder_roll.pos",
    "right_shoulder_yaw.pos",
    "right_elbow.pos",
    "right_wrist_roll.pos",
    "right_wrist_yaw.pos",
    "right_wrist_pitch.pos",
    "right_gripper.pos",
)

LEFT_ARM_SLICE = slice(0, 7)
LEFT_GRIPPER_INDEX = 7
RIGHT_ARM_SLICE = slice(8, 15)
RIGHT_GRIPPER_INDEX = 15


@dataclass(frozen=True)
class BridgeSnapshot:
    measured: np.ndarray
    control: np.ndarray
    updated_at: float
    sequence: int

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.updated_at


_lock = threading.Lock()
_snapshot: BridgeSnapshot | None = None
_sequence = 0


def packet32_to_vectors(packet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert existing 32D bridge feedback to 16D measured/control vectors.

    Packet layout:
      [0:7]   measured left arm
      [7:14]  measured right arm
      [14:21] control left arm
      [21:28] control right arm
      [28]    measured left gripper
      [29]    measured right gripper
      [30]    control left gripper
      [31]    control right gripper
    """
    packet = np.asarray(packet, dtype=np.float64)
    if packet.shape != (32,):
        raise ValueError(f"Expected 32D Areumii feedback, got shape={packet.shape}")

    measured = np.empty(16, dtype=np.float64)
    control = np.empty(16, dtype=np.float64)

    measured[LEFT_ARM_SLICE] = packet[0:7]
    measured[LEFT_GRIPPER_INDEX] = packet[28]
    measured[RIGHT_ARM_SLICE] = packet[7:14]
    measured[RIGHT_GRIPPER_INDEX] = packet[29]

    control[LEFT_ARM_SLICE] = packet[14:21]
    control[LEFT_GRIPPER_INDEX] = packet[30]
    control[RIGHT_ARM_SLICE] = packet[21:28]
    control[RIGHT_GRIPPER_INDEX] = packet[31]

    return measured, control


def update_from_packet32(packet: np.ndarray) -> BridgeSnapshot:
    global _snapshot, _sequence
    measured, control = packet32_to_vectors(packet)
    with _lock:
        _sequence += 1
        _snapshot = BridgeSnapshot(
            measured=measured.copy(),
            control=control.copy(),
            updated_at=time.monotonic(),
            sequence=_sequence,
        )
        return _snapshot


def get_snapshot() -> BridgeSnapshot | None:
    with _lock:
        if _snapshot is None:
            return None
        return BridgeSnapshot(
            measured=_snapshot.measured.copy(),
            control=_snapshot.control.copy(),
            updated_at=_snapshot.updated_at,
            sequence=_snapshot.sequence,
        )


def vector_to_action_dict(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (16,):
        raise ValueError(f"Expected 16D vector, got shape={values.shape}")
    return {key: float(value) for key, value in zip(FEATURE_KEYS, values, strict=True)}


def action_dict_to_vector(action: dict[str, float]) -> np.ndarray:
    try:
        values = np.asarray([float(action[key]) for key in FEATURE_KEYS], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Missing Areumii action feature: {exc.args[0]}") from exc

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite Areumii action: {values}")
    return values
