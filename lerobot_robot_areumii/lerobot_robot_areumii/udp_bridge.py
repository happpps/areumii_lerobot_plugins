from __future__ import annotations

import socket
import time

import numpy as np


class UdpBridge:
    """Python side of the existing Areumii UDP <-> CAN/SHM bridge."""

    def __init__(self, *, cmd_host: str, cmd_port: int, feedback_host: str, feedback_port: int):
        self.cmd_addr = (cmd_host, cmd_port)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind((feedback_host, feedback_port))
        self.rx.setblocking(False)

        self.latest_q: np.ndarray | None = None
        self.latest_time = 0.0

    def poll(self) -> np.ndarray | None:
        while True:
            try:
                data, _ = self.rx.recvfrom(4096)
            except BlockingIOError:
                break

            try:
                q = np.asarray([float(x) for x in data.decode().strip().split(",")], dtype=np.float64)
            except Exception:
                continue

            if q.shape[0] in (28, 32) and np.all(np.isfinite(q)):
                self.latest_q = q
                self.latest_time = time.monotonic()

        return None if self.latest_q is None else self.latest_q.copy()

    def wait(self, timeout_s: float) -> np.ndarray:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            q = self.poll()
            if q is not None:
                return q
            time.sleep(0.01)
        raise TimeoutError("No Areumii feedback on the configured UDP feedback port.")

    @property
    def age_s(self) -> float:
        if self.latest_q is None:
            return float("inf")
        return time.monotonic() - self.latest_time

    def send_arm(self, side: str, q7: np.ndarray) -> None:
        q7 = np.asarray(q7, dtype=np.float64)
        if q7.shape != (7,):
            raise ValueError(f"{side}: expected 7 arm joints, got {q7.shape}")
        tag = "L" if side == "left" else "R"
        msg = tag + "," + ",".join(f"{v:.12g}" for v in q7)
        self.tx.sendto(msg.encode(), self.cmd_addr)

    def send_gripper(self, side: str, motor_rad: float) -> None:
        tag = "GL" if side == "left" else "GR"
        self.tx.sendto(f"{tag},{float(motor_rad):.12g}".encode(), self.cmd_addr)

    def close(self) -> None:
        for sock in (self.rx, self.tx):
            try:
                sock.close()
            except Exception:
                pass
