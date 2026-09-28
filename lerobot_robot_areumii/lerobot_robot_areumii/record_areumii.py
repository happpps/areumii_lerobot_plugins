from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixed Areumii recording configuration.
# ---------------------------------------------------------------------------

ROBOT_ID = "areumii_c1"
CMD_PORT = 5005
FEEDBACK_PORT = 5006

TELEOP_ID = "areumii_xr"
ISAAC_TELEOP_PATH = str(
    Path(
        os.environ.get("AREUMII_TELEOP_PATH", Path.home() / "projects/IsaacLab_teleop/real_teleop")
    ).expanduser()
)
URDF = str(
    Path(
        os.environ.get("AREUMII_URDF", Path(ISAAC_TELEOP_PATH) / "areumii_c1_kinematics.urdf")
    ).expanduser()
)

# TASK = "Pick up the blue can and place it on the upper shelf."
# TASK = "Pick up the green can and place it on the upper shelf."
TASK = "Place the green can on the upper shelf, then place the blue can on the lower shelf."

EPISODE_TIME_S = 3000
RESET_TIME_S = 5
FPS = 30
VIDEO = True


def str2bool(value: str) -> bool:
    value = value.lower().strip()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact lerobot-record launcher for the real Areumii robot."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--push-to-hub", type=str2bool, required=True)
    parser.add_argument("--display-data", type=str2bool, required=True)
    parser.add_argument("--enable-command", type=str2bool, required=True)
    return parser.parse_args()


def _prepare_python_path() -> None:
    if ISAAC_TELEOP_PATH not in sys.path:
        sys.path.insert(0, ISAAC_TELEOP_PATH)

    old_pythonpath = os.environ.get("PYTHONPATH", "")
    paths = [p for p in old_pythonpath.split(":") if p]
    if ISAAC_TELEOP_PATH not in paths:
        os.environ["PYTHONPATH"] = (
            ISAAC_TELEOP_PATH
            if not old_pythonpath
            else f"{ISAAC_TELEOP_PATH}:{old_pythonpath}"
        )


def _install_areumii_reset_hook(lerobot_record_module) -> None:
    """Patch only this areumii-record process; site-packages is not edited."""

    original_record_loop = lerobot_record_module.record_loop

    def record_loop_with_areumii_reset(*args, **kwargs):
        robot = kwargs.get("robot")
        dataset = kwargs.get("dataset", None)
        events = kwargs.get("events")

        # Normal episode: use LeRobot exactly as-is.
        if dataset is not None:
            return original_record_loop(*args, **kwargs)

        # Reset phase: dataset is omitted/None in LeRobot.
        print(
            "[RESET HOOK] entered",
            f"robot={type(robot).__name__ if robot is not None else None}",
            f"has_reset_pose={hasattr(robot, 'reset_pose') if robot is not None else False}",
            flush=True,
        )

        # Do not carry the key that ended the previous episode into reset.
        if events is not None:
            events["exit_early"] = False

        if robot is not None and hasattr(robot, "reset_pose"):
            robot.reset_pose()

        try:
            # Keep the original LeRobot reset loop so the user can reposition
            # the object while no frames are stored in the dataset.
            return original_record_loop(*args, **kwargs)
        finally:
            # Do not carry the key that ends reset into the next episode.
            if events is not None:
                events["exit_early"] = False

    # Patch the module global.
    lerobot_record_module.record_loop = record_loop_with_areumii_reset

    # Also patch the exact globals used by the undecorated record() function.
    # This makes the hook robust even though record() is wrapped by parser.wrap().
    raw_record = inspect.unwrap(lerobot_record_module.record)
    raw_record.__globals__["record_loop"] = record_loop_with_areumii_reset


def main() -> int:
    args = parse_args()
    _prepare_python_path()

    from lerobot.scripts import lerobot_record

    cmd_args = [
        "--robot.type=areumii",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cmd_port={CMD_PORT}",
        f"--robot.feedback_port={FEEDBACK_PORT}",
        f"--robot.enable_command={str(args.enable_command).lower()}",

        "--teleop.type=areumii_xr",
        f"--teleop.id={TELEOP_ID}",
        f"--teleop.urdf={URDF}",

        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={TASK}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={EPISODE_TIME_S}",
        f"--dataset.reset_time_s={RESET_TIME_S}",
        f"--dataset.fps={FPS}",
        f"--dataset.video={str(VIDEO).lower()}",
        f"--dataset.push_to_hub={str(args.push_to_hub).lower()}",

        f"--display_data={str(args.display_data).lower()}",
    ]

    print("[Areumii] launching:")
    print("lerobot-record " + " ".join(cmd_args))
    print()

    old_argv = sys.argv[:]
    try:
        sys.argv = ["lerobot-record", *cmd_args]

        # IMPORTANT:
        # LeRobot's normal main() does:
        #   register_third_party_plugins()
        #   record()
        #
        # We explicitly keep that order, then install our temporary reset hook.
        lerobot_record.register_third_party_plugins()
        _install_areumii_reset_hook(lerobot_record)

        # Call the normal decorated record() directly.
        result = lerobot_record.record()
        return 0 if result is None else 0

    except KeyboardInterrupt:
        return 130

    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
