#!/usr/bin/env python3
"""Real-robot SmolVLA evaluation launcher for AReuMii.

Uses LeRobot's official `lerobot-rollout` deployment path.

The existing AReuMii LeRobot robot plugin already provides:
- measured encoder state
- head / left_wrist / right_wrist cameras
- 16-D joint-target actions
- UDP bridge communication to the real robot

Examples
--------
Dry-run / inference sanity check:
    python eval_areumii_smolvla.py \
      --step 10000 \
      --mode smoke \
      --enable-command false

Real 5-episode evaluation:
    python eval_areumii_smolvla.py \
      --step 10000 \
      --mode eval \
      --enable-command true \
      --repo-id 1ys1/eval-areumii-real-pickplace-v1-010000 \
      --num-episodes 5 \
      --push-to-hub true

Repeat with --step 20000 and --step 30000.

IMPORTANT:
- Start the AReuMii C++ UDP/SHM/CAN bridge first.
- Start the robot from the same reset pose used during data collection.
- Real motor commands are disabled by default.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROBOT_TYPE = "areumii"
ROBOT_ID = "areumii_c1"
CMD_PORT = 5005
FEEDBACK_PORT = 5006

DEFAULT_FPS = 30
DEFAULT_DEVICE = "cuda"
DEFAULT_TASK = "Pick up the blue can and place it in the box."

DEFAULT_TRAIN_DIR = (
    "outputs/train/areumii_smolvla_real_pickplace_v1_baseline"
)

DEFAULT_NUM_EPISODES = 5
DEFAULT_EPISODE_TIME_S = 60.0
DEFAULT_RESET_TIME_S = 12.0
DEFAULT_SMOKE_DURATION_S = 8.0


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def resolve_model(
    model: str | None,
    train_dir: str,
    step: int | None,
) -> tuple[str, str]:
    """Return (policy_path, checkpoint_label)."""
    if model:
        return model, Path(model.rstrip("/")).name or "model"

    if step is None:
        raise ValueError("Either --model or --step must be provided.")

    ckpt = (
        Path(train_dir).expanduser()
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    )

    if not ckpt.exists():
        checkpoints_dir = Path(train_dir).expanduser() / "checkpoints"
        available = []
        if checkpoints_dir.is_dir():
            available = sorted(
                p.name for p in checkpoints_dir.iterdir() if p.is_dir()
            )

        raise FileNotFoundError(
            f"Checkpoint not found:\n  {ckpt}\n"
            f"Available checkpoint dirs: {available}\n"
            "If the model is on the Hub, pass --model USER/MODEL instead."
        )

    return str(ckpt), f"{step:06d}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a fine-tuned SmolVLA checkpoint on real AReuMii."
    )

    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--model",
        default=None,
        help="Hugging Face model id or local pretrained_model path.",
    )
    source.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step, e.g. 10000, 20000, 30000.",
    )

    p.add_argument(
        "--train-dir",
        default=DEFAULT_TRAIN_DIR,
        help="Training output directory containing checkpoints/.",
    )
    p.add_argument(
        "--mode",
        choices=("smoke", "eval"),
        default="smoke",
        help="smoke = no eval dataset; eval = episodic rollout + recording.",
    )
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument(
        "--rename-map",
        default=None,
        help="JSON observation-key mapping used during training; forwarded to LeRobot.",
    )

    p.add_argument(
        "--enable-command",
        type=str2bool,
        default=False,
        help="false = dry-run, true = send commands to the real robot.",
    )
    p.add_argument(
        "--display-data",
        type=str2bool,
        default=True,
        help="Show observations/actions in Rerun.",
    )
    p.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=("cuda", "cpu"),
    )
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument(
        "--interpolation-multiplier",
        type=int,
        default=1,
        help="Keep 1 for the first checkpoint comparison.",
    )

    # Smoke mode.
    p.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_SMOKE_DURATION_S,
        help="Smoke-test duration in seconds.",
    )
    # Eval mode.
    p.add_argument(
        "--repo-id",
        default=None,
        help="Evaluation LeRobot dataset repo id.",
    )
    p.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
    )
    p.add_argument(
        "--episode-time-s",
        type=float,
        default=DEFAULT_EPISODE_TIME_S,
    )
    p.add_argument(
        "--reset-time-s",
        type=float,
        default=DEFAULT_RESET_TIME_S,
    )
    p.add_argument(
        "--push-to-hub",
        type=str2bool,
        default=False,
    )
    p.add_argument(
        "--reset-to-initial-position",
        type=str2bool,
        default=True,
        help="Return to startup position between episodic rollouts.",
    )

    # Optional RTC. Sync is the baseline.
    p.add_argument(
        "--inference",
        choices=("sync", "rtc"),
        default="sync",
    )
    p.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=10,
    )
    p.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=10.0,
    )

    # Manual real-world success labels.
    p.add_argument(
        "--label-results",
        type=str2bool,
        default=True,
    )
    p.add_argument(
        "--results-dir",
        default="./results",
    )

    args = p.parse_args()

    if args.model is None and args.step is None:
        p.error("Pass either --step 10000/20000/30000 or --model <path-or-HF-id>.")

    if args.mode == "eval" and not args.repo_id:
        p.error("--repo-id is required when --mode eval.")

    if args.num_episodes < 1:
        p.error("--num-episodes must be >= 1.")
    if args.fps <= 0:
        p.error("--fps must be > 0.")
    if args.interpolation_multiplier < 1:
        p.error("--interpolation-multiplier must be >= 1.")

    return args


def build_common_cmd(
    args: argparse.Namespace,
    policy_path: str,
) -> list[str]:
    cmd = [
        "lerobot-rollout",
        f"--policy.path={policy_path}",
        f"--robot.type={ROBOT_TYPE}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cmd_port={CMD_PORT}",
        f"--robot.feedback_port={FEEDBACK_PORT}",
        f"--robot.enable_command={str(args.enable_command).lower()}",
        f"--task={args.task}",
        f"--fps={args.fps}",
        f"--device={args.device}",
        f"--display_data={str(args.display_data).lower()}",
        f"--interpolation_multiplier={args.interpolation_multiplier}",
        "--play_sounds=false",
    ]

    if args.rename_map is not None:
        cmd.append(f"--rename_map={args.rename_map}")

    if args.inference == "rtc":
        cmd += [
            "--inference.type=rtc",
            f"--inference.rtc.execution_horizon={args.rtc_execution_horizon}",
            f"--inference.rtc.max_guidance_weight={args.rtc_max_guidance_weight}",
        ]

    return cmd


def build_smoke_cmd(
    args: argparse.Namespace,
    policy_path: str,
) -> list[str]:
    return build_common_cmd(args, policy_path) + [
        "--strategy.type=base",
        f"--duration={args.duration}",
    ]


def build_eval_cmd(
    args: argparse.Namespace,
    policy_path: str,
) -> list[str]:
    return build_common_cmd(args, policy_path) + [
        "--strategy.type=episodic",
        (
            "--strategy.reset_to_initial_position="
            f"{str(args.reset_to_initial_position).lower()}"
        ),
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time_s}",
        f"--dataset.reset_time_s={args.reset_time_s}",
        f"--dataset.push_to_hub={str(args.push_to_hub).lower()}",
    ]


def print_run_header(
    args: argparse.Namespace,
    policy_path: str,
    checkpoint_label: str,
    cmd: list[str],
) -> None:
    print("\n[AReuMii SmolVLA evaluation]")
    print(f"  mode             : {args.mode}")
    print(f"  checkpoint       : {checkpoint_label}")
    print(f"  policy           : {policy_path}")
    print(f"  task             : {args.task}")
    print(f"  fps              : {args.fps}")
    print(f"  inference        : {args.inference}")
    print(
        "  robot command    : "
        f"{'ENABLED' if args.enable_command else 'DISABLED / DRY RUN'}"
    )
    print(f"  UDP cmd/feedback : {CMD_PORT}/{FEEDBACK_PORT}")

    if args.mode == "eval":
        print(f"  eval repo        : {args.repo_id}")
        print(f"  episodes         : {args.num_episodes}")
        print(f"  episode time     : {args.episode_time_s:.1f} s")
        print(f"  reset time       : {args.reset_time_s:.1f} s")
        print(f"  reset to startup : {args.reset_to_initial_position}")
    else:
        print(f"  duration         : {args.duration:.1f} s")

    if args.enable_command:
        print(
            "\n[SAFETY] REAL COMMANDS ARE ENABLED.\n"
            "         Confirm bridge feedback is healthy and the robot starts\n"
            "         from the same reset pose used in the demonstrations."
        )
    else:
        print(
            "\n[SAFE MODE] Commands are disabled. "
            "Use this first to verify cameras/state/action."
        )

    print("\n[launch]")
    print(" ".join(cmd))
    print()


def label_eval_results(
    args: argparse.Namespace,
    policy_path: str,
    checkpoint_label: str,
) -> None:
    if not args.label_results:
        return

    print("\n[RESULT LABELING]")
    print("Enter y = success, n = failure, s = skip/unknown.")

    rows = []
    for episode_idx in range(args.num_episodes):
        while True:
            try:
                value = input(
                    f"Episode {episode_idx} success? [y/n/s]: "
                ).strip().lower()
            except EOFError:
                print("\n[WARN] stdin closed; skipping manual labels.")
                return

            if value in {"y", "yes"}:
                success = 1
                label = "success"
                break
            if value in {"n", "no"}:
                success = 0
                label = "failure"
                break
            if value in {"s", "skip", ""}:
                success = ""
                label = "unknown"
                break
            print("Please enter y, n, or s.")

        rows.append(
            {
                "checkpoint": checkpoint_label,
                "policy_path": policy_path,
                "eval_repo_id": args.repo_id,
                "episode_index": episode_idx,
                "success": success,
                "label": label,
                "task": args.task,
                "fps": args.fps,
                "inference": args.inference,
            }
        )

    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = (
        results_dir
        / f"areumii_real_smolvla_eval_{checkpoint_label}_{stamp}.csv"
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    known = [row for row in rows if row["success"] != ""]
    successes = sum(int(row["success"]) for row in known)

    print(f"\n[RESULTS] Saved labels: {csv_path}")
    if known:
        rate = successes / len(known)
        print(
            f"[RESULTS] Success: {successes}/{len(known)} "
            f"= {rate:.3f}"
        )
    else:
        print("[RESULTS] No episodes were labeled.")


def main() -> int:
    args = parse_args()

    try:
        policy_path, checkpoint_label = resolve_model(
            args.model,
            args.train_dir,
            args.step,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if args.mode == "smoke":
        cmd = build_smoke_cmd(args, policy_path)
    else:
        cmd = build_eval_cmd(args, policy_path)

    print_run_header(
        args,
        policy_path,
        checkpoint_label,
        cmd,
    )

    try:
        return_code = subprocess.call(cmd)
    except FileNotFoundError:
        print(
            "[ERROR] `lerobot-rollout` was not found in PATH.\n"
            "Activate env_lerobot and verify `lerobot-rollout --help`.",
            file=sys.stderr,
        )
        return 127
    except KeyboardInterrupt:
        return 130

    if return_code != 0:
        print(
            f"[ERROR] lerobot-rollout exited with code {return_code}.",
            file=sys.stderr,
        )
        return return_code

    if args.mode == "eval":
        label_eval_results(
            args,
            policy_path,
            checkpoint_label,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
