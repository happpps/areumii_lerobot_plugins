#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_VENV="${LEROBOT_VENV:-${VIRTUAL_ENV:-$HOME/projects/env_lerobot}}"
source "$LEROBOT_VENV/bin/activate"
cd "$SCRIPT_DIR"

echo "[UPLOAD] policy.push_to_hub=true: uploads to configured 1ys1 Hub repos; W&B logging enabled."

COMMON_ARGS=(
  --policy.path=lerobot/smolvla_base
  --policy.push_to_hub=true
  --batch_size=8
  --steps=30000
  --policy.device=cuda
  --wandb.enable=true
  --wandb.disable_artifact=true
  --save_freq=10000
  --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}'
)

echo "========================================"
echo "[1/3] GREEN training start"
echo "========================================"

lerobot-train \
  "${COMMON_ARGS[@]}" \
  --dataset.repo_id=1ys1/areumii-real-pickplace-v2-green \
  --policy.repo_id=1ys1/areumii-smolvla-real-pickplace-v2-green \
  --output_dir=outputs/train/areumii-smolvla-real-pickplace-v2-green \
  --job_name=areumii-smolvla-real-pickplace-v2-green

echo "========================================"
echo "[1/3] GREEN finished"
echo "[2/3] BLUE training start"
echo "========================================"

lerobot-train \
  "${COMMON_ARGS[@]}" \
  --dataset.repo_id=1ys1/areumii-real-pickplace-v2-blue \
  --policy.repo_id=1ys1/areumii-smolvla-real-pickplace-v2-blue \
  --output_dir=outputs/train/areumii-smolvla-real-pickplace-v2-blue \
  --job_name=areumii-smolvla-real-pickplace-v2-blue

echo "========================================"
echo "[2/3] BLUE finished"
echo "[3/3] BOTH training start"
echo "========================================"

lerobot-train \
  "${COMMON_ARGS[@]}" \
  --dataset.repo_id=1ys1/areumii-real-pickplace-v2-both \
  --policy.repo_id=1ys1/areumii-smolvla-real-pickplace-v2-both \
  --output_dir=outputs/train/areumii-smolvla-real-pickplace-v2-both \
  --job_name=areumii-smolvla-real-pickplace-v2-both

echo "========================================"
echo "ALL TRAINING FINISHED"
echo "========================================"
