#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_VENV="${LEROBOT_VENV:-${VIRTUAL_ENV:-$HOME/projects/env_lerobot}}"
source "$LEROBOT_VENV/bin/activate"
cd "$SCRIPT_DIR"

echo "[UPLOAD] policy.push_to_hub=true: uploads to configured 1ys1 Hub repos; W&B logging enabled."

echo "======================================"
echo "[1/2] V3 - sequential 25ep"
echo "======================================"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=1ys1/areumii-smolvla-real-pickplace-v3 \
  --policy.push_to_hub=true \
  --dataset.repo_id=1ys1/areumii-real-pickplace-v3 \
  --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}' \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --output_dir=outputs/train/areumii-smolvla-real-pickplace-v3 \
  --job_name=areumii-smolvla-real-pickplace-v3 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true

echo "======================================"
echo "[1/2] V3 DONE"
echo "[2/2] V3 + single 10"
echo "======================================"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=1ys1/areumii-smolvla-real-pickplace-v3-single10 \
  --policy.push_to_hub=true \
  --dataset.repo_id=1ys1/areumii-real-pickplace-v3-single10 \
  --rename_map='{"observation.images.head":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}' \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --output_dir=outputs/train/areumii-smolvla-real-pickplace-v3-single10 \
  --job_name=areumii-smolvla-real-pickplace-v3-single10 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true

echo "======================================"
echo "ALL TRAINING FINISHED"
echo "======================================"
