#!/usr/bin/env bash
set -euo pipefail
# Usage: phase3_wandb_launch.sh <mode> <seed> [run_name_suffix] [total_steps]
# Example: ./phase3_wandb_launch.sh mlp 42 trialA 20000
MODE=${1:-fixed}
SEED=${2:-0}
RUN_SUFFIX=${3:-""}
TOTAL_STEPS=${4:-20000}
PROJECT=${RUTH_WANDB_PROJECT:-"metakl_experiments"}
ENTITY=${RUTH_WANDB_ENTITY:-}
API_KEY=${WANDB_API_KEY:-${WANDB_API_KEY_ENV:-}}
RUN_NAME="${MODE}_s${SEED}${RUN_SUFFIX:+_${RUN_SUFFIX}}"
CONTAINER_NAME="phase3-${MODE}-s${SEED}"
RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_${MODE}_s${SEED}" 

echo "Launching W&B run: project=$PROJECT run=$RUN_NAME mode=$MODE seed=$SEED"

# Export W&B env or rely on user's wandb login
if [ -n "$API_KEY" ]; then
  export WANDB_API_KEY="$API_KEY"
fi

RUTH_MODE="$MODE" \
RUTH_SEED="$SEED" \
RUTH_CONTAINER_NAME="$CONTAINER_NAME" \
RUTH_RUN_DIR="$RUN_DIR" \
RUTH_WANDB=true \
RUTH_WANDB_PROJECT="$PROJECT" \
RUTH_WANDB_ENTITY="$ENTITY" \
RUTH_TOTAL_TRAINING_STEPS="$TOTAL_STEPS" \
RUTH_SAVE_FREQ=2000 \
RUTH_TEST_FREQ=2000 \
RUTH_REMOVE_PREVIOUS_CKPT=true \
RUTH_RESUME_MODE=never \
./simpleRL-reason/scripts/run_metakl_training_detached.sh &

echo "Launched container: $CONTAINER_NAME (background)"
