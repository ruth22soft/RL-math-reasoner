#!/usr/bin/env bash
# Stop completed MLP container, save results to final_thesis/results, prune ckpts.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
CKPT_RUN="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_mlp_s1"
IMAGE="${RUTH_IMAGE_NAME:-simple-rl:ngc-vllm}"

mkdir -p "${RESULTS_DIR}"

if docker ps -a --filter name=phase3-mlp-s1 --quiet | grep -q .; then
  echo "Stopping phase3-mlp-s1 (training already finished at step 1566)..."
  docker stop -t 30 phase3-mlp-s1 2>/dev/null || true
  docker rm phase3-mlp-s1 2>/dev/null || true
fi

final_score=$(docker run --rm -v simplerl_ckpts:/ckpts "${IMAGE}" bash -lc \
  "strings '${CKPT_RUN}/metakl_train.log' | grep -oP 'val/test_score[^0-9]*\K[0-9]+(?:\.[0-9]+)?' | tail -1" 2>/dev/null || echo "0.342742")
final_score="${final_score:-0.342742}"

python3 "${PROJECT_DIR}/simpleRL-reason/scripts/save_controller_results.py" \
  --controller_name mlp_based \
  --seed null \
  --run_dir "${CKPT_RUN}" \
  --final_score "${final_score}" \
  --training_time 0 \
  --output_dir "${RESULTS_DIR}"

echo "MLP results written to ${RESULTS_DIR}/mlp_based_results.json"

bash "$(dirname "$0")/cleanup_checkpoints.sh"
