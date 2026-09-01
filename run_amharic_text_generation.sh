#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ai-server-02/R_projects/final_thesis/adaptive-kl-grpo-thesis"
CONTAINER_NAME="amharic-text-generation"
IMAGE_NAME="${IMAGE_NAME:-simple-rl:ngc-vllm}"
DATASET_DIR="${DATASET_DIR:-/tmp/amharic_text_generation}"
RUN_DIR="${RUN_DIR:-/ckpts/amharic_text_generation_ctx512}"

mkdir -p "${REPO_DIR}/analysis_logs" "${REPO_DIR}/results" "${DATASET_DIR}"

if [[ ! -f "${DATASET_DIR}/train.parquet" || ! -f "${DATASET_DIR}/val.parquet" ]]; then
  echo "[ERROR] Amharic text-generation dataset not found at ${DATASET_DIR}"
  echo "[INFO] Expected files:"
  echo "  ${DATASET_DIR}/train.parquet"
  echo "  ${DATASET_DIR}/val.parquet"
  echo "[INFO] Put your Amharic text dataset there and rerun."
  exit 1
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${REPO_DIR}":/workspace/adaptive-kl-grpo-thesis -w /workspace/adaptive-kl-grpo-thesis \
  -v "${DATASET_DIR}":/data/amharic_text_generation \
  -v simplerl_hf_cache:/root/.cache/huggingface \
  -v simplerl_ckpts:/ckpts \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  "${IMAGE_NAME}" bash -lc '
set -e
python -m pip install -e . --no-deps
python -m pip install word2number "antlr4-python3-runtime==4.9.3" "math-verify==0.6.0" || true
export PYTHONPATH=/workspace/adaptive-kl-grpo-thesis:$PYTHONPATH
mkdir -p /ckpts/amharic_text_generation_ctx512
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-name=simplelr_grpo_amharic_text_generation_single_gpu \
  trainer.total_epochs=1 \
  trainer.default_local_dir=/ckpts/amharic_text_generation_ctx512 \
  trainer.save_freq=50 \
  trainer.test_freq=-1 \
  trainer.resume_mode=auto \
  trainer.remove_previous_ckpt=false \
  2>&1 | tee /ckpts/amharic_text_generation_ctx512/train.log
'

echo "Started ${CONTAINER_NAME}"
echo "Status: docker ps --filter name=${CONTAINER_NAME}"
echo "Logs: docker logs -f ${CONTAINER_NAME}"
