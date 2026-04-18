#!/usr/bin/env bash
set -euo pipefail

# Runs adaptive-KL training in a persistent Docker container.
# Designed for power-failure resilience via Docker restart policy + checkpoint resume.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${RUTH_CONTAINER_NAME:-ruth-training-run}"
IMAGE_NAME="${RUTH_IMAGE_NAME:-simple-rl:ngc-vllm}"

# Keep training parameters aligned with the current baseline preset.
TOTAL_EPOCHS="${RUTH_TOTAL_EPOCHS:-3}"
MAX_PROMPT_LEN="${RUTH_MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${RUTH_MAX_RESPONSE_LEN:-1024}"
MAX_BATCHED_TOKENS="${RUTH_MAX_BATCHED_TOKENS:-4096}"

# Checkpointing for resume after restarts/outages.
RUN_DIR="${RUTH_RUN_DIR:-/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive}"
SAVE_FREQ="${RUTH_SAVE_FREQ:-50}"
TEST_FREQ="${RUTH_TEST_FREQ:--1}"
REMOVE_PREV_CKPT="${RUTH_REMOVE_PREVIOUS_CKPT:-true}"

mkdir -p "${REPO_DIR}/analysis_logs"

cd "${REPO_DIR}"

echo "[INFO] Repo: ${REPO_DIR}"
echo "[INFO] Container: ${CONTAINER_NAME}"
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Epochs: ${TOTAL_EPOCHS}"
echo "[INFO] Prompt/Response: ${MAX_PROMPT_LEN}/${MAX_RESPONSE_LEN}"
echo "[INFO] max_num_batched_tokens: ${MAX_BATCHED_TOKENS}"
echo "[INFO] save_freq: ${SAVE_FREQ}"
echo "[INFO] resume_mode: auto"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --label simplerl.role=training \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${REPO_DIR}":/workspace/simpleRL-reason -w /workspace/simpleRL-reason \
  -v simplerl_data:/data \
  -v simplerl_hf_cache:/root/.cache/huggingface \
  -v simplerl_ckpts:/ckpts \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  "${IMAGE_NAME}" bash -lc '
set -e
python -m pip install -e . --no-deps
python -m pip install word2number "antlr4-python3-runtime==4.9.3" "math-verify==0.6.0"
export PYTHONPATH=/workspace/simpleRL-reason:$PYTHONPATH
mkdir -p "'"${RUN_DIR}"'"
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-name simplelr_grpo_qwen05_single_gpu \
  trainer.total_epochs='"${TOTAL_EPOCHS}"' \
  data.max_prompt_length='"${MAX_PROMPT_LEN}"' \
  data.max_response_length='"${MAX_RESPONSE_LEN}"' \
  actor_rollout_ref.rollout.max_num_batched_tokens='"${MAX_BATCHED_TOKENS}"' \
  trainer.default_local_dir='"${RUN_DIR}"' \
  trainer.save_freq='"${SAVE_FREQ}"' \
  trainer.test_freq='"${TEST_FREQ}"' \
  trainer.resume_mode=auto \
  trainer.remove_previous_ckpt='"${REMOVE_PREV_CKPT}"' \
  actor_rollout_ref.actor.actor_adaptive_kl.enable=true \
  actor_rollout_ref.actor.actor_adaptive_kl.mode=lstm \
  actor_rollout_ref.actor.actor_adaptive_kl.warmup_steps=1 \
  2>&1 | tee "'"${RUN_DIR}"'/metakl_train.log"
'

echo "[OK] Started ${CONTAINER_NAME}"
echo "[OK] Follow logs: docker logs -f ${CONTAINER_NAME}"
echo "[OK] Checkpoints: docker run --rm -v simplerl_ckpts:/ckpts ${IMAGE_NAME} bash -lc 'ls -lah ${RUN_DIR}'"
