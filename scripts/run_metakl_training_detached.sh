#!/usr/bin/env bash
set -euo pipefail

# Runs adaptive-KL training in a Docker container.
# Defaults to a fresh checkpoint directory per launch so runs do not silently resume.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${RUTH_CONTAINER_NAME:-ruth-training-run}"
IMAGE_NAME="${RUTH_IMAGE_NAME:-simple-rl:ngc-vllm}"

# Keep training parameters aligned with the current baseline preset.
TOTAL_EPOCHS="${RUTH_TOTAL_EPOCHS:-3}"
TOTAL_TRAINING_STEPS="${RUTH_TOTAL_TRAINING_STEPS:-}"
MAX_PROMPT_LEN="${RUTH_MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${RUTH_MAX_RESPONSE_LEN:-1024}"
MAX_BATCHED_TOKENS="${RUTH_MAX_BATCHED_TOKENS:-4096}"
ROLLOUT_N="${RUTH_ROLLOUT_N:-4}"
ROLLOUT_GPU_MEMORY_UTIL="${RUTH_ROLLOUT_GPU_MEMORY_UTIL:-0.5}"
ROLLOUT_MAX_NUM_SEQS="${RUTH_ROLLOUT_MAX_NUM_SEQS:-32}"

# Checkpointing for clear, explicit run boundaries.
RUN_BASE_DIR="${RUTH_RUN_BASE_DIR:-/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUTH_RUN_DIR:-${RUN_BASE_DIR}/${RUN_STAMP}}"
RESUME_CHECKPOINT="${RUTH_RESUME_CHECKPOINT:-}"
if [[ -n "${RESUME_CHECKPOINT}" && -z "${RUTH_RUN_DIR:-}" ]]; then
  RUN_DIR="$(dirname "${RESUME_CHECKPOINT}")"
fi
SAVE_FREQ="${RUTH_SAVE_FREQ:-50}"
TEST_FREQ="${RUTH_TEST_FREQ:--1}"
RESTART_POLICY="${RUTH_RESTART_POLICY:-no}"
RESUME_MODE="${RUTH_RESUME_MODE:-never}"
REMOVE_PREV_CKPT="${RUTH_REMOVE_PREVIOUS_CKPT:-true}"
TRAIN_MODE="${RUTH_MODE:-fixed}"
FIXED_BETA="${RUTH_FIXED_BETA:-0.001}"

case "${TRAIN_MODE}" in
  fixed)
    ADAPTIVE_KL_ENABLE="false"
    ADAPTIVE_KL_MODE="lstm"
    ;;
  rule|mlp|lstm)
    ADAPTIVE_KL_ENABLE="true"
    ADAPTIVE_KL_MODE="${TRAIN_MODE}"
    ;;
  *)
    echo "[ERROR] Unknown RUTH_MODE=${TRAIN_MODE}; expected fixed|rule|mlp|lstm" >&2
    exit 2
    ;;
esac

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_MODE="${RESUME_CHECKPOINT}"
fi

if [[ "${RESUME_MODE}" == "auto" ]]; then
  echo "[WARN] resume_mode=auto is disallowed for multi-seed runs; forcing never"
  RESUME_MODE="never"
fi

mkdir -p "${REPO_DIR}/analysis_logs"

cd "${REPO_DIR}"

echo "[INFO] Repo: ${REPO_DIR}"
echo "[INFO] Container: ${CONTAINER_NAME}"
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Run base dir: ${RUN_BASE_DIR}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "[INFO] Resume checkpoint: ${RESUME_CHECKPOINT}"
fi
echo "[INFO] mode: ${TRAIN_MODE}"
echo "[INFO] fixed_beta: ${FIXED_BETA}"
echo "[INFO] Epochs: ${TOTAL_EPOCHS}"
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  echo "[INFO] total_training_steps: ${TOTAL_TRAINING_STEPS}"
fi
echo "[INFO] Prompt/Response: ${MAX_PROMPT_LEN}/${MAX_RESPONSE_LEN}"
echo "[INFO] max_num_batched_tokens: ${MAX_BATCHED_TOKENS}"
echo "[INFO] rollout_n: ${ROLLOUT_N}"
echo "[INFO] rollout_gpu_memory_util: ${ROLLOUT_GPU_MEMORY_UTIL}"
echo "[INFO] rollout_max_num_seqs: ${ROLLOUT_MAX_NUM_SEQS}"
echo "[INFO] save_freq: ${SAVE_FREQ}"
echo "[INFO] resume_mode: ${RESUME_MODE}"
echo "[INFO] adaptive_kl_enable: ${ADAPTIVE_KL_ENABLE}"
echo "[INFO] adaptive_kl_mode: ${ADAPTIVE_KL_MODE}"
echo "[INFO] restart_policy: ${RESTART_POLICY}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --name "${CONTAINER_NAME}" \
  --restart "${RESTART_POLICY}" \
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
TRAINER_TOTAL_STEPS_ARGS=()
if [[ -n "'"${TOTAL_TRAINING_STEPS}"'" ]]; then
  TRAINER_TOTAL_STEPS_ARGS+=(trainer.total_training_steps='"${TOTAL_TRAINING_STEPS}"')
fi
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-name simplelr_grpo_qwen05_single_gpu \
  trainer.total_epochs='"${TOTAL_EPOCHS}"' \
  data.max_prompt_length='"${MAX_PROMPT_LEN}"' \
  data.max_response_length='"${MAX_RESPONSE_LEN}"' \
  actor_rollout_ref.rollout.max_num_batched_tokens='"${MAX_BATCHED_TOKENS}"' \
  actor_rollout_ref.rollout.n='"${ROLLOUT_N}"' \
  actor_rollout_ref.rollout.gpu_memory_utilization='"${ROLLOUT_GPU_MEMORY_UTIL}"' \
  actor_rollout_ref.rollout.max_num_seqs='"${ROLLOUT_MAX_NUM_SEQS}"' \
  trainer.default_local_dir='"${RUN_DIR}"' \
  trainer.save_freq='"${SAVE_FREQ}"' \
  trainer.test_freq='"${TEST_FREQ}"' \
  trainer.resume_mode='"${RESUME_MODE}"' \
  trainer.remove_previous_ckpt='"${REMOVE_PREV_CKPT}"' \
  actor_rollout_ref.actor.kl_loss_coef='"${FIXED_BETA}"' \
  actor_rollout_ref.actor.actor_adaptive_kl.enable='"${ADAPTIVE_KL_ENABLE}"' \
  actor_rollout_ref.actor.actor_adaptive_kl.mode='"${ADAPTIVE_KL_MODE}"' \
  actor_rollout_ref.actor.actor_adaptive_kl.warmup_steps=1 \
  "${TRAINER_TOTAL_STEPS_ARGS[@]}" \
  2>&1 | tee "'"${RUN_DIR}"'/metakl_train.log"
'

echo "[OK] Started ${CONTAINER_NAME}"
echo "[OK] Follow logs: docker logs -f ${CONTAINER_NAME}"
echo "[OK] Checkpoints: docker run --rm -v simplerl_ckpts:/ckpts ${IMAGE_NAME} bash -lc 'ls -lah ${RUN_DIR}'"
