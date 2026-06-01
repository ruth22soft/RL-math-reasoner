#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# SINGLE-SEED Training Launcher with Result Saving
#
# CRITICAL CHANGES FROM ORIGINAL:
#   1. Uses RUTH_GLOBAL_SEED (not multi-seed loop)
#   2. Enforces single-seed-only via guard assertions
#   3. Saves results to JSON after completion
#   4. If seed != GLOBAL_SEED: aborts immediately
# ============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${RUTH_CONTAINER_NAME:-ruth-training-run}"
IMAGE_NAME="${RUTH_IMAGE_NAME:-simple-rl:ngc-vllm}"

# ============================================================================
# GLOBAL SEED: All controllers must use this value (do NOT change per run)
# ============================================================================
GLOBAL_SEED="${RUTH_GLOBAL_SEED:-42}"

# Keep training parameters aligned with the current baseline preset.
TOTAL_EPOCHS="${RUTH_TOTAL_EPOCHS:-3}"
TOTAL_TRAINING_STEPS="${RUTH_TOTAL_TRAINING_STEPS:-}"
MAX_PROMPT_LEN="${RUTH_MAX_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${RUTH_MAX_RESPONSE_LEN:-1024}"
MAX_BATCHED_TOKENS="${RUTH_MAX_BATCHED_TOKENS:-4096}"
ROLLOUT_N="${RUTH_ROLLOUT_N:-2}"
ROLLOUT_GPU_MEMORY_UTIL="${RUTH_ROLLOUT_GPU_MEMORY_UTIL:-0.4}"
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

# ============================================================================
# SEED GUARD: Validate that only GLOBAL_SEED is being used
# ============================================================================

# If RUTH_SEED is passed (from old multi-seed orchestrator), verify it matches
if [[ -n "${RUTH_SEED:-}" ]]; then
  if [[ "${RUTH_SEED}" != "${GLOBAL_SEED}" ]]; then
    echo "[ERROR] RUTH_SEED=${RUTH_SEED} does not match GLOBAL_SEED=${GLOBAL_SEED}"
    echo "[ERROR] Only GLOBAL_SEED=${GLOBAL_SEED} is permitted for single-seed runs"
    echo "[ERROR] Aborting to prevent wasting GPU hours on wrong seed"
    exit 1
  fi
fi

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
  echo "[WARN] resume_mode=auto is disallowed for single-seed runs; forcing never"
  RESUME_MODE="never"
fi

mkdir -p "${REPO_DIR}/analysis_logs" "${REPO_DIR}/results"

cd "${REPO_DIR}"

echo "[INFO] ════════════════════════════════════════════════════════════"
echo "[INFO] Single-Seed Training Launch (GLOBAL_SEED=${GLOBAL_SEED})"
echo "[INFO] ════════════════════════════════════════════════════════════"
echo "[INFO] Repo: ${REPO_DIR}"
echo "[INFO] Container: ${CONTAINER_NAME}"
echo "[INFO] Run dir: ${RUN_DIR}"
echo "[INFO] Mode: ${TRAIN_MODE}"
echo "[INFO] Epochs: ${TOTAL_EPOCHS}"
echo "[INFO] Global seed (enforced): ${GLOBAL_SEED}"
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  echo "[INFO] total_training_steps: ${TOTAL_TRAINING_STEPS}"
fi
echo "[INFO] adaptive_kl_enable: ${ADAPTIVE_KL_ENABLE}"
echo "[INFO] adaptive_kl_mode: ${ADAPTIVE_KL_MODE}"
echo "[INFO] ════════════════════════════════════════════════════════════"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# Prepare Telegram credentials (pass to monitoring)
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

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
  -e TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}" \
  -e TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID}" \
  -e GLOBAL_SEED="${GLOBAL_SEED}" \
  "${IMAGE_NAME}" bash -lc '
set -e

# ========================================================================
# INSIDE CONTAINER: Strict Seed Enforcement
# ========================================================================

if [[ "${GLOBAL_SEED:-0}" -eq 0 ]]; then
  echo "[WARN] GLOBAL_SEED not set; using default 42"
  GLOBAL_SEED=42
fi

echo "[INFO] Container starting with GLOBAL_SEED=${GLOBAL_SEED}"

python -m pip install -e . --no-deps
python -m pip install word2number "antlr4-python3-runtime==4.9.3" "math-verify==0.6.0"
export PYTHONPATH=/workspace/simpleRL-reason:$PYTHONPATH
mkdir -p "'"${RUN_DIR}"'"

# Install result-saving utility if needed
python << "ENDPY"
import json
import os
from datetime import datetime

def save_training_results(controller_name, seed, final_score, training_time, run_dir):
    """Save training results to JSON for this controller."""
    results_dir = os.path.join("/workspace/simpleRL-reason", "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"{controller_name}_results.json")
    
    result_data = {
        "controller": controller_name,
        "seed": seed,
        "global_seed": seed,  # Always same as seed in single-seed mode
        "epochs": '"${TOTAL_EPOCHS}"',
        "final_test_score": final_score,
        "training_time_seconds": training_time,
        "completion_timestamp": datetime.utcnow().isoformat() + "Z",
        "checkpoint_dir": run_dir,
        "status": "completed"
    }
    
    with open(results_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"[INFO] Results saved to {results_file}")

# Export for use later
import sys
sys.modules["result_utils"] = sys.modules[__name__]
ENDPY

TRAINER_TOTAL_STEPS_ARGS=()
if [[ -n "'"${TOTAL_TRAINING_STEPS}"'" ]]; then
  TRAINER_TOTAL_STEPS_ARGS+=(trainer.total_training_steps='"${TOTAL_TRAINING_STEPS}"')
fi

# Launch training with seed enforcement
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

echo "[INFO] Training completed. Results saved to ${RUN_DIR}"
'

echo "[OK] Started ${CONTAINER_NAME}"
echo "[OK] Follow logs: docker logs -f ${CONTAINER_NAME}"
echo "[OK] Checkpoints: docker run --rm -v simplerl_ckpts:/ckpts ${IMAGE_NAME} bash -lc 'ls -lah ${RUN_DIR}'"

# Start background Telegram monitoring if credentials are provided
if [[ -n "${TELEGRAM_BOT_TOKEN}" && -n "${TELEGRAM_CHAT_ID}" ]]; then
  echo "[OK] Starting Telegram monitoring..."
  (
    # Wait for container to be ready, then send initial start notification
    sleep 10
    TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}" TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID}" \
    python3 "${REPO_DIR}/scripts/telegram_metakl_report.py" <<< "start" 2>/dev/null || true
    
    # Monitor and report every 5 minutes while container is running
    while docker ps --filter "name=${CONTAINER_NAME}" --quiet | grep -q .; do
      sleep 300
      TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}" TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID}" \
      python3 "${REPO_DIR}/scripts/telegram_metakl_report.py" 2>/dev/null || true
    done
    
    # Final report when container exits
    sleep 5
    TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}" TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID}" \
    python3 "${REPO_DIR}/scripts/telegram_metakl_report.py" <<< "final" 2>/dev/null || true
  ) &
  BG_PID=$!
  echo "[OK] Telegram monitor PID: ${BG_PID}"
else
  echo "[WARN] Telegram monitoring disabled (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set)"
fi
