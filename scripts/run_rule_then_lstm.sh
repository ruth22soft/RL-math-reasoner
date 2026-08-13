#!/usr/bin/env bash
# Run remaining benchmarks: rule_based then lstm_based.
# Preserves unless-stopped + resume_mode=auto for power-loss recovery.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="${PROJECT_DIR}/simpleRL-reason/scripts/run_metakl_training_detached.sh"
CKPTS_BASE="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_FILE="${RESULTS_DIR}/pipeline.log"

mkdir -p "${RESULTS_DIR}"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }

# Campaign hyperparameters (match completed zero_kl / fixed / mlp runs)
export RUTH_TOTAL_EPOCHS=3
export RUTH_ROLLOUT_N=4
export RUTH_ROLLOUT_GPU_MEMORY_UTIL=0.5
export RUTH_ROLLOUT_MAX_NUM_SEQS=32
export RUTH_MAX_BATCHED_TOKENS=4096
export RUTH_SAVE_FREQ=500
export RUTH_TEST_FREQ=-1
export RUTH_RESTART_POLICY=unless-stopped
export RUTH_RESUME_MODE=auto
export RUTH_REMOVE_PREVIOUS_CKPT=true
export RUTH_RESULTS_DIR="${RESULTS_DIR}"

RULE_RUN_DIR="${CKPTS_BASE}/runs/rule-based/seed_3/20260523_154450"
LSTM_RUN_DIR="${CKPTS_BASE}/phase3_lstm_s1"

wait_for_container() {
  local name="$1"
  while docker ps -a --filter "name=${name}" --filter "status=running" --quiet | grep -q .; do
    sleep 120
  done
  docker inspect -f '{{.State.ExitCode}}' "${name}" 2>/dev/null || echo 1
}

launch_rule() {
  log "Launching rule_based (resume from latest checkpoint if present)..."
  RUTH_MODE=rule \
  RUTH_CONTAINER_NAME=phase3-rule_based-s1 \
  RUTH_RUN_DIR="${RULE_RUN_DIR}" \
  RUTH_EXTRA_ARGS="actor_rollout_ref.actor.actor_adaptive_kl.target_kl_loss=0.08 actor_rollout_ref.actor.actor_adaptive_kl.up_gain=1.08 actor_rollout_ref.actor.actor_adaptive_kl.down_gain=0.94" \
  bash "${LAUNCHER}"
}

launch_lstm() {
  log "Launching lstm_based..."
  RUTH_MODE=lstm \
  RUTH_CONTAINER_NAME=phase3-lstm-s1 \
  RUTH_RUN_DIR="${LSTM_RUN_DIR}" \
  RUTH_EXTRA_ARGS="actor_rollout_ref.actor.actor_adaptive_kl.lstm_reset_on_epoch=true" \
  bash "${LAUNCHER}"
}

log "===== rule_based phase ====="
launch_rule
rule_exit=$(wait_for_container phase3-rule_based-s1)
if [[ "${rule_exit}" != "0" ]] && [[ ! -f "${RESULTS_DIR}/rule_based_results.json" ]]; then
  log "rule_based failed (exit ${rule_exit}) and no results JSON; stopping pipeline"
  exit 1
fi
log "rule_based phase done (container exit ${rule_exit})"

log "===== lstm_based phase ====="
launch_lstm
lstm_exit=$(wait_for_container phase3-lstm-s1)
if [[ "${lstm_exit}" != "0" ]] && [[ ! -f "${RESULTS_DIR}/lstm_based_results.json" ]]; then
  log "lstm_based failed (exit ${lstm_exit})"
  exit 1
fi
log "Pipeline complete."
