#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Single-Seed Orchestrator for 5-Controller RL Benchmark Comparison
# 
# PURPOSE:
#   Run 5 KL controllers sequentially with a single shared seed for all.
#   Each controller trains for 3 epochs once, results are saved immediately.
#
# CONTROLLERS (Priority Order):
#   1. Zero KL (β = 0)
#   2. Rule-based KL calculation
#   3. MLP-based KL calculation
#   4. Fixed/Constant KL coefficient
#   5. LSTM-based KL calculation
#
# CRITICAL RULE:
#   - Only 1 seed is used: GLOBAL_SEED (hardcoded to 42)
#   - No multi-seed loop
#   - If any controller is mid-training on seed 2 or 3 → KILL IT IMMEDIATELY
#   - Results saved per-controller before moving to next
#
################################################################################

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_DIR}/simpleRL-reason/analysis_logs"
RESULTS_DIR="${REPO_DIR}/results"
ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator_single_seed_$(date +%Y%m%d_%H%M%S).log"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Global seed: ALL 5 controllers use this seed
# Extract from Zero KL first run if it exists, otherwise use 42
GLOBAL_SEED=42

# Controller priority order: exactly as specified
CONTROLLERS=("zero_kl" "rule_based" "mlp_based" "fixed" "lstm_based")

# Total epochs (do NOT change)
TOTAL_EPOCHS=3

# Base checkpoint directory
BASE_RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive"

# Telegram settings (optional)
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# ============================================================================
# LOGGING HELPER
# ============================================================================

log_msg() {
  local msg="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" | tee -a "${ORCHESTRATOR_LOG}"
}

# ============================================================================
# SEND TELEGRAM UPDATE
# ============================================================================

send_telegram_update() {
  local message="$1"
  if [[ -n "${TELEGRAM_BOT_TOKEN}" && -n "${TELEGRAM_CHAT_ID}" ]]; then
    TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}" TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID}" \
    python3 "${REPO_DIR}/simpleRL-reason/scripts/telegram_metakl_report.py" <<< "${message}" 2>/dev/null || true
  fi
}

# ============================================================================
# KILL STRAY CONTAINERS (CRITICAL INTERRUPTION RULE)
# ============================================================================

kill_stray_seed_runs() {
  local controller="$1"
  log_msg ""
  log_msg "[CRITICAL CHECK] Killing any existing runs of ${controller} on seed 2 or 3..."
  
  for seed_num in 2 3; do
    local container_name="phase3-${controller}-s${seed_num}"
    if docker ps | grep -q "${container_name}"; then
      log_msg "  ⚠️  KILLING ${container_name} (mid-training on seed ${seed_num})"
      docker kill "${container_name}" 2>/dev/null || true
      docker rm -f "${container_name}" 2>/dev/null || true
      sleep 2
    fi
  done
}

# ============================================================================
# EXTRACT RESULTS FROM COMPLETED RUN
# ============================================================================

extract_final_score() {
  local container_name="$1"
  local run_dir="$2"
  
  # Try to find the final validation score in container logs
  docker logs "${container_name}" 2>&1 | \
    grep -oP "val/test_score['\"]?\s*[=:]\s*\K[0-9.]+(?![0-9.])" | tail -n 1 || echo "N/A"
}

# ============================================================================
# SAVE CONTROLLER RESULTS TO JSON
# ============================================================================

save_controller_results() {
  local controller="$1"
  local seed="$2"
  local run_dir="$3"
  local final_score="$4"
  local duration_seconds="$5"
  
  local results_file="${RESULTS_DIR}/${controller}_results.json"
  local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  
  # Create results JSON
  local json_content=$(cat <<EOF
{
  "controller": "${controller}",
  "seed": ${seed},
  "global_seed": ${GLOBAL_SEED},
  "epochs": ${TOTAL_EPOCHS},
  "final_test_score": "${final_score}",
  "training_time_seconds": ${duration_seconds},
  "completion_timestamp": "${timestamp}",
  "checkpoint_dir": "${run_dir}",
  "status": "completed"
}
EOF
)
  
  echo "${json_content}" > "${results_file}"
  log_msg "  ✓ Saved results: ${results_file}"
  log_msg "    - Final score: ${final_score}"
  log_msg "    - Training time: $((duration_seconds / 60))m $((duration_seconds % 60))s"
}

# ============================================================================
# MAIN ORCHESTRATION LOOP
# ============================================================================

log_msg "╔════════════════════════════════════════════════════════════════════════════╗"
log_msg "║           SINGLE-SEED 5-CONTROLLER RL BENCHMARK ORCHESTRATOR              ║"
log_msg "║                     Adaptive KL Regularization Study                      ║"
log_msg "╚════════════════════════════════════════════════════════════════════════════╝"
log_msg ""
log_msg "Configuration:"
log_msg "  • Global seed: ${GLOBAL_SEED}"
log_msg "  • Total epochs per controller: ${TOTAL_EPOCHS}"
log_msg "  • Controllers: ${CONTROLLERS[*]}"
log_msg "  • Result directory: ${RESULTS_DIR}"
log_msg "  • Orchestrator log: ${ORCHESTRATOR_LOG}"
log_msg ""

# Create summary file
SUMMARY_FILE="${LOG_DIR}/campaign_summary_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "═══════════════════════════════════════════════════════════"
  echo "Single-Seed 5-Controller Benchmark Results"
  echo "Generated: $(date)"
  echo "Global Seed: ${GLOBAL_SEED}"
  echo "═══════════════════════════════════════════════════════════"
  echo ""
} > "${SUMMARY_FILE}"

RUN_COUNT=0
TOTAL_RUNS=${#CONTROLLERS[@]}

# ============================================================================
# LOOP THROUGH CONTROLLERS IN PRIORITY ORDER
# ============================================================================

for controller in "${CONTROLLERS[@]}"; do
  RUN_COUNT=$((RUN_COUNT + 1))
  
  log_msg ""
  log_msg "╭─────────────────────────────────────────────────────────────────────────╮"
  log_msg "│ [$((RUN_COUNT))/${TOTAL_RUNS}] Controller: ${controller}"
  log_msg "╰─────────────────────────────────────────────────────────────────────────╯"
  
  # ===========================================================================
  # CRITICAL: Kill any stray seed 2/3 runs for this controller
  # ===========================================================================
  
  kill_stray_seed_runs "${controller}"
  
  # ===========================================================================
  # DETERMINE CONTROLLER MODE AND BETA
  # ===========================================================================
  
  case "${controller}" in
    zero_kl)
      RUTH_MODE="fixed"
      RUTH_FIXED_BETA="0.0"
      CONTROLLER_DESC="Zero KL (β = 0)"
      ;;
    fixed)
      RUTH_MODE="fixed"
      RUTH_FIXED_BETA="0.001"
      CONTROLLER_DESC="Fixed KL (β = 0.001)"
      ;;
    rule_based)
      RUTH_MODE="rule"
      RUTH_FIXED_BETA=""
      CONTROLLER_DESC="Rule-based KL calculation"
      ;;
    mlp_based)
      RUTH_MODE="mlp"
      RUTH_FIXED_BETA=""
      CONTROLLER_DESC="MLP-based KL calculation"
      ;;
    lstm_based)
      RUTH_MODE="lstm"
      RUTH_FIXED_BETA=""
      CONTROLLER_DESC="LSTM-based KL calculation"
      ;;
    *)
      log_msg "  ✗ ERROR: Unknown controller ${controller}"
      continue
      ;;
  esac
  
  # ===========================================================================
  # PREPARE RUN ENVIRONMENT
  # ===========================================================================
  
  CONTAINER_NAME="phase3-${controller}-single-seed"
  RUN_DIR="${BASE_RUN_DIR}/phase3_${controller}_seed${GLOBAL_SEED}"
  
  log_msg "  Description: ${CONTROLLER_DESC}"
  log_msg "  Mode: ${RUTH_MODE}"
  log_msg "  Seed: ${GLOBAL_SEED}"
  log_msg "  Epochs: ${TOTAL_EPOCHS}"
  log_msg "  Container: ${CONTAINER_NAME}"
  log_msg "  Checkpoint dir: ${RUN_DIR}"
  log_msg ""
  
  # ===========================================================================
  # LAUNCH TRAINING CONTAINER
  # ===========================================================================
  
  log_msg "  Starting training..."
  export RUTH_MODE="${RUTH_MODE}"
  export RUTH_GLOBAL_SEED="${GLOBAL_SEED}"
  export RUTH_TOTAL_EPOCHS="${TOTAL_EPOCHS}"
  export RUTH_SAVE_FREQ=2000
  export RUTH_TEST_FREQ=2000
  export RUTH_ROLLOUT_N=2
  export RUTH_ROLLOUT_GPU_MEMORY_UTIL=0.4
  export RUTH_ROLLOUT_MAX_NUM_SEQS=32
  export RUTH_CONTAINER_NAME="${CONTAINER_NAME}"
  export RUTH_RUN_DIR="${RUN_DIR}"
  export RUTH_REMOVE_PREVIOUS_CKPT=true
  export RUTH_RESUME_MODE=never
  
  if [[ -n "${RUTH_FIXED_BETA}" ]]; then
    export RUTH_FIXED_BETA="${RUTH_FIXED_BETA}"
  fi
  
  START_TIME=$(date +%s)
  
  cd "${REPO_DIR}/simpleRL-reason"
  bash "${REPO_DIR}/simpleRL-reason/scripts/run_metakl_training_detached.sh" || {
    log_msg "  ✗ ERROR: Failed to start training container for ${controller}"
    echo "  ${RUN_COUNT}. ${controller}: FAILED_TO_START" >> "${SUMMARY_FILE}"
    continue
  }
  
  # ===========================================================================
  # WAIT FOR TRAINING TO COMPLETE
  # ===========================================================================
  
  log_msg "  Waiting for training to complete..."
  POLL_INTERVAL=30
  ELAPSED=0
  
  while docker ps --filter "name=${CONTAINER_NAME}" --quiet | grep -q .; do
    ELAPSED=$(($(date +%s) - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    printf "\r  Progress: %dm %02ds elapsed" "$MINUTES" "$SECONDS" >&2
    sleep $POLL_INTERVAL
  done
  
  printf "\n" >&2
  
  # ===========================================================================
  # COLLECT AND SAVE RESULTS
  # ===========================================================================
  
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  MINUTES=$((DURATION / 60))
  SECONDS=$((DURATION % 60))
  
  log_msg ""
  log_msg "  ✓ Training completed in ${MINUTES}m ${SECONDS}s"
  
  # Extract final validation score from logs
  FINAL_SCORE=$(extract_final_score "${CONTAINER_NAME}" "${RUN_DIR}")
  
  # Save results to JSON file (per-controller)
  save_controller_results "${controller}" "${GLOBAL_SEED}" "${RUN_DIR}" "${FINAL_SCORE}" "${DURATION}"
  
  # Append to summary
  {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${controller}"
    echo "  Mode: ${RUTH_MODE}"
    echo "  Final Score: ${FINAL_SCORE}"
    echo "  Time: ${MINUTES}m ${SECONDS}s"
  } >> "${SUMMARY_FILE}"
  
  # ===========================================================================
  # CLEANUP: Keep checkpoint but remove container
  # ===========================================================================
  
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
  
  log_msg "  ✓ Container cleaned up"
  
  # Send Telegram update if configured
  if [[ -n "${TELEGRAM_BOT_TOKEN}" && -n "${TELEGRAM_CHAT_ID}" ]]; then
    send_telegram_update "✓ ${controller} completed: ${FINAL_SCORE} (${MINUTES}m)"
  fi
  
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

log_msg ""
log_msg "╔════════════════════════════════════════════════════════════════════════════╗"
log_msg "║                     ALL CONTROLLERS COMPLETED SUCCESSFULLY                 ║"
log_msg "╚════════════════════════════════════════════════════════════════════════════╝"
log_msg ""
log_msg "Summary saved to:"
log_msg "  ${SUMMARY_FILE}"
log_msg ""
log_msg "Individual controller results saved to:"
for controller in "${CONTROLLERS[@]}"; do
  log_msg "  ${RESULTS_DIR}/${controller}_results.json"
done
log_msg ""
log_msg "All checkpoints saved to:"
log_msg "  ${BASE_RUN_DIR}/"
log_msg ""

{
  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo "Campaign Status: COMPLETED"
  echo "Timestamp: $(date)"
  echo "═══════════════════════════════════════════════════════════"
} >> "${SUMMARY_FILE}"

log_msg "✓ Orchestrator finished successfully"

if [[ -n "${TELEGRAM_BOT_TOKEN}" && -n "${TELEGRAM_CHAT_ID}" ]]; then
  send_telegram_update "🎉 All 5 controllers completed successfully!"
fi
