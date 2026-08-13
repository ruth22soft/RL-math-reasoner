#!/usr/bin/env bash
set -euo pipefail

# Multi-seed orchestrator: Sequential execution of fixed, rule, mlp, lstm
# Each mode: 3 seeds (0, 1, 2) = 12 total runs
# Waits for phase3-nokl-baseline (5th baseline) to complete before starting

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_DIR}/simpleRL-reason/analysis_logs"
ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator_multiseed_v2_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

# Configuration
MODES=("fixed" "rule" "mlp" "lstm")
SEEDS=(0 1 2)
TOTAL_EPOCHS=3
BASE_RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive"

# Fixed mode specific
FIXED_BETA=0.001

# Telegram settings (if available)
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

# Helper function to log with timestamp
log_msg() {
  local msg="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" | tee -a "${ORCHESTRATOR_LOG}"
}

# Helper function to extract final val/test_score
extract_final_score() {
  local container_name="$1"
  local run_dir="$2"
  
  # Look for final validation score in logs
  docker logs "${container_name}" 2>&1 | \
    grep -oP "val/test_score['\"]?\s*[=:]\s*\K[0-9.]+(?![0-9.])" | tail -n 1 || echo "N/A"
}

# Helper function to send Telegram update
send_telegram_update() {
  local message="$1"
  if [[ -n "${TELEGRAM_BOT_TOKEN}" && -n "${TELEGRAM_CHAT_ID}" ]]; then
    TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}" TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID}" \
    python3 "${REPO_DIR}/simpleRL-reason/scripts/telegram_metakl_report.py" <<< "${message}" 2>/dev/null || true
  fi
}

log_msg "==============================================================="
log_msg "Multi-Seed Orchestrator v2 - Starting"
log_msg "==============================================================="
log_msg "Modes: ${MODES[*]}"
log_msg "Seeds per mode: ${SEEDS[*]}"
log_msg "Total runs: $((${#MODES[@]} * ${#SEEDS[@]}))"
log_msg "Log file: ${ORCHESTRATOR_LOG}"
log_msg ""

# Wait for phase3-nokl-baseline to complete
log_msg "Waiting for phase3-nokl-baseline (5th baseline, β=0) to complete..."
while docker ps --filter "name=phase3-nokl-baseline" --quiet | grep -q .; do
  log_msg "  ... phase3-nokl-baseline still running, checking again in 60s"
  sleep 60
done

# Collect no-KL baseline result
log_msg "Collecting phase3-nokl-baseline results..."
NOKL_SCORE=$(docker logs phase3-nokl-baseline 2>&1 | grep -oP "val/test_score['\"]?\s*[=:]\s*\K[0-9.]+(?![0-9.])" | tail -n 1 || echo "N/A")
log_msg "  No-KL baseline final score: ${NOKL_SCORE}"
docker rm -f phase3-nokl-baseline || true

log_msg ""
log_msg "==============================================================="
log_msg "Starting 4-Mode × 3-Seed Campaign (12 runs total)"
log_msg "==============================================================="
log_msg ""

RESULTS_SUMMARY="${LOG_DIR}/campaign_results_summary_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "Multi-Seed Campaign Results Summary"
  echo "Generated: $(date)"
  echo "========================================"
  echo ""
  echo "Baseline Comparison:"
  echo "  5. No-KL (β=0): ${NOKL_SCORE}"
  echo "  4. LSTM (reference, Apr21): 0.3387 (33.87%)"
  echo ""
  echo "Campaign Results:"
} > "${RESULTS_SUMMARY}"

RUN_COUNT=0
TOTAL_RUNS=$((${#MODES[@]} * ${#SEEDS[@]}))

for mode in "${MODES[@]}"; do
  log_msg "======= MODE: ${mode} ======="
  
  for seed in "${SEEDS[@]}"; do
    RUN_COUNT=$((RUN_COUNT + 1))
    CONTAINER_NAME="phase3-${mode}-s${seed}"
    RUN_DIR="${BASE_RUN_DIR}/phase3_${mode}_s${seed}"
    
    log_msg ""
    log_msg "[${RUN_COUNT}/${TOTAL_RUNS}] Starting ${CONTAINER_NAME}..."
    log_msg "  Mode: ${mode}, Seed: ${seed}, Epochs: ${TOTAL_EPOCHS}"
    
    # Build environment for this run
    export RUTH_MODE="${mode}"
    export RUTH_TOTAL_EPOCHS="${TOTAL_EPOCHS}"
    export RUTH_SAVE_FREQ=2000
    export RUTH_TEST_FREQ=2000
    export RUTH_ROLLOUT_N=4
    export RUTH_ROLLOUT_GPU_MEMORY_UTIL=0.5
    export RUTH_ROLLOUT_MAX_NUM_SEQS=32
    export RUTH_CONTAINER_NAME="${CONTAINER_NAME}"
    export RUTH_RUN_DIR="${RUN_DIR}"
    export RUTH_REMOVE_PREVIOUS_CKPT=true
    export RUTH_RESUME_MODE=never
    
    if [[ "${mode}" == "fixed" ]]; then
      export RUTH_FIXED_BETA="${FIXED_BETA}"
    fi
    
    # Launch container
    START_TIME=$(date +%s)
    cd "${REPO_DIR}/simpleRL-reason"
    bash "${REPO_DIR}/simpleRL-reason/scripts/run_metakl_training_detached.sh" || {
      log_msg "  ERROR: Failed to start ${CONTAINER_NAME}"
      continue
    }
    
    # Wait for container to complete
    log_msg "  Waiting for ${CONTAINER_NAME} to complete..."
    while docker ps --filter "name=${CONTAINER_NAME}" --quiet | grep -q .; do
      # Show progress every 5 minutes
      ELAPSED=$(($(date +%s) - START_TIME))
      MINUTES=$((ELAPSED / 60))
      log_msg "    Progress: ${MINUTES}min elapsed..."
      sleep 300
    done
    
    # Collect results
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    FINAL_SCORE=$(extract_final_score "${CONTAINER_NAME}" "${RUN_DIR}")
    
    log_msg "  COMPLETED in ${MINUTES}m ${SECONDS}s"
    log_msg "  Final validation score: ${FINAL_SCORE}"
    
    # Append to results summary
    echo "  ${RUN_COUNT}. ${mode} seed=${seed}: ${FINAL_SCORE} (${MINUTES}m ${SECONDS}s)" >> "${RESULTS_SUMMARY}"
    
    # Send Telegram update
    send_telegram_update "✅ Run ${RUN_COUNT}/${TOTAL_RUNS} complete: ${mode} s${seed} = ${FINAL_SCORE}"
    
    # Clean up container
    docker rm -f "${CONTAINER_NAME}" || true
    
    log_msg ""
  done
done

log_msg "==============================================================="
log_msg "Campaign Complete!"
log_msg "==============================================================="
log_msg ""
log_msg "Results Summary:"
cat "${RESULTS_SUMMARY}" | tail -n +3 | tee -a "${ORCHESTRATOR_LOG}"

log_msg ""
log_msg "Full logs: ${ORCHESTRATOR_LOG}"
log_msg "Results: ${RESULTS_SUMMARY}"
log_msg ""

# Final Telegram report
if [[ -n "${TELEGRAM_BOT_TOKEN}" && -n "${TELEGRAM_CHAT_ID}" ]]; then
  log_msg "Sending final Telegram report..."
  FINAL_MSG="🎉 Multi-Seed Campaign Complete!
Ran ${TOTAL_RUNS} experiments (4 modes × 3 seeds)
Results: $(cat "${RESULTS_SUMMARY}" | head -n 20)"
  send_telegram_update "${FINAL_MSG}"
fi

log_msg "Done!"
