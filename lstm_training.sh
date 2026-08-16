#!/usr/bin/env bash
# LSTM Training Quick Dashboard and Control
# Purpose: Simple one-command interface for LSTM training
# Usage: ./lstm_training.sh [command] [options]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="lstm-training-live"
RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training"
IMAGE_NAME="simple-rl:ngc-vllm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}$1${NC}"
  echo -e "${BLUE}========================================${NC}"
}

print_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
  echo -e "${RED}✗ $1${NC}"
}

print_info() {
  echo -e "${YELLOW}ℹ $1${NC}"
}

show_help() {
  cat << 'EOF'
LSTM Training Quick Control

USAGE:
  ./lstm_training.sh [command] [options]

COMMANDS:
  start [mode]       Start training (mode: lstm|rbf; resumes if checkpoint exists)
  status             Show current training status
  logs               Show live training logs (tail -f)
  logs N             Show last N lines of logs
  stop               Stop training container gracefully
  kill               Force stop training container
  gpu                Check GPU usage
  checkpoints        List saved checkpoints
  metrics            Show latest training metrics
  config             Show LSTM configuration
  clean              Remove container (won't delete checkpoints)
  help               Show this help message

EXAMPLES:
  ./lstm_training.sh start               # Start LSTM training
  ./lstm_training.sh start rbf           # Start RBF training
  ./lstm_training.sh logs                # Watch logs live
  ./lstm_training.sh status              # Check status
  ./lstm_training.sh logs 50             # Last 50 lines
  ./lstm_training.sh metrics             # Latest metrics
  ./lstm_training.sh checkpoints         # List checkpoints
  ./lstm_training.sh stop                # Stop gracefully
  ./lstm_training.sh clean               # Remove container

EOF
}

start_training() {
  local mode="${1:-lstm}"

  print_header "Starting LSTM Training"
  print_info "Mode: ${mode}"
  
  print_info "Container: $CONTAINER_NAME"
  print_info "Run Dir: $RUN_DIR"
  print_info "Epochs: 3"
  print_info "Save Freq: 500 steps"
  
  latest_checkpoint=""
  latest_checkpoint=$(docker run --rm -v simplerl_ckpts:/ckpts alpine:3.20 sh -lc '
    RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training"
    find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -name "global_step_*" 2>/dev/null | sort -V | tail -1 || true
  ')
  
  if [[ -n "${latest_checkpoint}" ]]; then
    resume_mode="auto"
    resume_checkpoint="${latest_checkpoint}"
    restart_policy="no"
    remove_previous_ckpt="false"
    print_info "Resume Mode: AUTO (continue from ${resume_checkpoint})"
    print_info "Remove Old Checkpoints: NO (keep checkpoint history)"
  else
    resume_mode="never"
    resume_checkpoint=""
    restart_policy="unless-stopped"
    remove_previous_ckpt="true"
    print_info "Resume Mode: NEVER (fresh run from step 1)"
    print_info "Remove Old Checkpoints: YES"
  fi
  
  cd "${REPO_DIR}"
  
  RUTH_MODE="${mode}" \
  RUTH_CONTAINER_NAME="${CONTAINER_NAME}" \
  RUTH_RUN_DIR="${RUN_DIR}" \
  RUTH_TOTAL_EPOCHS=3 \
  RUTH_ROLLOUT_N=4 \
  RUTH_SAVE_FREQ=500 \
  RUTH_TEST_FREQ=-1 \
  RUTH_RESTART_POLICY="${restart_policy}" \
  RUTH_RESUME_MODE="${resume_mode}" \
  RUTH_RESUME_CHECKPOINT="${resume_checkpoint}" \
  RUTH_REMOVE_PREVIOUS_CKPT="${remove_previous_ckpt}" \
  bash simpleRL-reason/scripts/run_metakl_training_detached.sh
  
  print_success "Training started!"
  print_info "Watch logs with: $0 logs"
  print_info "Check status with: $0 status"
}

show_status() {
  print_header "Training Status"
  
  if docker ps -a --filter "name=${CONTAINER_NAME}" --quiet | grep -q .; then
    state=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
    running=$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || echo "false")
    exit_code=$(docker inspect -f '{{.State.ExitCode}}' "${CONTAINER_NAME}" 2>/dev/null || echo "N/A")
    started_at=$(docker inspect -f '{{.State.StartedAt}}' "${CONTAINER_NAME}" 2>/dev/null || echo "N/A")
    
    echo -e "${BLUE}Container:${NC} $CONTAINER_NAME"
    echo -e "${BLUE}Status:${NC} $state"
    if [[ "$running" == "true" ]]; then
      echo -e "${GREEN}Running:${NC} Yes"
    else
      echo -e "${RED}Running:${NC} No (exit code: $exit_code)"
    fi
    echo -e "${BLUE}Started:${NC} $started_at"
    
    # Show latest training step using the actual step log line, not timing metrics like timing_s/step:XX
    recent_logs=$(docker logs "${CONTAINER_NAME}" 2>&1 | tail -n 200 || true)
    latest_step=$(printf '%s\n' "$recent_logs" | grep -Eo 'step:[[:space:]]*[0-9]+[[:space:]]+-' | tail -1 | sed -E 's/.*step:[[:space:]]*([0-9]+)[[:space:]]+-/\1/' || echo "N/A")
    echo -e "${BLUE}Latest Step:${NC} $latest_step"
    
    # Show latest metrics from recent logs only
    latest_line=$(printf '%s\n' "$recent_logs" | grep "actor/" | tail -1 || echo "")
    if [[ -n "$latest_line" ]]; then
      echo -e "\n${BLUE}Latest Metrics:${NC}"
      echo "$latest_line" | sed 's/ - /\n  /g' | head -10
    fi
  else
    print_error "Container not found"
  fi
}

show_logs() {
  local lines="${1:-0}"
  
  if [[ $lines -eq 0 ]]; then
    print_info "Showing live logs... (Press Ctrl+C to exit)"
    docker logs -f "${CONTAINER_NAME}"
  else
    print_info "Showing last $lines lines of logs:"
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -n "$lines"
  fi
}

stop_training() {
  print_header "Stopping Training"
  
  if docker ps --filter "name=${CONTAINER_NAME}" --quiet | grep -q .; then
    print_info "Sending SIGTERM to container (waiting 30 seconds)..."
    docker stop -t 30 "${CONTAINER_NAME}"
    print_success "Container stopped gracefully"
  else
    print_error "Container is not running"
  fi
}

kill_training() {
  print_header "Force Stopping Training"
  
  if docker ps -a --filter "name=${CONTAINER_NAME}" --quiet | grep -q .; then
    print_info "Force stopping container..."
    docker kill "${CONTAINER_NAME}" || true
    print_success "Container force stopped"
  else
    print_error "Container not found"
  fi
}

show_gpu() {
  print_header "GPU Usage"
  nvidia-smi
}

list_checkpoints() {
  print_header "Saved Checkpoints"
  
  docker run --rm -v simplerl_ckpts:/ckpts "${IMAGE_NAME}" \
    bash -lc "ls -lh ${RUN_DIR} 2>/dev/null | grep global_step || echo 'No checkpoints yet'" | head -10
  
  print_info "Latest iteration tracker:"
  docker run --rm -v simplerl_ckpts:/ckpts "${IMAGE_NAME}" \
    bash -lc "cat ${RUN_DIR}/latest_checkpointed_iteration.txt 2>/dev/null || echo 'None'" || true
}

show_metrics() {
  print_header "Latest Training Metrics"
  
  if docker ps -a --filter "name=${CONTAINER_NAME}" --quiet | grep -q .; then
    print_info "Last 3 metric lines from training:"
    docker logs "${CONTAINER_NAME}" 2>&1 | grep "actor/" | tail -3
  else
    print_error "Container not running"
  fi
}

show_config() {
  print_header "LSTM Configuration"
  
  cat << 'EOF'
LSTM Mode Settings:
  Mode: lstm (LSTM-based adaptive KL control)
  Enable: true
  
LSTM Architecture:
  Hidden Dimension: 32
  Learning Rate: 5e-6
  Reset on Epoch: true (resets LSTM state between epochs)
  State Features: [kl_loss, reward_mean, reward_std, lagged_grad_norm]
  
KL Control Parameters:
  Target KL Loss: 0.003
  Initial Beta (KL coefficient): 0.001
  Min Beta: 0.0001
  Max Beta: 0.01
  
Training Setup:
  Total Epochs: 3
  Train Batch Size: 256
  Rollout N: 4
  Prompt Length: 1024
  Response Length: 1024
  Max Batched Tokens: 4096
  
Checkpoint Management:
  Save Frequency: 500 steps
  Test Frequency: -1 (disabled)
  Remove Previous Checkpoints: YES
  Resume Mode: auto
  
Expected Timeline:
  Total Steps: ~1566 (522 per epoch)
  Time per Epoch: ~45-60 min
  Full Training: ~2.5-3 hours (single GPU)
  Checkpoint Interval: ~15 minutes

EOF
}

clean_container() {
  print_header "Cleaning Up Container"
  
  if docker ps -a --filter "name=${CONTAINER_NAME}" --quiet | grep -q .; then
    print_info "Removing container $CONTAINER_NAME..."
    docker rm "${CONTAINER_NAME}" || print_error "Failed to remove container"
    print_success "Container removed (checkpoints preserved)"
  else
    print_info "Container not found, nothing to clean"
  fi
}

# Main command parsing
if [[ $# -eq 0 ]]; then
  show_status
  exit 0
fi

case "${1:-}" in
  start)
    start_training "${2:-lstm}"
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs "${2:-0}"
    ;;
  stop)
    stop_training
    ;;
  kill)
    kill_training
    ;;
  gpu)
    show_gpu
    ;;
  checkpoints)
    list_checkpoints
    ;;
  metrics)
    show_metrics
    ;;
  config)
    show_config
    ;;
  clean)
    clean_container
    ;;
  help|--help|-h)
    show_help
    ;;
  *)
    print_error "Unknown command: $1"
    echo ""
    show_help
    exit 1
    ;;
esac
