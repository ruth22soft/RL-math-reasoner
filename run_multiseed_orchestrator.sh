#!/usr/bin/env bash
set -euo pipefail
# Sequential multi-seed orchestrator: modes x seeds
MODES=(fixed rule mlp lstm)
SEEDS=(0 1 2)
TOTAL_STEPS=${1:-20000}
RUN_WAIT_SLEEP=10
ROOT_DIR=$(pwd)
LAUNCHER="$ROOT_DIR/phase3_wandb_launch.sh"
if [ ! -x "$LAUNCHER" ]; then
  echo "Launcher not found or not executable: $LAUNCHER" >&2
  exit 2
fi
for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Starting run: mode=$mode seed=$seed total_steps=$TOTAL_STEPS"
    # fixed mode needs explicit beta override
    if [ "$mode" = "fixed" ]; then
      RUTH_FIXED_BETA=0.001 RUTH_TOTAL_TRAINING_STEPS="$TOTAL_STEPS" "$LAUNCHER" "$mode" "$seed" "run"
    else
      RUTH_TOTAL_TRAINING_STEPS="$TOTAL_STEPS" "$LAUNCHER" "$mode" "$seed" "run"
    fi
    CONTAINER="phase3-${mode}-s${seed}"
    echo "Launched container: $CONTAINER — waiting to finish"
    # Wait for container to appear
    until docker inspect "$CONTAINER" --format '{{.State.Status}}' >/dev/null 2>&1; do
      echo "Waiting for container $CONTAINER to be created..."
      sleep 2
    done
    # Poll until container is not running
    while true; do
      status=$(docker inspect "$CONTAINER" --format '{{.State.Status}}' 2>/dev/null || echo missing)
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $CONTAINER status=$status"
      if [ "$status" != "running" ]; then
        echo "Container $CONTAINER finished with status=$status"
        break
      fi
      sleep $RUN_WAIT_SLEEP
    done
    echo "Run $mode seed $seed completed — proceeding to next."
    sleep 3
  done
done

echo "All runs completed."
