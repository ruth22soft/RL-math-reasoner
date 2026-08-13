#!/usr/bin/env bash
set -euo pipefail
OUT_DIR=/home/ai-server-02/R_projects/final_thesis/phase2_monitor_smoke_fixed_n4
mkdir -p "$OUT_DIR"
GPU_CSV="$OUT_DIR/gpu_usage.csv"
LOG_TXT="$OUT_DIR/training_steps.log"
CONTAINER=smoke-phase2-fixed-n4
echo "timestamp,memory_used_mib,memory_total_mib,gpu_util_pct" > "$GPU_CSV"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Monitor started" > "$LOG_TXT"
while true; do
  status=$(docker inspect "$CONTAINER" --format '{{.State.Status}}' 2>/dev/null || echo missing)
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [ "$status" = "missing" ]; then
    echo "$ts container_missing" >> "$LOG_TXT"
    break
  fi
  # sample nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    read mem_used mem_total util <<<$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -n1 | awk -F, '{gsub(" ","",$1);gsub(" ","",$2);gsub(" ","",$3); print $1" "$2" "$3}') || true
    if [ -n "${mem_used:-}" ]; then
      echo "$ts,$mem_used,$mem_total,$util" >> "$GPU_CSV"
    fi
  fi
  # capture latest step line
  last_line=$(docker logs "$CONTAINER" 2>&1 | grep -n "step:" | tail -n1 || true)
  echo "$ts | STATUS=$status | $last_line" >> "$LOG_TXT"
  if [ "$status" != "running" ]; then
    echo "$ts container_not_running" >> "$LOG_TXT"
    break
  fi
  sleep 5
done

echo "Monitor finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_TXT"
