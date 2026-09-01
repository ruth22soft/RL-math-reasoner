#!/usr/bin/env bash
set -euo pipefail

# Wait for the amharic fixed-KL training container to finish, then start LSTM
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMHARIC_CONTAINER="amharic-text-generation"
LOGFILE="${REPO_DIR}/start_lstm_after_fixed.log"

echo "[INFO] Watcher started at $(date)" | tee -a "${LOGFILE}"

while docker ps --filter "name=${AMHARIC_CONTAINER}" --filter "status=running" --quiet | grep -q .; do
  echo "[INFO] ${AMHARIC_CONTAINER} still running at $(date)" | tee -a "${LOGFILE}"
  sleep 30
done

echo "[INFO] ${AMHARIC_CONTAINER} not running anymore at $(date). Starting LSTM." | tee -a "${LOGFILE}"
cd "${REPO_DIR}"
./lstm_training.sh start lstm 2>&1 | tee -a "${LOGFILE}" &

echo "[INFO] LSTM start triggered (background) at $(date)" | tee -a "${LOGFILE}"
