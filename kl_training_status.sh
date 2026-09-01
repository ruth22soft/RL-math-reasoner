#!/usr/bin/env bash
set -euo pipefail

for name in amharic-fixed-kl amharic-lstm-kl amharic-grpo; do
  echo "===== ${name} ====="
  if docker ps -a --filter "name=${name}" --quiet | grep -q .; then
    docker inspect -f 'status={{.State.Status}} running={{.State.Running}} exit={{.State.ExitCode}} started={{.State.StartedAt}}' "${name}"
    echo "--- last lines ---"
    docker logs "${name}" 2>&1 | tail -n 20 || true
  else
    echo "not found"
  fi
  echo
 done
