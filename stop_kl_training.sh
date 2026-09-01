#!/usr/bin/env bash
set -euo pipefail

for name in amharic-fixed-kl amharic-lstm-kl amharic-grpo; do
  if docker ps -a --filter "name=${name}" --quiet | grep -q .; then
    echo "Stopping ${name}"
    docker stop -t 30 "${name}" || docker kill "${name}" || true
  fi
done

echo "All matching KL training containers stopped."
