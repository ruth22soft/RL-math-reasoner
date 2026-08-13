#!/usr/bin/env bash
set -euo pipefail
while true; do
  last=$(docker logs smoke-phase2-fixed-n4 2>&1 | grep -n 'step:' | tail -n1 || true)
  echo "LAST=$last"
  if echo "$last" | grep -q 'step:20'; then
    echo "REACHED_STEP_20"
    exit 0
  fi
  status=$(docker inspect smoke-phase2-fixed-n4 --format '{{.State.Status}}' 2>/dev/null || echo missing)
  echo "STATUS=$status"
  if [ "$status" != "running" ]; then
    echo "CONTAINER_NOT_RUNNING"
    exit 0
  fi
  sleep 5
done
