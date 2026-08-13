#!/usr/bin/env bash
# Deprecated: use scripts/run_rule_then_lstm.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_rule_then_lstm.sh" "$@"
