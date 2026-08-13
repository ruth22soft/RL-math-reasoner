#!/usr/bin/env bash
# Prune obsolete checkpoints on simplerl_ckpts volume. Keeps one latest ckpt per active run dir.
set -euo pipefail

CKPT_ROOT="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive"
IMAGE="${RUTH_IMAGE_NAME:-simple-rl:ngc-vllm}"

log() { echo "[cleanup $(date '+%H:%M:%S')] $*"; }

docker run --rm -v simplerl_ckpts:/ckpts "${IMAGE}" bash -lc "
set -euo pipefail
CKPT_ROOT='${CKPT_ROOT}'

log() { echo \"[cleanup] \$*\"; }

prune_run_dir() {
  local run_dir=\"\$1\"
  local keep_last=\"\${2:-1}\"
  [[ -d \"\$run_dir\" ]] || return 0
  mapfile -t steps < <(find \"\$run_dir\" -maxdepth 1 -type d -name 'global_step_*' | sort -V)
  local n=\${#steps[@]}
  if (( n <= keep_last )); then
    return 0
  fi
  local delete_count=\$((n - keep_last))
  for ((i=0; i<delete_count; i++)); do
    log \"Removing \${steps[i]}\"
    rm -rf \"\${steps[i]}\"
  done
}

# Completed MLP: keep only latest checkpoint
prune_run_dir \"\${CKPT_ROOT}/phase3_mlp_s1\" 1

# Rule resume run: keep only latest (step 400) before continuing
prune_run_dir \"\${CKPT_ROOT}/runs/rule-based/seed_3/20260523_154450\" 1

# Completed fixed benchmark dir (logs only; no global_step dirs)
# Remove failed / empty phase3 seed sweep dirs
for d in phase3_mlp_s0 phase3_mlp_s2 phase3_rule_s0 phase3_rule_s1 phase3_rule_s2 \
         phase3_lstm_s0 phase3_lstm_s1 phase3_lstm_s2 phase3_fixed_s0 phase3_fixed_s1 \
         phase3_rule_based_s1; do
  if [[ -d \"\${CKPT_ROOT}/\${d}\" ]]; then
    log \"Removing obsolete run dir \${d}\"
    rm -rf \"\${CKPT_ROOT}/\${d}\"
  fi
done

# Old experiments outside phase3 campaign
for d in 20260420_110659 20260420_110947 20260421_072158 global_step_1550 smoke_phase2_fixed_n4; do
  if [[ -e \"\${CKPT_ROOT}/\${d}\" ]]; then
    log \"Removing legacy \${d}\"
    rm -rf \"\${CKPT_ROOT}/\${d}\"
  fi
done

# Stray root log
rm -f \"\${CKPT_ROOT}/metakl_train.log\" \"\${CKPT_ROOT}/latest_checkpointed_iteration.txt\" 2>/dev/null || true

du -sh \"\${CKPT_ROOT}\"/* 2>/dev/null | sort -hr | head -15
"

log "Checkpoint cleanup done."
