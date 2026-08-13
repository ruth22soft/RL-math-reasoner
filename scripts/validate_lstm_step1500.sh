#!/usr/bin/env bash
# One-off validation of the LSTM controller's global_step_1500 checkpoint.
# Reproduces the trainer's own _validate() (val_only=true) using the IDENTICAL
# config that produced the val/test_score for the other controllers, so the
# resulting number is directly comparable. Writes results to the LSTM run dir.
set -euo pipefail

REPO_DIR="/home/ai-server-02/R_projects/final_thesis/simpleRL-reason"
IMAGE_NAME="simple-rl:ngc-vllm"
RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_lstm_s1"
CONTAINER_NAME="phase3-lstm-s1-validate"
VAL_LOG="${RUN_DIR}/val_only_step1500.log"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --rm --name "${CONTAINER_NAME}" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${REPO_DIR}":/workspace/simpleRL-reason -w /workspace/simpleRL-reason \
  -v simplerl_data:/data \
  -v simplerl_hf_cache:/root/.cache/huggingface \
  -v simplerl_ckpts:/ckpts \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${IMAGE_NAME}" bash -lc '
set -e
python -m pip install -e . --no-deps >/dev/null 2>&1
python -m pip install word2number "antlr4-python3-runtime==4.9.3" "math-verify==0.6.0" >/dev/null 2>&1

# Same grader nan/zoo patch the training run uses, so scoring matches exactly.
python3 << "PATCH_GRADER"
grader_path = "/usr/local/lib/python3.10/dist-packages/math_verify/grader.py"
try:
    with open(grader_path) as f:
        content = f.read()
    if "raise ValueError" in content and "Can'"'"'t evaluate nan or zoo" in content:
        content = content.replace("raise ValueError(\"Can'"'"'t evaluate nan or zoo\")", "return False")
        with open(grader_path, "w") as f:
            f.write(content)
        print("[OK] Patched grader")
    else:
        print("[WARN] grader pattern not found (already patched?)")
except Exception as e:
    print("[ERROR] grader patch failed:", e)
PATCH_GRADER

export PYTHONPATH=/workspace/simpleRL-reason:$PYTHONPATH
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-name simplelr_grpo_qwen05_single_gpu \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  trainer.default_local_dir='"${RUN_DIR}"' \
  trainer.resume_mode=auto \
  +trainer.val_only=true \
  trainer.val_before_train=true \
  actor_rollout_ref.actor.actor_adaptive_kl.enable=true \
  actor_rollout_ref.actor.actor_adaptive_kl.mode=lstm \
  actor_rollout_ref.actor.actor_adaptive_kl.warmup_steps=1 \
  2>&1 | tee "'"${VAL_LOG}"'"
'
