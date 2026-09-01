#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ai-server-02/R_projects/final_thesis/simpleRL-reason"
CONTAINER_NAME="amharic-lstm-kl"
IMAGE_NAME="${IMAGE_NAME:-simple-rl:ngc-vllm}"
RUN_DIR="/ckpts/amharic_lstm_kl"

mkdir -p /home/ai-server-02/R_projects/final_thesis/results

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${REPO_DIR}":/workspace/simpleRL-reason -w /workspace/simpleRL-reason \
  -v simplerl_data:/data \
  -v simplerl_hf_cache:/root/.cache/huggingface \
  -v simplerl_ckpts:/ckpts \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  "${IMAGE_NAME}" bash -lc '
set -e
python -m pip install -e . --no-deps
python -m pip install word2number "antlr4-python3-runtime==4.9.3" "math-verify==0.6.0" || true
export PYTHONPATH=/workspace/simpleRL-reason:$PYTHONPATH
mkdir -p /ckpts/amharic_lstm_kl
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-name simplelr_grpo_amharic_single_gpu \
  trainer.total_epochs=1 \
  trainer.default_local_dir=/ckpts/amharic_lstm_kl \
  trainer.save_freq=50 \
  trainer.test_freq=-1 \
  trainer.resume_mode=never \
  trainer.remove_previous_ckpt=false \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.actor_adaptive_kl.enable=true \
  actor_rollout_ref.actor.actor_adaptive_kl.mode=lstm \
  actor_rollout_ref.actor.actor_adaptive_kl.lstm_hidden_dim=32 \
  actor_rollout_ref.actor.actor_adaptive_kl.lstm_reset_on_epoch=true \
  2>&1 | tee /ckpts/amharic_lstm_kl/train.log
'

echo "Started ${CONTAINER_NAME}"
echo "Status: docker ps --filter name=${CONTAINER_NAME}"
echo "Logs: docker logs -f ${CONTAINER_NAME}"
