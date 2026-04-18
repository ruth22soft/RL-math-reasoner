# Docker GRPO training guide (SimpleRL)

This guide documents the Docker-based workflow we’ve been using in this repo to run GRPO training and keep **data + HF cache + checkpoints** persisted via Docker volumes.

## 0) Prerequisites (host)

- Linux host with NVIDIA GPU + driver.
- Docker installed.
- NVIDIA Container Toolkit installed (so `--gpus all` works).

Quick sanity checks:

```bash
docker --version
nvidia-smi
```

Inside a CUDA container, you should see a GPU (run on host):

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 1) Build the Docker image (only when needed)

If you already have the image `simple-rl:ngc-vllm`, you can skip this section.

From the repo root:

```bash
docker build -t simple-rl:ngc-vllm -f docker/Dockerfile.ngc.vllm .
```

## 2) One-time: create persistent Docker volumes

We use 3 volumes:

- `simplerl_data` → mounted at `/data` (datasets)
- `simplerl_hf_cache` → mounted at `/root/.cache/huggingface` (HF cache)
- `simplerl_ckpts` → mounted at `/ckpts` (logs + checkpoints)

Create them once:

```bash
docker volume create simplerl_data
docker volume create simplerl_hf_cache
docker volume create simplerl_ckpts
```

## 3) One-time (or as-needed): put the dataset into the `/data` volume

This downloads the SimpleRL-Zoo parquet files into the `simplerl_data` volume so you don’t have to manage host paths.

```bash
docker run --rm \
  -v simplerl_data:/data \
  simple-rl:ngc-vllm bash -lc '
set -e
mkdir -p /data/simplelr_qwen_level3to5
cd /data/simplelr_qwen_level3to5
python - <<"PY"
import os
import urllib.request

base = "https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_qwen_level3to5"
for fn in ("train.parquet", "test.parquet"):
    dst = os.path.join(os.getcwd(), fn)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("exists", fn)
        continue
    print("downloading", fn)
    urllib.request.urlretrieve(f"{base}/{fn}", dst)
print("dataset_ready", os.listdir(os.getcwd()))
PY
'
```

## 4) Run training (next time + every time)

### 4.1) Recommended: run using the checked-in Hydra preset

Preset config file:

- [verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml](verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml)

It is designed for:

- single GPU
- GRPO (`algorithm.adv_estimator=grpo`)
- vLLM rollout
- context length: 1024 prompt + 1024 response
- outputs: `/ckpts/simplelr_grpo_qwen05_ctx1024`
- **checkpoint saving enabled** (see “Checkpoints” below)

Run it inside the container (recommended: use a fixed container name so monitoring is reliable):

```bash
docker run --rm --name ruth-training-run \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/simpleRL-reason -w /workspace/simpleRL-reason \
  -v simplerl_data:/data \
  -v simplerl_hf_cache:/root/.cache/huggingface \
  -v simplerl_ckpts:/ckpts \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  simple-rl:ngc-vllm bash -lc '
set -e
python -m pip install -e . --no-deps

# Reward deps used by SimpleRL math scoring
python -m pip install word2number "antlr4-python3-runtime==4.9.3" "math-verify==0.6.0"

export PYTHONPATH=/workspace/simpleRL-reason:$PYTHONPATH

# Run the preset config
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-name simplelr_grpo_qwen05_single_gpu \
  2>&1 | tee /ckpts/simplelr_grpo_qwen05_ctx1024/run.log
'
```

### 4.2) Common overrides (optional)

Epochs (the preset defaults to 3):

```bash
# add to the python command:
trainer.total_epochs=1   # or 3, 5, ...
```

Change where outputs go (new run directory):

```bash
trainer.default_local_dir=/ckpts/my_new_run_name
```

Disable periodic validation during training (final validation still runs at the end):

```bash
trainer.test_freq=-1
```

Enable periodic validation every N steps:

```bash
trainer.test_freq=50
```

## 5) Checkpoints (what persists and how resuming works)

### 5.1) Where checkpoints live

All logs/checkpoints are written into the Docker volume `simplerl_ckpts` mounted at `/ckpts`.

For the preset config, outputs go under:

- `/ckpts/simplelr_grpo_qwen05_ctx1024/`

To list what exists in the volume:

```bash
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm bash -lc 'ls -lah /ckpts'
```

### 5.2) Checkpoint saving frequency (already enabled in the preset)

In the preset config file, we set:

- `trainer.save_freq: 50` (save every 50 steps)
- `trainer.remove_previous_ckpt: True` (keep only the latest checkpoint to avoid filling disk)
- `trainer.resume_mode: auto`

If you want to keep **all** checkpoints, change:

- `trainer.remove_previous_ckpt: False`

### 5.3) Resuming a run

Resuming is controlled by:

- `trainer.default_local_dir` (points at the run directory)
- `trainer.resume_mode` (set to `auto`)
- having at least one saved checkpoint folder like `global_step_50/` inside the run directory

To resume the same run, re-run training with the **same** `trainer.default_local_dir`.

If there are **no** `global_step_*` folders yet, there is nothing to resume from (it will start from scratch).

## 6) Monitoring

### 6.1) Live logs

Find the running container:

```bash
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
```

Follow logs:

```bash
# replace <container_name>
docker logs -f <container_name>
```

### 6.2) GPU usage

On host:

```bash
watch -n 1 nvidia-smi
```

### 6.3) Telegram monitor (if you enabled it via systemd)

The Telegram monitor we’ve been using runs from systemd and sends periodic status messages.

- It reads **live `docker logs`** from the training container.
- For reliability (and to avoid false positives), you should run training with a fixed container name:
  - `--name ruth-training-run`

Optional environment overrides (if you want to force behavior):

- `RUTH_CONTAINER_NAME`: force a specific container name (default: `ruth-training-run`)
- `RUTH_TOTAL_STEPS`: used only for the progress bar and “complete” detection (dataset-dependent)
- `RUTH_CKPT_DIR`: checkpoint directory shown in messages (default `/ckpts`)

Useful commands:

```bash
# check timer
sudo systemctl status ruth-monitor.timer
sudo systemctl list-timers | grep ruth-monitor

# force an immediate update (sends a message now)
sudo systemctl start ruth-monitor.service
```

## 7) What you can change (safe knobs)

### 7.1) Persisted paths (Docker volumes)

- Dataset path (inside container): `/data/...`
- HF caches (inside container): `/root/.cache/huggingface/...`
- Logs/checkpoints (inside container): `/ckpts/...`

### 7.2) The main config to edit

- [verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml](verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml)

Common fields:

- `data.train_files`, `data.val_files`
- `data.max_prompt_length`, `data.max_response_length`
- `actor_rollout_ref.model.path` (base model)
- `actor_rollout_ref.rollout.*` (vLLM memory/concurrency knobs)
- `trainer.total_epochs`
- `trainer.default_local_dir` (run output directory under `/ckpts`)
- `trainer.save_freq` (checkpoint frequency)

## 8) Auto-restart / auto-resume after power outage (recommended)

If you want training to come back automatically after a reboot/power fluctuation, you need two things:

1) **Periodic checkpoints enabled** (so resuming is possible). In the preset we enabled:
   - `trainer.save_freq: 50`
   - `trainer.resume_mode: auto`

2) A **persistent container** that Docker can restart (do **not** use `--rm`) plus a restart policy.

Example: start training as a persistent container that will restart only if it crashes (non-zero exit):

```bash
# If a previous container exists with the same name, remove it first
docker rm -f ruth-training-run 2>/dev/null || true

# Start a persistent training container
docker run -d --name ruth-training-run \
  --restart on-failure:20 \
  --label simplerl.role=training \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/simpleRL-reason -w /workspace/simpleRL-reason \
  -v simplerl_data:/data \
  -v simplerl_hf_cache:/root/.cache/huggingface \
  -v simplerl_ckpts:/ckpts \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  simple-rl:ngc-vllm bash -lc '
set -e
python -m pip install -e . --no-deps
python -m pip install word2number "antlr4-python3-runtime==4.9.3" "math-verify==0.6.0"
export PYTHONPATH=/workspace/simpleRL-reason:$PYTHONPATH
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  --config-name simplelr_grpo_qwen05_single_gpu \
  2>&1 | tee /ckpts/simplelr_grpo_qwen05_ctx1024/run.log
'
```

Notes:

- After a hard power loss, Docker will restart containers that have a restart policy.
- Because `resume_mode=auto`, the trainer will look inside `trainer.default_local_dir` for the latest `global_step_*` checkpoint and continue.
- This works only after the first checkpoint has been saved (e.g., after step 50 with `save_freq=50`).

## 9) Troubleshooting quick notes

- If `docker run` prints “NVIDIA Driver was not detected”, you didn’t start the container with GPU support (missing `--gpus all` or NVIDIA Container Toolkit isn’t configured).
- If reward deps break Hydra/OmegaConf, keep `antlr4-python3-runtime==4.9.3` pinned (as in the commands above).
