# Training Operations Commands (Docker + Adaptive KL)

This is a practical command sheet for daily use.

## 0) Go to repo

```bash
cd /home/ai-server-02/R_projects/final_thesis/simpleRL-reason
```

## 1) Start training (recommended)

Uses the resilient script with locked settings:
- total_epochs=3
- max_prompt_length=1024
- max_response_length=1024
- max_num_batched_tokens=4096
- adaptive KL enabled
- fresh timestamped checkpoint directory by default
- no automatic resume unless you explicitly set `RUTH_RESUME_MODE=auto`

```bash
bash scripts/run_metakl_training_detached.sh
```

## 2) Stop training

```bash
docker stop ruth-training-run
```

Force stop if needed:

```bash
docker rm -f ruth-training-run
```

## 3) Check if training is running

```bash
docker inspect ruth-training-run --format 'Status={{.State.Status}} Running={{.State.Running}} ExitCode={{.State.ExitCode}} RestartCount={{.RestartCount}} StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}}'
```

Quick running-list check:

```bash
docker ps --format '{{.Names}}\t{{.Status}}' | grep '^ruth-training-run\b' || true
```

## 4) Watch logs

Live logs:

```bash
docker logs -f ruth-training-run
```

Latest 200 lines:

```bash
docker logs --tail 200 ruth-training-run
```

Find step/KL lines:

```bash
docker logs ruth-training-run 2>&1 | grep -nE 'step:[0-9]+|actor/kl_coef_dynamic|actor/kl_loss|critic/score/mean' | tail -n 40
```

## 5) Check checkpoints (power-failure recovery)

Current adaptive checkpoint directory base:

```bash
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm bash -lc 'ls -lah /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive'
```

Interpretation:
- Each launch gets its own run directory under the base path unless you override `RUTH_RUN_DIR`.
- If `latest_checkpointed_iteration.txt` exists and has a number, that run directory tracks its last saved step.
- global_step_* folders are saved checkpoints.

## 6) Resume behavior

The launcher already sets:
- trainer.resume_mode=never by default
- trainer.default_local_dir=/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/<timestamp>

So restarting the container with the same launcher starts a new run directory unless you explicitly override the run dir and resume mode.

## 7) Send Telegram report

### Option A: Repo script (requires env vars in your shell)

```bash
export TELEGRAM_BOT_TOKEN='<your_bot_token>'
export TELEGRAM_CHAT_ID='<your_chat_id>'
bash scripts/send_metakl_telegram_report.sh
```

### Option B: Existing monitor script (already configured on host)

```bash
/home/ai-server-02/monitor/training_monitor.sh
```

## 8) Important health checks

GPU check:

```bash
nvidia-smi
```

Container restart policy check:

```bash
docker inspect ruth-training-run --format 'RestartPolicy={{.HostConfig.RestartPolicy.Name}} MaximumRetryCount={{.HostConfig.RestartPolicy.MaximumRetryCount}}'
```

## 9) One-command workflow (daily)

1. Start:

```bash
bash scripts/run_metakl_training_detached.sh
```

2. Check status:

```bash
docker inspect ruth-training-run --format 'Status={{.State.Status}} Running={{.State.Running}} RestartCount={{.RestartCount}} StartedAt={{.State.StartedAt}}'
```

3. Follow logs:

```bash
docker logs -f ruth-training-run
```

4. Send Telegram update:

```bash
/home/ai-server-02/monitor/training_monitor.sh
```

5. Stop when needed:

```bash
docker stop ruth-training-run
```

## 11) Resume from an exact checkpoint after power loss

If you want to continue from a specific saved checkpoint, set `RUTH_RESUME_CHECKPOINT` to the exact `global_step_*` path. The launcher will reuse the parent run directory automatically unless you override `RUTH_RUN_DIR`.

Example:

```bash
RUTH_RESUME_CHECKPOINT=/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/20260420_110659/global_step_1550 \
RUTH_RESTART_POLICY=no \
bash scripts/run_metakl_training_detached.sh
```

This is the safest option when recovering from a power failure because it resumes from one exact checkpoint instead of auto-scanning the directory.

## 10) If training does not start

- Check logs first:

```bash
docker logs --tail 300 ruth-training-run
```

- Relaunch cleanly:

```bash
docker rm -f ruth-training-run >/dev/null 2>&1 || true
bash scripts/run_metakl_training_detached.sh
```

- Confirm command inside container:

```bash
docker inspect ruth-training-run --format '{{json .Config.Cmd}}'
```
