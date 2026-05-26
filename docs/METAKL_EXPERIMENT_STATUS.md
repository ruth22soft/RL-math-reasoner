# MetaKL Experiment Status

This document summarizes the current verified setup for the Qwen2.5-0.5B GRPO experiment in `simpleRL-reason` and the phase-gated workflow we are using.

## Verified Hardware

- GPU: NVIDIA RTX A4000
- VRAM: 16 GiB total
- Current GPU state during inspection: idle, effectively no training process running
- Host disk: 1.8 TiB volume, with about 412 GiB free after checkpoint cleanup

## Storage Volumes

- `simplerl_ckpts`: checkpoint volume mounted at `/ckpts`
- `simplerl_data`: dataset volume mounted at `/data`
- `simplerl_hf_cache`: Hugging Face cache volume

## Launcher Precedence

When the launcher passes a value on the command line, that value wins over the YAML default for the same setting.

The launcher currently overrides:

- `trainer.total_epochs`
- `trainer.resume_mode`
- `trainer.remove_previous_ckpt`
- `trainer.save_freq`
- `actor_rollout_ref.rollout.n`
- `actor_rollout_ref.rollout.gpu_memory_utilization`
- `actor_rollout_ref.rollout.max_num_seqs`

Important safety rule:

- `resume_mode` is forced to `never` by default in both launchers.
- `resume_mode=auto` is rejected for the campaign workflow to avoid cross-run resume between seeds.

## Current Launcher Defaults

### `scripts/run_metakl_training_detached.sh`

- `RUTH_MODE=fixed|rule|mlp|lstm`
- `RUTH_FIXED_BETA=0.001`
- `RUTH_TOTAL_EPOCHS=3` for real runs
- `RUTH_TOTAL_TRAINING_STEPS` may be set explicitly if you want a step-based cap
- `RUTH_ROLLOUT_N=4`
- `RUTH_ROLLOUT_GPU_MEMORY_UTIL=0.5`
- `RUTH_ROLLOUT_MAX_NUM_SEQS=32`
- `RUTH_REMOVE_PREVIOUS_CKPT=true`
- `RUTH_RESUME_MODE=never`

### `train_grpo_math_tune_ray.sh`

- `TOTAL_EPOCHS=3`
- `ROLLOUT_N=4`
- `ROLLOUT_GPU_MEMORY_UTIL=0.5`
- `ROLLOUT_MAX_NUM_SEQS=32`
- `REMOVE_PREVIOUS_CKPT=True`
- `RESUME_MODE=never`
- `FIXED_BETA=0.001`
- `TRAIN_MODE=fixed|rule|mlp|lstm`

## KL Modes

The launcher now exposes four selectable modes:

- `fixed`: adaptive KL disabled, constant beta used via `actor_rollout_ref.actor.kl_loss_coef`
- `rule`: rule-based adaptive KL controller
- `mlp`: learned MLP controller
- `lstm`: learned LSTM controller

The constant-beta baseline is explicit and set to:

- `RUTH_FIXED_BETA=0.001`

That value is passed directly as the actor KL coefficient in fixed mode.

## Preset YAML Defaults

File: `verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml`

Verified defaults in the preset:

- `actor_adaptive_kl.enable: False`
- `actor_adaptive_kl.mode: lstm`
- `actor_adaptive_kl.init_beta: 0.001`
- `actor_adaptive_kl.min_beta: 0.0001`
- `actor_adaptive_kl.max_beta: 0.01`
- `actor_adaptive_kl.warmup_steps: 10`
- `actor_adaptive_kl.update_interval: 1`
- `actor_adaptive_kl.ema_alpha: 0.75`
- `actor_adaptive_kl.target_kl_loss: 0.003`
- `actor_adaptive_kl.lstm_hidden_dim: 32`
- `actor_adaptive_kl.mlp_hidden_dim: 32`
- `trainer.logger: ['console']`
- `trainer.save_freq: 50`
- `trainer.resume_mode: never`
- `trainer.total_epochs: 3`
- `trainer.remove_previous_ckpt: True`
- `rollout.gpu_memory_utilization: 0.4`
- `rollout.max_num_batched_tokens: 4096`
- `rollout.micro_rollout_batch_size: 8`
- `rollout.max_num_seqs: 16`
- `rollout.enable_chunked_prefill: True`
- `rollout.n: 2`

## Checkpoint Policy

Current behavior:

- Only the latest `global_step_*` checkpoint is kept in the active run tree.
- Old `global_step_*` folders were deleted from the historical checkpoint set.
- The adaptive checkpoint tree is now much smaller and the host disk has more free space.

Recommended policy for new runs:

- keep the final checkpoint for evaluation
- keep at most one or two intermediate checkpoints if you need restart safety
- delete older checkpoints inside the active run directory only

## GPU Efficiency Notes

Current tuned values are conservative and should be adjusted empirically in Phase 2 smoke testing:

- `n=4`
- `gpu_memory_utilization=0.5`
- `max_num_seqs=32`
- gradient checkpointing is enabled in the preset

The exact best `n` still needs a short smoke test because response length and KV cache usage trade off against parallel samples per prompt.

## Phase Gates

- Phase 0: inspect and verify settings, hardware, and disk state
- Phase 1: define the four selectable modes and safe resume behavior
- Phase 2: short smoke test for GPU tuning and max stable `n`
- Phase 3: enable W&B logging and dashboard visibility
- Phase 4: checkpoint cleanup policy with resume verification
- Phase 5: one-command launcher for a single run
- Phase 6: multi-seed campaign orchestration
- Phase 7: results aggregation and thesis-ready plots/tables

## Next Action

Before starting Phase 2, we should confirm the exact smoke-test parameters, then run a short job and record memory, response length, and the largest stable `n`.