# Adaptive KL Approach and Logic (Implementation Note)

Prepared for: RL-math-reasoner (branch v1)
Date: 2026-04-18

## 1) Objective

The objective of adaptive KL is to replace a fixed KL penalty coefficient with a dynamic coefficient that responds to training state.

Why this is needed:
- A fixed KL coefficient can be too weak in unstable phases and too strong in stable phases.
- Policy drift pressure is not constant across steps.
- Reward quality and gradient scale change over training.

Target behavior:
- Increase KL pressure when drift risk rises.
- Decrease KL pressure when learning can safely move faster.
- Keep policy updates bounded while preserving progress.

## 2) Core Loss Logic

Baseline actor update uses PPO terms plus optional KL penalty.

With adaptive KL, the actor-side loss is:

L_final = L_ppo + beta_t * L_kl

Where:
- L_ppo includes policy gradient term and entropy regularization.
- L_kl is computed between actor and reference log-probs.
- beta_t is dynamic and updated step-by-step by the controller.

This is implemented in actor update flow so control happens at the exact optimization point.

## 3) Why Actor-Side Control

Adaptive control is applied in the actor update path (not only trainer-side reward shaping) because:
- KL penalty is part of policy optimization itself.
- We can use immediate local signals (current KL, grad norm, batch reward stats).
- We can log and debug the control effect directly from step metrics.

Practically, this gives transparent step-level metrics:
- actor/kl_coef_dynamic
- actor/kl_penalty_term
- actor/kl_loss

## 4) State Signals Used by Controller

Controller state is built from:
- kl_loss: direct measure of actor-reference divergence
- reward_mean: batch reward center
- reward_std: batch reward dispersion/noise proxy
- lagged_grad_norm: recent optimizer pressure proxy

Design rationale:
- kl_loss captures what we are controlling.
- reward_mean/reward_std indicate reward quality and uncertainty.
- lagged_grad_norm captures update magnitude/stability pressure.

## 5) Dataflow of Signals

Signal path during training:
1. Trainer computes reward statistics on the current batch.
2. Trainer injects reward_signals into batch metadata.
3. Actor reads reward_signals and local KL/grad context.
4. Controller produces beta_t.
5. Actor applies beta_t * L_kl in policy loss.
6. Reduced step metrics are logged.

This keeps the interface lightweight and avoids changing core tensor schemas.

## 6) Controller Modes

Two controller modes are supported:

### 6.1 Rule-Based Controller
- Deterministic, interpretable behavior.
- Useful as fallback and sanity baseline.
- Uses configured gains and thresholds to scale beta.

### 6.2 LSTM Meta-Controller
- Learns temporal control behavior from state sequence.
- Can react to longer-horizon patterns that simple thresholds miss.
- Uses bounded output transform to keep beta in safe range.

Mode is selected from config block actor_adaptive_kl.

## 7) Stability and Safety Guards

To prevent oscillation and runaway penalties, the implementation includes:
- Warmup steps: conservative behavior at startup.
- Min/max beta bounds: hard safety envelope.
- EMA-style smoothing: reduces noise sensitivity.
- Epoch reset support: optional state reset to avoid stale hidden-state carryover.

Operationally this gives smoother KL control and fewer abrupt shifts in update pressure.

## 8) Config Design and Operational Defaults

The operational pipeline currently uses:
- total_epochs = 3
- max_prompt_length = 1024
- max_response_length = 1024
- max_num_batched_tokens = 4096
- adaptive KL enabled, mode=lstm
- checkpoint save_freq = 50
- resume_mode = auto
- dedicated adaptive checkpoint directory

This combination was chosen for reproducibility and restart safety.

## 9) Logging Interpretation Guide

How to read the main adaptive fields:

- actor/kl_coef_dynamic:
  dynamic controller output beta_t. If this moves over time, controller is active.

- actor/kl_loss:
  current divergence penalty term value. Near-zero values can occur early or on easier batches.

- actor/kl_penalty_term:
  beta_t * kl_loss contribution into actor loss.

- critic/score/mean and critic/rewards/mean:
  batch-level reward/accuracy proxy for training health.

Important note on occasional zero reward/accuracy:
- Individual samples can produce reward 0 frequently in strict math grading.
- This is expected and does not imply collapse.
- Health is determined by evolving batch aggregates and sustained step progression.

## 10) Failure Mode Addressed During Integration

Observed issue:
- Resume from legacy fixed-KL checkpoint tree produced optimizer parameter-group mismatch.

Root cause:
- Old optimizer state layout incompatible with updated actor parameter structure.

Resolution:
- Use dedicated adaptive checkpoint directory for adaptive runs.
- Keep resume_mode=auto within that dedicated run namespace.

Result:
- Restart behavior remains robust after outages without mixing incompatible optimizer states.

## 11) Practical Validation Signals (What Confirms It Works)

Adaptive mechanism is considered active when all are true:
- Step lines progress normally (step:N increases).
- actor/kl_coef_dynamic appears in logs.
- actor/kl_loss appears in logs.
- No recurring optimizer-state mismatch tracebacks in current adaptive checkpoint namespace.

## 12) Summary

The implemented adaptive KL mechanism is an actor-side dynamic regularization controller that:
- uses KL, reward, and gradient context,
- outputs a bounded per-step KL coefficient,
- integrates directly into policy loss,
- logs transparent control metrics,
- and operates with restart-safe checkpointing.

This design is intentionally practical: minimal invasive changes, clear observability, and stable operations for long Docker-based runs.
