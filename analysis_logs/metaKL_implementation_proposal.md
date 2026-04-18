# MetaKLController Implementation Proposal

## 1) Repository and Environment Context

- Repository name: RL-math-reasoner
- Owner: ruth22soft
- Active branch: v1
- Default branch: v1
- Workspace path: /home/ai-server-02/R_projects/final_thesis/simpleRL-reason
- Date: 2026-04-18
- Proposal target: integrate adaptive KL coefficient control for GRPO training in this codebase.

## 2) Objective

Implement an adaptive KL coefficient mechanism for actor-side GRPO KL regularization, starting with a stable rule-based controller and then enabling your LSTM MetaKLController.

Primary goal:
- Improve training stability and final validation correctness by adapting KL pressure over training phases.

Secondary goals:
- Preserve reproducibility.
- Keep baseline behavior available behind config switches.
- Add clear telemetry for diagnosis and ablation.

## 3) Current Code Behavior (Verified)

Current active path for your run:
- Actor-side KL loss is enabled with fixed coefficient.
- Actor loss term: policy_loss = policy_loss + kl_loss * kl_loss_coef
- Logged metrics include actor/kl_loss and actor/kl_coef.

Important distinction in this repo:
- There is also a trainer-side KL controller path (fixed/adaptive) used in reward shaping.
- That trainer-side path is bypassed when actor use_kl_loss=True.
- Therefore, setting algorithm.kl_ctrl to adaptive alone does not make actor/kl_coef adaptive in your current setup.

## 4) Proposed Implementation Strategy

### Phase A (Recommended first): Adaptive Rule-Based Actor KL

Purpose:
- Validate adaptive KL control interface with low engineering risk.
- Build stable control signals and logging first.

What changes:
1. Compute dynamic beta_t each training step in trainer loop.
2. Pass beta_t to actor update (runtime override of static actor kl_loss_coef).
3. Keep strict bounds and smoothing.
4. Log dynamic beta and penalty diagnostics.

### Phase B: LSTM MetaKLController (Your Thesis Method)

Purpose:
- Replace rule-based beta_t predictor with learned recurrent controller f_phi.
- Use state vector from live training signals.

What changes:
1. Add LSTM controller module.
2. Add state assembly from current metrics.
3. Predict beta_t and feed into actor update.
4. Keep same safety guards (bounds, warmup, smoothing).

## 5) File-Level Plan

### New files

1. verl/trainer/ppo/actor_kl_controller.py
- Rule-based adaptive actor KL controller class.
- APIs:
  - reset()
  - update(state_dict) -> beta_t
  - get_state()

2. verl/trainer/ppo/meta_kl_controller.py
- LSTM-based controller for Phase B.
- APIs:
  - initialize_hidden(batch_size=1)
  - forward(state_t, hidden)
  - predict_beta(state_dict) -> beta_t

### Modified files

1. verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml
- Add actor adaptive KL config block.
- Keep backward-compatible defaults (disabled by default).

2. verl/trainer/ppo/ray_trainer.py
- Instantiate actor KL controller based on config.
- Build step-level state signals.
- Compute beta_t before actor update.
- Inject beta_t into batch/meta_info for actor worker.
- Log controller metrics.

3. verl/workers/actor/dp_actor.py
- Read runtime beta_t override from meta info.
- Use override if present, else fallback to static config value.
- Log actor/kl_coef_dynamic and actor/kl_penalty_term.

## 6) Parameter Plan

### 6.1 Baseline parameters (current)

- actor_rollout_ref.actor.use_kl_loss: True
- actor_rollout_ref.actor.kl_loss_coef: 0.001
- actor_rollout_ref.actor.kl_loss_type: low_var_kl
- algorithm.adv_estimator: grpo

### 6.2 Phase A parameters (rule-based adaptive)

Proposed defaults:
- actor_adaptive_kl.enable: true
- actor_adaptive_kl.mode: rule
- actor_adaptive_kl.init_beta: 0.001
- actor_adaptive_kl.min_beta: 0.00001
- actor_adaptive_kl.max_beta: 0.005
- actor_adaptive_kl.warmup_steps: 150
- actor_adaptive_kl.update_interval: 5
- actor_adaptive_kl.ema_alpha: 0.95
- actor_adaptive_kl.target_kl_loss: 0.08
- actor_adaptive_kl.up_gain: 1.08
- actor_adaptive_kl.down_gain: 0.94
- actor_adaptive_kl.reward_std_weight: 0.15
- actor_adaptive_kl.grad_norm_weight: 0.10

Behavior:
- If observed kl_loss > target band, beta_t increases by up_gain.
- If observed kl_loss < target band, beta_t decreases by down_gain.
- Final beta_t is smoothed by EMA and clamped to [min_beta, max_beta].

### 6.3 Phase B parameters (LSTM MetaKLController)

Proposed initial defaults:
- actor_adaptive_kl.enable: true
- actor_adaptive_kl.mode: lstm
- actor_adaptive_kl.init_beta: 0.001
- actor_adaptive_kl.min_beta: 0.00001
- actor_adaptive_kl.max_beta: 0.005
- actor_adaptive_kl.warmup_steps: 150
- actor_adaptive_kl.update_interval: 1
- actor_adaptive_kl.ema_alpha: 0.90
- actor_adaptive_kl.state_features:
  - kl_loss
  - reward_mean
  - reward_std
  - grad_norm
- actor_adaptive_kl.lstm_hidden_dim: 32
- actor_adaptive_kl.lstm_num_layers: 1
- actor_adaptive_kl.mlp_hidden_dim: 32
- actor_adaptive_kl.beta_output_activation: sigmoid_scaled

Beta transform:
- raw = controller(state_t)
- beta_t = min_beta + (max_beta - min_beta) * sigmoid(raw)
- optional EMA smoothing afterwards.

## 7) Controller State Definition

State vector at step t:
- s_t = [kl_loss_t, reward_mean_t, reward_std_t, grad_norm_t]

State normalization (recommended):
- running mean/std normalization with epsilon 1e-6.
- clip normalized features to [-5, 5].

## 8) Logging and Diagnostics

New metrics to log per step:
- actor/kl_coef_dynamic
- actor/kl_loss
- actor/kl_penalty_term
- actor_adaptive_kl/beta_raw
- actor_adaptive_kl/beta_smoothed
- actor_adaptive_kl/controller_mode
- actor_adaptive_kl/state_kl_loss
- actor_adaptive_kl/state_reward_mean
- actor_adaptive_kl/state_reward_std
- actor_adaptive_kl/state_grad_norm

Stability alerts:
- beta at min/max for too many consecutive steps.
- kl_loss spikes above configured threshold.
- entropy collapse sustained for N steps.

## 9) Experimental Plan

### Stage 0: Baseline replication
- Run current fixed coefficient setup.
- Record final val/test_correctness and learning curves.

### Stage 1: Rule-based adaptive (Phase A)
- Keep all training params same except adaptive KL settings.
- Compare to baseline using same seeds and dataset splits.

### Stage 2: LSTM MetaKL (Phase B)
- Replace rule controller with LSTM predictor.
- Keep safety bounds identical.

### Ablation matrix

1. Fixed beta = 0.001
2. Fixed beta grid = [0.0005, 0.001, 0.002]
3. Rule-based adaptive beta
4. LSTM adaptive beta

Metrics to compare:
- val/test_correctness
- val/test_score
- sample efficiency (score vs steps)
- KL stability (variance of actor/kl_loss)
- training stability (entropy, grad_norm behavior)

## 10) Success Criteria

Primary success:
- LSTM adaptive beta achieves higher final val/test_correctness than fixed-beta baseline by a meaningful margin (target: +1.5 to +3.0 points absolute).

Secondary success:
- Lower KL volatility than baseline.
- No increase in collapse/instability events.
- Similar or better wall-clock convergence behavior.

## 11) Risks and Mitigations

Risk 1: Controller oscillation
- Mitigation: EMA, update_interval, hard clipping, warmup.

Risk 2: Over-regularization
- Mitigation: conservative max_beta and lower initial beta.

Risk 3: Under-regularization drift
- Mitigation: floor min_beta and target-band control.

Risk 4: Attribution confusion with other changes
- Mitigation: strict A/B protocol and seed control.

## 12) Backward Compatibility

- If actor_adaptive_kl.enable = false, behavior remains exactly current fixed-beta implementation.
- Existing scripts/configs remain valid.

## 13) Proposed Immediate Next Action

Implement Phase A first (rule-based adaptive actor KL) with the exact parameters listed in Section 6.2, run one controlled A/B experiment, and then promote to Phase B (LSTM MetaKLController) after stability is confirmed.

## 14) Notes for Thesis Alignment

Your thesis contribution is preserved in Phase B:
- Learned recurrent controller for beta_t.
- Live signal-driven adaptation.
- GRPO-specific integration for LLM math reasoning.

Phase A is an engineering bridge that de-risks the integration and gives a strong baseline for comparison in your thesis experiments.
