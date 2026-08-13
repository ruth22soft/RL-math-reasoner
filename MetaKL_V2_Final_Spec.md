# MetaKLController — V2 Final Implementation Spec

**Author:** Ruth Tamiru  
**Reviewed by:** Claude critique (v1) + repo agent (v2 corrections)  
**Repository:** ruth22soft/RL-math-reasoner, branch: v1  
**Path:** /home/ai-server-02/R_projects/final_thesis/simpleRL-reason  
**Date:** 2026-04-18  
**Status:** Final spec — safe to implement

---

## Meta: what this document resolves

Three inputs fed into this V2 spec:

| Source | Contribution |
|---|---|
| Original proposal (metaKL_implementation_proposal.md) | Architecture, file plan, ablation matrix — all good |
| Claude critique (v1) | Identified 8 structural/parameter errors — directionally correct |
| Repo agent (v2 corrections) | Caught 4 errors IN the Claude critique itself |

The table below shows where the three sources agree or conflict, and what V2 adopts:

| Issue | Original | Claude critique | Repo agent | V2 decision |
|---|---|---|---|---|
| Baseline kl_loss_coef | 0.001 | "must be 0.0001" | 0.001 (confirmed from repo files) | **0.001** — repo agent is correct |
| target_kl_loss | 0.08 | 0.003 (hardcoded) | Derive from baseline run stats | **Derive from baseline** — see Section 3 |
| EMA alpha consistency | 0.95/0.90 mismatch | Fix to 0.90 both | Agrees | **0.90 both** |
| update_interval consistency | 5/1 mismatch | Fix to 1 both | Agrees | **1 both** |
| LSTM hidden state reset | Not stated | Reset per epoch | Agrees | **Reset per epoch** |
| Gradient flow | Not stated | Explicit statement needed | Agrees, but flags detach issue | **See corrected code below** |
| Controller placement | ray_trainer.py | Move to dp_actor.py | Partial — state availability issue | **Hybrid: see Section 4** |
| LSTM detach logic | N/A | Detach (h,c) each step | This breaks TBPTT — correct only partially | **See corrected LSTM below** |

---

## Section 1 — Confirmed baseline (from actual repo files)

From `train_grpo_math_tune_ray.sh` and `simplelr_grpo_qwen05_single_gpu.yaml`:

```yaml
actor_rollout_ref.actor.use_kl_loss: true
actor_rollout_ref.actor.kl_loss_coef: 0.001      # confirmed from repo — NOT 0.0001
actor_rollout_ref.actor.kl_loss_type: low_var_kl
algorithm.adv_estimator: grpo
```

> **Note for thesis:** If your baseline report PDF stated 0.0001, double-check your actual
> log files. The repo agent read the live files and found 0.001. Use whichever matches
> your actual logged `actor/kl_coef` value from the completed baseline run. The key
> principle: `init_beta` in Phase A and Phase B **must equal the exact value from your
> baseline run** — not a value from a different config file or a previous experiment.

---

## Section 2 — The core error that remains from both the original and the v1 critique

The original proposal set `target_kl_loss: 0.08`.  
The v1 Claude critique changed it to `0.003` — also hardcoded.  
The repo agent correctly identified that both are wrong.

**Why hardcoding target_kl_loss is dangerous:**  
The rule-based controller reacts when `kl_loss > target`. If you hardcode target without
looking at your actual baseline KL loss distribution, the controller will either:
- Never fire (target too high — original's 0.08 problem)
- Fire too aggressively (target too low — v1 critique's risk with 0.003)

**The correct approach: derive target from your baseline run**

After your 3-epoch baseline completes, run:

```bash
# Extract actor/kl_loss values from your training log
grep "actor/kl_loss" auto_train.log | awk '{print $NF}' > kl_loss_values.txt

# Get the statistics
python3 - << 'PY'
import numpy as np
vals = [float(x) for x in open('kl_loss_values.txt') if x.strip()]
print(f"Mean:   {np.mean(vals):.6f}")
print(f"Median: {np.median(vals):.6f}")
print(f"P70:    {np.percentile(vals, 70):.6f}")
print(f"P90:    {np.percentile(vals, 90):.6f}")
print(f"Max:    {np.max(vals):.6f}")
PY
```

Then set:
```yaml
actor_adaptive_kl.target_kl_loss: <P70 value from above>
# This means: "if KL rises above the 70th percentile of baseline,
# increase beta — we are in abnormally high KL territory."
```

This makes target_kl_loss principled and data-driven rather than guessed.

---

## Section 3 — Corrected Phase A and Phase B parameters

### Phase A (rule-based adaptive) — corrected

```yaml
actor_adaptive_kl.enable: true
actor_adaptive_kl.mode: rule
actor_adaptive_kl.init_beta: 0.001          # matches confirmed baseline
actor_adaptive_kl.min_beta: 0.0001          # floor = baseline value
actor_adaptive_kl.max_beta: 0.005           # 5× baseline
actor_adaptive_kl.warmup_steps: 30          # was 150 — too long
actor_adaptive_kl.update_interval: 1        # was 5 — mismatch with Phase B
actor_adaptive_kl.ema_alpha: 0.90           # was 0.95 — mismatch with Phase B
actor_adaptive_kl.target_kl_loss: DERIVE    # from baseline P70 — see Section 2
actor_adaptive_kl.up_gain: 1.08
actor_adaptive_kl.down_gain: 0.94
actor_adaptive_kl.reward_std_weight: 0.15
actor_adaptive_kl.grad_norm_weight: 0.10
```

### Phase B (LSTM MetaKLController) — corrected

```yaml
actor_adaptive_kl.enable: true
actor_adaptive_kl.mode: lstm
actor_adaptive_kl.init_beta: 0.001          # matches confirmed baseline
actor_adaptive_kl.min_beta: 0.0001
actor_adaptive_kl.max_beta: 0.005
actor_adaptive_kl.warmup_steps: 30
actor_adaptive_kl.update_interval: 1        # consistent with Phase A
actor_adaptive_kl.ema_alpha: 0.90           # consistent with Phase A
actor_adaptive_kl.state_features:
  - kl_loss
  - reward_mean
  - reward_std
  - grad_norm
actor_adaptive_kl.lstm_hidden_dim: 32
actor_adaptive_kl.lstm_num_layers: 1
actor_adaptive_kl.beta_output_activation: sigmoid_scaled
actor_adaptive_kl.lstm_reset_on_epoch: true
actor_adaptive_kl.lstm_gradient_flow: joint  # jointly trained with actor
```

---

## Section 4 — Corrected architecture: state plumbing

The repo agent correctly flagged that `reward_mean` and `reward_std` are not naturally
available inside `dp_actor.py` at actor update time — they live in the rollout batch which
is assembled in `ray_trainer.py`.

**The correct hybrid pattern:**

```
ray_trainer.py              assembles state_dict from rollout batch
        |
        | passes state_dict via meta_info (single dict, no tensors)
        v
dp_actor.py                 reads state_dict, calls controller, gets beta_t
        |                   applies: loss = pg_loss + beta_t * kl_loss
        v
        gradients flow back through beta_t into LSTM weights
```

**What goes in meta_info (assembled in ray_trainer.py):**

```python
# In ray_trainer.py, before calling actor_worker.update_policy():
state_dict = {
    'kl_loss': float(metrics.get('actor/kl_loss', 0.0)),
    'reward_mean': float(batch['token_level_scores'].mean()),
    'reward_std': float(batch['token_level_scores'].std()),
    'grad_norm': float(metrics.get('actor/grad_norm', 0.0)),
}
# Pass into actor update:
actor_worker.update_policy(data=batch, meta_info={'kl_state': state_dict})
```

**What dp_actor.py reads:**

```python
# In dp_actor.py update_policy():
state = meta_info.get('kl_state', None)
if state is not None and self.kl_controller is not None:
    beta_t = self.kl_controller(state)
    loss = pg_loss + beta_t * kl_loss
else:
    loss = pg_loss + self.config.actor.kl_loss_coef * kl_loss
```

This gives you reward signals (from trainer) and KL/gradient signals (computed in actor)
without full controller duplication in either place.

---

## Section 5 — Corrected LSTM implementation

The v1 critique had a subtle error: detaching `(h, c)` completely from the computational
graph every step breaks temporal backpropagation. However, the repo agent is also partially
wrong — for this use case (joint training with the actor), full TBPTT is not needed and
creates gradient instability. The correct approach is to detach only the hidden state
(stopping gradients through time) but allow gradients to flow from `beta_t` back to the
LSTM weights at the current step.

```python
import torch
import torch.nn as nn

class MetaKLController(nn.Module):
    """
    LSTM-based adaptive KL coefficient controller.

    Gradient flow: gradients flow from (beta_t * kl_loss) into LSTM
    weights at each step via the output head. Hidden state is detached
    to prevent gradient explosion through time (truncated BPTT).

    Hidden state reset: call reset_hidden() at the START of each epoch.
    Do NOT reset every step — this destroys temporal memory.
    """

    def __init__(self, cfg):
        super().__init__()
        self.input_size = 4
        self.hidden_size = cfg.lstm_hidden_dim
        self.min_beta = cfg.min_beta
        self.max_beta = cfg.max_beta
        self.warmup = cfg.warmup_steps
        self.alpha = cfg.ema_alpha
        self._step = 0
        self._beta_ema = cfg.init_beta

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.head = nn.Linear(self.hidden_size, 1)

        # Running normalisation buffers
        self.register_buffer('_run_mean', torch.zeros(self.input_size))
        self.register_buffer('_run_var', torch.ones(self.input_size))
        self.register_buffer('_run_n', torch.tensor(0.0))

        # Hidden state — None until first forward call
        self._h: torch.Tensor | None = None
        self._c: torch.Tensor | None = None

    def reset_hidden(self):
        """Call at start of each epoch. Resets memory and step counter."""
        self._h = None
        self._c = None
        self._step = 0

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Welford online normalisation. Clip to [-5, 5]."""
        self._run_n += 1
        delta = x.detach() - self._run_mean
        self._run_mean += delta / self._run_n
        delta2 = x.detach() - self._run_mean
        self._run_var += delta * delta2
        std = (self._run_var / self._run_n + 1e-6).sqrt()
        return torch.clamp((x - self._run_mean.detach()) / std.detach(), -5.0, 5.0)

    def forward(self, state: dict) -> torch.Tensor:
        """
        Returns beta_t as a scalar tensor with gradient.
        Caller: loss = pg_loss + beta_t * kl_loss
        This allows gradients to flow into LSTM head weights.
        """
        self._step += 1

        x = torch.tensor([
            state.get('kl_loss', 0.0),
            state.get('reward_mean', 0.0),
            state.get('reward_std', 0.0),
            state.get('grad_norm', 0.0),
        ], dtype=torch.float32, requires_grad=False)

        x_norm = self._normalise(x)

        # Initialise hidden state if needed
        if self._h is None:
            self._h = torch.zeros(1, self.hidden_size)
            self._c = torch.zeros(1, self.hidden_size)

        # Forward through LSTM — hidden state detached (truncated BPTT)
        # This stops gradients flowing through time but allows them
        # to flow from beta_t into lstm.weight_ih and weight_hh at this step
        h_new, c_new = self.lstm(x_norm.unsqueeze(0), (self._h, self._c))

        # Update hidden for next step (detached — no gradient through time)
        self._h = h_new.detach()
        self._c = c_new.detach()

        # Output head — gradients flow here from beta_t * kl_loss
        raw = self.head(h_new)   # h_new is NOT detached — gradient flows
        beta_t = self.min_beta + (self.max_beta - self.min_beta) * torch.sigmoid(raw)

        # Warmup: return constant init value, no gradient
        if self._step <= self.warmup:
            return torch.tensor(self._beta_ema, requires_grad=False)

        # EMA smoothing on the scalar for logging
        self._beta_ema = self.alpha * self._beta_ema + (1 - self.alpha) * beta_t.item()

        return beta_t.squeeze()

    def get_log_dict(self) -> dict:
        """Returns metrics for logging at each step."""
        return {
            'actor_adaptive_kl/beta_smoothed': self._beta_ema,
            'actor_adaptive_kl/step': self._step,
        }
```

---

## Section 6 — Rule-based controller (corrected)

```python
class RuleBasedActorKLController:
    """
    Phase A: rule-based adaptive beta_t.
    target_kl_loss should be set from baseline P70 value.
    """
    def __init__(self, cfg):
        self._beta = cfg.init_beta
        self.min_beta = cfg.min_beta
        self.max_beta = cfg.max_beta
        self.target = cfg.target_kl_loss    # set from baseline P70
        self.up = cfg.up_gain               # 1.08
        self.down = cfg.down_gain           # 0.94
        self.alpha = cfg.ema_alpha          # 0.90
        self.warmup = cfg.warmup_steps      # 30
        self.interval = cfg.update_interval # 1
        self._step = 0
        self._beta_ema = cfg.init_beta

    def __call__(self, state: dict) -> float:
        self._step += 1
        if self._step <= self.warmup:
            return self._beta_ema
        if self._step % self.interval != 0:
            return self._beta_ema

        kl = state.get('kl_loss', 0.0)
        raw = self._beta * (self.up if kl > self.target else self.down)
        raw = max(self.min_beta, min(self.max_beta, raw))
        self._beta_ema = self.alpha * self._beta_ema + (1 - self.alpha) * raw
        self._beta = self._beta_ema
        return self._beta_ema

    def reset(self):
        self._step = 0
        self._beta_ema = self._beta

    def get_log_dict(self) -> dict:
        return {
            'actor_adaptive_kl/beta_smoothed': self._beta_ema,
            'actor_adaptive_kl/step': self._step,
        }
```

---

## Section 7 — File change summary

| File | Change type | What changes |
|---|---|---|
| `verl/trainer/ppo/meta_kl_controller.py` | **NEW** | LSTM controller class (Section 5) |
| `verl/trainer/ppo/actor_kl_controller.py` | **NEW** | Rule-based controller class (Section 6) |
| `verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml` | **EDIT** | Add `actor_adaptive_kl` config block |
| `verl/trainer/ppo/ray_trainer.py` | **EDIT** | Assemble state_dict, pass via meta_info, reset on epoch |
| `verl/workers/actor/dp_actor.py` | **EDIT** | Read state_dict, call controller, use beta_t in loss |

---

## Section 8 — Implementation order

```
Step 1  Run 3-epoch baseline with fixed beta — extract kl_loss statistics
Step 2  Compute target_kl_loss from P70 of baseline actor/kl_loss log
Step 3  Implement RuleBasedActorKLController — run 5-step smoke test
Step 4  Run Phase A full 3-epoch experiment — verify beta_t varies
Step 5  Implement MetaKLController — verify LSTM weights change after step 1
Step 6  Run Phase B full 3-epoch experiment — compare all four ablation conditions
Step 7  Generate beta trajectory plot — your key thesis figure
```

---

## Section 9 — Ablation matrix (unchanged from original, confirmed correct)

| Condition | β mechanism | Purpose |
|---|---|---|
| A | Fixed β = 0.001 (baseline) | Lower bound — your comparison point |
| B | Fixed β grid {0.0005, 0.001, 0.002} | Upper bound for fixed approach |
| C | Rule-based adaptive (Phase A) | Validates interface, ablation against LSTM |
| D | LSTM MetaKLController (Phase B) | **Your thesis claim** |

Thesis claim: D significantly outperforms A. If D also outperforms C, the LSTM
adds value beyond simple rule adaptation — this strengthens your contribution.

---

## Section 10 — Success criteria

| Metric | Minimum | Strong |
|---|---|---|
| MATH500 accuracy vs fixed β | +1.0 pp | +2.0 pp |
| β trajectory | Non-trivial variation | Correlates with reward phases |
| KL stability | No worse than baseline | Noticeably lower variance |
| Training stability | No collapse | Stable entropy throughout |

---

*V2 Final Spec — 2026-04-18*  
*Supersedes: MetaKL_Critique_And_Revised_Proposal.md (v1)*  
*For: Ruth Tamiru MSc thesis — Adaptive KL Regularization via Meta-Learning*
