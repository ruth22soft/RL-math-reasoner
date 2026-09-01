# CHAPTER 3 METHODOLOGY - QUICK REFERENCE GUIDE

**Full Report:** [CHAPTER3_METHODOLOGY_EXTRACTION.md](CHAPTER3_METHODOLOGY_EXTRACTION.md)

## One-Page Summary

### System Overview
```
Dataset (math problems) → Qwen2.5-0.5B-Instruct → GRPO Training
  ↓ Rollout: n=4 responses via vLLM
  ↓ Reward: Binary correctness (math_verify)
  ↓ KL: low_var_kl divergence from reference policy
  ↓ KL Controller: {LSTM|MLP|RBF|Rule|Zero/Fixed}
  ↓ Beta: Adaptive or fixed coefficient
  ↓ Loss: policy_loss - entropy + beta × kl_loss
  ↓ Optimizer: Adam (actor + controller joint)
```

### Key Numbers (for thesis tables)

| Metric | Value |
|--------|-------|
| Model | Qwen2.5-0.5B |
| Seed | 42 |
| Epochs | 3 |
| Batch size | 16 |
| Actor LR | 1e-6 |
| Controller LR | 5e-6 |
| PPO clip | 0.2 |
| Entropy coeff | 0.001 |
| Beta bounds | [0.0001, 0.01] |
| Rollout n | 4 (or 2 for zero-KL) |
| State dim | 4 |
| LSTM hidden | 32 |
| MLP hidden | 32 |
| RBF kernels | 16 |

### Controller Architectures

**Rule-Based:** Multiplicative heuristic (no learning)
```
if kl_loss > target: β ← β × 1.08
else: β ← β × 0.94
```

**MLP:** 4 → 32 (Tanh) → 1 → sigmoid → β

**LSTM:** LSTMCell(4, 32) → Linear(32, 1) → sigmoid → β
- Recurrent memory per epoch
- Reset every epoch
- Same learning rate as MLP

**RBF:** Gaussian basis expansion
```
φᵢ = exp(-γ × ||x - cᵢ||²)
β = sigmoid(Linear(φ₁...φ₁₆))
```

### State Vector Features

```python
state = {
    'kl_loss': scalar (masked mean of KL divergence)
    'reward_mean': scalar (batch reward mean)
    'reward_std': scalar (batch reward std)
    'lagged_grad_norm': scalar (actor grad norm from prev step)
}
```

All normalized via EMA normalizer (α=0.01) before entering controller.

### Five Experimental Conditions

1. **Zero-KL:** β = 0 (no KL penalty)
2. **Fixed:** β = 0.001 (constant)
3. **Rule-Based:** Heuristic target_kl=0.003
4. **MLP:** Learned stateless
5. **LSTM:** Learned recurrent

### Critical Implementation Details

✅ **Controllers are learned** (except rule-based): Gradients flow through β_t to controller parameters  
✅ **Joint training:** Actor + controller share optimizer  
✅ **Single seed:** All 5 controllers use seed=42  
✅ **GRPO advantages:** Group-relative outcome normalization  
✅ **Binary reward:** 1.0 if correct, 0.0 if incorrect  
✅ **LSTM reset:** Per epoch  
✅ **State detachment:** Input is detached, output (beta) is not  

### Dataset
- **Path:** /data/simplelr_qwen_level3to5/{train,test}.parquet
- **Type:** MATH-style symbolic problems
- **Verification:** math_verify library (symbolic comparison)

### Files You'll Need to Reference

**For architecture:**
- `verl/trainer/ppo/meta_kl_controller.py` (LSTM, MLP, RBF)
- `verl/trainer/ppo/actor_kl_controller.py` (rule-based)

**For training:**
- `verl/workers/actor/dp_actor.py` (policy update loop)
- `verl/trainer/ppo/core_algos.py` (loss computation)

**For configuration:**
- `verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml` (base config)

**For orchestration:**
- `run_single_seed_orchestrator_v1.sh` (benchmark campaign)

### Important Notes for Thesis Writing

1. **Verified facts:** All numerical values come from config files or code
2. **Single-seed design:** Not a limitation; intentional ablation design
3. **Rollout n=2 for zero-KL:** Pragmatic choice (baseline cost-saving)
4. **LSTM OOM interruption:** Documented; 0.334677 is step 1500 score
5. **RBF status:** Code exists but no successful run results preserved
6. **Do NOT cite:** The unverified 0.366935 score (it's not in the repository)

---

## For Sections to Write

- **Introduction:** "This study systematically compares five KL regularization strategies..."
- **Model:** "Qwen2.5-0.5B-Instruct, loaded from HuggingFace, bfloat16 precision, FSDP strategy..."
- **Training:** "GRPO with group-relative outcome advantages, 3 epochs, 16 batch size..."
- **Controllers:** Describe each in 1-2 paragraphs with architecture diagram
- **Evaluation:** "Mathematical correctness via symbolic verification (math_verify library)..."

Use the full [CHAPTER3_METHODOLOGY_EXTRACTION.md](CHAPTER3_METHODOLOGY_EXTRACTION.md) for detailed citations and equations.
