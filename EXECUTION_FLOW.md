# Complete Execution Flow - Step-by-Step

## Starting the Orchestrator

```bash
cd /home/ai-server-02/R_projects/final_thesis
./run_single_seed_orchestrator_v1.sh
```

---

## Timeline of Execution

### TIME: T+0:00 — Orchestrator Startup

```
═══════════════════════════════════════════════════════════════════════════════
[2026-05-28 14:30:00] ╔════════════════════════════════════════════════════════╗
[2026-05-28 14:30:00] ║    SINGLE-SEED 5-CONTROLLER RL BENCHMARK ORCHESTRATOR  ║
[2026-05-28 14:30:00] ║           Adaptive KL Regularization Study             ║
[2026-05-28 14:30:00] ╚════════════════════════════════════════════════════════╝
[2026-05-28 14:30:00]
[2026-05-28 14:30:00] Configuration:
[2026-05-28 14:30:00]   • Global seed: 42
[2026-05-28 14:30:00]   • Total epochs per controller: 3
[2026-05-28 14:30:00]   • Controllers: zero_kl rule_based mlp_based fixed lstm_based
[2026-05-28 14:30:00]   • Result directory: /home/ai-server-02/R_projects/final_thesis/results
[2026-05-28 14:30:00]
═══════════════════════════════════════════════════════════════════════════════
```

**What happens**:
1. Parse configuration (GLOBAL_SEED=42, controller list, epochs=3)
2. Create result directories
3. Log initialization info to `orchestrator_single_seed_20260528_143000.log`

---

### TIME: T+0:01 — Controller 1: Zero KL (β = 0)

```
═══════════════════════════════════════════════════════════════════════════════
[2026-05-28 14:30:01] ╭─────────────────────────────────────────────────────────╮
[2026-05-28 14:30:01] │ [1/5] Controller: zero_kl                               │
[2026-05-28 14:30:01] ╰─────────────────────────────────────────────────────────╯
[2026-05-28 14:30:01]
[2026-05-28 14:30:01] [CRITICAL CHECK] Killing any existing runs of zero_kl on seed 2 or 3...
[2026-05-28 14:30:02]
[2026-05-28 14:30:02]   Description: Zero KL (β = 0)
[2026-05-28 14:30:02]   Mode: fixed
[2026-05-28 14:30:02]   Seed: 42
[2026-05-28 14:30:02]   Epochs: 3
[2026-05-28 14:30:02]   Container: phase3-zero_kl-single-seed
[2026-05-28 14:30:02]   Checkpoint dir: /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42
[2026-05-28 14:30:02]
[2026-05-28 14:30:02]   Starting training...
```

**What happens**:
1. Kill any stray `phase3-zero_kl-s2` or `phase3-zero_kl-s3` containers
2. Set environment variables:
   - `RUTH_MODE="fixed"`
   - `RUTH_FIXED_BETA="0.0"`
   - `RUTH_GLOBAL_SEED="42"`
   - `RUTH_CONTAINER_NAME="phase3-zero_kl-single-seed"`
   - `RUTH_RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42"`
3. Call: `bash simpleRL-reason/scripts/run_metakl_training_single_seed.sh`

---

### TIME: T+0:02 — Training Script Execution

```
Inside run_metakl_training_single_seed.sh:

[INFO] ════════════════════════════════════════════════════════════
[INFO] Single-Seed Training Launch (GLOBAL_SEED=42)
[INFO] ════════════════════════════════════════════════════════════
[INFO] Container: phase3-zero_kl-single-seed
[INFO] Run dir: /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42
[INFO] Mode: fixed
[INFO] Epochs: 3
[INFO] Global seed (enforced): 42
[INFO] ════════════════════════════════════════════════════════════

[OK] Started phase3-zero_kl-single-seed
[OK] Follow logs: docker logs -f phase3-zero_kl-single-seed
```

**What happens**:
1. Validate RUTH_SEED == GLOBAL_SEED (if both set)
2. Start Docker container with:
   - GPU access: `--gpus all`
   - Memory: `--ulimit memlock=-1`
   - Volumes: mounted simplerl_data, simplerl_ckpts, simplerl_hf_cache
   - Environment: GLOBAL_SEED=42 passed
3. Inside container:
   - Install dependencies
   - Initialize PYTHONPATH
   - Create run directory `/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42`

---

### TIME: T+0:03 — Container Startup & Seed Initialization

```
Inside Docker container:

[SEED INIT] Global seed set to: 42
[SEED INIT] torch.seed: 9223372036854775807  (derived from 42)
[SEED INIT] np.random seed: 42

[INFO] ════════════════════════════════════════════════════════════
[INFO] Starting training with config:
[INFO]   trainer.total_epochs: 3
[INFO]   data.max_prompt_length: 1024
[INFO]   data.max_response_length: 1024
[INFO]   actor_rollout_ref.rollout.n: 2
[INFO]   actor_rollout_ref.rollout.gpu_memory_utilization: 0.4
[INFO]   actor_rollout_ref.actor.kl_loss_coef: 0.0  (β = 0)
[INFO]   actor_rollout_ref.actor.actor_adaptive_kl.enable: false
[INFO] ════════════════════════════════════════════════════════════

[ACTOR INIT] GLOBAL_SEED from env: 42
[ACTOR INIT] torch.initial_seed(): 9223372036854775807
```

**What happens**:
1. **main_ppo.py** initializes:
   - `torch.manual_seed(42)`
   - `torch.cuda.manual_seed_all(42)`
   - `np.random.seed(42)`
   - `random.seed(42)`
   - Sets `torch.backends.cudnn.deterministic = True`
   - Logs seed values

2. **dp_actor.py** initializes:
   - Logs GLOBAL_SEED from environment
   - Verifies seed consistency
   - Sets up KL controller (disabled for Zero KL mode)

---

### TIME: T+0:04 — Training Loop Execution

```
Epoch 1/3:
  ├─ Loading data...
  ├─ Rollout generation (vLLM, gpu_util=0.4)
  │  ├─ Sample 16 prompts
  │  ├─ Generate 2 responses per prompt (n=2)
  │  └─ Compute rewards (hf_math_verify)
  ├─ Advantage computation (GRPO)
  ├─ Actor update (PPO)
  │  ├─ KL penalty applied: β=0 (no KL component)
  │  └─ Actor loss = policy_loss + 0 × KL_loss
  ├─ Validation run on MATH500
  │  └─ val/test_score: 0.2456
  └─ Checkpoint saved

Epoch 2/3:
  ├─ [Similar to Epoch 1]
  │  └─ val/test_score: 0.2534
  └─ Checkpoint saved

Epoch 3/3 (Final):
  ├─ [Similar to Epoch 1]
  │  └─ val/test_score: 0.2567  ← FINAL SCORE
  └─ Checkpoint saved
```

**What happens**:
1. For each of 3 epochs:
   - Load training batch (batch_size=16)
   - Generate rollouts with vLLM (n=2 responses per prompt)
   - Compute rewards
   - Compute advantages (GRPO)
   - Update actor with KL penalty
   - Validate on MATH500 test set
   - Save checkpoint

2. Seed maintains consistency throughout because all RNGs initialized once with same seed

---

### TIME: T+2:15 — Training Completes

```
[INFO] Training completed successfully
[INFO] Final validation score: val/test_score = 0.2567
[INFO] Checkpoints saved to: /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42/
[INFO] Total training time: 135 minutes

✓ Training completed.
```

**What happens**:
1. Last checkpoint saved
2. Container exits normally
3. Training logs written to:
   - Container logs (retrievable via `docker logs`)
   - Log file: `/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42/metakl_train.log`

---

### TIME: T+2:16 — Results Extraction & Saving

**Back in orchestrator:**

```
[2026-05-28 14:30:00 + 2:16] [1/5] Zero KL completed in 135m 00s

[2026-05-28 16:46:00]   ✓ Training completed in 135m 0s
[2026-05-28 16:46:00]
[2026-05-28 16:46:00]   ✓ Saved results: /home/ai-server-02/R_projects/final_thesis/results/zero_kl_results.json
[2026-05-28 16:46:00]     - Final score: 0.2567
[2026-05-28 16:46:00]     - Training time: 135m 0s
[2026-05-28 16:46:00]
[2026-05-28 16:46:00]   ✓ Container cleaned up
```

**What happens**:
1. Extract final score from container logs:
   ```bash
   docker logs phase3-zero_kl-single-seed 2>&1 | \
     grep -oP "val/test_score['\"]?\s*[=:]\s*\K[0-9.]+(?![0-9.])"
   ```
   Result: `0.2567`

2. Call `save_controller_results.py`:
   ```bash
   python save_controller_results.py \
     --controller_name "zero_kl" \
     --seed 42 \
     --run_dir "/ckpts/.../phase3_zero_kl_seed42" \
     --final_score "0.2567" \
     --training_time 8100 \
     --output_dir "/path/to/results"
   ```

3. Create JSON file:
   ```json
   {
     "controller": "zero_kl",
     "seed": 42,
     "global_seed": 42,
     "epochs": 3,
     "final_test_score": "0.2567",
     "training_time_seconds": 8100,
     "training_time_readable": "135m 0s",
     "completion_timestamp": "2026-05-28T16:46:00Z",
     "checkpoint_dir": "/ckpts/.../phase3_zero_kl_seed42",
     "status": "completed"
   }
   ```
   Location: `results/zero_kl_results.json` ✓ SAVED

4. Clean up Docker container

---

### TIME: T+2:17 — Move to Controller 2

```
[2026-05-28 16:46:01] ╭─────────────────────────────────────────────────────────╮
[2026-05-28 16:46:01] │ [2/5] Controller: rule_based                            │
[2026-05-28 16:46:01] ╰─────────────────────────────────────────────────────────╯
[2026-05-28 16:46:01]
[2026-05-28 16:46:01] [CRITICAL CHECK] Killing any existing runs of rule_based on seed 2 or 3...
[2026-05-28 16:46:02]
[2026-05-28 16:46:02]   Description: Rule-based KL calculation
[2026-05-28 16:46:02]   Mode: rule
[2026-05-28 16:46:02]   Seed: 42
[2026-05-28 16:46:02]   Epochs: 3
[2026-05-28 16:46:02]   Container: phase3-rule_based-single-seed
```

**What happens**:
1. Repeat entire process for controller 2 (rule_based)
2. Same GLOBAL_SEED=42 used
3. Different RUTH_MODE="rule"
4. Same 3 epochs, same MATH500 dataset
5. Different adaptive KL logic (rule-based instead of fixed)

---

### TIME: T+4:30 → T+6:45 → T+9:00 → T+12:00

```
[2/5] Rule-based:      Completes at T+4:30 (150 min)
      Results → rule_based_results.json
      
[3/5] MLP-based:       Completes at T+6:45 (135 min more)
      Results → mlp_based_results.json
      
[4/5] Fixed (β=0.001): Completes at T+9:00 (135 min more)
      Results → fixed_results.json
      
[5/5] LSTM-based:      Completes at T+12:00 (180 min more)
      Results → lstm_based_results.json
```

---

### TIME: T+12:00 — Campaign Complete

```
[2026-05-28 16:46:00 + 12h]
[2026-05-28 19:46:00]  ╔════════════════════════════════════════════════════════╗
[2026-05-28 19:46:00]  ║         ALL CONTROLLERS COMPLETED SUCCESSFULLY          ║
[2026-05-28 19:46:00]  ╚════════════════════════════════════════════════════════╝
[2026-05-28 19:46:00]
[2026-05-28 19:46:00] Summary saved to:
[2026-05-28 19:46:00]   simpleRL-reason/analysis_logs/campaign_summary_20260528_143000.txt
[2026-05-28 19:46:00]
[2026-05-28 19:46:00] Individual controller results saved to:
[2026-05-28 19:46:00]   results/zero_kl_results.json
[2026-05-28 19:46:00]   results/rule_based_results.json
[2026-05-28 19:46:00]   results/mlp_based_results.json
[2026-05-28 19:46:00]   results/fixed_results.json
[2026-05-28 19:46:00]   results/lstm_based_results.json
[2026-05-28 19:46:00]
[2026-05-28 19:46:00] All checkpoints saved to:
[2026-05-28 19:46:00]   /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/
[2026-05-28 19:46:00]
[2026-05-28 19:46:00] ✓ Orchestrator finished successfully
```

**What happens**:
1. Generate final summary file
2. List all saved results
3. Exit cleanly
4. All 5 result JSON files exist with complete metrics

---

## Output File Structure After Completion

```
/home/ai-server-02/R_projects/final_thesis/
├── results/                          ← NEW DIRECTORY
│   ├── zero_kl_results.json         ✓ Final score, time, seed
│   ├── rule_based_results.json      ✓ 
│   ├── mlp_based_results.json       ✓ 
│   ├── fixed_results.json           ✓ 
│   └── lstm_based_results.json      ✓ 
│
├── simpleRL-reason/analysis_logs/
│   ├── orchestrator_single_seed_20260528_143000.log  ← Execution trace
│   └── campaign_summary_20260528_143000.txt          ← Final summary
│
└── /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/
    ├── phase3_zero_kl_seed42/
    │   ├── checkpoint_*.pt  ← Trained weights
    │   └── metakl_train.log ← Training logs
    ├── phase3_rule_based_seed42/
    ├── phase3_mlp_based_seed42/
    ├── phase3_fixed_seed42/
    └── phase3_lstm_based_seed42/
```

---

## What Happens If Something Goes Wrong

### Scenario 1: Container Crashes During Controller 2 (Rule-based)

```
[2026-05-28 16:46:00 + 4:30]
[2026-05-28 18:16:00] ✗ Container phase3-rule_based-single-seed exited with error

Container logs show:
  OutOfMemory Error: CUDA out of memory

Orchestrator response:
[2026-05-28 18:16:01]   ✗ ERROR: Training crashed for rule_based
[2026-05-28 18:16:02]   Proceeding to next controller...
[2026-05-28 18:16:02] ╭─────────────────────────────────────────────────────────╮
[2026-05-28 18:16:02] │ [3/5] Controller: mlp_based                             │
[2026-05-28 18:16:02] ╰─────────────────────────────────────────────────────────╯

Note: rule_based_results.json NOT created (not saved)
      Checkpoint /ckpts/.../phase3_rule_based_seed42/ may be partially saved
      Next controller starts automatically
```

**User action**: Restart at rule_based:
```bash
# Edit run_single_seed_orchestrator_v1.sh line 59:
# CONTROLLERS=("zero_kl" "rule_based" "mlp_based" "fixed" "lstm_based")
CONTROLLERS=("rule_based" "mlp_based" "fixed" "lstm_based")  # Skip zero_kl

./run_single_seed_orchestrator_v1.sh
```

---

### Scenario 2: User Interrupts (Ctrl+C) During Controller 3

```
[2026-05-28 16:46:00 + 6:45]
[2026-05-28 18:31:00] User presses Ctrl+C

Orchestrator response:
[2026-05-28 18:31:00] ✗ Orchestrator interrupted by user
[2026-05-28 18:31:01] Stopping current container: docker kill phase3-mlp_based-single-seed
[2026-05-28 18:31:02] ✓ Container stopped

Current status:
  ✓ zero_kl_results.json         (SAVED)
  ✓ rule_based_results.json      (SAVED)
  ✗ mlp_based_results.json       (NOT SAVED - partial training)
  ✗ fixed_results.json           (NOT RUN)
  ✗ lstm_based_results.json      (NOT RUN)
```

**User action**: Resume:
```bash
CONTROLLERS=("mlp_based" "fixed" "lstm_based") ./run_single_seed_orchestrator_v1.sh
```

---

## Monitoring Commands at Different Stages

### During Orchestrator Startup (T+0:00 to T+0:05)
```bash
tail -f simpleRL-reason/analysis_logs/orchestrator_single_seed_*.log
```

### During Controller 1 Training (T+0:05 to T+2:15)
```bash
docker logs -f phase3-zero_kl-single-seed
# See seed init, data loading, epochs 1-3, validation scores
```

### During Result Saving (T+2:15 to T+2:20)
```bash
ls -la results/  # Watch for JSON files appearing
cat results/zero_kl_results.json | jq .  # View saved result
```

### During Later Controllers (T+2:20 onwards)
```bash
# Switch to monitoring next controller
docker logs -f phase3-rule_based-single-seed
```

---

## Seed Validation Points

The orchestrator performs seed validation at these points:

1. **T+0:01** — Before Controller 1 starts
   ```bash
   kill_stray_seed_runs() checks for seed 2/3 containers
   ```

2. **T+0:02** — Training script validates
   ```bash
   if [[ "${RUTH_SEED}" != "${GLOBAL_SEED}" ]]; then exit 1; fi
   ```

3. **T+0:03** — Main trainer validates
   ```python
   global_seed = int(os.environ.get('GLOBAL_SEED', 42))
   torch.manual_seed(global_seed)  # All RNGs seeded
   ```

4. **T+0:04** — Actor logs seed
   ```python
   print(f"[ACTOR INIT] GLOBAL_SEED: {global_seed}")
   ```

All 5 controllers pass through all 4 validation points → **Seed consistency maintained**

---

## Summary

✅ **Orchestrator starts** (T+0:00)
✅ **Each controller runs sequentially** (Zero KL → Rule → MLP → Fixed → LSTM)
✅ **Results saved after each** (no overwrites)
✅ **Seed enforced at 4 levels** (no silent failures)
✅ **Auto-kills seed 2/3** (prevents GPU waste)
✅ **Campaign completes** (T+12-14h)
✅ **5 result files created** (ready for analysis)

