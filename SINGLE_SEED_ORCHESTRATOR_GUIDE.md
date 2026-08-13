# Single-Seed 5-Controller RL Benchmark Orchestrator

## Overview

This orchestrator runs your 5-controller Adaptive KL benchmark comparison **using a single shared seed** for all controllers, eliminating the computational waste of multi-seed training.

**Five Controllers (Priority Order):**
1. **Zero KL** (β = 0)
2. **Rule-based** KL calculation  
3. **MLP-based** KL calculation
4. **Fixed/Constant** KL coefficient (β = 0.001)
5. **LSTM-based** KL calculation

## Key Features

✓ **Single Seed Mode**: All 5 controllers use `GLOBAL_SEED=42`  
✓ **Sequential Execution**: Controllers run one at a time (no parallelism)  
✓ **Per-Controller Results**: Each controller's results saved immediately after completion  
✓ **Critical Interruption**: Auto-kills any stray seed 2/3 runs  
✓ **Seed Guards**: Enforces single-seed-only at orchestrator and trainer levels  
✓ **Result Persistence**: Completed controller results never overwritten  

## Prerequisites

- RTX A4000 GPU (or similar)
- Docker with `simple-rl:ngc-vllm` image ready
- All data volumes mounted (simplerl_data, simplerl_ckpts, simplerl_hf_cache)
- 3 epochs × 5 controllers ≈ 8-12 hours total GPU time

## Quick Start

### 1. Extract/Set GLOBAL_SEED (if Zero KL already ran)

```bash
# If you have a completed Zero KL run, extract its seed:
ZERO_KL_LOG="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_s0/metakl_train.log"

# Search for seed initialization (usually 42)
grep -i "seed" "${ZERO_KL_LOG}" | head -3

# Set as GLOBAL_SEED in orchestrator (line 50):
GLOBAL_SEED=42
```

Otherwise, use the default `GLOBAL_SEED=42`.

### 2. Make Scripts Executable

```bash
chmod +x /home/ai-server-02/R_projects/final_thesis/run_single_seed_orchestrator_v1.sh
chmod +x /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/scripts/run_metakl_training_single_seed.sh
```

### 3. Run the Orchestrator

```bash
cd /home/ai-server-02/R_projects/final_thesis

# Start the orchestrator (runs in foreground, ~12 hours)
./run_single_seed_orchestrator_v1.sh

# Or run in background with nohup:
nohup ./run_single_seed_orchestrator_v1.sh > orchestrator.out 2>&1 &
```

### 4. Monitor Progress

```bash
# Watch main orchestrator log
tail -f simpleRL-reason/analysis_logs/orchestrator_single_seed_*.log

# Watch current training container
docker logs -f phase3-CONTROLLER_NAME-single-seed

# Check GPU usage
watch nvidia-smi
```

## Output Structure

After completion, results are saved to:

```
results/
├── zero_kl_results.json
├── rule_based_results.json
├── mlp_based_results.json
├── fixed_results.json
└── lstm_based_results.json

simpleRL-reason/analysis_logs/
├── orchestrator_single_seed_20260528_143022.log
└── campaign_summary_20260528_143022.txt

/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/
├── phase3_zero_kl_seed42/
├── phase3_rule_based_seed42/
├── phase3_mlp_based_seed42/
├── phase3_fixed_seed42/
└── phase3_lstm_based_seed42/
```

## Result JSON Format

Each `*_results.json` contains:

```json
{
  "controller": "fixed",
  "seed": 42,
  "global_seed": 42,
  "epochs": 3,
  "final_test_score": 0.3456,
  "training_time_seconds": 3600,
  "training_time_readable": "60m 0s",
  "completion_timestamp": "2026-05-28T14:30:22Z",
  "checkpoint_dir": "/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_fixed_seed42",
  "status": "completed"
}
```

## Critical Interruption Rule

If the orchestrator detects a controller running on seed 2 or 3:

1. **Immediately kills the container**  
2. **Discards the partial results**  
3. **Moves to next controller**  
4. **Logs warning** to orchestrator log

Example log output:
```
[2026-05-28 14:30:00] ⚠️  KILLING phase3-fixed-s2 (mid-training on seed 2)
[2026-05-28 14:30:02] [CRITICAL CHECK] Killing any existing runs of fixed on seed 2 or 3...
```

## Seed Guards in Code

### 1. Orchestrator Level (`run_single_seed_orchestrator_v1.sh`)
```bash
# Before each controller:
kill_stray_seed_runs "${controller}"  # Kill seeds 2 and 3
```

### 2. Trainer Script Level (`run_metakl_training_single_seed.sh`)
```bash
# Validate GLOBAL_SEED matches RUTH_SEED if both provided
if [[ "${RUTH_SEED}" != "${GLOBAL_SEED}" ]]; then
  echo "[ERROR] Seed mismatch - aborting"
  exit 1
fi
```

### 3. Main Trainer Level (`verl/trainer/main_ppo.py`)
```python
# Initialize all RNG with same seed
global_seed = int(os.environ.get('GLOBAL_SEED', 42))
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)
```

### 4. Actor Level (`verl/workers/actor/dp_actor.py`)
```python
# Log current seed for debugging
print(f"[ACTOR INIT] GLOBAL_SEED from env: {global_seed}")
print(f"[ACTOR INIT] torch.initial_seed(): {current_seed}")
```

## Troubleshooting

### Q: "Container failed to start for controller X"
**A:** Check available disk space and Docker status:
```bash
docker ps -a
docker logs phase3-X-single-seed
df -h /ckpts
```

### Q: "Training seems to be hanging"
**A:** Check if the container is actually running:
```bash
docker exec phase3-zero_kl-single-seed nvidia-smi
docker top phase3-zero_kl-single-seed
```

### Q: "Results not saved for controller X"
**A:** Verify the run_dir exists and has results:
```bash
docker run --rm -v simplerl_ckpts:/ckpts alpine ls -lah /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_X_seed42/
```

### Q: "How do I resume from a particular controller?"
**A:** Edit `run_single_seed_orchestrator_v1.sh` and comment out controllers before the one you want:
```bash
# CONTROLLERS=("zero_kl" "rule_based" "mlp_based" "fixed" "lstm_based")
CONTROLLERS=("mlp_based" "fixed" "lstm_based")  # Skip zero_kl and rule_based
```

## Configuration Changes

### To change GLOBAL_SEED:
Edit line ~50 in `run_single_seed_orchestrator_v1.sh`:
```bash
GLOBAL_SEED=42  # Change to desired seed value
```

### To change training parameters:
Edit lines in `run_single_seed_orchestrator_v1.sh`:
```bash
TOTAL_EPOCHS=3                    # Epochs per controller (keep as 3)
RUTH_ROLLOUT_N=2                  # Rollout n (keep as 2)
RUTH_ROLLOUT_GPU_MEMORY_UTIL=0.4  # vLLM GPU util (keep as 0.4)
```

### To disable Telegram notifications:
Leave empty:
```bash
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
```

## Cleaning Up

### Remove completed container
```bash
docker rm phase3-zero_kl-single-seed
```

### Remove all orchestrator containers
```bash
docker rm -f $(docker ps -aq -f "label=simplerl.role=training")
```

### Clear old logs (keep recent)
```bash
cd simpleRL-reason/analysis_logs
ls -1t orchestrator_single_seed_*.log | tail -n +6 | xargs rm -f
```

## Expected Timing

- **Zero KL**: ~2-2.5 hours
- **Rule-based**: ~2-2.5 hours  
- **MLP-based**: ~2.5-3 hours (MLP training overhead)
- **Fixed**: ~2-2.5 hours
- **LSTM-based**: ~3-3.5 hours (LSTM training overhead)

**Total: ~12-14 hours** on RTX A4000

## References

- **Orchestrator**: `run_single_seed_orchestrator_v1.sh`
- **Training Script**: `simpleRL-reason/scripts/run_metakl_training_single_seed.sh`
- **Result Saver**: `simpleRL-reason/scripts/save_controller_results.py`
- **Config**: `simpleRL-reason/verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml`

---

**Note**: This orchestrator enforces single-seed-only mode and will **abort loudly** if:
- A different seed is detected
- A stray seed 2/3 container is found  
- Any seed validation fails

This prevents silent GPU hour waste on wrong seeds.
