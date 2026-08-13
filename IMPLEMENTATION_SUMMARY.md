# Single-Seed 5-Controller Benchmark - Implementation Summary

## Overview

Ruth requested a critical transformation of the multi-seed orchestrator to a **single-seed-only orchestrator** for the MSc thesis on Adaptive KL Regularization. This reduces GPU hours from ~36-42 hours (3 seeds × 4 modes) to ~12-14 hours (1 seed × 5 controllers).

## Key Changes Implemented

### ✅ STEP 1: SEED CHANGES

**Status: COMPLETED**

- **Removed multi-seed loop entirely**
  - Old: Looped through SEEDS=(0 1 2) for each mode
  - New: Single GLOBAL_SEED=42 used for all 5 controllers
  
- **Hardcoded GLOBAL_SEED constant**
  - Location: `run_single_seed_orchestrator_v1.sh` line 50
  - ```bash
    GLOBAL_SEED=42  # All 5 controllers use this seed
    ```
  - Can be extracted from Zero KL's first run if it already completed
  
- **Added seed guards at multiple levels**
  1. **Orchestrator level**: Validates GLOBAL_SEED before each run
  2. **Trainer script level**: Rejects mismatched seed with loud error
  3. **Main trainer level**: Initializes ALL RNGs with same seed
  4. **Actor level**: Logs seed for debugging/verification

### ✅ STEP 2: CRITICAL INTERRUPTION RULE

**Status: COMPLETED**

Implemented 3-layer interruption system:

**Layer 1: Orchestrator Pre-check**
```bash
kill_stray_seed_runs() {
  # Kills any container running seed 2 or 3 BEFORE starting new run
  for seed_num in 2 3; do
    docker kill "phase3-${controller}-s${seed_num}" 2>/dev/null || true
  done
}
```

**Layer 2: Script-level Guard**
```bash
# If RUTH_SEED passed from old orchestrator, validate it matches GLOBAL_SEED
if [[ "${RUTH_SEED}" != "${GLOBAL_SEED}" ]]; then
  echo "[ERROR] Only GLOBAL_SEED=${GLOBAL_SEED} permitted"
  exit 1
fi
```

**Layer 3: Trainer-level Initialization**
```python
# Initialize all RNG with GLOBAL_SEED
global_seed = int(os.environ.get('GLOBAL_SEED', 42))
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
np.random.seed(global_seed)
random.seed(global_seed)
```

### ✅ STEP 3: EXECUTION PRIORITY ORDER

**Status: COMPLETED**

Controllers execute in exact priority order:

```bash
CONTROLLERS=("zero_kl" "rule_based" "mlp_based" "fixed" "lstm_based")
```

**Mapping to config values:**
- `zero_kl` → RUTH_MODE="fixed", RUTH_FIXED_BETA="0.0"
- `rule_based` → RUTH_MODE="rule"
- `mlp_based` → RUTH_MODE="mlp"
- `fixed` → RUTH_MODE="fixed", RUTH_FIXED_BETA="0.001"
- `lstm_based` → RUTH_MODE="lstm"

**Execution pattern:**
- Each controller trains for exactly 3 epochs
- Results saved immediately after completion
- Container cleaned up before moving to next

### ✅ STEP 4: RESULT SAVING

**Status: COMPLETED**

**Per-controller results saved after EACH completes:**

```json
results/
├── zero_kl_results.json
├── rule_based_results.json
├── mlp_based_results.json
├── fixed_results.json
└── lstm_based_results.json
```

**Each JSON contains:**
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

**Checkpoints preserved in:**
```
/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/
├── phase3_zero_kl_seed42/
├── phase3_rule_based_seed42/
├── phase3_mlp_based_seed42/
├── phase3_fixed_seed42/
└── phase3_lstm_based_seed42/
```

### ✅ STEP 5: DO NOT CHANGE

**Status: VERIFIED**

All requested parameters kept exactly as-is:
- ✓ 3 epochs for all controllers
- ✓ All existing reward functions (hf_math_verify, compute_score)
- ✓ Model config (Qwen2.5-0.5B, verl framework)
- ✓ vllm gpu_util = 0.4
- ✓ Actor and reference model parameter offload settings
- ✓ Checkpoint saving logic and auto-resume
- ✓ MATH500 evaluation protocol
- ✓ Batch size = 16, n = 2, max sequence length = 512 tokens
- ✓ Learning rate = 5×10⁻⁷

## Files Created/Modified

### NEW FILES

1. **`run_single_seed_orchestrator_v1.sh`** (274 lines)
   - Main orchestrator script
   - Runs 5 controllers sequentially with GLOBAL_SEED
   - Implements interruption logic
   - Saves per-controller results
   
2. **`simpleRL-reason/scripts/run_metakl_training_single_seed.sh`** (185 lines)
   - Updated training launch script
   - Validates GLOBAL_SEED before starting
   - Container-level seed enforcement
   - Result persistence support

3. **`simpleRL-reason/scripts/save_controller_results.py`** (95 lines)
   - Python utility for saving results to JSON
   - Called after each controller completes
   - Generates per-controller results file

4. **`SINGLE_SEED_ORCHESTRATOR_GUIDE.md`** (Complete user guide)
   - Quick start instructions
   - Monitoring/debugging guide
   - Configuration options
   - Troubleshooting section

5. **`IMPLEMENTATION_SUMMARY.md`** (This file)
   - Technical overview of all changes
   - File-by-file breakdown
   - How to run the system

### MODIFIED FILES

1. **`simpleRL-reason/verl/trainer/main_ppo.py`**
   - Added seed initialization at line ~125
   - Initializes torch, numpy, random with GLOBAL_SEED
   - Sets CUDA deterministic mode
   - Logs seed values for verification

2. **`simpleRL-reason/verl/workers/actor/dp_actor.py`**
   - Added seed guard in __init__ (lines ~48-52)
   - Logs GLOBAL_SEED and torch.initial_seed()
   - Validates seed consistency
   - Helps catch seed mismatches during training

## Usage Instructions

### Quick Start (3 steps)

**Step 1: Set GLOBAL_SEED**
```bash
# Already set to 42 in the script
# If you have a prior Zero KL run, extract its seed:
grep -i "seed" /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_s0/metakl_train.log | head -3
# Then update line 50 in run_single_seed_orchestrator_v1.sh
```

**Step 2: Make executable**
```bash
chmod +x /home/ai-server-02/R_projects/final_thesis/run_single_seed_orchestrator_v1.sh
chmod +x /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/scripts/run_metakl_training_single_seed.sh
```

**Step 3: Launch orchestrator**
```bash
cd /home/ai-server-02/R_projects/final_thesis
./run_single_seed_orchestrator_v1.sh
```

### Monitoring

**Main log:**
```bash
tail -f simpleRL-reason/analysis_logs/orchestrator_single_seed_*.log
```

**Current training container:**
```bash
docker logs -f phase3-zero_kl-single-seed  # Or other controller name
```

**GPU usage:**
```bash
watch nvidia-smi
```

## Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ run_single_seed_orchestrator_v1.sh (MAIN ORCHESTRATOR)      │
│ GLOBAL_SEED=42                                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
     ┌─────────────┴─────────────┐
     │   For each controller:     │
     └─────────────┬─────────────┘
                   │
     ┌─────────────▼─────────────┐
     │  1. kill_stray_seed_runs   │
     │     (kill seed 2/3)        │
     └─────────────┬─────────────┘
                   │
     ┌─────────────▼─────────────────────────────┐
     │  2. Launch run_metakl_training_single_seed │
     │     env: GLOBAL_SEED=42, RUTH_MODE,etc    │
     └─────────────┬─────────────────────────────┘
                   │
     ┌─────────────▼──────────────────────────────────┐
     │  3. Docker Container Starts                    │
     │     - Validates RUTH_SEED == GLOBAL_SEED       │
     │     - Initializes torch/numpy/random seeds     │
     │     - Trains for 3 epochs                      │
     └─────────────┬──────────────────────────────────┘
                   │
     ┌─────────────▼──────────────────────┐
     │  4. Container Completes             │
     │     Extract final_score from logs   │
     └─────────────┬──────────────────────┘
                   │
     ┌─────────────▼──────────────────────────────┐
     │  5. save_controller_results.py              │
     │     Write results/CONTROLLER_results.json   │
     └─────────────┬──────────────────────────────┘
                   │
     ┌─────────────▼──────────────────────┐
     │  6. docker rm (cleanup container)   │
     │     Move to NEXT controller         │
     └─────────────────────────────────────┘
                   │
                   └─── Loop for all 5 controllers ───┐
                                                       │
                                                       ▼
                        ┌──────────────────────────────────┐
                        │  Campaign Complete!              │
                        │  5 × results/..._results.json    │
                        │  5 × /ckpts/.../phase3_*_seed42/ │
                        └──────────────────────────────────┘
```

## Expected Outputs

### After Successful Run

**Files created:**
- `results/zero_kl_results.json` (final score, time, seed info)
- `results/rule_based_results.json`
- `results/mlp_based_results.json`
- `results/fixed_results.json`
- `results/lstm_based_results.json`
- `simpleRL-reason/analysis_logs/orchestrator_single_seed_20260528_143022.log`
- `simpleRL-reason/analysis_logs/campaign_summary_20260528_143022.txt`

**Checkpoints saved to:**
- `/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42/`
- `/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_rule_based_seed42/`
- etc. (one per controller)

### Sample Result JSON

```json
{
  "controller": "zero_kl",
  "seed": 42,
  "global_seed": 42,
  "epochs": 3,
  "final_test_score": "0.2845",
  "training_time_seconds": 7200,
  "training_time_readable": "120m 0s",
  "completion_timestamp": "2026-05-28T15:30:00Z",
  "checkpoint_dir": "/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42",
  "status": "completed"
}
```

## Timing Estimates

| Controller | GPU Time | Notes |
|-----------|----------|-------|
| Zero KL | 2-2.5h | Baseline, β=0 |
| Rule-based | 2-2.5h | Deterministic control |
| MLP-based | 2.5-3h | Small MLP training overhead |
| Fixed (β=0.001) | 2-2.5h | Constant β |
| LSTM-based | 3-3.5h | LSTM state tracking overhead |
| **TOTAL** | **~12-14h** | Single RTX A4000 |

## Verification Checklist

Before running:
- [ ] GLOBAL_SEED set to 42 (or extracted value) in `run_single_seed_orchestrator_v1.sh` line 50
- [ ] Scripts are executable: `ls -la *.sh simpleRL-reason/scripts/*.sh`
- [ ] Docker image exists: `docker images | grep simple-rl`
- [ ] Data volumes exist: `docker volume ls | grep simplerl`
- [ ] At least 50GB free in `/ckpts`
- [ ] Read `SINGLE_SEED_ORCHESTRATOR_GUIDE.md`

After running:
- [ ] All 5 results files created in `results/`
- [ ] Each results JSON has valid data
- [ ] Orchestrator log shows no errors
- [ ] All 5 checkpoint directories exist in `/ckpts/`

## Troubleshooting

### "Container failed to start"
→ Check Docker status and logs:
```bash
docker ps -a | grep phase3
docker logs phase3-X-single-seed
```

### "Training seems stuck"
→ Check if container is actually running:
```bash
docker exec phase3-zero_kl-single-seed nvidia-smi
docker top phase3-zero_kl-single-seed
```

### "Results not saved"
→ Check if run_dir exists and has output:
```bash
docker run --rm -v simplerl_ckpts:/ckpts alpine \
  ls -lah /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42/
```

### "Different results than before"
→ Seed should be same (42), but:
- Model weights may vary by PyTorch version
- CUDA version affects numerics
- GPU model matters (memory layout)
→ If you need exact reproducibility, pin versions in Dockerfile

## References

**Main Scripts:**
- Orchestrator: `run_single_seed_orchestrator_v1.sh`
- Training: `simpleRL-reason/scripts/run_metakl_training_single_seed.sh`
- Result saving: `simpleRL-reason/scripts/save_controller_results.py`

**Configuration:**
- Config: `simpleRL-reason/verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml`

**Documentation:**
- User guide: `SINGLE_SEED_ORCHESTRATOR_GUIDE.md`
- Implementation: `IMPLEMENTATION_SUMMARY.md` (this file)

**Modified Trainer Code:**
- Main: `simpleRL-reason/verl/trainer/main_ppo.py` (added seed init)
- Actor: `simpleRL-reason/verl/workers/actor/dp_actor.py` (added guards)

---

**Last Updated**: May 28, 2026  
**Author**: GitHub Copilot  
**Status**: Ready for Production
