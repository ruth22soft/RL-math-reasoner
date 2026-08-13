# KL Controller Orchestration System

## Overview

This system automatically runs 4 adaptive KL controllers **sequentially** without manual intervention:

1. **rule_based** → Proportional feedback control (β adjusts based on KL loss)
2. **mlp_based** → Neural network learns optimal β from [kl_loss, reward_mean, reward_std, lagged_grad_norm]
3. **fixed** → Fixed β=0.001 (no adaptation)
4. **lstm_based** → Recurrent network with temporal state (your thesis contribution)

## Critical Configuration

✅ **NO SEED INITIALIZATION** - All controllers use random initialization (matches zero_kl baseline)
- No `GLOBAL_SEED` environment variable
- No `torch.manual_seed()` calls
- No `np.random.seed()` calls

✅ **Correct Hyperparameters** (matching zero_kl baseline):
- `n=4` responses per prompt
- `gpu_memory_utilization=0.5`
- `max_num_seqs=32`
- `batch_size=16`
- `epochs=3`
- `learning_rate=1e-6`
- Dataset: MATH500 (8,360 train samples)

✅ **Step Count**:
- Steps per epoch: ⌈8,360 ÷ 16⌉ = 523 steps
- Total steps: 523 × 3 = 1,569 steps per controller
- Expected time: ~111 seconds/step → ~29 hours per controller

## How to Use

### Quick Start

Run the orchestrator:
```bash
cd /home/ai-server-02/R_projects/final_thesis
./orchestrate_kl_controllers.sh
```

The script will:
1. Launch rule_based controller
2. Monitor every 60 seconds (% progress)
3. Auto-extract final test_score when complete
4. Save results to `results/rule_based_results.json`
5. Kill rule_based container
6. Automatically launch mlp_based
7. Repeat for fixed and lstm_based
8. Create combined `results/all_controllers_results.json`

### What You See

```
[2026-05-28 15:05:30] Launching rule_based controller...
✓ Container phase3-rule_based-s1 started
[2026-05-28 15:05:30] Monitoring rule_based (est. 29 hours)
[2026-05-28 15:06:30] Initializing...
[2026-05-28 15:10:30] Step 15/1569 (0.95%)
[2026-05-28 15:15:30] Step 30/1569 (1.91%)
... (updates every 60 seconds) ...
[2026-05-28 15:28:11] Step 345/1569 (21.99%) ← continues for ~29 hours
```

### Monitoring During Execution

While orchestrator runs, in another terminal:
```bash
# Watch latest step count
watch -n 10 'docker logs phase3-rule_based-s1 2>&1 | strings | grep -oP "step:\K[0-9]+" | tail -1'

# View full training logs
docker logs -f phase3-rule_based-s1

# Check GPU usage
nvidia-smi
```

## Results

After each controller completes, results are saved to:
```
results/
├── zero_kl_results.json          # From previous run (0.3246 score)
├── rule_based_results.json       # Test score, steps, timestamp
├── mlp_based_results.json
├── fixed_results.json
├── lstm_based_results.json
└── all_controllers_results.json  # Combined summary
```

### Example Result File

```json
{
  "controller": "rule_based",
  "seed": null,
  "seed_initialization": "random (no seed set)",
  "epochs": 3,
  "total_steps": 1569,
  "expected_steps": 1569,
  "final_test_score": "0.3456",
  "training_time_seconds": 174159,
  "completion_timestamp": "2026-05-30T20:15:00Z",
  "checkpoint_dir": "/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_rule_based_s1",
  "status": "completed"
}
```

## Important Notes

### ⚠️ Long Execution Time
- **Total time: ~4 × 29 hours = 116 hours (5 days)**
- Run in a `tmux` or `screen` session to survive SSH disconnections:
  ```bash
  tmux new-session -d -s orchestrator './orchestrate_kl_controllers.sh'
  
  # Later, attach to check status:
  tmux attach -t orchestrator
  ```

### ⚠️ GPU Resources
- Requires 1× RTX A4000 (or compatible NVIDIA GPU)
- Script allocates `--gpus all` to the container
- Monitor with: `watch -n 5 nvidia-smi`

### ⚠️ Seed Verification
To verify no seed is set:
```bash
# Check environment variables in running container
docker inspect phase3-rule_based-s1 | grep -i seed
# Output: (should be empty - no seed env vars)

# Check training logs for seed initialization
docker logs phase3-rule_based-s1 | strings | grep -i "seed"
# Output: (should show NO torch.manual_seed or np.random.seed calls)
```

## Troubleshooting

### Container stuck at initialization
```bash
# Check why it's not starting
docker logs phase3-rule_based-s1

# If pip install fails, restart:
docker stop phase3-rule_based-s1
docker rm phase3-rule_based-s1
./orchestrate_kl_controllers.sh  # Restart orchestrator
```

### Want to manually stop
```bash
# Stop current container
docker stop phase3-rule_based-s1

# The orchestrator will eventually detect stuck state and error out
# Then you can restart it to continue from next controller
./orchestrate_kl_controllers.sh
```

### Check progress without waiting for next log
```bash
# Get latest step count
docker exec phase3-rule_based-s1 bash -c 'strings /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_rule_based_s1/metakl_train.log | grep -oP "step:\K[0-9]+" | tail -1'

# Example output: 345
```

## File Locations

- **Orchestrator script:** `/home/ai-server-02/R_projects/final_thesis/orchestrate_kl_controllers.sh`
- **Results:** `/home/ai-server-02/R_projects/final_thesis/results/`
- **Checkpoints:** `/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_*_s1/`
- **Training logs:** `phase3_*/metakl_train.log` (inside container volumes)

## Key Differences from Your Initial Approach

| Aspect | Previous | This System |
|--------|----------|------------|
| **Seed** | Had GLOBAL_SEED=42 | NO SEED (random) ✓ |
| **N (rollout)** | n=2 (WRONG) | n=4 (CORRECT) ✓ |
| **GPU Util** | 0.4 (WRONG) | 0.5 (CORRECT) ✓ |
| **Automation** | Manual each run | Automatic sequential ✓ |
| **Results** | Manual extraction | Auto JSON save ✓ |
| **Monitoring** | Manual polling | Auto 60s checks ✓ |

## Next Steps

1. **Start orchestrator:**
   ```bash
   ./orchestrate_kl_controllers.sh
   ```

2. **Detach and let it run:**
   - Use `tmux` or `screen`
   - Or run with `nohup` and `&`

3. **Check back in 5 days** for results

4. **Review results** in `results/all_controllers_results.json`

---

**Created:** May 28, 2026
**Total Expected Runtime:** ~5 days (116 hours)
**Status:** Ready to run
