# SimpleRL GRPO Adaptive KL Multi-Seed Campaign Documentation

**Project**: RL Math Reasoner - GRPO Training with Adaptive KL Control  
**Repository**: `/home/ai-server-02/R_projects/final_thesis/simpleRL-reason/`  
**Campaign Start**: May 26, 2026 @ 15:04 UTC  
**Status**: ACTIVE - Baseline running, orchestrator waiting for completion

---

## 1. CURRENT STATUS (May 26, 2026, 15:04+ UTC)

### Active Experiment
```
Container: phase3-nokl-baseline
Mode: Fixed KL, β=0 (No KL regularization baseline)
Status: RUNNING (step 2/1569)
Est. Completion: ~20 hours (~11:00 UTC May 27)
Epochs: 3
Dataset: 8,360 filtered samples
```

### Orchestrator Status
```
Process: Running (PID ~3358053)
Log: /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/orchestrator_multiseed_v2_20260526_150408.log
Status: WAITING for phase3-nokl-baseline to complete
Queued: 12 runs (4 modes × 3 seeds)
```

### Campaign Timeline
- **Phase 0 (Baseline)**: No-KL (β=0) - NOW RUNNING
- **Phase 1-4 (Sequential)**: Will start after baseline completes
  - Fixed β=0.001 (3 seeds)
  - Rule adaptive KL (3 seeds)
  - MLP adaptive KL (3 seeds)
  - LSTM adaptive KL (3 seeds, reference: 33.87%)
- **Total Duration**: ~60-70 hours

---

## 2. ARCHITECTURE & SYSTEM DESIGN

### Training Stack
```
┌─────────────────────────────────────────┐
│ Launcher: run_metakl_training_detached.sh
│ (Converts RUTH_* env vars to CLI args)  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Docker Container (simple-rl:ngc-vllm)   │
│ ├─ PyTorch 2.4.0 + FSDP               │
│ ├─ vLLM (inference/rollout)           │
│ ├─ Transformers + Qwen2.5-0.5B        │
│ └─ Ray (distributed workers)          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ Trainer: verl.trainer.ppo.ray_trainer   │
│ ├─ fit() loop (epoch/step control)    │
│ ├─ Actor (policy): PPO/GRPO           │
│ ├─ Critic (value function)            │
│ ├─ Reference policy (frozen)          │
│ └─ Checkpoint manager (pruning)       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ KL Controllers (actor-side)             │
│ ├─ Fixed: constant β                   │
│ ├─ Rule: proportional adaptive          │
│ ├─ MLP: learned network                │
│ └─ LSTM: your thesis method            │
└─────────────────────────────────────────┘

Orchestrator: run_multiseed_orchestrator_v2.sh
├─ Waits for baseline completion
├─ Launches 12 runs sequentially
├─ Collects final metrics (val/test_score)
├─ Saves results to file
└─ Sends Telegram updates (if credentials set)

Volumes:
├─ simplerl_data: Training/test datasets
├─ simplerl_ckpts: Checkpoints (pruned each run)
├─ simplerl_hf_cache: HuggingFace model cache
└─ Workspace mount: /workspace/simpleRL-reason → local repo
```

### Data Flow
```
Dataset: simplelr_qwen_level3to5 (parquet)
├─ Original: 8,523 samples
├─ Filtered: 8,360 samples (math content validation)
├─ Train/Val split: ~16.35:1 (8,360 train, 496 test)
├─ Batch size: 16
└─ Steps per epoch: ceil(8360/16) = 523

Per Run:
├─ Epochs: 3
├─ Total steps: 523 × 3 = 1,569
├─ Duration: ~20-22 hours (89s/step avg)
└─ Checkpoint saves: at freq=2000 (after training ends at 1,569)
```

### KL Coefficient Control

**Fixed Mode (Baseline)**
```yaml
actor_rollout_ref.actor.kl_loss_coef: 0.001
actor_rollout_ref.actor.actor_adaptive_kl.enable: false
```
- Constant β throughout training
- Fixed β=0.001 (default baseline)
- No-KL variant (β=0) for comparison

**Rule-Based Adaptive**
```yaml
actor_rollout_ref.actor.actor_adaptive_kl.enable: true
actor_rollout_ref.actor.actor_adaptive_kl.mode: rule
actor_rollout_ref.actor.actor_adaptive_kl.init_beta: 0.001
actor_rollout_ref.actor.actor_adaptive_kl.target_kl_loss: 0.08
actor_rollout_ref.actor.actor_adaptive_kl.up_gain: 1.08
actor_rollout_ref.actor.actor_adaptive_kl.down_gain: 0.94
```
- Proportional feedback controller
- Adjusts β based on KL loss deviation from target

**MLP Adaptive**
```yaml
actor_rollout_ref.actor.actor_adaptive_kl.enable: true
actor_rollout_ref.actor.actor_adaptive_kl.mode: mlp
```
- Neural network learns optimal KL adjustment
- Input: recent metrics (reward std, grad norm, KL loss)
- Output: KL coefficient multiplier

**LSTM Adaptive (Thesis Method)**
```yaml
actor_rollout_ref.actor.actor_adaptive_kl.enable: true
actor_rollout_ref.actor.actor_adaptive_kl.mode: lstm
```
- LSTM processes historical metric sequences
- Learns temporal patterns for KL control
- Reference: 33.87% validation score (Apr 21, 2026)

---

## 3. KEY FILES & MODIFICATIONS

### Launcher Script
**File**: `simpleRL-reason/scripts/run_metakl_training_detached.sh`

**Purpose**: Converts RUTH_* environment variables to Hydra CLI arguments and launches Docker training container

**Key Environment Variables**:
```bash
# Mode & KL Settings
RUTH_MODE=fixed|rule|mlp|lstm          # KL controller mode
RUTH_FIXED_BETA=0.001                   # Fixed β (only for fixed mode)

# Training Budget
RUTH_TOTAL_EPOCHS=3                     # Epochs (primary control)
RUTH_TOTAL_TRAINING_STEPS=              # Total steps (overrides epochs if set)
                                         # LEAVE EMPTY to use epoch-based training

# Checkpoint Management
RUTH_RUN_DIR=/ckpts/...                # Output directory for this run
RUTH_SAVE_FREQ=2000                     # Save checkpoint frequency
RUTH_TEST_FREQ=2000                     # Validation test frequency
RUTH_RESUME_MODE=never                  # never|auto - prevent cross-seed resumption
RUTH_REMOVE_PREVIOUS_CKPT=true          # Enable pruning of old checkpoints

# Rollout/Inference Settings
RUTH_ROLLOUT_N=4                        # Num samples per prompt
RUTH_ROLLOUT_GPU_MEMORY_UTIL=0.5        # vLLM GPU memory budget
RUTH_ROLLOUT_MAX_NUM_SEQS=32            # Max batched sequences in vLLM

# Container
RUTH_CONTAINER_NAME=phase3-nokl         # Docker container name
RUTH_IMAGE_NAME=simple-rl:ngc-vllm      # Docker image (default shown)

# Telegram (Optional)
TELEGRAM_BOT_TOKEN=...                  # Telegram bot token
TELEGRAM_CHAT_ID=...                    # Telegram chat ID
```

**Recent Modifications**:
- Added Telegram credential passthrough to container
- Added background monitoring loop (reports every 5 min + start/end)
- Verified Hydra CLI override precedence (CLI wins over YAML)

### Checkpoint Manager (Patched)
**File**: `simpleRL-reason/verl/utils/checkpoint/checkpoint_manager.py`

**Purpose**: Manages model/optimizer/scheduler/state checkpointing across distributed training

**Patch Applied**: `remove_previous_save_local_path()` method enhanced
```python
# Rank-0 logic added after per-rank file removal:
if self.rank == 0:
    abs_path = os.path.join(self.strategy.save_dir, ...)
    if os.path.exists(abs_path):
        shutil.rmtree(abs_path)  # Recursively delete entire old checkpoint dir
```

**Impact**: Prevents checkpoint directory accumulation (e.g., global_step_200...900 no longer persist)

### Trainer Loop
**File**: `simpleRL-reason/verl/trainer/ppo/ray_trainer.py`

**Key Code Segment**: `fit()` method (lines ~900-980)
```python
self.global_steps = 0
self._load_checkpoint()  # Respects resume_mode

for epoch in range(self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        # Training update...
        self.global_steps += 1
        
        if self.global_steps >= self.total_training_steps:
            return  # Exit early if step cap reached
        
        if (self.global_steps + 1) % save_freq == 0:
            self._save_checkpoint()
```

**Critical**: `total_training_steps` defaults to -1 (disabled); overridable via CLI

### Training Config
**File**: `simpleRL-reason/verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml`

**Key Defaults**:
```yaml
trainer:
  total_epochs: 3
  total_training_steps: -1           # -1 = use epochs, override on CLI
  resume_mode: auto                  # Overridden by RUTH_RESUME_MODE
  remove_previous_ckpt: true         # Enables checkpoint pruning
  save_freq: 50
  test_freq: -1

data:
  train_batch_size: 16
  train_files: /data/simplelr_qwen_level3to5/train.parquet

actor_rollout_ref.actor:
  kl_loss_coef: 0.001                # Overridden by RUTH_FIXED_BETA
  actor_adaptive_kl:
    enable: false                    # Overridden by RUTH_MODE
    mode: lstm
    init_beta: 0.001
    warmup_steps: 1
    ...
```

**Note**: All RUTH_* environment variables override YAML defaults via CLI

### Orchestrator Script (NEW)
**File**: `run_multiseed_orchestrator_v2.sh`

**Purpose**: Sequentially executes 12 training runs (4 modes × 3 seeds) with automatic logging and result collection

**Logic Flow**:
1. Waits for `phase3-nokl-baseline` container to exit
2. Collects final validation score from no-KL run
3. For each mode in [fixed, rule, mlp, lstm]:
   - For each seed in [0, 1, 2]:
     - Launches container with RUTH_MODE, seed-specific run dir
     - Waits for container completion
     - Extracts final val/test_score
     - Saves to results file
     - Sends Telegram update
     - Cleans up container
4. Generates summary report with all results

**Output Files**:
- Main log: `/simpleRL-reason/analysis_logs/orchestrator_multiseed_v2_YYYYMMDD_HHMMSS.log`
- Results summary: `/simpleRL-reason/analysis_logs/campaign_results_summary_YYYYMMDD_HHMMSS.txt`

---

## 4. HOW TO RUN EXPERIMENTS

### Option A: Full Multi-Seed Campaign (Automated)

**Prerequisite**: Orchestrator already running (started at 15:04 UTC May 26)

**To check status**:
```bash
# View orchestrator process
ps aux | grep orchestrator_v2 | grep -v grep

# View live orchestrator log
tail -f /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/orchestrator_multiseed_v2_*.log

# View results as they complete
tail -f /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/campaign_results_summary_*.txt
```

**To start fresh orchestrator** (if needed):
```bash
cd /home/ai-server-02/R_projects/final_thesis

# Optional: Set Telegram credentials
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Launch orchestrator in background
nohup bash ./run_multiseed_orchestrator_v2.sh > orchestrator_v2_startup.log 2>&1 &

# Verify it started
ps aux | grep orchestrator_v2
```

### Option B: Single Run (Manual)

**Launch a single experiment**:
```bash
cd /home/ai-server-02/R_projects/final_thesis

# Example: Fixed mode, seed 0, 3 epochs
RUTH_MODE=fixed \
RUTH_FIXED_BETA=0.001 \
RUTH_TOTAL_EPOCHS=3 \
RUTH_SAVE_FREQ=2000 \
RUTH_TEST_FREQ=2000 \
RUTH_ROLLOUT_N=4 \
RUTH_ROLLOUT_GPU_MEMORY_UTIL=0.5 \
RUTH_ROLLOUT_MAX_NUM_SEQS=32 \
RUTH_CONTAINER_NAME=phase3-fixed-s0 \
RUTH_RUN_DIR=/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_fixed_s0 \
RUTH_REMOVE_PREVIOUS_CKPT=true \
RUTH_RESUME_MODE=never \
./simpleRL-reason/scripts/run_metakl_training_detached.sh
```

**Monitor the run**:
```bash
# Live logs
docker logs -f phase3-fixed-s0

# Extract specific metrics
docker logs phase3-fixed-s0 2>&1 | grep 'val/test_score'

# Check final result when done
docker logs phase3-fixed-s0 2>&1 | grep -oP 'val/test_score.*?:\s*\K[0-9.]+' | tail -1
```

**Example variants**:

Rule adaptive:
```bash
RUTH_MODE=rule \
RUTH_CONTAINER_NAME=phase3-rule-s1 \
RUTH_RUN_DIR=/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_rule_s1 \
...
```

LSTM adaptive:
```bash
RUTH_MODE=lstm \
RUTH_CONTAINER_NAME=phase3-lstm-s2 \
RUTH_RUN_DIR=/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_lstm_s2 \
...
```

No-KL baseline:
```bash
RUTH_MODE=fixed \
RUTH_FIXED_BETA=0 \
RUTH_CONTAINER_NAME=phase3-nokl-baseline \
RUTH_RUN_DIR=/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_nokl_baseline \
...
```

---

## 5. HOW TO VIEW RESULTS

### Monitoring Active Runs

**Show all training containers**:
```bash
docker ps --filter label=simplerl.role=training --format 'table {{.Names}}\t{{.Status}}'
```

**Real-time metrics from a container**:
```bash
# Watch training steps (updates ~every 90 seconds)
watch -n 10 'docker logs phase3-nokl-baseline 2>&1 | grep "step:" | tail -3'

# Monitor validation scores periodically
watch -n 300 'docker logs phase3-nokl-baseline 2>&1 | grep "val/test_score" | tail -5'
```

### Extracting Final Results

**Get final validation score from completed run**:
```bash
# Method 1: Grep logs directly
docker logs phase3-nokl-baseline 2>&1 | grep -oP 'val/test_score["\']?\s*[=:]\s*\K[0-9.]+' | tail -1

# Method 2: Check saved log file in container
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'tail -100 /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_nokl_baseline/metakl_train.log | grep "val/test_score"'

# Method 3: Orchestrator extracts automatically
cat /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/campaign_results_summary_*.txt
```

### Checkpoint Directory Structure

**Browse checkpoints**:
```bash
# List checkpoint directories for a run
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'ls -lah /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_nokl_baseline/'

# Show latest checkpoint
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'ls -dt /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_nokl_baseline/*/ | head -1'

# Check checkpoint manager cleaned up old steps
# (Only final checkpoint should exist if pruning worked)
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'find /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_nokl_baseline -name "*.pt" -type f | wc -l'
```

### Orchestrator Results

**View campaign summary**:
```bash
# Main orchestrator log (updated every 60s while waiting, then per-run)
tail -100 /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/orchestrator_multiseed_v2_*.log

# Results summary (populated as runs complete)
cat /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/campaign_results_summary_*.txt

# Follow live updates
tail -f /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/orchestrator_multiseed_v2_*.log
```

---

## 6. BENCHMARK REFERENCE & COMPARISONS

### Prior Results (Apr 16-21, 2026)

**Fixed KL (β=0.001), 3 epochs**
- Dataset: 8,360 filtered samples
- Final validation: 31.25%
- Baseline for adaptive improvements

**LSTM Adaptive KL, 3 epochs** (Apr 21, 2026, 07:24:31)
- Dataset: 8,360 filtered samples (IDENTICAL)
- Epochs: 3 (IDENTICAL)
- Total steps: ~1566 (IDENTICAL to new 1,569)
- Final validation: **33.87%** ← REFERENCE FOR CAMPAIGN
- Improvement: +2.62 percentage points vs. fixed baseline

### Current Campaign (May 26, 2026)

**Benchmark Conditions**:

| Condition | KL Type | β | Status | ETA |
|-----------|---------|---|--------|-----|
| 5. No-KL | Fixed | 0.0 | RUNNING | May 27, ~11:00 |
| 1. Fixed | Fixed | 0.001 | Queued | After baseline |
| 2. Rule | Adaptive | Dynamic | Queued | After fixed |
| 3. MLP | Adaptive | Learned | Queued | After rule |
| 4. LSTM | Adaptive | Learned | Queued | After MLP |

**Expected Outcomes**:
- No-KL (β=0): Performance without KL penalty (lower bound)
- Fixed (β=0.001): ~31% (baseline replication check)
- Rule: ~32-33% (proportional feedback)
- MLP: ~33-34% (learned network)
- LSTM: ~33.87% (thesis method, should replicate Apr 21 result)

**Comparability**: All runs use:
- Identical dataset (8,360 samples)
- Identical batch size (16)
- Identical epochs (3 = 1,569 steps)
- Identical checkpoint pruning
- Identical seed/resume control (resume_mode=never)

---

## 7. COMMON TASKS & TROUBLESHOOTING

### Task: Stop a Running Experiment

```bash
# Find container
docker ps | grep phase3

# Stop gracefully
docker stop phase3-nokl-baseline

# Force remove if needed
docker rm -f phase3-nokl-baseline

# Stop orchestrator
pkill -f orchestrator_v2
```

### Task: Check GPU Memory Usage

```bash
# GPU stats on host
nvidia-smi

# Inside container during training
docker exec phase3-nokl-baseline nvidia-smi
```

### Task: Clean Old Checkpoints

```bash
# List all checkpoint dirs
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'find /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive -type d -name "global_step_*" | sort'

# Remove ALL checkpoints for fresh start (careful!)
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'rm -rf /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/*'
```

### Task: Verify Dataset Consistency

```bash
# Check current dataset size in running container
docker logs phase3-nokl-baseline 2>&1 | grep 'filter dataset len'

# View data config
docker logs phase3-nokl-baseline 2>&1 | grep -A5 'train_files:'
```

### Troubleshooting: Container Won't Start

**Check image exists**:
```bash
docker images | grep simple-rl
```

**View startup errors**:
```bash
# Full launch output
docker logs phase3-nokl-baseline 2>&1 | head -100

# Python errors
docker logs phase3-nokl-baseline 2>&1 | grep -i error | head -20
```

**Verify volumes**:
```bash
docker volume ls | grep simplerl
```

### Troubleshooting: Training Hangs

```bash
# Check container is responsive
docker exec phase3-nokl-baseline nvidia-smi

# Check process inside container
docker exec phase3-nokl-baseline ps aux | head -20

# Check memory/disk
docker exec phase3-nokl-baseline df -h
docker exec phase3-nokl-baseline free -h
```

---

## 8. KEY HYPERPARAMETERS & TUNING

### Fixed KL Controller
```yaml
actor_rollout_ref.actor.kl_loss_coef: 0.001  # Adjust for stronger/weaker regularization
# 0.000: No KL penalty (no-KL baseline)
# 0.001: Default (this campaign)
# 0.005: Stronger KL constraint
# 0.01: Very strong KL constraint
```

### Rule-Based Controller
```yaml
actor_adaptive_kl.init_beta: 0.001            # Starting β
actor_adaptive_kl.min_beta: 0.00001           # Lower bound
actor_adaptive_kl.max_beta: 0.005             # Upper bound
actor_adaptive_kl.target_kl_loss: 0.08        # Target loss deviation
actor_adaptive_kl.up_gain: 1.08               # Multiplier when below target
actor_adaptive_kl.down_gain: 0.94             # Multiplier when above target
actor_adaptive_kl.update_interval: 5          # Update every N steps
actor_adaptive_kl.warmup_steps: 150           # Warmup before adaptation
```

### Training Budget
```yaml
trainer.total_epochs: 3                       # Primary control
trainer.total_training_steps: -1              # -1 = use epochs only
# For 3 epochs: 523 steps/epoch × 3 = 1,569 total steps
# At ~89s/step: ~20-22 hours per run
```

### Rollout Settings (affects memory/speed)
```yaml
actor_rollout_ref.rollout.n: 4                # Samples per prompt
actor_rollout_ref.rollout.gpu_memory_utilization: 0.5  # vLLM memory target
actor_rollout_ref.rollout.max_num_seqs: 32   # vLLM batch size
actor_rollout_ref.rollout.max_num_batched_tokens: 4096  # Token batching
```

---

## 9. USEFUL COMMANDS REFERENCE

### Quick Status Checks
```bash
# All training containers
docker ps --filter label=simplerl.role=training

# Orchestrator status
pgrep -a -f orchestrator_v2

# Last 5 steps from current run
docker logs $(docker ps -q --filter label=simplerl.role=training) 2>&1 | grep 'step:' | tail -5

# Latest validation scores
docker logs $(docker ps -q --filter label=simplerl.role=training) 2>&1 | grep 'val/test_score' | tail -3
```

### Metrics Extraction
```bash
# Count training steps completed
docker logs phase3-nokl-baseline 2>&1 | grep -c 'step:'

# Average step duration
docker logs phase3-nokl-baseline 2>&1 | grep 'timing_s/step' | awk -F':' '{sum+=$NF; count++} END {print sum/count " s/step"}'

# Final KL loss value
docker logs phase3-nokl-baseline 2>&1 | grep 'actor/kl_loss' | tail -1

# Reward statistics
docker logs phase3-nokl-baseline 2>&1 | grep 'critic/rewards' | tail -3
```

### Log Inspection
```bash
# View entire training log
docker logs phase3-nokl-baseline > /tmp/nokl_full_log.txt

# Search for errors
docker logs phase3-nokl-baseline 2>&1 | grep -i 'error\|warning\|exception' | head -20

# Find all validation events
docker logs phase3-nokl-baseline 2>&1 | grep 'test_score'
```

### Checkpoint Inspection
```bash
# List all saved checkpoints
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'ls -lh /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_nokl_baseline/global_step_*/pytorch_model.bin'

# Verify checkpoint pruning worked (only latest saved)
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'find /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_nokl_baseline -name "pytorch_model.bin" | wc -l'
```

---

## 10. ENVIRONMENT SETUP (For Future Runs)

### Python Environment
```bash
# From workspace root
cd /home/ai-server-02/R_projects/final_thesis/simpleRL-reason
pip install -e . --no-deps
pip install word2number antlr4-python3-runtime==4.9.3 math-verify==0.6.0
```

### Docker Volumes (Pre-created, persistent)
```bash
# Training data
docker volume inspect simplerl_data
# → /var/lib/docker/volumes/simplerl_data/_data

# Checkpoints
docker volume inspect simplerl_ckpts
# → /var/lib/docker/volumes/simplerl_ckpts/_data

# HuggingFace cache
docker volume inspect simplerl_hf_cache
# → /var/lib/docker/volumes/simplerl_hf_cache/_data
```

### Data Files
```bash
# Dataset location in container
/data/simplelr_qwen_level3to5/train.parquet   (8,360 samples)
/data/simplelr_qwen_level3to5/test.parquet    (496 samples)

# Models (cached on first pull)
Qwen/Qwen2.5-0.5B-Instruct                    (500M params)
```

### Telegram Setup (Optional)
```bash
# Get bot token from @BotFather on Telegram
# Get chat ID by: /getid command in chat with IDBot

# Set environment variables before running
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Verify credentials
echo "Token: $TELEGRAM_BOT_TOKEN"
echo "Chat: $TELEGRAM_CHAT_ID"

# Run orchestrator with Telegram
TELEGRAM_BOT_TOKEN="..." TELEGRAM_CHAT_ID="..." \
  nohup bash ./run_multiseed_orchestrator_v2.sh > orchestrator.log 2>&1 &
```

---

## 11. FILES & DIRECTORIES MAP

### Core Training
```
simpleRL-reason/
├── scripts/
│   ├── run_metakl_training_detached.sh    ← Launcher (MODIFIED)
│   ├── telegram_metakl_report.py          ← Telegram reporter
│   └── send_metakl_telegram_report.sh     ← Telegram CLI
├── verl/
│   ├── trainer/
│   │   ├── ppo/ray_trainer.py             ← Main trainer loop
│   │   └── config/
│   │       └── simplelr_grpo_qwen05_single_gpu.yaml  ← Base config
│   └── utils/
│       └── checkpoint/checkpoint_manager.py   ← Pruning (PATCHED)
└── analysis_logs/
    ├── orchestrator_multiseed_v2_*.log   ← Orchestrator log
    ├── campaign_results_summary_*.txt    ← Results file
    └── ruth-training-run.log             ← Old reference
```

### Orchestration
```
Root: /home/ai-server-02/R_projects/final_thesis/
├── run_multiseed_orchestrator_v2.sh       ← Sequential orchestrator (NEW)
├── orchestrator_v2_startup.log            ← Startup output
└── (working directory for launches)
```

### Outputs
```
Docker volumes (persistent):
simplerl_ckpts/
└── simplelr_grpo_qwen05_ctx1024_adaptive/
    ├── phase3_nokl_baseline/
    │   └── metakl_train.log               ← Training log in container
    ├── phase3_fixed_s0/
    ├── phase3_fixed_s1/
    ├── phase3_fixed_s2/
    ├── phase3_rule_s0/
    ├── phase3_rule_s1/
    ├── phase3_rule_s2/
    ├── phase3_mlp_s0/
    ├── phase3_mlp_s1/
    ├── phase3_mlp_s2/
    ├── phase3_lstm_s0/
    ├── phase3_lstm_s1/
    └── phase3_lstm_s2/
```

---

## 12. NEXT STEPS

### Immediate (Next 20-24 hours)
1. ✅ Monitor no-KL baseline completion
2. ✅ Orchestrator automatically starts fixed mode (3 seeds)
3. ✅ Collect metrics for all runs

### After Campaign Completes
1. Compare all 5 conditions (no-KL, fixed, rule, mlp, lstm)
2. Extract final validation scores for each seed
3. Compute mean ± std for each condition
4. Compare to April 21 LSTM reference (33.87%)
5. Generate comparison tables/figures
6. Document which controller mode performed best

### Analysis
```bash
# When all runs complete, generate summary
cat /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/campaign_results_summary_*.txt

# Example expected format:
# 1. Fixed (s0): 0.3125
# 2. Fixed (s1): 0.3128
# 3. Fixed (s2): 0.3132
# 4. Rule (s0): 0.3215
# ...
# 13. LSTM (s2): 0.3387
```

---

## Quick Reference Card

**Start Orchestrator**:
```bash
cd /home/ai-server-02/R_projects/final_thesis
nohup bash ./run_multiseed_orchestrator_v2.sh > orchestrator_v2.log 2>&1 &
```

**Monitor Status**:
```bash
tail -f /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/orchestrator_multiseed_v2_*.log
```

**View Results**:
```bash
cat /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/campaign_results_summary_*.txt
```

**Stop Everything**:
```bash
pkill -f orchestrator_v2
docker stop $(docker ps -q --filter label=simplerl.role=training)
```

**Single Run (Manual)**:
```bash
cd /home/ai-server-02/R_projects/final_thesis
RUTH_MODE=fixed RUTH_FIXED_BETA=0.001 RUTH_TOTAL_EPOCHS=3 \
RUTH_CONTAINER_NAME=phase3-test RUTH_RUN_DIR=/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/test \
./simpleRL-reason/scripts/run_metakl_training_detached.sh
```

---

**Documentation Generated**: May 26, 2026, 15:10 UTC  
**Campaign Status**: ACTIVE (1 of 13 runs, baseline in progress)  
**Last Updated**: May 26, 2026, 15:10 UTC
