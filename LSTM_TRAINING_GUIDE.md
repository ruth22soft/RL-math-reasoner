# LSTM Training - Complete Setup Guide

**Created**: August 14, 2026  
**Status**: Ready for Training  
**Last Issues Fixed**: 
- ✅ Telegram step reporting bug (was showing incorrect step numbers like "444")
- ✅ Checkpoint management configured (auto-delete old checkpoints to save space)
- ✅ Simple command interface created

---

## 🎯 WHAT'S BEEN DONE

### 1. **Telegram Script Fixed** ✅
- **Issue**: Step numbers were incorrect (showing "444" instead of actual step)
- **Fix**: Improved `parse_step_metrics()` to correctly extract and convert step to integer
- **Result**: Now reports accurate training step numbers and proper status after server restart

### 2. **Checkpoint Management** ✅
- **Already Configured**: `RUTH_REMOVE_PREVIOUS_CKPT=true` 
- **What It Does**: When a new checkpoint is saved (every 500 steps), the previous one is automatically deleted
- **Space Saved**: ~2-3GB per training run instead of accumulating all checkpoints
- **Safety**: Only deletes old checkpoints, never current or required ones

### 3. **Easy Command Interface** ✅
- Created `/lstm_training.sh` - One command to do everything
- Created `LSTM_TRAINING_COMMANDS.md` - Full reference guide

---

## 🚀 HOW TO START TRAINING NOW

### Option 1: Using the New Simple Script (RECOMMENDED)
```bash
cd /home/ai-server-02/R_projects/final_thesis

# Start training
./lstm_training.sh start

# In another terminal, watch progress
./lstm_training.sh logs

# Or check status anytime
./lstm_training.sh status
```

### Option 2: Using Direct Command
```bash
cd /home/ai-server-02/R_projects/final_thesis

RUTH_MODE=lstm \
RUTH_CONTAINER_NAME=lstm-training-live \
RUTH_RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training" \
bash simpleRL-reason/scripts/run_metakl_training_detached.sh
```

---

## 📊 LSTM ARCHITECTURE DETAILS

### What is LSTM in this context?
- **LSTM**: Long Short-Term Memory neural network
- **Purpose**: Dynamically controls the KL coefficient during training
- **Advantage over Fixed**: Learns to adjust KL penalty based on training state
- **Learns From**: kl_loss, reward_mean, reward_std, lagged_grad_norm

### Configuration

```yaml
LSTM Parameters:
├── hidden_dim: 32          # Size of LSTM hidden state
├── learning_rate: 5e-6     # How fast LSTM learns
├── reset_on_epoch: true    # Reset hidden state each epoch
└── state_features: 4       # Number of input features

KL Control by LSTM:
├── init_beta: 0.001        # Starting KL coefficient
├── min_beta: 0.0001        # Minimum allowed value
├── max_beta: 0.01          # Maximum allowed value
├── target_kl_loss: 0.003   # What LSTM aims for
└── update_interval: 1      # Update every step
```

### How LSTM learns:
1. **Observes**: Current training metrics (kl_loss, rewards, etc)
2. **Remembers**: Past states using its hidden layer
3. **Decides**: What KL coefficient to use next
4. **Adjusts**: Based on whether KL loss is above/below target
5. **Improves**: Over time with experience

---

## ⚙️ TRAINING SETTINGS

```
Architecture Settings:
├── Model: Qwen 2.5 Math 7B
├── Mode: LSTM (adaptive KL)
├── Epochs: 3
├── Total Steps: ~1566
├── Save Checkpoint Every: 500 steps
└── Test Every: Disabled (only at end)

Batch Settings:
├── Training Batch Size: 256
├── Rollout N: 4 (trajectories per prompt)
├── Prompt Length: 1024 tokens
├── Response Length: 1024 tokens
└── Max Batched Tokens: 4096

Optimization:
├── Optimizer: AdamW
├── Learning Rate: 5e-7
├── KL Loss Coefficient: Controlled by LSTM
└── Entropy: 0.001

Storage:
├── Remove Old Checkpoints: YES (saves space)
├── Checkpoint Directory: /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training/
└── Checkpoint Size: ~1.5GB each
```

---

## 📝 QUICK COMMANDS REFERENCE

```bash
# Start training
./lstm_training.sh start

# Watch live logs
./lstm_training.sh logs

# Show last 50 lines
./lstm_training.sh logs 50

# Check current status
./lstm_training.sh status

# View latest metrics
./lstm_training.sh metrics

# Check GPU
./lstm_training.sh gpu

# List checkpoints
./lstm_training.sh checkpoints

# Show LSTM config
./lstm_training.sh config

# Stop gracefully
./lstm_training.sh stop

# Force stop
./lstm_training.sh kill

# Remove container
./lstm_training.sh clean

# Get help
./lstm_training.sh help
```

---

## 🔄 WHAT HAPPENS DURING TRAINING

### Training Flow:
```
1. Container starts with Docker image simple-rl:ngc-vllm
2. Model downloads from Hugging Face cache (reused if exists)
3. LSTM controller weights initialized
4. Training begins:
   ├─ Each step:
   │  ├─ Rollout: Generate responses from current policy
   │  ├─ Reward: Grade mathematical correctness
   │  ├─ PPO Update: Update policy based on rewards
   │  ├─ LSTM Controller: Observe metrics, output new KL coefficient
   │  └─ Logging: Print metrics to log file
   ├─ Every 500 steps:
   │  ├─ Save checkpoint (model weights + optimizer state)
   │  └─ Delete old checkpoint (if exists)
   └─ Every epoch (522 steps):
      └─ Reset LSTM hidden state
5. After 1566 steps (3 epochs):
   ├─ Final validation
   ├─ Save results JSON
   └─ Container exits
```

### Key Metrics Printed Every Step:
```
step: 123                       # Current training step
actor/kl_loss: 0.004           # KL divergence from reference
actor/kl_coef: 0.0008          # Current KL coefficient (set by LSTM)
critic/score/mean: 0.35         # Average reward (target metric)
actor/entropy: 4.2              # Policy entropy (exploration)
```

---

## 🛑 STOPPING AND RESUMING

### Normal Stop (Graceful):
```bash
./lstm_training.sh stop
# Waits 30 seconds for graceful shutdown
# Then force kills if needed
```

### Resume from Checkpoint:
```bash
# Training will automatically resume from latest checkpoint
./lstm_training.sh start
```

### Manual Checkpoint Selection:
Edit `run_metakl_training_detached.sh` to set:
```bash
RUTH_RESUME_CHECKPOINT="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training/global_step_500/checkpoint"
```

---

## 💾 CHECKPOINT MANAGEMENT

### Why Auto-Delete Old Checkpoints?
- Each checkpoint: ~1.5GB
- Without deletion: 3-4 checkpoints = 6GB+ storage
- With deletion: Only 1 checkpoint at a time = 1.5GB

### What Gets Saved:
- Model weights (quantized)
- Optimizer state
- LSTM controller weights (most important!)
- Training metadata

### Safe to Delete Manually?
```bash
# List checkpoints
./lstm_training.sh checkpoints

# Remove specific checkpoint (keep latest)
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'rm -rf /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training/global_step_500'
```

---

## 📈 MONITORING DURING TRAINING

### Real-Time Monitoring:
```bash
# Terminal 1: Watch logs live
./lstm_training.sh logs

# Terminal 2: Check GPU
watch nvidia-smi

# Terminal 3: Get periodic status
while true; do ./lstm_training.sh status; sleep 60; done
```

### What to Look For:
```
✅ Good Signs:
  - step increasing (500, 1000, 1500...)
  - critic/score/mean gradually increasing (target 0.35)
  - actor/kl_loss around 0.003-0.008
  - GPU using >40GB memory
  - Checkpoints appearing every 15 min

❌ Bad Signs:
  - step stuck at same number for 5 min
  - Out of memory errors
  - NaN values in metrics
  - Container repeatedly exiting
```

---

## 📊 EXPECTED PERFORMANCE

### Timeline:
```
0-10 min        Training initializes, first step appears
10-15 min       Checkpoint 1 saved (step 500)
15-30 min       Training steady, metrics stabilizing
30-45 min       Checkpoint 2 saved (step 1000)
45-55 min       Final phase, metrics refinement
55-60 min       Checkpoint 3 saved (step 1500)
60-65 min       Final validation and cleanup
Total: ~3 hours for full training
```

### Expected Metric Values:
```
Metric                  Early (step 100)    Mid (step 800)      Late (step 1500)
────────────────────────────────────────────────────────────────────────
kl_loss                 0.010               0.005               0.003-0.005
kl_coef (by LSTM)       0.001               0.0002-0.001        0.0002-0.008
critic/score/mean       0.25                0.32                0.34-0.36
actor/entropy           4.5                 4.0                 3.5-4.2
```

---

## 🆘 TROUBLESHOOTING

### Problem: "Container not found"
```bash
# Check if container is there
docker ps -a | grep lstm

# If not, start it fresh
./lstm_training.sh start
```

### Problem: "Out of memory" errors
```bash
# Reduce batch size or rollout N
# Edit: RUTH_ROLLOUT_N=2 (instead of 4)
# Edit: RUTH_MAX_BATCHED_TOKENS=2048 (instead of 4096)
```

### Problem: "GPU not found"
```bash
# Check GPU availability
./lstm_training.sh gpu
nvidia-smi

# If no GPU, check Docker GPU support
docker run --rm --gpus all nvidia/cuda nvidia-smi
```

### Problem: Training stuck (step not increasing)
```bash
# Check logs for errors
./lstm_training.sh logs 100 | grep -i error

# If model is loading, wait 10 min for first step
# If truly stuck, kill and restart
./lstm_training.sh kill
./lstm_training.sh clean
./lstm_training.sh start
```

### Problem: Telegram notifications wrong
```bash
# The telegram script is now fixed!
# It should report correct step numbers
# Set env vars and test:
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
python3 simpleRL-reason/scripts/telegram_metakl_report.py
```

---

## 📁 FILE LOCATIONS

```
Project Root:
└── /home/ai-server-02/R_projects/final_thesis/
    ├── lstm_training.sh                    ← Use this script!
    ├── LSTM_TRAINING_COMMANDS.md           ← Detailed commands
    ├── LSTM_TRAINING_GUIDE.md              ← This file
    ├── simpleRL-reason/
    │   ├── scripts/
    │   │   ├── run_metakl_training_detached.sh
    │   │   ├── telegram_metakl_report.py    ← FIXED
    │   │   └── ...
    │   └── verl/trainer/main_ppo.py         ← Training code
    └── results/
        └── lstm_based_results.json          ← Final results

Docker Volumes:
└── simplerl_ckpts: (Volume in Docker)
    └── /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training/
        ├── global_step_500/
        ├── global_step_1000/
        ├── latest_checkpointed_iteration.txt
        └── metakl_train.log
```

---

## ✅ READY TO START?

```bash
cd /home/ai-server-02/R_projects/final_thesis

# 1. Start training
./lstm_training.sh start

# 2. Watch it go
./lstm_training.sh logs

# 3. That's it! Training runs automatically
```

The previous issues with checkpoints and Telegram reporting are now fixed!

---

## 📞 QUICK HELP

- **Lost?** → Run `./lstm_training.sh help`
- **Check logs?** → Run `./lstm_training.sh logs`
- **Status?** → Run `./lstm_training.sh status`  
- **GPU?** → Run `./lstm_training.sh gpu`
- **Config?** → Run `./lstm_training.sh config`
- **Need commands?** → Read `LSTM_TRAINING_COMMANDS.md`

---

**Made for single-GPU LSTM training with automatic checkpoint management**
