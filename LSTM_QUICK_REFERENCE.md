# LSTM Training - Quick Reference Card

## 🚀 START TRAINING
```bash
cd /home/ai-server-02/R_projects/final_thesis
./lstm_training.sh start
```

---

## 📊 MONITOR TRAINING

| Command | Purpose |
|---------|---------|
| `./lstm_training.sh logs` | Watch live training logs |
| `./lstm_training.sh status` | Check current status |
| `./lstm_training.sh metrics` | Latest training metrics |
| `./lstm_training.sh gpu` | GPU usage |
| `./lstm_training.sh checkpoints` | List saved checkpoints |

---

## ⚙️ CONFIGURATION

**LSTM Architecture:**
- Hidden Dimension: 32
- Learning Rate: 5e-6
- Reset on Epoch: YES

**Training Setup:**
- Total Epochs: 3
- Total Steps: ~1566
- Save Checkpoint Every: 500 steps
- Auto-Delete Old Checkpoints: YES

**Expected Timeline:** ~3 hours (single GPU)

---

## 🎯 KEY METRICS TO MONITOR

| Metric | Good Range | Meaning |
|--------|-----------|---------|
| `step` | 0→1566 | Training progress |
| `actor/kl_loss` | 0.003-0.008 | KL divergence |
| `critic/score/mean` | 0.32-0.36 | Accuracy/reward |
| `actor/kl_coef` | 0.0001-0.01 | LSTM output |

---

## 🛑 STOP TRAINING

| Command | Effect |
|---------|--------|
| `./lstm_training.sh stop` | Graceful stop (30 sec wait) |
| `./lstm_training.sh kill` | Force stop immediately |
| `./lstm_training.sh clean` | Remove container |

---

## ✅ WHAT'S FIXED

- ✅ Telegram now reports correct step numbers
- ✅ Auto-deletes old checkpoints (saves space)
- ✅ Simple command interface created
- ✅ LSTM config documented

---

## 📁 IMPORTANT FILES

| File | Purpose |
|------|---------|
| `lstm_training.sh` | Main control script |
| `LSTM_TRAINING_COMMANDS.md` | Detailed commands |
| `LSTM_TRAINING_GUIDE.md` | Complete guide |
| `LSTM_CHANGES_SUMMARY.md` | What was fixed |

---

## 🆘 COMMON ISSUES

**"Container not found"**
```bash
./lstm_training.sh start
```

**"Out of memory"**
- Edit `RUTH_ROLLOUT_N=2` (reduce from 4)

**"GPU not available"**
```bash
nvidia-smi
```

**"Training stuck"**
```bash
./lstm_training.sh logs 100 | grep error
```

---

## 📍 CHECKPOINT LOCATIONS

**Saved Checkpoints:**
```
/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training/
├── global_step_500/
├── global_step_1000/
└── latest_checkpointed_iteration.txt
```

**Training Logs:**
```
Same directory + metakl_train.log
```

**Final Results:**
```
results/lstm_based_results.json
```

---

## ⏱️ EXPECTED TIMELINE

| Time | Progress |
|------|----------|
| 0-5 min | Setup |
| 5-15 min | Step 0→500 (1st checkpoint) |
| 15-30 min | Step 500→1000 |
| 30-45 min | Step 1000→1500 (2nd checkpoint) |
| 45-60 min | Step 1500→1566 |
| 60-67 min | Validation & results |

---

## 🎯 ONE-LINER COMMANDS

```bash
# Start and watch immediately
./lstm_training.sh start && sleep 5 && ./lstm_training.sh logs

# Check status every minute
while true; do ./lstm_training.sh status; sleep 60; done

# Get latest step
docker logs lstm-training-live 2>&1 | grep -o 'step: *[0-9]*' | tail -1

# List all checkpoints
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'ls -1d /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training/global_step_*'
```

---

## 📞 GET HELP

```bash
./lstm_training.sh help
```

Read full documentation:
- `LSTM_TRAINING_GUIDE.md` - Complete setup guide
- `LSTM_TRAINING_COMMANDS.md` - All available commands
- `LSTM_CHANGES_SUMMARY.md` - What was fixed

---

**Everything is ready. Just run:**
```
./lstm_training.sh start
```

Timestamp: August 14, 2026 | Status: ✅ Production Ready
