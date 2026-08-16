# LSTM Training - Quick Command Reference

**Last Updated**: August 14, 2026
**Status**: Ready for LSTM Training

---

## 🚀 QUICK START - One Command to Start Training

```bash
cd /home/ai-server-02/R_projects/final_thesis

# Start LSTM training with automatic checkpoint management
RUTH_MODE=lstm \
RUTH_CONTAINER_NAME=lstm-training-live \
RUTH_RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training" \
RUTH_TOTAL_EPOCHS=3 \
RUTH_ROLLOUT_N=4 \
RUTH_SAVE_FREQ=500 \
RUTH_TEST_FREQ=-1 \
RUTH_REMOVE_PREVIOUS_CKPT=true \
bash simpleRL-reason/scripts/run_metakl_training_detached.sh
```

---

## 📊 MONITORING COMMANDS

### 1. **Watch Training Progress (LIVE)**
```bash
docker logs -f lstm-training-live
```

### 2. **Check Training Status (Quick Overview)**
```bash
docker inspect -f '{{.State.Status}}' lstm-training-live
```

### 3. **Check GPU Usage**
```bash
watch nvidia-smi
# Press Ctrl+C to exit
```

### 4. **View Latest Step Metrics**
```bash
docker logs lstm-training-live 2>&1 | tail -20
```

### 5. **Check Checkpoints Saved**
```bash
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'ls -lah /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training'
```

### 6. **Get Telegram Status Report** (if configured)
```bash
RUTH_CONTAINER_NAME=lstm-training-live \
RUTH_RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training" \
python3 simpleRL-reason/scripts/telegram_metakl_report.py
```

---

## ⚙️ LSTM CONFIGURATION

### Current Settings:
- **Mode**: LSTM-based adaptive KL control
- **Architecture**: LSTM with 32 hidden dimensions
- **Learning Rate**: 5e-6
- **Reset on Epoch**: Yes (resets LSTM state between epochs)
- **State Features**: kl_loss, reward_mean, reward_std, lagged_grad_norm
- **Total Epochs**: 3
- **Batch Size**: 256
- **Rollout N**: 4
- **Save Frequency**: 500 steps
- **Remove Old Checkpoints**: Yes (saves disk space)

### LSTM Hyperparameters:
```yaml
actor_adaptive_kl:
  mode: lstm
  enable: true
  lstm_hidden_dim: 32         # LSTM hidden state size
  lstm_lr: 5e-06              # LSTM learning rate
  lstm_reset_on_epoch: true   # Reset LSTM between epochs
  beta_output_activation: sigmoid_scaled
  target_kl_loss: 0.003       # Target KL divergence
  init_beta: 0.001            # Initial KL coefficient
  min_beta: 0.0001            # Minimum KL coefficient
  max_beta: 0.01              # Maximum KL coefficient
  warmup_steps: 1
  update_interval: 1
  ema_alpha: 0.75
```

---

## 🔄 CHECKPOINT MANAGEMENT

### Understanding Checkpoints:
- **Saved at**: Every 500 steps (configurable via `RUTH_SAVE_FREQ`)
- **Automatic Cleanup**: Old checkpoints deleted when new one saved (enabled)
- **Checkpoint Format**: `global_step_XXXX/` directories
- **Latest Tracker**: `latest_checkpointed_iteration.txt`

### View Recent Checkpoints:
```bash
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'ls -1d /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training/global_step_* | tail -5'
```

### Resume from Checkpoint:
```bash
# Find the step number you want to resume from
CHECKPOINT_STEP=500

RUTH_MODE=lstm \
RUTH_CONTAINER_NAME=lstm-training-resume \
RUTH_RUN_DIR="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training" \
RUTH_RESUME_MODE=auto \
bash simpleRL-reason/scripts/run_metakl_training_detached.sh
```

---

## 🛑 STOPPING & CLEANUP

### Stop Training Container:
```bash
docker stop lstm-training-live
```

### Clean Stop (let it finish gracefully):
```bash
docker stop -t 30 lstm-training-live  # Wait 30s before killing
```

### View Exit Status:
```bash
docker inspect -f '{{.State.ExitCode}}' lstm-training-live
```

### Remove Container After Stopping:
```bash
docker rm lstm-training-live
```

---

## 📈 UNDERSTANDING METRICS

### Key Metrics in Logs:
- **step**: Current training step
- **actor/kl_loss**: KL divergence between policy and reference
- **actor/kl_coef**: Current KL coefficient (dynamic, set by LSTM)
- **critic/score/mean**: Mean reward/accuracy (primary objective)
- **actor/entropy**: Policy entropy
- **actor/ratio**: Importance sampling ratio

### Expected Values:
```
kl_loss:        0.001 - 0.01  (lower is better)
kl_coef:        0.0001 - 0.01 (adaptive, set by LSTM)
score_mean:     0.3 - 0.5     (for math reasoning, target ~0.35)
entropy:        3.0 - 5.0     (encourages exploration)
ratio:          0.95 - 1.05   (should stay close to 1)
```

---

## 💾 OUTPUT LOCATIONS

### Checkpoints:
```
/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training/
├── global_step_500/
├── global_step_1000/
├── global_step_1500/
└── latest_checkpointed_iteration.txt
```

### Training Logs:
```
/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/lstm_training/
└── metakl_train.log
```

### Results Summary:
```
results/lstm_based_results.json
```

---

## 🔧 TROUBLESHOOTING

### Container Not Found:
```bash
docker ps -a | grep lstm-training
# Container should be in the list; if not, it crashed
```

### View Container Logs with Errors:
```bash
docker logs lstm-training-live 2>&1 | grep -i error | tail -20
```

### Check if GPU is Available:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

### Container Keeps Restarting:
```bash
# Check restart policy
docker inspect -f '{{.HostConfig.RestartPolicy}}' lstm-training-live
# If policy is `unless-stopped`, it will restart on crash
```

---

## 📋 TRAINING PHASES

### Expected Timeline (3 Epochs):
- **Total Steps**: ~1566 steps (522 per epoch)
- **Per Epoch Time**: ~45-60 minutes (single GPU)
- **Full Training**: ~2.5-3 hours
- **Checkpoint Frequency**: Every 500 steps (~15 min intervals)

### Progress Checkpoints:
```
Step 500   (~15 min)  - First checkpoint
Step 1000  (~30 min)  - Mid-training checkpoint  
Step 1500  (~45 min)  - Near completion
Step 1566  (~47 min)  - Training complete
```

---

## ✅ VERIFICATION CHECKLIST

After starting training, verify:

- [ ] Container is running: `docker ps | grep lstm-training-live`
- [ ] GPU is in use: `nvidia-smi` shows memory usage
- [ ] Logs are being produced: `docker logs lstm-training-live` shows output
- [ ] Metrics are appearing: Look for `step:` and `actor/` lines in logs
- [ ] Checkpoints are being saved: Check `/ckpts/` directory

---

## 🎯 COMMON TASKS

### See live step count:
```bash
docker logs lstm-training-live 2>&1 | grep -o 'step: *[0-9]*' | tail -1
```

### Get current KL coefficient:
```bash
docker logs lstm-training-live 2>&1 | grep 'kl_coef' | tail -1
```

### Get current accuracy:
```bash
docker logs lstm-training-live 2>&1 | grep 'score/mean' | tail -1
```

### Count total lines logged:
```bash
docker logs lstm-training-live 2>&1 | wc -l
```

---

## 📞 NEED HELP?

1. **Check logs first**: `docker logs lstm-training-live 2>&1 | tail -50`
2. **Look for errors**: `docker logs lstm-training-live 2>&1 | grep -i error`
3. **Check container status**: `docker inspect lstm-training-live`
4. **Verify GPU access**: `nvidia-smi`
