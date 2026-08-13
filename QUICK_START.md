# Single-Seed Orchestrator — Quick Reference Card

## 🚀 ONE-COMMAND START

```bash
cd /home/ai-server-02/R_projects/final_thesis
./run_single_seed_orchestrator_v1.sh
```

**That's it!** Everything else runs automatically.

---

## 📊 What Will Run

```
✓ Zero KL (β=0)        — ~2-2.5h
✓ Rule-based           — ~2-2.5h
✓ MLP-based            — ~2.5-3h
✓ Fixed (β=0.001)      — ~2-2.5h
✓ LSTM-based           — ~3-3.5h
  ─────────────────────────────
  TOTAL                ~12-14h (single GPU)
```

---

## 📍 Monitoring Commands

**Main progress log:**
```bash
tail -f simpleRL-reason/analysis_logs/orchestrator_single_seed_*.log
```

**Current training (replace CONTROLLER_NAME):**
```bash
docker logs -f phase3-CONTROLLER_NAME-single-seed
docker logs -f phase3-zero_kl-single-seed      # Example
```

**GPU status:**
```bash
watch nvidia-smi
```

**List all outputs:**
```bash
ls -lah results/
docker volume inspect simplerl_ckpts
```

---

## 💾 Output Locations

**Results (final scores, times):**
```
results/
├── zero_kl_results.json        ← Final score here
├── rule_based_results.json
├── mlp_based_results.json
├── fixed_results.json
└── lstm_based_results.json
```

**Checkpoints (model weights):**
```
/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/
├── phase3_zero_kl_seed42/
├── phase3_rule_based_seed42/
├── phase3_mlp_based_seed42/
├── phase3_fixed_seed42/
└── phase3_lstm_based_seed42/
```

**Logs:**
```
simpleRL-reason/analysis_logs/
├── orchestrator_single_seed_20260528_*.log
└── campaign_summary_20260528_*.txt
```

---

## ⚙️ Configuration

**To change seed (default=42):**
Edit `run_single_seed_orchestrator_v1.sh` line 50:
```bash
GLOBAL_SEED=42  # ← Change this
```

**To skip some controllers:**
Edit `run_single_seed_orchestrator_v1.sh` line 59:
```bash
# CONTROLLERS=("zero_kl" "rule_based" "mlp_based" "fixed" "lstm_based")
CONTROLLERS=("mlp_based" "fixed" "lstm_based")  # Skip first two
```

**To disable Telegram notifications:**
```bash
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
```

---

## 🛑 Emergency Stop

**Kill current container:**
```bash
docker kill phase3-CONTROLLER_NAME-single-seed
```

**Kill all orchestrator containers:**
```bash
docker kill $(docker ps -q -f "label=simplerl.role=training")
```

**Force stop orchestrator script:**
```
Press Ctrl+C in terminal
```

---

## ⚠️ Critical Rules

🔴 **NEVER run multi-seed manually** — orchestrator handles it

🔴 **NEVER change GLOBAL_SEED mid-run** — it will abort

🔴 **NEVER interrupt seeds 2 or 3** — orchestrator auto-kills them

🔴 **NEVER run > 1 controller at a time** — they run sequentially

---

## ✅ Verification Checklist

Before running:
```bash
# 1. Scripts are executable
ls -la run_single_seed_orchestrator_v1.sh
ls -la simpleRL-reason/scripts/run_metakl_training_single_seed.sh

# 2. Docker image exists
docker images | grep simple-rl

# 3. Data volumes exist
docker volume ls | grep simplerl

# 4. Disk space OK (need ~50GB)
df -h /ckpts
```

After running:
```bash
# 1. Check results saved
ls -lah results/*_results.json

# 2. Check checkpoints saved
docker run --rm -v simplerl_ckpts:/ckpts alpine ls /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/ | grep phase3

# 3. Check no errors in log
grep -i error simpleRL-reason/analysis_logs/orchestrator_single_seed_*.log

# 4. Verify seed consistency
grep -i "seed" simpleRL-reason/analysis_logs/orchestrator_single_seed_*.log | head -5
```

---

## 📞 Troubleshooting

**Problem**: "Container failed to start"  
**Solution**: Check logs  
```bash
docker logs phase3-zero_kl-single-seed
# or check disk space:
df -h /ckpts
```

**Problem**: "Training stuck for hours"  
**Solution**: Check if actually running  
```bash
docker exec phase3-zero_kl-single-seed nvidia-smi
# or:
docker top phase3-zero_kl-single-seed
```

**Problem**: "Results not appearing"  
**Solution**: Check if run_dir is mounted  
```bash
docker run --rm -v simplerl_ckpts:/ckpts alpine ls -lah /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_zero_kl_seed42/
```

**Problem**: "Want to resume from controller N"  
**Solution**: Edit `run_single_seed_orchestrator_v1.sh` to skip earlier ones  
```bash
# Line 59: comment out controllers 0-N-1
```

---

## 📝 Key Files

| File | Purpose |
|------|---------|
| `run_single_seed_orchestrator_v1.sh` | Main orchestrator (YOU RUN THIS) |
| `simpleRL-reason/scripts/run_metakl_training_single_seed.sh` | Training launcher |
| `simpleRL-reason/scripts/save_controller_results.py` | Result saver |
| `SINGLE_SEED_ORCHESTRATOR_GUIDE.md` | Full documentation |
| `IMPLEMENTATION_SUMMARY.md` | Technical details |

---

## 🔐 Seed Guards Explained

The system has 4-layer seed validation to prevent wasting GPU hours:

1. **Orchestrator** checks no seed 2/3 containers exist before starting
2. **Training script** rejects mismatched seeds
3. **Main trainer** initializes all RNGs (torch, numpy, random) with same seed
4. **Actor** logs seed for debugging

If ANY layer detects a mismatch → **ABORTS LOUDLY** to save GPU hours

---

## ✨ What's New (vs Old Multi-Seed Approach)

| Aspect | Old (Multi-Seed) | New (Single-Seed) |
|--------|------------------|-------------------|
| Runs per controller | 3 (seeds 0,1,2) | 1 (seed 42 only) |
| Total GPU time | 36-42 hours | 12-14 hours |
| Result saving | End of all runs | After each controller |
| Interruption | Manual | Auto-kill seed 2/3 |
| Seed enforcement | None | 4-layer validation |

---

## 💡 Pro Tips

✨ Run in `screen` or `tmux` for detached execution:
```bash
screen -S orchestrator
./run_single_seed_orchestrator_v1.sh
# Ctrl+A then D to detach
screen -r orchestrator  # Reattach later
```

✨ Redirect logs to file:
```bash
nohup ./run_single_seed_orchestrator_v1.sh > orchestrator.out 2>&1 &
tail -f orchestrator.out
```

✨ Get Telegram notifications (if configured):
- Script automatically sends progress updates
- Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` env vars before running

---

**Status**: ✅ Ready to Run  
**Last Updated**: May 28, 2026  
**GPU Estimate**: 12-14 hours (RTX A4000)
