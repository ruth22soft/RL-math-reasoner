# Analysis Tools Summary & Quick Start

## 🎯 What You Now Have

Two powerful analysis tools for your GRPO campaign:

### Tool #1: `monitor_campaign_live.py` ✅ [READY NOW]
**For tracking progress DURING campaign execution**

```bash
# Watch campaign in real-time
python3 /home/ai-server-02/R_projects/final_thesis/monitor_campaign_live.py --watch
```

**Shows**:
- Orchestrator status (running/waiting)
- Currently active training container
- Real-time training metrics (steps, epochs, losses)
- Partial results as runs complete
- Auto-updates every 30 seconds

### Tool #2: `analyze_campaign_results.py` ✅ [READY AFTER CAMPAIGN]
**For comprehensive analysis AFTER all runs complete**

```bash
# Analyze completed campaign
python3 /home/ai-server-02/R_projects/final_thesis/analyze_campaign_results.py
```

**Generates**:
- Summary statistics table (mean, std, min, max per mode)
- Detailed per-run results
- Comparison vs. Apr 21 LSTM reference (33.87%)
- Markdown report for sharing
- JSON data for programmatic access
- Visualization plots (if matplotlib available)

---

## 📊 Current Status (May 26, 15:11 UTC)

```
ORCHESTRATOR:     ✓ Running
ACTIVE TRAINING:  phase3-nokl-baseline (5.1% complete, ~20 hours remaining)
MONITOR STATUS:   ✓ Working (just tested above)
ANALYSIS STATUS:  ✓ Ready (waiting for results)
```

---

## 🚀 How to Use

### RIGHT NOW (During Campaign)

```bash
cd /home/ai-server-02/R_projects/final_thesis

# Open one terminal for monitoring
python3 monitor_campaign_live.py --watch

# In another terminal, check detailed logs
tail -f simpleRL-reason/analysis_logs/orchestrator_multiseed_v2_*.log
```

**Monitor will show**:
```
ORCHESTRATOR STATUS:
  Running: ✓ YES
  Waiting for: phase3-nokl-baseline
  Progress: 1/13 runs (7.7%)

ACTIVE TRAINING:
  Container: phase3-nokl-baseline
  Progress: 80/1569 steps (5.1%)
  KL Loss: 0.000574
```

### WHEN CAMPAIGN FINISHES (~May 27 @ 11:00 UTC + 60 hours)

```bash
cd /home/ai-server-02/R_projects/final_thesis

# Run analysis (auto-finds latest results file)
python3 analyze_campaign_results.py

# This will generate:
# 1. campaign_analysis_report_*.md       (markdown, human-readable)
# 2. campaign_analysis_report_*.json     (JSON, programmatic)
# 3. campaign_results_plot.png           (visualization)

# View results
cat simpleRL-reason/analysis_logs/campaign_analysis_report_*.md

# Or view as JSON
cat simpleRL-reason/analysis_logs/campaign_analysis_report_*.json | python3 -m json.tool
```

---

## 📋 Expected Output Example

When you run the analysis script, you'll get something like:

```markdown
# Campaign Results Analysis Report
Generated: 2026-05-27 21:45:32

## Summary Statistics

| Mode | Mean | Std | Min | Max | Runs | vs. Reference |
|------|------|-----|-----|-----|------|---------------|
| nokl | 0.2956 | 0.0000 | 0.2956 | 0.2956 | 1 | -12.77% |
| fixed | 0.3125 | 0.0012 | 0.3108 | 0.3142 | 3 | -7.75% |
| rule | 0.3245 | 0.0018 | 0.3223 | 0.3265 | 3 | -4.19% |
| mlp | 0.3321 | 0.0025 | 0.3290 | 0.3347 | 3 | -1.95% |
| lstm | 0.3387 | 0.0008 | 0.3378 | 0.3395 | 3 | +0.00% |

## Key Findings

**Best Performing Mode**: LSTM
- Mean Score: 0.3387
- Improvement vs. Reference: +0.00%

**Consistency**: Fixed > LSTM > Rule > MLP > NoKL
**Winner**: LSTM (thesis method successfully replicated!)
```

---

## 📚 Documentation Files

All created and ready to use:

| File | Purpose | Size | Status |
|------|---------|------|--------|
| `CAMPAIGN_DOCUMENTATION.md` | Complete reference guide | 27 KB | ✅ Ready |
| `ANALYSIS_SCRIPTS_GUIDE.md` | Analysis tools manual | 10 KB | ✅ Ready |
| `analyze_campaign_results.py` | Full analysis script | 20 KB | ✅ Executable |
| `monitor_campaign_live.py` | Live monitoring script | 10 KB | ✅ Executable |
| `run_multiseed_orchestrator_v2.sh` | Campaign orchestrator | 7 KB | ✅ Running |

---

## 🎬 Next Steps Timeline

### Now (T+0 hours, May 26 15:11)
- ✅ Orchestrator running
- ✅ phase3-nokl-baseline executing (5.1% done)
- ✅ Monitor scripts ready

### T+20 hours (May 27 ~11:00 UTC)
- No-KL baseline completes
- Orchestrator starts fixed mode runs (phase3-fixed-s0, s1, s2)
- Use monitor script to track progress

### T+40 hours (May 27 ~13:00 UTC next day)
- Fixed mode completes
- Orchestrator starts rule mode runs

### T+60 hours (May 27 ~15:00 UTC)
- Rule mode completes
- MLP mode starts

### T+80 hours (May 28 ~11:00 UTC)
- MLP mode completes
- LSTM mode starts

### T+100 hours (May 28 ~13:00 UTC)
- ✅ ENTIRE CAMPAIGN COMPLETE
- Run analysis script
- Review results!

---

## 🔍 Detailed Usage Examples

### Example 1: Check Status Once
```bash
python3 /home/ai-server-02/R_projects/final_thesis/monitor_campaign_live.py
```

### Example 2: Continuous Monitoring
```bash
# Terminal 1: Monitor progress
python3 /home/ai-server-02/R_projects/final_thesis/monitor_campaign_live.py --watch

# Terminal 2: Watch detailed logs
tail -f /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/orchestrator_multiseed_v2_*.log
```

### Example 3: Analyze Results
```bash
# After campaign finishes
python3 /home/ai-server-02/R_projects/final_thesis/analyze_campaign_results.py

# View the generated report
cat /home/ai-server-02/R_projects/final_thesis/simpleRL-reason/analysis_logs/campaign_analysis_report_*.md

# Access data as JSON
python3 << 'EOF'
import json, glob
with open(glob.glob('*/campaign_analysis_report_*.json')[0]) as f:
    data = json.load(f)
    for mode in sorted(data['summary_stats']):
        m = data['summary_stats'][mode]
        print(f"{mode:10} -> mean={m['mean']:.4f} ± {m['std']:.4f}")
EOF
```

### Example 4: Programmatic Access
```python
import json

# Load results
with open('simpleRL-reason/analysis_logs/campaign_analysis_report_*.json') as f:
    results = json.load(f)

# Extract key info
ref_score = results['reference']['Apr21_LSTM']['score']
lstm_mean = results['summary_stats']['lstm']['mean']
improvement = (lstm_mean - ref_score) / ref_score * 100

print(f"LSTM Performance:")
print(f"  Score: {lstm_mean:.4f}")
print(f"  Reference: {ref_score:.4f}")
print(f"  Improvement: {improvement:+.2f}%")
```

---

## 📞 Help & Troubleshooting

**Monitor script shows "No results yet"?**
- Normal - results only appear as runs complete
- Check orchestrator log: `tail -f orchestrator_*.log`

**Analysis script can't find results file?**
- Campaign hasn't finished yet, or
- Manually specify: `python3 analyze_campaign_results.py /path/to/file`

**Want to stop everything?**
```bash
pkill -f orchestrator_v2
docker stop $(docker ps -q --filter label=simplerl.role=training)
```

**Want to see individual run logs?**
```bash
# While running
docker logs phase3-nokl-baseline

# After completed (from volume)
docker run --rm -v simplerl_ckpts:/ckpts simple-rl:ngc-vllm \
  bash -lc 'tail -100 /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_nokl_baseline/metakl_train.log'
```

---

## ✅ Verification Checklist

- [x] Both analysis scripts created
- [x] Both scripts are executable
- [x] Monitor script tested and working
- [x] Complete documentation provided
- [x] Usage guide created
- [x] Quick reference guide provided
- [x] Campaign currently running

**Ready for full multi-seed execution!** 🚀

---

**Created**: May 26, 2026 15:11 UTC  
**Campaign Status**: ACTIVE  
**Last Verified**: Monitor script tested, showing real-time status
