# Campaign Analysis Scripts - Usage Guide

This directory contains automated tools for analyzing multi-seed GRPO campaign results.

## Scripts Overview

### 1. `monitor_campaign_live.py` - Real-Time Campaign Monitor

Monitor campaign progress **while it's running** without waiting for completion.

**Features**:
- Shows orchestrator status (running/waiting/progress)
- Displays active training container and metrics
- Shows partial results as they complete
- Continuous monitoring mode with auto-refresh

**Usage**:

```bash
# Single snapshot of current status
python3 monitor_campaign_live.py

# Continuous monitoring (updates every 30s)
python3 monitor_campaign_live.py --watch

# Exit monitoring (Ctrl+C)
```

**Output Example**:
```
================================================================================
CAMPAIGN LIVE MONITOR
================================================================================
Time: 2026-05-27 12:34:56

ORCHESTRATOR STATUS:
  Running: ✓ YES
  Current run: phase3-fixed-s0
  Progress: 2/13 runs (15.4%)
  Latest: [2026-05-27 12:34:00] Waiting for phase3-nokl-baseline to complete...

ACTIVE TRAINING:
  Container: phase3-fixed-s0
  Status: Up 2 hours
  Progress: 847/1569 steps (54.0%)
  Epoch: 2/3
  KL Loss: 0.000312
  Reward (mean): 0.125400

COMPLETED RESULTS:
  nokl    : mean=0.2987, runs=1, scores=['0.2987']

================================================================================
```

### 2. `analyze_campaign_results.py` - Complete Results Analysis

Comprehensive analysis of **completed campaign** with statistics, comparisons, and visualizations.

**Features**:
- Parses campaign results summary file
- Computes mean, std, min, max per KL mode
- Generates markdown and JSON reports
- Creates comparison visualizations (if matplotlib available)
- Compares against Apr 21 LSTM reference (33.87%)

**Usage**:

```bash
# Analyze latest campaign results (auto-finds file)
python3 analyze_campaign_results.py

# Analyze specific results file
python3 analyze_campaign_results.py /path/to/campaign_results_summary_*.txt

# Output files generated:
# - campaign_analysis_report_YYYYMMDD_HHMMSS.md    (markdown report)
# - campaign_analysis_report_YYYYMMDD_HHMMSS.json  (JSON data)
# - campaign_results_plot.png                      (visualization, if matplotlib available)
```

**Output Files**:

1. **Markdown Report** (`campaign_analysis_report_*.md`)
   - Summary statistics table
   - Detailed per-run results
   - Interpretation of findings
   - Statistical summary

2. **JSON Report** (`campaign_analysis_report_*.json`)
   - Machine-readable format
   - All statistics and run details
   - Reference information
   - Timestamped

3. **Visualization** (`campaign_results_plot.png`)
   - Bar chart: Mean scores with error bars
   - Improvement vs. reference
   - Score distribution by mode
   - Summary statistics box

**Example Report Structure**:

```markdown
# Campaign Results Analysis Report
Generated: 2026-05-27 21:45:32

## Summary Statistics

### Results by KL Controller Mode

| Mode | Mean | Std | Min | Max | Runs | vs. Reference |
|------|------|-----|-----|-----|------|---------------|
| fixed | 0.3125 | 0.0012 | 0.3108 | 0.3142 | 3 | -7.75% |
| lstm | 0.3387 | 0.0008 | 0.3378 | 0.3395 | 3 | +0.00% |
| mlp | 0.3321 | 0.0025 | 0.3290 | 0.3347 | 3 | -1.95% |
| nokl | 0.2956 | 0.0000 | 0.2956 | 0.2956 | 1 | -12.77% |
| rule | 0.3245 | 0.0018 | 0.3223 | 0.3265 | 3 | -4.19% |

## Key Findings

**Best Performing Mode**: LSTM
- Mean Score: 0.3387
- Improvement vs. Reference: +0.00%
...
```

---

## Step-by-Step Examples

### During Campaign (Monitoring)

```bash
# Watch progress live
cd /home/ai-server-02/R_projects/final_thesis

# Option 1: Single status check
python3 monitor_campaign_live.py

# Option 2: Continuous monitoring (every 30 seconds)
python3 monitor_campaign_live.py --watch

# In another terminal, check orchestrator log
tail -f simpleRL-reason/analysis_logs/orchestrator_multiseed_v2_*.log
```

**Expected Progress**:
- T+0-25 hours: No-KL baseline (phase3-nokl-baseline)
- T+25-45 hours: Fixed β=0.001 seeds (phase3-fixed-s0, s1, s2)
- T+45-65 hours: Rule adaptive (phase3-rule-s0, s1, s2)
- T+65-85 hours: MLP adaptive (phase3-mlp-s0, s1, s2)
- T+85-105 hours: LSTM adaptive (phase3-lstm-s0, s1, s2)

### After Campaign Completes

```bash
cd /home/ai-server-02/R_projects/final_thesis

# Analyze results automatically (finds latest summary file)
python3 analyze_campaign_results.py

# View generated markdown report
cat simpleRL-reason/analysis_logs/campaign_analysis_report_*.md

# View visualization
open simpleRL-reason/analysis_logs/campaign_results_plot.png  # macOS
xdg-open simpleRL-reason/analysis_logs/campaign_results_plot.png  # Linux

# Access JSON data programmatically
python3 << 'EOF'
import json
with open('simpleRL-reason/analysis_logs/campaign_analysis_report_*.json') as f:
    results = json.load(f)
    print(f"Best mode: {max(results['summary_stats'], key=lambda m: results['summary_stats'][m]['mean'])}")
EOF
```

---

## Data Flow

```
Campaign Execution:
  └─ orchestrator_multiseed_v2.sh
      ├─ Launches 13 containers sequentially
      ├─ Collects final metrics from each
      └─ Saves to campaign_results_summary_YYYYMMDD_HHMMSS.txt

During Execution (Monitoring):
  └─ monitor_campaign_live.py --watch
      ├─ Reads orchestrator log
      ├─ Reads active container logs
      ├─ Reads partial summary file
      └─ Displays live status

After Execution (Analysis):
  └─ analyze_campaign_results.py
      ├─ Reads completed summary file
      ├─ Parses all run results
      ├─ Computes statistics
      ├─ Generates reports (MD, JSON)
      └─ Creates visualizations
```

---

## Prerequisites

### Required
- Python 3.6+
- Docker (for container queries)

### Optional (for full features)
- `numpy`: Statistical computations (falls back to pure Python)
- `matplotlib`: Visualization (skipped if unavailable)

**Install optional dependencies**:
```bash
pip install numpy matplotlib
```

### If running in container
Most dependencies are pre-installed. For standalone analysis on host:
```bash
python3 -m pip install numpy matplotlib
```

---

## Troubleshooting

### Monitor shows no active training
**Possible causes**:
- Orchestrator waiting for baseline to complete
- Container just finished and being cleaned up
- Check with: `docker ps --filter label=simplerl.role=training`

### Analysis script not finding summary file
**Solution**:
```bash
# Manually specify file
python3 analyze_campaign_results.py /path/to/campaign_results_summary_20260527_150408.txt

# Or check if file exists
ls -lah simpleRL-reason/analysis_logs/campaign_results_summary_*.txt
```

### Matplotlib not available for visualization
**Result**: Script will warn and skip plot generation  
**Solution**: `pip install matplotlib`

### Permission denied when running scripts
```bash
chmod +x monitor_campaign_live.py analyze_campaign_results.py
python3 monitor_campaign_live.py  # Use python3 explicitly
```

---

## Integration with Other Tools

### Use analysis results in scripts
```python
import json

# Load JSON report
with open('campaign_analysis_report_*.json') as f:
    data = json.load(f)

# Access statistics
stats = data['summary_stats']
lstm_mean = stats['lstm']['mean']
lstm_std = stats['lstm']['std']

# Check vs reference
ref = data['reference']['Apr21_LSTM']['score']
improvement = (lstm_mean - ref) / ref * 100
print(f"LSTM: {lstm_mean:.4f} ({improvement:+.2f}% vs reference)")
```

### Automated comparison
```bash
# Check if LSTM beat the reference
python3 << 'EOF'
import json
with open('simpleRL-reason/analysis_logs/campaign_analysis_report_*.json') as f:
    results = json.load(f)
    lstm_mean = results['summary_stats']['lstm']['mean']
    ref = results['reference']['Apr21_LSTM']['score']
    if lstm_mean > ref:
        print(f"✓ LSTM improved: {lstm_mean:.4f} > {ref:.4f}")
    else:
        print(f"✗ LSTM regressed: {lstm_mean:.4f} < {ref:.4f}")
EOF
```

---

## Command Reference Card

```bash
# MONITORING (During Campaign)
python3 monitor_campaign_live.py              # Single snapshot
python3 monitor_campaign_live.py --watch      # Continuous (Ctrl+C to stop)
tail -f orchestrator_multiseed_v2_*.log       # Raw orchestrator log
docker logs $(docker ps -q) 2>&1 | grep step  # Training metrics

# ANALYSIS (After Campaign)
python3 analyze_campaign_results.py           # Full analysis
cat campaign_analysis_report_*.md             # View markdown report
cat campaign_analysis_report_*.json | jq .   # View JSON (with pretty-print)

# CLEANUP
docker rm -f $(docker ps -q --filter label=simplerl.role=training)  # Stop all training
pkill -f orchestrator_v2                      # Stop orchestrator
```

---

## Files Generated

After running both monitoring and analysis:

```
simpleRL-reason/analysis_logs/
├── orchestrator_multiseed_v2_20260526_150408.log       # Orchestrator execution log
├── campaign_results_summary_20260527_200000.txt        # Results (auto-generated)
├── campaign_analysis_report_20260527_210000.md         # Analysis (generated by script)
├── campaign_analysis_report_20260527_210000.json       # Machine-readable (generated by script)
└── campaign_results_plot.png                           # Visualization (generated by script)
```

---

## Next Steps

1. **During campaign**:
   ```bash
   python3 monitor_campaign_live.py --watch
   ```

2. **After campaign completes**:
   ```bash
   python3 analyze_campaign_results.py
   ```

3. **Review results**:
   ```bash
   cat simpleRL-reason/analysis_logs/campaign_analysis_report_*.md
   ```

4. **Share findings**:
   ```bash
   # Share JSON report and plot with stakeholders
   cat simpleRL-reason/analysis_logs/campaign_analysis_report_*.json
   ```

---

**Last Updated**: May 26, 2026  
**Version**: 1.0  
**Status**: Ready for campaign execution
