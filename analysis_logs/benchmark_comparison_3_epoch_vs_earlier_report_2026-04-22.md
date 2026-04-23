# 3-Epoch Benchmark Comparison Report (Updated)

Prepared by: GitHub Copilot  
Repository: RL-math-reasoner (branch: v1)  
Date: 2026-04-23

## Summary

This updated report includes the newer completed run in the Apr 20/21 window (started 2026-04-21 and finished 2026-04-22) and compares it against the two earlier results.

Across the three checkpoints, the latest run is the strongest:

- Older baseline: 29.5% (0.2950)
- Earlier 3-epoch fixed-KL benchmark: 31.25% (0.3125)
- Newer Apr21->Apr22 run: 33.8709677419% (0.3387096774)

## Sources Compared

- Earlier summary: `analysis_logs/training_run_summary_2026-04-16.md`
- Prior benchmark comparison: `analysis_logs/GRPO_Benchmark_Report_2026-04-18.md`
- New run evidence: raw container logs from `ruth-training-run` and container metadata from `docker inspect`

## New Run Verification (Apr21->Apr22)

Container identity and timing:

- Container: `ruth-training-run`
- StartedAt: `2026-04-21T07:21:58Z`
- FinishedAt: `2026-04-22T07:10:29Z`
- ExitCode: `0`

Final metrics from raw container log:

- `val/test_score/simplelr_qwen = 0.3387096774193548`
- `val/test_correctness/simplelr_qwen = 0.3387096774193548`
- Final line: `step:1566 - val/test_score/simplelr_qwen:0.338710 - val/test_correctness/simplelr_qwen:0.338710`

Workspace artifact note:

- Under `outputs/2026-04-20` and `outputs/2026-04-21`, only Hydra config artifacts were present and `main_ppo.log` files were empty for inspected run directories, so final metrics were verified from container logs.

## Three-Way Comparison

| Run | Validation score | Validation correctness |
| --- | ---: | ---: |
| Older baseline | 0.2950 | 29.50% |
| Earlier 3-epoch fixed-KL benchmark | 0.3125 | 31.25% |
| Newer Apr21->Apr22 run | 0.3387096774 | 33.8709677419% |

Pairwise deltas:

- Fixed-KL benchmark vs older baseline: +0.0175 score, +1.75 percentage points correctness, +5.93% relative
- New run vs fixed-KL benchmark: +0.0262096774 score, +2.620968 percentage points correctness, +8.387097% relative
- New run vs older baseline: +0.0437096774 score, +4.370968 percentage points correctness, +14.816840% relative

## Interpretation

The new Apr21->Apr22 result establishes a stronger 3-epoch reference than the previous fixed-KL benchmark. The improvement over both earlier points is not marginal: it is +2.62 points over the 31.25% run and +4.37 points over the 29.5% baseline.

Given the run completed cleanly (exit code 0) and reports final validation at the same terminal step format (`step:1566`), this result is suitable to use as the latest benchmark in thesis progress reporting.

## Thesis-Ready Takeaway

Use the following statement in the thesis draft:

> The newer completed run (Apr21->Apr22) reached 33.87% final validation correctness (0.3387), surpassing the earlier 3-epoch fixed-KL benchmark of 31.25% by 2.62 percentage points (8.39% relative), and the older 29.5% baseline by 4.37 points (14.82% relative).

## File Location

This updated report is saved at:

- `analysis_logs/benchmark_comparison_3_epoch_vs_earlier_report_2026-04-22.md`