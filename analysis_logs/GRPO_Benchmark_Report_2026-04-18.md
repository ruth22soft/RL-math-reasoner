# GRPO Benchmark Training Report

Prepared by: Ruth Tamiru  
Repository: RL-math-reasoner (branch: v1)  
Date: 2026-04-18

## Abstract
This report summarizes the latest completed GRPO training benchmark run executed in the Docker environment for the Qwen2.5-0.5B setup. The run completed successfully (exit code 0) and achieved final validation correctness of 31.25% on the configured validation benchmark, improving over the earlier baseline documented in thesis_progress_1.

## 1. Experiment Context

- Training framework: verl GRPO
- Execution environment: Docker container run
- Container name: ruth-training-run
- Config family: simplelr_grpo_qwen05_single_gpu
- Run log source: /ckpts/simplelr_grpo_qwen05_ctx1024/run.log (copied to workspace)
- Local copied log: analysis_logs/ruth-training-run.log

## 2. Verified Run Status

- Container status: exited
- Exit code: 0
- Start time (container): 2026-04-16T15:30:14Z
- End time (container): 2026-04-17T15:31:40Z
- Log timestamp span observed: 2026-04-16 15:32:07 to 2026-04-17 14:11:35

Conclusion: the benchmark run finished normally without process crash.

## 3. Final Benchmark Results

Final metrics from training log:

- val/test_score/simplelr_qwen = 0.3125
- val/test_correctness/simplelr_qwen = 0.3125

Accuracy interpretation:

- Final validation accuracy = 31.25%

## 4. Comparison to Prior Baseline

Prior baseline in thesis_progress_1:

- Post-training result reported there: 29.5%

Current benchmark:

- 31.25%

Absolute improvement over prior baseline:

- +1.75 percentage points

Relative improvement over prior baseline:

- (31.25 - 29.5) / 29.5 = 5.93%

## 5. Training Dynamics Summary

- Parsed step lines with actor metrics: 1565
- Final logged validation step index: 1566
- Checkpoint cadence detected: every 50 steps
- Checkpoint steps observed: 50 through 1550 (31 checkpoints)

Representative sampled dynamics:

- step 1: critic/score/mean=0.062, actor/kl_loss=0.000
- step 200: critic/score/mean=0.094, actor/kl_loss=0.004
- step 400: critic/score/mean=0.125, actor/kl_loss=0.023
- step 800: critic/score/mean=0.188, actor/kl_loss=0.014
- step 1000: critic/score/mean=0.125, actor/kl_loss=0.096
- step 1400: critic/score/mean=0.125, actor/kl_loss=0.117
- step 1565: critic/score/mean=0.156, actor/kl_loss=0.124

## 6. KL and Reward-Parsing Notes

Observed in log:

- actor/kl_coef remained constant at 0.001 across parsed steps
- actor/kl_loss varied dynamically during training (approximately 0.0 to 0.654)

Reward grading anomalies detected but non-fatal:

- "Error comparing" occurrences: 9
- "ValueError: Can't evaluate nan or zoo" occurrences: 6

These did not terminate training and did not prevent final validation logging.

## 7. Benchmark Interpretation

This run establishes a stronger fixed-KL benchmark than the prior report value, with measurable gains in final validation correctness. The result is suitable as the baseline reference for upcoming adaptive-KL experiments (Phase A and Phase B of MetaKLController).

## 8. Benchmark Values to Use in Thesis Draft (Current)

Use the following benchmark values in the next thesis update:

- Final validation correctness (current benchmark): 31.25%
- Final validation score (current benchmark): 0.3125
- Prior benchmark reference (thesis_progress_1): 29.5%
- Improvement over prior benchmark: +1.75 percentage points

## 9. Reproducibility Artifacts

- Main benchmark log: analysis_logs/ruth-training-run.log
- Previous summary: analysis_logs/training_run_summary_2026-04-16.md
- MetaKL plan docs:
  - analysis_logs/metaKL_implementation_proposal.md

## 10. Operational Pipeline Update (Power-Failure Safe)

To make future runs resilient and reproducible, the project now includes a fixed run pipeline:

- Startup script: scripts/run_metakl_training_detached.sh
- Telegram report scripts:
  - scripts/send_metakl_telegram_report.sh
  - scripts/telegram_metakl_report.py

Locked runtime parameters in the new startup script:

- trainer.total_epochs = 3
- data.max_prompt_length = 1024
- data.max_response_length = 1024
- actor_rollout_ref.rollout.max_num_batched_tokens = 4096
- adaptive KL = enabled (lstm)

Checkpoint/restart policy in the new startup script:

- trainer.save_freq = 50
- trainer.resume_mode = auto
- trainer.default_local_dir = /ckpts/simplelr_grpo_qwen05_ctx1024_adaptive
- container restart policy = unless-stopped

This ensures that if the machine loses power and comes back, Docker restarts the training container and the trainer resumes from latest_checkpointed_iteration.txt in the checkpoint directory.

## 11. Telegram Report Fields (Requested)

The Telegram report now includes:

- container status and start time
- current/latest training step
- accuracy proxy (critic/score/mean)
- KL metrics:
  - actor/kl_loss
  - actor/kl_coef_dynamic (adaptive)
  - actor/kl_coef (fixed fallback)
- checkpoint status:
  - latest_checkpointed_iteration
  - recent global_step_* checkpoints

Telegram delivery requires host environment variables:

- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

---
Report prepared from verified run artifacts in the current workspace and Docker execution records.
