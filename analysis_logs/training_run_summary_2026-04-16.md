# Training Run Summary (Docker)

## Run Identity
- Container: `ruth-training-run`
- Image: `simple-rl:ngc-vllm`
- Container status: `exited`
- Exit code: `0`
- Started: `2026-04-16T15:30:14Z`
- Finished: `2026-04-17T15:31:40Z`
- Log timestamp window observed: `2026-04-16 15:32:07` to `2026-04-17 14:11:35`

## Log Locations
- Copied workspace log: `analysis_logs/ruth-training-run.log`
- Original in container volume: `/ckpts/simplelr_grpo_qwen05_ctx1024/run.log`

## Final Validation Metrics
- `val/test_score/simplelr_qwen = 0.3125`
- `val/test_correctness/simplelr_qwen = 0.3125`
- Final reported step line: `step:1566`

## Training Progress Snapshot
- Parsed step lines with `critic/score/mean`: `1565`
- First parsed step: `step:1`, `critic/score/mean:0.062`, `actor/kl_loss:0.000`
- Last parsed step: `step:1565`, `critic/score/mean:0.156`, `actor/kl_loss:0.124`
- Sampled checkpoints:
  - step 200: score `0.094`, kl `0.004`
  - step 400: score `0.125`, kl `0.023`
  - step 600: score `0.094`, kl `0.036`
  - step 800: score `0.188`, kl `0.014`
  - step 1000: score `0.125`, kl `0.096`
  - step 1200: score `0.188`, kl `0.086`
  - step 1400: score `0.125`, kl `0.117`
  - step 1565: score `0.156`, kl `0.124`

## Checkpointing
- Checkpoints found every 50 steps from `50` through `1550`
- Total checkpoint save steps detected: `31`

## Anomalies Observed
- `Error comparing` occurrences in grader logs: `9`
- `ValueError: Can't evaluate nan or zoo` occurrences: `6`
- These appeared in reward grading paths and did not terminate training (job completed with exit code 0).

## Commit Correlation (Repository: RL-math-reasoner, branch v1)
- Commit before run start:
  - `4668a1f` at `2026-04-16 13:42:47 +0000`
  - Message: `updates the training to have a 0.5B benchmark`
  - Files changed:
    - `docs/DOCKER_TRAINING_GUIDE.md`
    - `tests/e2e/arithmetic_sequence/rl/main_trainer.py`
    - `verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml`
    - `verl/utils/reward_score/hf_math_verify.py`
- Commit after run:
  - `a6bc856` at `2026-04-18 10:00:17 +0000`
  - Message: `update-2`
  - Files changed:
    - `docs/DOCKER_TRAINING_GUIDE.md`
    - `verl/trainer/config/simplelr_grpo_qwen05_single_gpu.yaml`

### Correlation Note
The run started after commit `4668a1f` and before commit `a6bc856`, so this result is attributable to code/config state including `4668a1f` changes and excluding `a6bc856` changes.
