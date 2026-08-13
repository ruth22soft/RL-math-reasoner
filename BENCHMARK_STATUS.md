# Benchmark campaign status

All JSON results live in **`results/`** (project root).

| Controller   | Status      | Score    | Checkpoint / notes |
|-------------|-------------|----------|-------------------|
| zero_kl     | Done        | 0.3246   | `phase3_nokl_baseline` |
| fixed       | Done        | 0.324597 | `phase3_fixed_s2` |
| mlp_based   | Done        | 0.342742 | `phase3_mlp_s1` (ckpt pruned to latest) |
| rule_based  | **Running** | —        | Resuming from `runs/rule-based/seed_3/20260523_154450` @ step 400 |
| lstm_based  | Queued      | —        | Starts after rule_based exits |

## Operations

```bash
# Monitor pipeline (rule → lstm)
tail -f results/pipeline.log

# Monitor active training
docker logs -f phase3-rule_based-s1

# Re-run remaining benchmarks only
nohup bash scripts/run_rule_then_lstm.sh >> results/pipeline.log 2>&1 &
```

## Settings preserved

- Docker `--restart unless-stopped` (survives host reboot)
- `trainer.resume_mode=auto` (continues from latest `global_step_*`)
- `trainer.remove_previous_ckpt=true` (keeps disk usage bounded)

## Hang fix

`verl/trainer/main_ppo.py` now calls `ray.shutdown()` after training so the container exits cleanly and the next benchmark can start.
