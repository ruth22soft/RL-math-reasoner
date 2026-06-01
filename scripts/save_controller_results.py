#!/usr/bin/env python3
"""
Single-Seed Controller Result Saver

Saves training results from a completed run to JSON format.
Called after each controller finishes training.

Usage:
    python save_controller_results.py \
        --controller_name "fixed" \
        --seed 42 \
        --run_dir "/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_fixed_seed42" \
        --final_score 0.3456 \
        --training_time 1234 \
        --output_dir "/path/to/results"
"""

import json
import argparse
import os
from datetime import datetime
from pathlib import Path


def save_results(controller_name, seed, run_dir, final_score, training_time, output_dir):
    """
    Save training results to JSON file.
    
    Args:
        controller_name: Name of the controller (e.g., 'fixed', 'mlp_based')
        seed: Random seed used (should be same as GLOBAL_SEED)
        run_dir: Path to checkpoint/results directory
        final_score: Final validation accuracy
        training_time: Total training time in seconds
        output_dir: Directory to save results JSON
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f"{controller_name}_results.json")
    seed_value = None if seed in (None, "", "null", "none", "None") else int(seed)
    
    result_data = {
        "controller": controller_name,
        "seed": seed_value,
        "global_seed": seed_value,
        "epochs": 3,  # Always 3 epochs
        "final_test_score": float(final_score) if final_score != "N/A" else None,
        "training_time_seconds": int(training_time),
        "training_time_readable": f"{int(training_time // 60)}m {int(training_time % 60)}s",
        "completion_timestamp": datetime.utcnow().isoformat() + "Z",
        "checkpoint_dir": str(run_dir),
        "status": "completed"
    }
    
    # Write results with pretty formatting
    with open(results_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    print(f"  - Controller: {controller_name}")
    print(f"  - Seed: {seed}")
    print(f"  - Final Score: {final_score}")
    print(f"  - Time: {result_data['training_time_readable']}")
    
    return results_file


def main():
    parser = argparse.ArgumentParser(
        description="Save controller training results to JSON"
    )
    parser.add_argument("--controller_name", type=str, required=True,
                        help="Controller name (e.g., 'fixed', 'mlp_based')")
    parser.add_argument("--seed", type=str, default="null",
                        help="Random seed used, or null for unseeded runs")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to run directory with checkpoints")
    parser.add_argument("--final_score", type=str, default="N/A",
                        help="Final validation score")
    parser.add_argument("--training_time", type=int, required=True,
                        help="Total training time in seconds")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results JSON")
    
    args = parser.parse_args()
    
    seed_value = None
    if args.seed not in ("", "null", "none", "None"):
        seed_value = int(args.seed)

    save_results(
        controller_name=args.controller_name,
        seed=seed_value,
        run_dir=args.run_dir,
        final_score=args.final_score,
        training_time=args.training_time,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
