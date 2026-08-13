#!/usr/bin/env python3
"""
Campaign Results Analysis Script
Analyzes multi-seed GRPO adaptive KL campaign results

Usage:
    python3 analyze_campaign_results.py [campaign_summary_file]
    
    If no file provided, will search for latest campaign_results_summary_*.txt
"""

import os
import sys
import glob
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import subprocess

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CampaignAnalyzer:
    """Analyzes GRPO campaign results and generates comparison reports"""
    
    def __init__(self, repo_dir="/home/ai-server-02/R_projects/final_thesis"):
        self.repo_dir = repo_dir
        self.analysis_dir = Path(repo_dir) / "simpleRL-reason" / "analysis_logs"
        self.ckpt_volume_base = "/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive"
        
        # Benchmark reference
        self.reference = {
            "Apr21_LSTM": {
                "score": 0.3387,
                "mode": "lstm",
                "date": "2026-04-21",
                "steps": 1566
            }
        }
        
        # Result storage
        self.results = defaultdict(list)  # {mode: [scores]}
        self.run_details = []  # Detailed per-run info
        
    def find_latest_summary(self) -> Optional[Path]:
        """Find latest campaign results summary file"""
        pattern = str(self.analysis_dir / "campaign_results_summary_*.txt")
        files = glob.glob(pattern)
        if files:
            return Path(sorted(files)[-1])
        return None
    
    def extract_run_metrics(self, container_name: str, run_dir: str) -> Dict:
        """Extract metrics from a completed run's logs"""
        metrics = {
            "container": container_name,
            "run_dir": run_dir,
            "val_score": None,
            "final_kl_loss": None,
            "final_reward": None,
            "total_steps": None,
            "duration_sec": None
        }
        
        # Try to get logs from running container or saved file
        try:
            # Method 1: Check docker logs
            try:
                result = subprocess.run(
                    ["docker", "logs", container_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                logs = result.stdout + result.stderr
            except:
                # Method 2: Check saved log file in checkpoint dir
                log_file = f"{self.ckpt_volume_base}/{run_dir.split('/')[-1]}/metakl_train.log"
                # Try to read from volume
                try:
                    result = subprocess.run(
                        ["docker", "run", "--rm", "-v", "simplerl_ckpts:/ckpts",
                         "simple-rl:ngc-vllm", "bash", "-lc", f"cat {log_file}"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    logs = result.stdout + result.stderr
                except:
                    logs = ""
            
            # Extract validation score (look for last occurrence)
            score_matches = re.findall(
                r"val[/']?test[/']?_?score['\"]?\s*[=:]\s*([0-9.]+)",
                logs,
                re.IGNORECASE
            )
            if score_matches:
                metrics["val_score"] = float(score_matches[-1])
            
            # Extract final KL loss
            kl_matches = re.findall(
                r"actor[/']?kl[/']?_?loss['\"]?\s*[=:]\s*([0-9.e+-]+)",
                logs,
                re.IGNORECASE
            )
            if kl_matches:
                metrics["final_kl_loss"] = float(kl_matches[-1])
            
            # Extract reward
            reward_matches = re.findall(
                r"critic[/']?rewards[/']?_?mean['\"]?\s*[=:]\s*([0-9.e+-]+)",
                logs,
                re.IGNORECASE
            )
            if reward_matches:
                metrics["final_reward"] = float(reward_matches[-1])
            
            # Count total steps
            step_matches = re.findall(r"step:(\d+)", logs)
            if step_matches:
                metrics["total_steps"] = max(int(s) for s in step_matches)
        
        except Exception as e:
            print(f"Warning: Could not extract metrics for {container_name}: {e}")
        
        return metrics
    
    def parse_summary_file(self, filepath: Path) -> bool:
        """Parse campaign results summary file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Extract result lines (format: "N. mode seed=S: score (time)")
            pattern = r"(\d+)\.\s+(\w+)\s+seed[=_](\d)\s*:\s*([0-9.]+|N/A)"
            
            for match in re.finditer(pattern, content):
                run_num, mode, seed, score = match.groups()
                seed = int(seed)
                
                try:
                    score = float(score)
                    self.results[mode].append(score)
                    self.run_details.append({
                        "run_num": int(run_num),
                        "mode": mode,
                        "seed": seed,
                        "score": score
                    })
                except ValueError:
                    print(f"Skipping invalid score: {score}")
            
            return len(self.run_details) > 0
        
        except Exception as e:
            print(f"Error parsing summary file: {e}")
            return False
    
    def compute_statistics(self) -> Dict:
        """Compute mean, std, min, max per mode"""
        stats = {}
        
        for mode, scores in self.results.items():
            if scores:
                scores_array = np.array(scores) if HAS_NUMPY else scores
                
                if HAS_NUMPY:
                    stats[mode] = {
                        "mean": float(np.mean(scores_array)),
                        "std": float(np.std(scores_array)),
                        "min": float(np.min(scores_array)),
                        "max": float(np.max(scores_array)),
                        "n": len(scores)
                    }
                else:
                    # Fallback without numpy
                    mean = sum(scores) / len(scores)
                    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
                    std = variance ** 0.5
                    stats[mode] = {
                        "mean": mean,
                        "std": std,
                        "min": min(scores),
                        "max": max(scores),
                        "n": len(scores)
                    }
        
        return stats
    
    def generate_markdown_report(self, stats: Dict, output_file: Optional[Path] = None) -> str:
        """Generate markdown comparison report"""
        lines = [
            "# Campaign Results Analysis Report\n",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
            "## Summary Statistics\n",
        ]
        
        # Mode comparison table
        lines.append("### Results by KL Controller Mode\n")
        lines.append("| Mode | Mean | Std | Min | Max | Runs | vs. Reference |\n")
        lines.append("|------|------|-----|-----|-----|------|----------------|\n")
        
        ref_score = self.reference["Apr21_LSTM"]["score"]
        
        for mode in sorted(stats.keys()):
            s = stats[mode]
            mean = s["mean"]
            std = s["std"]
            improvement = (mean - ref_score) / ref_score * 100
            
            if improvement > 0:
                improvement_str = f"+{improvement:.2f}%"
            else:
                improvement_str = f"{improvement:.2f}%"
            
            lines.append(
                f"| {mode:12} | {mean:.4f} | {std:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['n']:3d} | {improvement_str:>8} |\n"
            )
        
        lines.append("\n")
        
        # Reference comparison
        lines.append("### Reference Benchmark\n")
        lines.append(f"**Apr 21, 2026 (LSTM Adaptive)**: {ref_score:.4f} (33.87%)\n\n")
        
        # Detailed run breakdown
        lines.append("## Detailed Run Results\n")
        lines.append("| Run # | Mode | Seed | Score | Diff vs. Ref |\n")
        lines.append("|-------|------|------|-------|---------------|\n")
        
        for detail in sorted(self.run_details, key=lambda x: x["run_num"]):
            diff = (detail["score"] - ref_score) / ref_score * 100
            if diff > 0:
                diff_str = f"+{diff:.2f}%"
            else:
                diff_str = f"{diff:.2f}%"
            
            lines.append(
                f"| {detail['run_num']:3d} | {detail['mode']:12} | {detail['seed']} | {detail['score']:.4f} | {diff_str:>8} |\n"
            )
        
        lines.append("\n")
        
        # Interpretation
        lines.append("## Interpretation\n\n")
        
        best_mode = max(stats.keys(), key=lambda m: stats[m]["mean"])
        best_score = stats[best_mode]["mean"]
        best_improvement = (best_score - ref_score) / ref_score * 100
        
        lines.append(f"**Best Performing Mode**: {best_mode.upper()}\n")
        lines.append(f"- Mean Score: {best_score:.4f}\n")
        lines.append(f"- Improvement vs. Reference: {best_improvement:+.2f}%\n\n")
        
        lines.append("**Key Findings**:\n")
        for mode in sorted(stats.keys()):
            s = stats[mode]
            consistency = "High" if s["std"] < 0.01 else "Moderate" if s["std"] < 0.02 else "Low"
            lines.append(f"- **{mode.upper()}**: Mean={s['mean']:.4f}, Consistency={consistency}\n")
        
        lines.append("\n")
        
        # Statistical summary
        all_scores = [d["score"] for d in self.run_details]
        if HAS_NUMPY:
            overall_mean = np.mean(all_scores)
            overall_std = np.std(all_scores)
        else:
            overall_mean = sum(all_scores) / len(all_scores)
            overall_std = (sum((x - overall_mean) ** 2 for x in all_scores) / len(all_scores)) ** 0.5
        
        lines.append("## Overall Campaign Statistics\n\n")
        lines.append(f"- **Total Runs**: {len(all_scores)}\n")
        lines.append(f"- **Overall Mean**: {overall_mean:.4f}\n")
        lines.append(f"- **Overall Std**: {overall_std:.4f}\n")
        lines.append(f"- **Best Run**: {max(all_scores):.4f}\n")
        lines.append(f"- **Worst Run**: {min(all_scores):.4f}\n")
        
        report = "".join(lines)
        
        # Save if output file provided
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def generate_json_report(self, stats: Dict, output_file: Optional[Path] = None) -> str:
        """Generate JSON report for programmatic access"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary_stats": stats,
            "run_details": self.run_details,
            "reference": self.reference,
            "overall": {
                "total_runs": len(self.run_details),
                "modes": list(self.results.keys()),
                "date": "2026-05-26"
            }
        }
        
        json_str = json.dumps(report, indent=2)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_str)
            print(f"JSON report saved to: {output_file}")
        
        return json_str
    
    def generate_comparison_table(self, stats: Dict) -> str:
        """Generate simple text comparison table"""
        lines = [
            "\n" + "="*80,
            "CAMPAIGN RESULTS - COMPARISON TABLE",
            "="*80 + "\n"
        ]
        
        ref_score = self.reference["Apr21_LSTM"]["score"]
        
        # Header
        lines.append(f"{'Mode':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Runs':>6} {'vs. Ref':>10}")
        lines.append("-" * 80)
        
        # Data rows
        for mode in sorted(stats.keys()):
            s = stats[mode]
            mean = s["mean"]
            improvement = (mean - ref_score) / ref_score * 100
            
            if improvement > 0:
                imp_str = f"+{improvement:.2f}%"
            else:
                imp_str = f"{improvement:.2f}%"
            
            lines.append(
                f"{mode:<15} {mean:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f} {s['n']:>6d} {imp_str:>10}"
            )
        
        lines.append("-" * 80)
        lines.append(f"{'Reference':<15} {ref_score:>10.4f}  (Apr 21, 2026 LSTM Adaptive)")
        lines.append("=" * 80 + "\n")
        
        return "\n".join(lines)
    
    def generate_visualization(self, stats: Dict, output_file: Optional[Path] = None) -> bool:
        """Generate comparison plots if matplotlib available"""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping visualization")
            return False
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('GRPO Adaptive KL Campaign Results', fontsize=16, fontweight='bold')
            
            modes = sorted(stats.keys())
            ref_score = self.reference["Apr21_LSTM"]["score"]
            
            # Plot 1: Mean scores with error bars
            ax = axes[0, 0]
            means = [stats[m]["mean"] for m in modes]
            stds = [stats[m]["std"] for m in modes]
            colors = plt.cm.Set2(np.linspace(0, 1, len(modes)))
            
            bars = ax.bar(range(len(modes)), means, yerr=stds, capsize=5, color=colors, alpha=0.7)
            ax.axhline(ref_score, color='red', linestyle='--', linewidth=2, label=f'Reference: {ref_score:.4f}')
            ax.set_ylabel('Validation Score')
            ax.set_title('Mean Validation Score by Mode')
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, rotation=45)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Plot 2: Improvement vs reference
            ax = axes[0, 1]
            improvements = [(stats[m]["mean"] - ref_score) / ref_score * 100 for m in modes]
            colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
            
            ax.bar(range(len(modes)), improvements, color=colors_imp, alpha=0.7)
            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.set_ylabel('Improvement (%)')
            ax.set_title('% Improvement vs. Reference (Apr 21 LSTM: 33.87%)')
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Plot 3: Score distribution (scatter)
            ax = axes[1, 0]
            for i, mode in enumerate(modes):
                scores = self.results[mode]
                x = [i + np.random.normal(0, 0.02) for _ in scores]
                ax.scatter(x, scores, alpha=0.6, s=100, label=mode)
            
            ax.axhline(ref_score, color='red', linestyle='--', linewidth=2, label='Reference')
            ax.set_ylabel('Validation Score')
            ax.set_title('Score Distribution by Mode (with jitter)')
            ax.set_xticks(range(len(modes)))
            ax.set_xticklabels(modes, rotation=45)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Plot 4: Statistics summary
            ax = axes[1, 1]
            ax.axis('off')
            
            summary_text = "SUMMARY STATISTICS\n" + "="*40 + "\n\n"
            for mode in modes:
                s = stats[mode]
                summary_text += f"{mode.upper()}\n"
                summary_text += f"  Mean:  {s['mean']:.4f}\n"
                summary_text += f"  Std:   {s['std']:.4f}\n"
                summary_text += f"  Range: [{s['min']:.4f}, {s['max']:.4f}]\n"
                summary_text += f"  Runs:  {s['n']}\n\n"
            
            summary_text += f"Reference (Apr 21):\n  {ref_score:.4f}"
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                   fontfamily='monospace', fontsize=9, verticalalignment='top')
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to: {output_file}")
            else:
                plt.savefig(self.analysis_dir / "campaign_results_plot.png", dpi=150, bbox_inches='tight')
                print(f"Visualization saved to: {self.analysis_dir / 'campaign_results_plot.png'}")
            
            plt.close()
            return True
        
        except Exception as e:
            print(f"Error generating visualization: {e}")
            return False
    
    def run_analysis(self, summary_file: Optional[Path] = None) -> bool:
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("CAMPAIGN RESULTS ANALYSIS")
        print("="*80 + "\n")
        
        # Find/use summary file
        if not summary_file:
            summary_file = self.find_latest_summary()
            if not summary_file:
                print("ERROR: No campaign results summary file found!")
                return False
        
        print(f"Analyzing: {summary_file}")
        print(f"File size: {summary_file.stat().st_size} bytes\n")
        
        # Parse results
        if not self.parse_summary_file(summary_file):
            print("ERROR: Could not parse summary file!")
            return False
        
        print(f"Parsed {len(self.run_details)} runs across {len(self.results)} modes\n")
        
        # Compute statistics
        stats = self.compute_statistics()
        
        # Generate reports
        print("Generating reports...\n")
        
        # Markdown report
        md_report = self.generate_markdown_report(stats)
        md_file = self.analysis_dir / f"campaign_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        self.generate_markdown_report(stats, md_file)
        
        # JSON report
        json_file = self.analysis_dir / f"campaign_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.generate_json_report(stats, json_file)
        
        # Text table
        text_table = self.generate_comparison_table(stats)
        print(text_table)
        
        # Visualization
        if HAS_MATPLOTLIB:
            self.generate_visualization(stats)
        
        # Print markdown report
        print("\n" + md_report)
        
        return True


def main():
    """Main entry point"""
    # Check arguments
    summary_file = None
    if len(sys.argv) > 1:
        summary_file = Path(sys.argv[1])
        if not summary_file.exists():
            print(f"ERROR: File not found: {summary_file}")
            sys.exit(1)
    
    # Run analysis
    analyzer = CampaignAnalyzer()
    success = analyzer.run_analysis(summary_file)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
