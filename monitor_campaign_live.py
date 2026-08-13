#!/usr/bin/env python3
"""
Campaign Live Monitor
Real-time monitoring of campaign progress during execution

Usage:
    python3 monitor_campaign_live.py              # Monitor default orchestrator
    python3 monitor_campaign_live.py --watch      # Continuous monitoring (updates every 30s)
    python3 monitor_campaign_live.py --logdir /path  # Monitor specific directory
"""

import os
import sys
import glob
import re
import time
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import curses
    HAS_CURSES = True
except ImportError:
    HAS_CURSES = False


class CampaignMonitor:
    """Live monitoring of campaign execution"""
    
    def __init__(self, analysis_dir: Path):
        self.analysis_dir = analysis_dir
        self.results = defaultdict(list)
        self.orchestrator_logs = list(self.analysis_dir.glob("orchestrator_multiseed_v2_*.log"))
        self.campaign_summaries = list(self.analysis_dir.glob("campaign_results_summary_*.txt"))
        
    def get_orchestrator_status(self) -> Dict:
        """Get orchestrator status from log"""
        status = {
            "is_running": False,
            "current_run": None,
            "runs_completed": 0,
            "total_runs": 0,
            "waiting_for": None,
            "latest_message": None,
            "last_update": None
        }
        
        if not self.orchestrator_logs:
            return status
        
        latest_log = self.orchestrator_logs[-1]
        
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            # Parse log
            for line in reversed(lines):
                status["last_update"] = line.strip()
                
                if "Starting" in line and "phase3" in line:
                    match = re.search(r"Starting (phase3-\w+-s\d+)", line)
                    if match:
                        status["current_run"] = match.group(1)
                
                if "Waiting for" in line:
                    match = re.search(r"Waiting for (\S+)", line)
                    if match:
                        status["waiting_for"] = match.group(1)
                
                if re.search(r"\[\d+/\d+\]", line):
                    match = re.search(r"\[(\d+)/(\d+)\]", line)
                    if match:
                        status["runs_completed"] = int(match.group(1)) - 1
                        status["total_runs"] = int(match.group(2))
            
            # Check if orchestrator process is running
            try:
                result = subprocess.run(
                    ["pgrep", "-f", "orchestrator_v2"],
                    capture_output=True,
                    timeout=5
                )
                status["is_running"] = result.returncode == 0
            except:
                pass
            
        except Exception as e:
            print(f"Error reading orchestrator log: {e}")
        
        return status
    
    def get_active_training(self) -> Tuple[Optional[str], Dict]:
        """Get currently active training container info"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "label=simplerl.role=training",
                 "--format", "{{.Names}}\t{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                parts = lines[0].split('\t')
                if len(parts) >= 2:
                    return parts[0], {"status": parts[1]}
        except:
            pass
        
        return None, {}
    
    def get_container_metrics(self, container_name: str) -> Dict:
        """Get metrics from running container"""
        metrics = {
            "current_step": 0,
            "total_steps": 1569,
            "epoch": 0,
            "last_kl_loss": None,
            "last_reward": None,
            "duration_sec": 0
        }
        
        try:
            result = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            logs = result.stdout + result.stderr
            
            # Get latest step number
            steps = re.findall(r"step:(\d+)", logs)
            if steps:
                metrics["current_step"] = int(steps[-1])
            
            # Get epoch
            epochs = re.findall(r"Epoch (\d+)", logs)
            if epochs:
                metrics["epoch"] = int(epochs[-1])
            
            # Get KL loss
            kl_losses = re.findall(r"actor/kl_loss['\"]?\s*[=:]\s*([0-9.e+-]+)", logs)
            if kl_losses:
                metrics["last_kl_loss"] = float(kl_losses[-1])
            
            # Get reward
            rewards = re.findall(r"critic/rewards['\"]?.*?mean['\"]?\s*[=:]\s*([0-9.e+-]+)", logs)
            if rewards:
                metrics["last_reward"] = float(rewards[-1])
        
        except Exception as e:
            pass
        
        return metrics
    
    def get_completed_results(self) -> Dict:
        """Get completed results from summary file"""
        results = defaultdict(list)
        
        if not self.campaign_summaries:
            return results
        
        latest_summary = self.campaign_summaries[-1]
        
        try:
            with open(latest_summary, 'r') as f:
                content = f.read()
            
            # Extract result lines
            pattern = r"(\d+)\.\s+(\w+)\s+seed[=_](\d)\s*:\s*([0-9.]+|N/A)"
            
            for match in re.finditer(pattern, content):
                run_num, mode, seed, score = match.groups()
                try:
                    results[mode].append(float(score))
                except:
                    pass
        
        except:
            pass
        
        return results
    
    def print_status_text(self):
        """Print status in text mode"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("="*80)
        print("CAMPAIGN LIVE MONITOR")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Orchestrator status
        orch_status = self.get_orchestrator_status()
        
        print("ORCHESTRATOR STATUS:")
        print(f"  Running: {'✓ YES' if orch_status['is_running'] else '✗ NO'}")
        if orch_status['waiting_for']:
            print(f"  Waiting for: {orch_status['waiting_for']}")
        if orch_status['current_run']:
            print(f"  Current run: {orch_status['current_run']}")
        if orch_status['total_runs'] > 0:
            pct = (orch_status['runs_completed'] / orch_status['total_runs']) * 100
            print(f"  Progress: {orch_status['runs_completed']}/{orch_status['total_runs']} runs ({pct:.1f}%)")
        if orch_status['last_update']:
            print(f"  Latest: {orch_status['last_update'][:80]}")
        
        print("\nACTIVE TRAINING:")
        container, info = self.get_active_training()
        if container:
            print(f"  Container: {container}")
            print(f"  Status: {info.get('status', 'Unknown')}")
            
            # Get metrics
            metrics = self.get_container_metrics(container)
            pct_complete = (metrics["current_step"] / metrics["total_steps"]) * 100
            
            print(f"  Progress: {metrics['current_step']}/{metrics['total_steps']} steps ({pct_complete:.1f}%)")
            if metrics['epoch'] > 0:
                print(f"  Epoch: {metrics['epoch']}/3")
            if metrics['last_kl_loss'] is not None:
                print(f"  KL Loss: {metrics['last_kl_loss']:.6f}")
            if metrics['last_reward'] is not None:
                print(f"  Reward (mean): {metrics['last_reward']:.6f}")
        else:
            print("  None currently running")
        
        # Completed results
        print("\nCOMPLETED RESULTS:")
        results = self.get_completed_results()
        
        if results:
            for mode in sorted(results.keys()):
                scores = results[mode]
                if scores:
                    mean = sum(scores) / len(scores)
                    print(f"  {mode:12}: mean={mean:.4f}, runs={len(scores)}, scores={[f'{s:.4f}' for s in scores]}")
        else:
            print("  No results yet")
        
        print("\n" + "="*80)
        print("Press Ctrl+C to exit\n")
    
    def monitor_loop(self, interval: int = 30):
        """Continuous monitoring loop"""
        try:
            while True:
                self.print_status_text()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


def main():
    """Main entry point"""
    repo_dir = Path("/home/ai-server-02/R_projects/final_thesis")
    analysis_dir = repo_dir / "simpleRL-reason" / "analysis_logs"
    
    if not analysis_dir.exists():
        print(f"ERROR: Analysis directory not found: {analysis_dir}")
        sys.exit(1)
    
    monitor = CampaignMonitor(analysis_dir)
    
    # Parse arguments
    watch_mode = "--watch" in sys.argv or "--continuous" in sys.argv
    interval = 30
    
    if watch_mode:
        print("Starting continuous monitor (updates every 30 seconds)...")
        print("(Press Ctrl+C to stop)\n")
        time.sleep(2)
        monitor.monitor_loop(interval)
    else:
        # Single snapshot
        monitor.print_status_text()


if __name__ == "__main__":
    main()
