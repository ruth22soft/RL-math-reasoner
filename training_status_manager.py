#!/usr/bin/env python3
"""Simple training status/start/stop manager with JSON summary output.

Examples:
  python3 training_status_manager.py status --name amharic-grpo
  python3 training_status_manager.py start --name amharic-grpo
  python3 training_status_manager.py stop --name amharic-grpo
  python3 training_status_manager.py wait --name lstm-training-live --json-out results/lstm_training_run.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def docker_inspect(name: str) -> Dict[str, Any]:
    try:
        proc = run_cmd(["docker", "inspect", name], check=False)
        if proc.returncode != 0:
            return {}
        return json.loads(proc.stdout)[0]
    except Exception:
        return {}


def container_state(name: str) -> Dict[str, Any]:
    info = docker_inspect(name)
    if not info:
        return {
            "exists": False,
            "status": "missing",
            "running": False,
        }
    state = info.get("State", {})
    return {
        "exists": True,
        "status": state.get("Status", "unknown"),
        "running": bool(state.get("Running", False)),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
        "exit_code": state.get("ExitCode"),
        "name": info.get("Name", name),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def start_training(name: str) -> Dict[str, Any]:
    if name == "amharic-grpo":
        cmd = [
            "bash",
            "-lc",
            "cd /home/ai-server-02/R_projects/final_thesis/adaptive-kl-grpo-thesis && "
            "nohup env PYTHONPATH=/workspace/adaptive-kl-grpo-thesis "
            "python3 -m verl.trainer.main_ppo --config-name=simplelr_grpo_amharic_single_gpu "
            "> /tmp/amharic_train.log 2>&1 & echo $!",
        ]
        proc = run_cmd(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "start failed")
        pid = proc.stdout.strip().splitlines()[-1]
        result = {
            "name": name,
            "status": "started",
            "pid": pid,
            "started_at": now_iso(),
            "source": "docker-exec-amharic",
        }
        return result

    if name == "lstm-training-live":
        cmd = ["bash", "-lc", "cd /home/ai-server-02/R_projects/final_thesis && ./lstm_training.sh start lstm"]
        proc = run_cmd(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "lstm start failed")
        result = {
            "name": name,
            "status": "started",
            "started_at": now_iso(),
            "source": "lstm_training.sh",
        }
        return result

    raise ValueError(f"Unsupported training name: {name}")


def stop_training(name: str) -> Dict[str, Any]:
    state = container_state(name)
    if not state["exists"]:
        return {"name": name, "status": "not_found", "stopped_at": now_iso()}

    proc = run_cmd(["docker", "stop", "-t", "30", name], check=False)
    if proc.returncode == 0:
        return {"name": name, "status": "stopped", "stopped_at": now_iso()}
    return {"name": name, "status": "stop_failed", "stderr": proc.stderr.strip(), "stopped_at": now_iso()}


def wait_for_completion(name: str, poll_seconds: int = 15, timeout_minutes: int = 360) -> Dict[str, Any]:
    started = now_iso()
    deadline = time.time() + timeout_minutes * 60

    while time.time() < deadline:
        st = container_state(name)
        if not st["exists"]:
            return {
                "name": name,
                "status": "finished",
                "started_at": started,
                "finished_at": now_iso(),
                "exit_code": "missing",
                "reason": "container not found",
            }
        if st["running"] is False and st.get("exit_code") is not None:
            return {
                "name": name,
                "status": "finished",
                "started_at": started,
                "finished_at": now_iso(),
                "exit_code": st.get("exit_code"),
                "status_detail": st.get("status"),
            }
        time.sleep(poll_seconds)

    return {
        "name": name,
        "status": "timeout",
        "started_at": started,
        "finished_at": now_iso(),
        "exit_code": "timeout",
    }


def status_payload(name: str) -> Dict[str, Any]:
    st = container_state(name)
    payload = {"name": name, "timestamp": now_iso()}
    payload.update(st)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage training status and write JSON state snapshots.")
    parser.add_argument("command", choices=["status", "start", "stop", "wait"], help="Action to run")
    parser.add_argument("--name", default="amharic-grpo", help="Container / training handle name")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON results")
    parser.add_argument("--timeout-minutes", type=int, default=360)
    # optional start commands for amharic/lstm only
    args = parser.parse_args()

    try:
        if args.command == "status":
            payload = status_payload(args.name)
        elif args.command == "start":
            payload = start_training(args.name)
        elif args.command == "stop":
            payload = stop_training(args.name)
        elif args.command == "wait":
            payload = wait_for_completion(args.name, timeout_minutes=args.timeout_minutes)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        payload = {
            "name": args.name,
            "status": "error",
            "error": str(exc),
            "timestamp": now_iso(),
        }
        if args.json_out:
            write_json(Path(args.json_out), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    if args.json_out:
        write_json(Path(args.json_out), payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
