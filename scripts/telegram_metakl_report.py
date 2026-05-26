#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)


def parse_step_metrics(log_text):
    step_line = None
    for line in reversed(log_text.splitlines()):
        if "step:" in line and "actor/" in line:
            step_line = line
            break

    if not step_line:
        return {}

    metrics = {}
    for part in step_line.split(" - "):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k == "step":
            metrics[k] = v
            continue
        try:
            metrics[k] = float(v)
        except ValueError:
            metrics[k] = v
    return metrics


def parse_total_training_steps(log_text):
    matches = re.findall(r"Total (?:training )?steps:\s*(\d+)", log_text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def read_latest_checkpoint(run_dir):
    # Try local path first (works if run_dir is a host path).
    tracker = Path(run_dir) / "latest_checkpointed_iteration.txt"
    latest_iter = None
    ckpt_names = []

    if tracker.exists():
        try:
            latest_iter = tracker.read_text(encoding="utf-8").strip()
        except Exception:
            latest_iter = None
        ckpts = sorted(Path(run_dir).glob("global_step_*"))
        ckpt_names = [c.name for c in ckpts[-5:]]
        return latest_iter, ckpt_names

    # Fallback: inspect Docker checkpoint volume directly.
    image = os.environ.get("RUTH_IMAGE_NAME", "simple-rl:ngc-vllm")
    cmd = (
        "docker run --rm -v simplerl_ckpts:/ckpts "
        f"{image} bash -lc '"
        f"cat {run_dir}/latest_checkpointed_iteration.txt 2>/dev/null || true; "
        f"ls -1d {run_dir}/global_step_* 2>/dev/null | tail -n 5 || true'"
    )
    res = run_cmd(cmd)
    if res.returncode != 0:
        return None, []

    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    if lines:
        if re.fullmatch(r"\d+", lines[0]):
            latest_iter = lines[0]
            ckpt_lines = lines[1:]
        else:
            ckpt_lines = lines
        ckpt_names = [Path(p).name for p in ckpt_lines if "global_step_" in p]

    return latest_iter, ckpt_names


def docker_container_state(container_name):
    res = run_cmd(f"docker inspect {container_name} --format '{{{{json .State}}}}'")
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout.strip())
    except Exception:
        return None


def post_telegram(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read().decode("utf-8", errors="ignore")
    return data


def main():
    container_name = os.environ.get("RUTH_CONTAINER_NAME", "ruth-training-run")
    run_dir = os.environ.get("RUTH_RUN_DIR", "/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive")
    log_file = os.environ.get("RUTH_TRAIN_LOG", f"{run_dir}/metakl_train.log")

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return 2

    log_res = run_cmd(f"docker logs {container_name} 2>&1")
    log_text = log_res.stdout if log_res.returncode == 0 else ""

    if not log_text and log_file:
        # fallback: read from persisted run log in docker volume using a throwaway container
        image = os.environ.get("RUTH_IMAGE_NAME", "simple-rl:ngc-vllm")
        cat_res = run_cmd(
            "docker run --rm -v simplerl_ckpts:/ckpts "
            f"{image} bash -lc 'test -f {log_file} && cat {log_file} || true'"
        )
        log_text = cat_res.stdout

    metrics = parse_step_metrics(log_text)

    state = docker_container_state(container_name)
    status = "not-found"
    running = False
    exit_code = None
    started_at = "-"
    if state:
        status = state.get("Status", "unknown")
        running = bool(state.get("Running", False))
        exit_code = state.get("ExitCode", None)
        started_at = state.get("StartedAt", "-")

    latest_iter, ckpts = read_latest_checkpoint(run_dir)
    total_training_steps = parse_total_training_steps(log_text)

    step = metrics.get("step", "n/a")
    kl_loss = metrics.get("actor/kl_loss", "n/a")
    kl_coef_dynamic = metrics.get("actor/kl_coef_dynamic", "n/a")
    kl_coef_fixed = metrics.get("actor/kl_coef", "n/a")
    # In this training stream, critic/score/mean acts as an online accuracy-like reward signal.
    accuracy_proxy = metrics.get("critic/score/mean", "n/a")

    # Only declare FINISHED when we've actually reached the configured total
    if running:
        training_state = "RUNNING"
    elif status == "exited":
        # If we have a total_training_steps and a recorded step, prefer that
        if isinstance(step, str) and step.isdigit() and total_training_steps:
            if int(step) >= int(total_training_steps):
                training_state = "FINISHED"
            else:
                training_state = f"EXITED({exit_code})"
        else:
            training_state = "FINISHED" if exit_code == 0 else f"EXITED({exit_code})"
    else:
        training_state = status.upper()

    progress = "n/a"
    if isinstance(step, str) and step.isdigit() and total_training_steps:
        progress = f"{int(step)}/{total_training_steps}"

    lines = [
        "*MetaKL Training Progress*",
        f"- container: `{container_name}`",
        f"- status: `{status}`",
        f"- training_state: `{training_state}`",
        f"- started_at: `{started_at}`",
        f"- run_dir: `{run_dir}`",
        "",
        "*Training Parameters (locked)*",
        "- total_epochs: `1`",
        "- max_prompt_length: `1024`",
        "- max_response_length: `1024`",
        "- max_num_batched_tokens: `4096`",
        "- adaptive_kl: `enabled (lstm)`",
        "",
        "*Latest Metrics*",
        f"- step: `{step}`",
        f"- progress: `{progress}`",
        f"- accuracy_proxy (critic/score/mean): `{accuracy_proxy}`",
        f"- actor/kl_loss: `{kl_loss}`",
        f"- actor/kl_coef_dynamic: `{kl_coef_dynamic}`",
        f"- actor/kl_coef (fixed fallback): `{kl_coef_fixed}`",
        f"- latest_total_training_steps: `{total_training_steps if total_training_steps is not None else 'n/a'}`",
        "",
        "*Checkpoint Status*",
        f"- latest_checkpointed_iteration: `{latest_iter if latest_iter else 'none'}`",
        f"- recent_checkpoints: `{', '.join(ckpts) if ckpts else 'none'}`",
    ]
    message = "\n".join(lines)

    result = post_telegram(bot_token=bot_token, chat_id=chat_id, message=message)
    print(result)
    return 0


def watch_mode(container_name, run_dir, bot_token, chat_id, poll_interval=5, rate_limit_seconds=30):
    """Tail docker logs and post updates when a new step line appears.
    This is a simple, resilient watcher that posts only when the step increases.
    """
    import time
    from subprocess import Popen, PIPE

    last_step = None
    last_post_ts = 0

    cmd = ["docker", "logs", "-f", container_name]
    with Popen(cmd, stdout=PIPE, stderr=PIPE, text=True, bufsize=1) as p:
        try:
            for raw_line in p.stdout:
                line = raw_line.strip()
                if "step:" not in line or "actor/" not in line:
                    continue
                metrics = parse_step_metrics(line)
                step = metrics.get("step")
                if isinstance(step, str) and step.isdigit():
                    step_i = int(step)
                else:
                    continue

                now = time.time()
                if last_step is None or step_i > last_step:
                    # prepare a compact status message
                    progress = None
                    total_training_steps = parse_total_training_steps(line) or None
                    if isinstance(step, str) and step.isdigit() and total_training_steps:
                        progress = f"{step_i}/{total_training_steps}"

                    kl_loss = metrics.get("actor/kl_loss", "n/a")
                    acc = metrics.get("critic/score/mean", "n/a")
                    msg_lines = ["*MetaKL Live Update*",
                                 f"- run_dir: `{run_dir}`",
                                 f"- step: `{step_i}`",
                                 f"- progress: `{progress if progress else 'n/a'}`",
                                 f"- critic/score/mean: `{acc}`",
                                 f"- actor/kl_loss: `{kl_loss}`"]
                    if now - last_post_ts > rate_limit_seconds:
                        try:
                            post_telegram(bot_token=bot_token, chat_id=chat_id, message="\n".join(msg_lines))
                            last_post_ts = now
                        except Exception:
                            pass
                    last_step = step_i
        except KeyboardInterrupt:
            try:
                p.terminate()
            except Exception:
                pass
        finally:
            try:
                p.kill()
            except Exception:
                pass


if __name__ == "__main__":
    # support a watch mode: `python telegram_metakl_report.py --watch`
    if "--watch" in sys.argv:
        container_name = os.environ.get("RUTH_CONTAINER_NAME", "ruth-training-run")
        run_dir = os.environ.get("RUTH_RUN_DIR", "/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive")
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not bot_token or not chat_id:
            print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID for watch mode")
            sys.exit(2)
        sys.exit(watch_mode(container_name, run_dir, bot_token, chat_id))
    else:
        sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
