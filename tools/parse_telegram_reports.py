#!/usr/bin/env python3
import json
import re
from pathlib import Path
import csv
import datetime
import math
import matplotlib.pyplot as plt

EXPORT_DIR = Path(__file__).resolve().parents[1] / 'telegram_exports' / 'ChatExport_2026-09-07'
OUT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = OUT_DIR / 'assets' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def join_text(text_field):
    if isinstance(text_field, str):
        return text_field
    parts = []
    for t in text_field:
        if isinstance(t, str):
            parts.append(t)
        elif isinstance(t, dict) and 'text' in t:
            parts.append(t['text'])
    return ''.join(parts)


def parse_messages(json_path):
    j = json.loads(Path(json_path).read_text())
    msgs = []
    for m in j.get('messages', []):
        text = join_text(m.get('text', ''))
        date = m.get('date')
        ts = None
        try:
            ts = datetime.datetime.fromisoformat(date)
        except Exception:
            ts = None
        msgs.append({'id': m.get('id'), 'date': date, 'ts': ts, 'text': text})
    return msgs


RX_STEP = re.compile(r"Step:\s*([0-9]+)\s*/\s*([0-9]+)")
RX_VAL = re.compile(r"Val score:\s*([0-9.]+|not yet)")
RX_KL = re.compile(r"KL loss:\s*([0-9.]+|not yet)")
RX_TAU = re.compile(r"KL coef:\s*([0-9.]+|not yet)")
RX_REWARD = re.compile(r"Reward:\s*([0-9.]+|unknown)")


def extract_fields(msg):
    t = msg['text']
    step_cur = None
    step_tot = None
    m = RX_STEP.search(t)
    if m:
        step_cur = int(m.group(1))
        step_tot = int(m.group(2))
    m = RX_VAL.search(t)
    val = m.group(1) if m else None
    m = RX_KL.search(t)
    kl = m.group(1) if m else None
    m = RX_TAU.search(t)
    tau = m.group(1) if m else None
    m = RX_REWARD.search(t)
    reward = m.group(1) if m else None
    service = None
    if 'failed' in t.lower():
        service = 'failed'
    if 'active' in t.lower():
        service = 'active'
    return {'step_cur': step_cur, 'step_tot': step_tot, 'val': val, 'kl': kl, 'tau': tau, 'reward': reward, 'service': service}


def to_number(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x
    try:
        return float(x)
    except Exception:
        return None


def main():
    jsonp = EXPORT_DIR / 'result.json'
    if not jsonp.exists():
        print('No export JSON found in', EXPORT_DIR)
        return 1
    msgs = parse_messages(jsonp)
    rows = []
    for m in msgs:
        f = extract_fields(m)
        rows.append({
            'id': m['id'],
            'date': m['date'],
            'ts': m['ts'].isoformat() if m['ts'] else '',
            'step_cur': f['step_cur'],
            'step_tot': f['step_tot'],
            'val': f['val'],
            'kl': f['kl'],
            'tau': f['tau'],
            'reward': f['reward'],
            'service': f['service'],
            'text': m['text'][:1000]
        })

    csvp = OUT_DIR / 'telegram_parsed.csv'
    with open(csvp, 'w', newline='', encoding='utf8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Wrote', csvp)

    times = []
    steps = []
    kls = []
    taus = []
    vals = []
    for r in rows:
        if r['ts']:
            times.append(datetime.datetime.fromisoformat(r['ts']))
        else:
            times.append(None)
        steps.append(to_number(r['step_cur']))
        kls.append(to_number(r['kl']))
        taus.append(to_number(r['tau']))
        vals.append(to_number(r['val']))

    if any(t is None for t in times):
        base = datetime.datetime.now()
        times = [base + datetime.timedelta(seconds=60*i) for i in range(len(times))]

    plt.figure(figsize=(8,4))
    plt.plot(times, kls, '-o')
    plt.xlabel('Time')
    plt.ylabel('KL loss')
    plt.title('KL loss from Telegram monitor')
    p1 = FIG_DIR / 'telegram_kl_loss.png'
    plt.tight_layout()
    plt.savefig(p1)
    print('Saved', p1)

    plt.figure(figsize=(8,4))
    plt.plot(times, taus, '-o')
    plt.xlabel('Time')
    plt.ylabel('KL coef (tau)')
    plt.title('KL coefficient from Telegram monitor')
    p2 = FIG_DIR / 'telegram_tau.png'
    plt.tight_layout()
    plt.savefig(p2)
    print('Saved', p2)

    plt.figure(figsize=(8,4))
    plt.plot(times, [s if s is not None else math.nan for s in steps], '-o')
    plt.xlabel('Time')
    plt.ylabel('Reported step')
    plt.title('Reported training step over time')
    p3 = FIG_DIR / 'telegram_step_progress.png'
    plt.tight_layout()
    plt.savefig(p3)
    print('Saved', p3)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
