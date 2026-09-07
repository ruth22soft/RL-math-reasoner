#!/usr/bin/env python3
import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TPATH = ROOT / 'telegram_parsed.csv'
RESULTS_DIR = ROOT / 'results'
OUT_DIR = ROOT / 'assets' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_telegram():
    rows = []
    with open(TPATH, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # parse timestamp
            ts = r.get('ts') or r.get('date')
            try:
                if ts and ts.endswith('Z'):
                    t = datetime.fromisoformat(ts.replace('Z',''))
                else:
                    t = datetime.fromisoformat(ts) if ts else None
                # normalize to naive datetime (drop tzinfo)
                if t and t.tzinfo is not None:
                    t = t.replace(tzinfo=None)
            except Exception:
                t = None
            # extract gpu utilization and memory from text
            text = r.get('text','')
            m_util = re.search(r'Utilization:\s*([0-9]+)%', text)
            m_mem = re.search(r'Memory:\s*([0-9]+)\s*/\s*([0-9]+)\s*MiB', text)
            util = int(m_util.group(1)) if m_util else None
            mem_used = int(m_mem.group(1)) if m_mem else None
            rows.append({**r, 'datetime': t, 'gpu_util': util, 'gpu_mem_used': mem_used})
    rows = [r for r in rows if r['datetime'] is not None]
    rows.sort(key=lambda x: x['datetime'])
    return rows

def read_results():
    controllers = ['fixed','rule_based','mlp_based','lstm_based']
    res = {}
    for c in controllers:
        p = RESULTS_DIR / f"{c}_results.json"
        if p.exists():
            j = json.loads(p.read_text())
            # parse timestamp
            ts = j.get('completion_timestamp')
            try:
                if ts and ts.endswith('Z'):
                    comp = datetime.fromisoformat(ts.replace('Z',''))
                else:
                    comp = datetime.fromisoformat(ts) if ts else None
                if comp and comp.tzinfo is not None:
                    comp = comp.replace(tzinfo=None)
            except Exception:
                comp = None
            res[c] = { 'meta': j, 'completion': comp }
        else:
            res[c] = { 'meta': None, 'completion': None }
    return res

def find_start_in_window(rows, comp_ts, hours=48, util_threshold=10):
    if comp_ts is None:
        return None
    start_window = comp_ts - timedelta(hours=hours)
    window_rows = [r for r in rows if start_window <= r['datetime'] <= comp_ts]
    if not window_rows:
        return None
    # find first index where util > threshold and previous util <= threshold for at least one sample
    prev_util = None
    for r in window_rows:
        util = r['gpu_util'] if r['gpu_util'] is not None else 0
        if prev_util is None:
            prev_util = util
            continue
        if util > util_threshold and prev_util <= util_threshold:
            return r['datetime']
        prev_util = util
    # fallback: return time of max util in window
    max_row = max(window_rows, key=lambda x: (x['gpu_util'] or 0))
    if (max_row['gpu_util'] or 0) > util_threshold:
        return max_row['datetime']
    return None

def plot_segment(rows, start, end, controller_name):
    seg = [r for r in rows if r['datetime'] >= start and r['datetime'] <= end]
    if not seg:
        print('No telegram data for', controller_name)
        return
    times = [r['datetime'] for r in seg]
    kl = [float(r['kl']) if r['kl'] not in (None,'','not yet') else float('nan') for r in seg]
    tau = [float(r['tau']) if r['tau'] not in (None,'','not yet') else float('nan') for r in seg]

    plt.figure(figsize=(8,3))
    plt.plot(times, kl, '-o')
    plt.title(f'{controller_name} — monitor Actor KL during run')
    plt.ylabel('Actor KL')
    plt.xticks(rotation=20)
    plt.tight_layout()
    out1 = OUT_DIR / f'{controller_name}_telegram_kl.png'
    plt.savefig(out1)
    plt.close()

    plt.figure(figsize=(8,3))
    plt.plot(times, tau, '-o')
    plt.title(f'{controller_name} — monitor tau during run')
    plt.ylabel('tau')
    plt.xticks(rotation=20)
    plt.tight_layout()
    out2 = OUT_DIR / f'{controller_name}_telegram_tau.png'
    plt.savefig(out2)
    plt.close()
    print('Saved', out1, out2)

def main():
    rows = parse_telegram()
    res = read_results()
    mapping = {}
    for c, info in res.items():
        comp = info['completion']
        start = find_start_in_window(rows, comp, hours=48, util_threshold=8)
        mapping[c] = { 'completion': comp.isoformat() if comp else None, 'start': start.isoformat() if start else None }
        if start and comp:
            plot_segment(rows, start, comp, c)
    # write mapping
    out = OUT_DIR / 'controller_start_times.json'
    out.write_text(json.dumps(mapping, indent=2))
    print('Wrote', out)

if __name__ == '__main__':
    main()
