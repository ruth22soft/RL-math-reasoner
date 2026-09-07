#!/usr/bin/env python3
from pathlib import Path
import csv
import datetime

CSV = Path(__file__).resolve().parents[1] / 'telegram_parsed.csv'
OUT = Path(__file__).resolve().parents[1] / 'analysis' / 'experiments_table.tex'
OUT.parent.mkdir(parents=True, exist_ok=True)

def parse():
    rows = []
    with open(CSV, newline='', encoding='utf8') as fh:
        r = csv.DictReader(fh)
        for line in r:
            rows.append(line)
    return rows

def summarize(rows):
    if not rows:
        return None
    times = [datetime.datetime.fromisoformat(r['ts']) for r in rows if r['ts']]
    start = min(times) if times else None
    end = max(times) if times else None
    steps = [int(r['step_cur']) for r in rows if r['step_cur']] if any(r['step_cur'] for r in rows) else []
    max_step = max(steps) if steps else None
    services = set(r['service'] for r in rows if r['service'])
    final_val = None
    for r in reversed(rows):
        if r['val'] and r['val']!='not yet':
            final_val = r['val']
            break
    return {'start': start, 'end': end, 'messages': len(rows), 'max_step': max_step, 'services': services, 'final_val': final_val}

def write_tex(s):
    lines = []
    lines.append('\\begin{table}[ht]')
    lines.append('\\centering')
    lines.append('\\begin{tabular}{ll}')
    lines.append('\\toprule')
    lines.append('Metric & Value\\\\')
    lines.append('\\midrule')
    lines.append(f'Start & {s["start"]}\\\\')
    lines.append(f'End & {s["end"]}\\\\')
    lines.append(f'Messages parsed & {s["messages"]}\\\\')
    lines.append(f'Max reported step & {s["max_step"]}\\\\')
    lines.append(f'Final reported val & {s["final_val"]}\\\\')
    lines.append(f'Services observed & {", ".join(s["services"]) if s["services"] else "none"}\\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\caption{Summary of training monitor messages parsed from Telegram export.}')
    lines.append('\\label{tab:telegram_summary}')
    lines.append('\\end{table}')
    OUT.write_text('\n'.join(lines))
    print('Wrote', OUT)

def main():
    rows = parse()
    s = summarize(rows)
    write_tex(s)

if __name__=='__main__':
    main()
