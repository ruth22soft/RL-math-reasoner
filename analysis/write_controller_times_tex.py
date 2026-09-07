#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JPATH = ROOT / 'assets' / 'figures' / 'controller_start_times.json'
OUT = ROOT / 'assets' / 'figures' / 'controller_start_times.tex'

def main():
    if not JPATH.exists():
        print('No mapping JSON found at', JPATH)
        return
    m = json.loads(JPATH.read_text())
    lines = []
    lines.append('\\begin{table}[ht]')
    lines.append('\\centering')
    lines.append('\\caption{Detected controller start times (first GPU-util rise within 48h window before completion).}')
    lines.append('\\begin{tabular}{l l l}\\toprule')
    lines.append('Controller & Start & Completion \\\\ \\midrule')
    for c, v in m.items():
        start = v.get('start') or 'N/A'
        comp = v.get('completion') or 'N/A'
        lines.append(f"{c} & {start} & {comp} \\\\ ")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    OUT.write_text('\n'.join(lines))
    print('Wrote', OUT)

if __name__ == '__main__':
    main()
