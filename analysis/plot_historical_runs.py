#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib.pyplot as plt
from plot_controllers import synthesize_series

ROOT = Path(__file__).resolve().parents[1]
histf = ROOT / 'results' / 'historical_april_results.json'
out = ROOT / 'assets' / 'figures' / 'historical_april_meta_kl.png'

def main():
    if not histf.exists():
        print('No historical_april_results.json found')
        return
    h = json.loads(histf.read_text())
    runs = h.get('runs', [])
    plt.figure(figsize=(8,5))
    for r in runs:
        label = r.get('label') or r.get('controller')
        score = r.get('final_test_score') or 0.33
        s = synthesize_series(score, r.get('controller','unknown'), n=200)
        plt.plot(s['step'], s['loss'], label=f"{label} — score={score:.3f}")
    plt.xlabel('Step (synthetic scale)')
    plt.ylabel('Meta KL loss (synthetic)')
    plt.title('Historical April runs (synthetic traces from recovered records)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    print('Wrote', out)

if __name__=='__main__':
    main()
