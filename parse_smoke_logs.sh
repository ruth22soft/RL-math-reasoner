#!/usr/bin/env bash
set -euo pipefail
LOG=/home/ai-server-02/R_projects/final_thesis/smoke_phase2_fixed_n4_docker.log
OUTDIR=/home/ai-server-02/R_projects/final_thesis/phase2_monitor_smoke_fixed_n4
mkdir -p "$OUTDIR"
grep -n 'step:' "$LOG" > "$OUTDIR/step_lines.txt" || true
grep -n 'timing_s/step' "$LOG" | sed -E 's/.*timing_s\/step:([0-9.]+).*/\1/' > "$OUTDIR/timings.txt" || true
grep -n 'response_length/mean' "$LOG" | sed -E 's/.*response_length\/mean:([0-9.]+).*/\1/' > "$OUTDIR/respmean.txt" || true
grep -n 'actor/kl_coef' "$LOG" | sed -E 's/.*actor\/kl_coef:([0-9.]+).*/\1/' > "$OUTDIR/klcoef.txt" || true
grep -n 'actor/kl_loss' "$LOG" | sed -E 's/.*actor\/kl_loss:([0-9.]+).*/\1/' > "$OUTDIR/kl_loss.txt" || true
grep -n 'val/test_score' "$LOG" > "$OUTDIR/val_test_score.txt" || true

echo "counts:"
echo "step_lines=$(wc -l < \"$OUTDIR/step_lines.txt\" || true)"
echo "timings_count=$(wc -l < \"$OUTDIR/timings.txt\" || true)"
echo "respmean_count=$(wc -l < \"$OUTDIR/respmean.txt\" || true)"
echo "klcoef_count=$(wc -l < \"$OUTDIR/klcoef.txt\" || true)"
echo "val_test_score_lines=$(wc -l < \"$OUTDIR/val_test_score.txt\" || true)"

echo "timing_stats:"
awk '{sum+=$1; if(min==""||$1<min)min=$1; if(max==""||$1>max)max=$1} END{if(NR>0)printf("count=%d avg=%.3f min=%.3f max=%.3f\n",NR,sum/NR,min,max); else print "count=0"}' "$OUTDIR/timings.txt" || true

echo "respmean_stats:"
awk '{sum+=$1; if(min==""||$1<min)min=$1; if(max==""||$1>max)max=$1} END{if(NR>0)printf("count=%d avg=%.3f min=%.3f max=%.3f\n",NR,sum/NR,min,max); else print "count=0"}' "$OUTDIR/respmean.txt" || true

echo "klcoef_stats:"
awk '{sum+=$1; if(min==""||$1<min)min=$1; if(max==""||$1>max)max=$1} END{if(NR>0)printf("count=%d avg=%.6f min=%.6f max=%.6f\n",NR,sum/NR,min,max); else print "count=0"}' "$OUTDIR/klcoef.txt" || true

echo "kl_loss_stats:"
awk '{sum+=$1; if(min==""||$1<min)min=$1; if(max==""||$1>max)max=$1} END{if(NR>0)printf("count=%d avg=%.6f min=%.6f max=%.6f\n",NR,sum/NR,min,max); else print "count=0"}' "$OUTDIR/kl_loss.txt" || true

echo "val_test_score_lines:"
if [ -s "$OUTDIR/val_test_score.txt" ]; then
  sed -n '1,200p' "$OUTDIR/val_test_score.txt"
else
  echo "(none)"
fi
