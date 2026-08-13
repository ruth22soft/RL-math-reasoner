#!/bin/bash

################################################################################
# SIMPLE START SCRIPT FOR ORCHESTRATOR
# Just run: ./START_ORCHESTRATION.sh
# Then forget about it for 5 days
################################################################################

cd "$(dirname "$0")"

echo "================================================"
echo "KL CONTROLLER ORCHESTRATOR - AUTO MODE"
echo "================================================"
echo ""
echo "This will run:"
echo "  1. rule_based → mlp_based → fixed → lstm_based"
echo "  2. Automatically monitor and save results"
echo "  3. NO SEED (random initialization)"
echo "  4. ~5 days total (~29 hours per controller)"
echo ""
echo "Results saved to: ./results/"
echo ""
read -p "Press ENTER to start, or Ctrl+C to cancel: "

echo ""
echo "Starting orchestrator in background..."
echo ""

# Run in background and save output
nohup ./orchestrate_kl_controllers.sh > orchestration.log 2>&1 &
ORCHESTRATOR_PID=$!

echo "✓ Orchestrator started (PID: $ORCHESTRATOR_PID)"
echo ""
echo "To monitor progress:"
echo "  tail -f orchestration.log"
echo ""
echo "To check current step:"
echo "  docker logs phase3-rule_based-s1 2>&1 | strings | grep -oP 'step:\K[0-9]+' | tail -1"
echo ""
echo "To attach to running process:"
echo "  fg  # if still attached"
echo "  jobs  # list background jobs"
echo ""
echo "To stop orchestrator:"
echo "  kill $ORCHESTRATOR_PID"
echo ""
