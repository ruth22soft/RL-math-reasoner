#!/bin/bash

################################################################################
# AUTOMATIC KL CONTROLLER ORCHESTRATOR
# Runs 4 adaptive KL controllers sequentially: rule_based → mlp_based → fixed → lstm_based
# 
# NO SEED: Random initialization (matches zero_kl baseline)
# Correct hyperparameters: n=4, gpu_util=0.5, batch_size=16, epochs=3
# 
# Automatically:
#   - Monitors container for completion
#   - Extracts final validation score
#   - Saves results to JSON
#   - Kills previous container
#   - Launches next controller
#   - Continues until all 4 complete
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Configuration
CONTROLLERS=("rule_based" "mlp_based" "fixed" "lstm_based")
IMAGE="simple-rl:ngc-vllm"
RESULTS_DIR="$PROJECT_DIR/results"
CKPTS_BASE="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive"
TOTAL_STEPS=1569
EXPECTED_STEP_TIME=111  # seconds per step

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

mkdir -p "$RESULTS_DIR"

################################################################################
# Function: Launch a KL controller container
################################################################################
launch_controller() {
    local controller=$1
    local container_name="phase3-${controller}-s1"
    local ckpt_dir="${CKPTS_BASE}/phase3_${controller}_s1"
    
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Launching ${controller} controller...${NC}"
    
    # Set mode-specific parameters
    case "$controller" in
        rule_based)
            MODE_PARAMS="actor_rollout_ref.actor.actor_adaptive_kl.enable=true \
                        actor_rollout_ref.actor.actor_adaptive_kl.mode=rule \
                        actor_rollout_ref.actor.actor_adaptive_kl.init_beta=0.001 \
                        actor_rollout_ref.actor.actor_adaptive_kl.target_kl_loss=0.08 \
                        actor_rollout_ref.actor.actor_adaptive_kl.up_gain=1.08 \
                        actor_rollout_ref.actor.actor_adaptive_kl.down_gain=0.94 \
                        actor_rollout_ref.actor.actor_adaptive_kl.min_beta=0.0001 \
                        actor_rollout_ref.actor.actor_adaptive_kl.max_beta=0.01"
            ;;
        mlp_based)
            MODE_PARAMS="actor_rollout_ref.actor.actor_adaptive_kl.enable=true \
                        actor_rollout_ref.actor.actor_adaptive_kl.mode=mlp \
                        actor_rollout_ref.actor.actor_adaptive_kl.init_beta=0.001 \
                        actor_rollout_ref.actor.actor_adaptive_kl.mlp_hidden_dim=32 \
                        actor_rollout_ref.actor.actor_adaptive_kl.min_beta=0.0001 \
                        actor_rollout_ref.actor.actor_adaptive_kl.max_beta=0.01"
            ;;
        fixed)
            MODE_PARAMS="actor_rollout_ref.actor.actor_adaptive_kl.enable=false \
                        actor_rollout_ref.actor.kl_loss_coef=0.001"
            ;;
        lstm_based)
            MODE_PARAMS="actor_rollout_ref.actor.actor_adaptive_kl.enable=true \
                        actor_rollout_ref.actor.actor_adaptive_kl.mode=lstm \
                        actor_rollout_ref.actor.actor_adaptive_kl.init_beta=0.001 \
                        actor_rollout_ref.actor.actor_adaptive_kl.lstm_hidden_dim=32 \
                        actor_rollout_ref.actor.actor_adaptive_kl.lstm_reset_on_epoch=true \
                        actor_rollout_ref.actor.actor_adaptive_kl.min_beta=0.0001 \
                        actor_rollout_ref.actor.actor_adaptive_kl.max_beta=0.01"
            ;;
    esac
    
    # Launch container with correct config (NO SEED - random initialization)
    docker run --name "$container_name" -d --gpus all \
        -v simplerl_data:/data \
        -v simplerl_ckpts:/ckpts \
        -v simplerl_hf_cache:/root/.cache/huggingface \
        "$IMAGE" \
        bash -lc "
set -e
python -m pip install -e . --no-deps > /dev/null 2>&1
python -m pip install word2number 'antlr4-python3-runtime==4.9.3' 'math-verify==0.6.0' > /dev/null 2>&1
export PYTHONPATH=/workspace/simpleRL-reason:\\\$PYTHONPATH
mkdir -p '$ckpt_dir'
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \\
  --config-name simplelr_grpo_qwen05_single_gpu \\
  trainer.total_epochs=3 \\
  data.max_prompt_length=1024 \\
  data.max_response_length=1024 \\
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \\
  actor_rollout_ref.rollout.n=4 \\
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \\
  actor_rollout_ref.rollout.max_num_seqs=32 \\
  trainer.default_local_dir='$ckpt_dir' \\
  trainer.save_freq=2000 \\
  trainer.test_freq=2000 \\
  trainer.resume_mode=never \\
  trainer.remove_previous_ckpt=true \\
  $MODE_PARAMS \\
  2>&1 | tee '$ckpt_dir/metakl_train.log'
" > /dev/null 2>&1 &
    
    echo -e "${GREEN}✓ Container $container_name started (PID available via docker ps)${NC}"
}

################################################################################
# Function: Monitor container until completion
################################################################################
monitor_container() {
    local container_name=$1
    local controller=$2
    local log_file="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_${controller}_s1/metakl_train.log"
    
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring $controller (est. 29 hours for 1569 steps @ 111s/step)${NC}"
    
    local last_step=0
    local stuck_count=0
    local max_stuck=300  # 300 checks * 60s = 5 hours without progress
    
    while true; do
        sleep 60  # Check every minute
        
        # Get latest step from container logs
        current_step=$(docker logs "$container_name" 2>&1 | strings 2>/dev/null | grep -oP 'step:\K[0-9]+' | tail -1 || echo "0")
        
        if [ "$current_step" -eq 0 ]; then
            # Container may still be initializing
            echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Initializing...${NC}"
            ((stuck_count++))
        elif [ "$current_step" -eq "$last_step" ]; then
            # No progress
            ((stuck_count++))
            echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Step $current_step/$TOTAL_STEPS (no progress, count: $stuck_count/$max_stuck)${NC}"
        else
            # Progress detected
            stuck_count=0
            percentage=$((current_step * 100 / TOTAL_STEPS))
            echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Step $current_step/$TOTAL_STEPS ($percentage%)${NC}"
        fi
        
        last_step=$current_step
        
        # Check for completion
        if [ "$current_step" -ge $((TOTAL_STEPS - 5)) ]; then
            echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Nearing completion at step $current_step${NC}"
            # Wait for actual completion
            sleep 300
            break
        fi
        
        # Check if stuck too long
        if [ "$stuck_count" -gt "$max_stuck" ]; then
            echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: No progress for 5+ hours, aborting${NC}"
            return 1
        fi
    done
}

################################################################################
# Function: Extract results and save to JSON
################################################################################
extract_and_save_results() {
    local controller=$1
    local container_name="phase3-${controller}-s1"
    local log_file="/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_${controller}_s1/metakl_train.log"
    local results_file="$RESULTS_DIR/${controller}_results.json"
    
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Extracting results for $controller...${NC}"
    
    # Extract final test score from log
    final_test_score=$(docker exec "$container_name" bash -c "strings '$log_file' 2>/dev/null | grep -oP \"val/test_score/simplelr_qwen:\K[0-9.]+\" | tail -1" 2>/dev/null || echo "N/A")
    
    # Extract final step count
    final_step=$(docker logs "$container_name" 2>&1 | strings 2>/dev/null | grep -oP 'step:\K[0-9]+' | tail -1 || echo "0")
    
    # Get checkpoint directory timestamp
    ckpt_timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    
    # Create JSON result
    cat > "$results_file" << EOF
{
  "controller": "$controller",
  "seed": null,
  "global_seed": null,
  "seed_initialization": "random (no seed set)",
  "epochs": 3,
  "total_steps": $final_step,
  "expected_steps": $TOTAL_STEPS,
  "final_test_score": "$final_test_score",
  "training_time_seconds": $((final_step * EXPECTED_STEP_TIME)),
  "completion_timestamp": "$ckpt_timestamp",
  "checkpoint_dir": "/ckpts/simplelr_grpo_qwen05_ctx1024_adaptive/phase3_${controller}_s1",
  "status": "completed",
  "configuration": {
    "n_responses_per_prompt": 4,
    "gpu_memory_util": 0.5,
    "max_num_seqs": 32,
    "batch_size": 16,
    "learning_rate": 1e-6,
    "dataset": "MATH500",
    "model": "Qwen2.5-0.5B-Instruct",
    "dtype": "bfloat16"
  },
  "notes": "Adaptive KL controller. Random initialization (no seed). Steps: $final_step/$TOTAL_STEPS"
}
EOF
    
    echo -e "${GREEN}✓ Results saved to $results_file${NC}"
    echo "  Test Score: $final_test_score"
    echo "  Final Step: $final_step/$TOTAL_STEPS"
}

################################################################################
# Function: Clean up and launch next controller
################################################################################
cleanup_and_continue() {
    local current_controller=$1
    local current_index=$2
    local container_name="phase3-${current_controller}-s1"
    
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning up $current_controller container...${NC}"
    docker stop "$container_name" 2>/dev/null || true
    docker rm "$container_name" 2>/dev/null || true
    echo -e "${GREEN}✓ Cleaned up${NC}"
    
    # Launch next controller if available
    if [ $((current_index + 1)) -lt ${#CONTROLLERS[@]} ]; then
        next_controller="${CONTROLLERS[$((current_index + 1))]}"
        echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Launching next controller: $next_controller${NC}"
        sleep 5
        launch_controller "$next_controller"
    else
        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ ALL CONTROLLERS COMPLETED!${NC}"
    fi
}

################################################################################
# MAIN EXECUTION LOOP
################################################################################
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}KL CONTROLLER ORCHESTRATOR${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Controllers: ${CONTROLLERS[*]}"
echo -e "Total steps per controller: $TOTAL_STEPS"
echo -e "Est. time per controller: ~29 hours"
echo -e "Seed: RANDOM (no initialization)"
echo -e "Results: $RESULTS_DIR/"
echo -e "${BLUE}========================================${NC}"
echo ""

# Process each controller
for i in "${!CONTROLLERS[@]}"; do
    controller="${CONTROLLERS[$i]}"
    container_name="phase3-${controller}-s1"
    
    echo -e "${BLUE}\n>>> CONTROLLER $((i+1))/${#CONTROLLERS[@]}: $controller${NC}"
    
    # If container already exists (running or exited), skip launching and monitor/extract
    if docker ps --filter "name=${container_name}" --quiet | grep -q .; then
        echo -e "${YELLOW}Found running container ${container_name}; will monitor${NC}"
    elif docker ps -a --filter "name=${container_name}" --quiet | grep -q .; then
        echo -e "${YELLOW}Found existing container ${container_name} (exited); will attempt to extract/monitor${NC}"
    else
        # Launch controller
        launch_controller "$controller"
    fi

    # Monitor until completion (only if container exists)
    if docker ps -a --filter "name=${container_name}" --quiet | grep -q .; then
        if ! monitor_container "$container_name" "$controller"; then
            echo -e "${RED}✗ $controller monitoring failed${NC}"
            # proceed to extraction attempt even if monitor reported failure
        fi
    else
        echo -e "${RED}✗ ${container_name} not found after launch attempt${NC}"
        continue
    fi
    
    # Extract and save results
    extract_and_save_results "$controller"
    
    # Cleanup and continue
    cleanup_and_continue "$controller" "$i"
    
    # Small delay before next launch
    sleep 5
done

echo -e "${GREEN}\n========================================${NC}"
echo -e "${GREEN}ORCHESTRATION COMPLETE!${NC}"
echo -e "${GREEN}Results: $RESULTS_DIR/${NC}"
echo -e "${GREEN}========================================${NC}"

# Summary
echo -e "\n${BLUE}Results Summary:${NC}"
for controller in "${CONTROLLERS[@]}"; do
    if [ -f "$RESULTS_DIR/${controller}_results.json" ]; then
        test_score=$(grep -o '"final_test_score": "[^"]*"' "$RESULTS_DIR/${controller}_results.json" | cut -d'"' -f4)
        echo "  $controller: $test_score"
    fi
done

# Merge all results into single file (only include files that exist)
echo -e "\n${BLUE}Creating combined results file...${NC}"
all_file="$RESULTS_DIR/all_controllers_results.json"
echo "{" > "$all_file"
first=true
for ctrl in "${CONTROLLERS[@]}"; do
    results_path="$RESULTS_DIR/${ctrl}_results.json"
    if [ -f "$results_path" ]; then
        content=$(cat "$results_path")
    else
        content="null"
    fi
    if [ "$first" = true ]; then
        first=false
    else
        echo "," >> "$all_file"
    fi
    echo "  \"${ctrl}\": ${content}" >> "$all_file"
done
echo "\n}" >> "$all_file"

echo -e "${GREEN}✓ All results saved to $all_file${NC}"
