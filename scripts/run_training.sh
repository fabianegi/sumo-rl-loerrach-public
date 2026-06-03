#!/usr/bin/env bash
# Training runner: trains all model variants sequentially, then evaluates.
# Usage: ./scripts/run_training.sh [--timesteps 500000] [--seed 42]
#
# Trains 4 variants:
#   1. DQN + diff-waiting-time
#   2. DQN + pressure
#   3. PPO + diff-waiting-time
#   4. PPO + pressure
#
# Requires: train_dqn.py and train_ppo.py to be implemented (Week 5+)
set -euo pipefail

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# --- Defaults ---
TIMESTEPS=1000000
SEED=42
NETWORK="single"
DATE=$(date +%Y-%m-%d)
LOG_FILE="logs/training_${DATE}.log"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --timesteps) TIMESTEPS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --network) NETWORK="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--timesteps N] [--seed N] [--network single|corridor]"
            echo "  --timesteps  Total training steps per model (default: 1000000)"
            echo "  --seed       Random seed (default: 42)"
            echo "  --network    Network type: single (default) or corridor"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --- Pre-flight checks ---
echo -e "${BLUE}=== sumo-rl-loerrach Training Runner ===${NC}"
echo "Date:      $DATE"
echo "Timesteps: $TIMESTEPS"
echo "Seed:      $SEED"
echo "Network:   $NETWORK"
echo ""

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "Activated .venv ($(python --version))"
else
    echo -e "${RED}ERROR: .venv not found! Run setup_linux_training.sh first.${NC}"
    exit 1
fi

# Select network files based on --network flag
if [ "$NETWORK" = "corridor" ]; then
    NET_FILE="data/sumo_config/loerrach_corridor.net.xml"
    ROUTE_FILE="data/sumo_config/loerrach_corridor_medium.rou.xml"
else
    NET_FILE="data/sumo_config/loerrach.net.xml"
    ROUTE_FILE="data/sumo_config/loerrach.rou.xml"
fi

if [ ! -f "$NET_FILE" ]; then
    echo -e "${RED}ERROR: Network file not found: $NET_FILE${NC}"
    exit 1
fi
if [ ! -f "$ROUTE_FILE" ]; then
    echo -e "${RED}ERROR: Route file not found: $ROUTE_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}SUMO files OK ($NETWORK network)${NC}"

# Enable libsumo for headless speed (~8x faster)
export LIBSUMO_AS_TRACI=1
echo "LIBSUMO_AS_TRACI=1 (headless mode)"

# Detect GPU
DEVICE=$(python -c "
import torch
if torch.cuda.is_available():
    print(f'cuda ({torch.cuda.get_device_name(0)})')
else:
    print('cpu')
" 2>&1)
echo "Device: $DEVICE"

# Create directories
mkdir -p logs models/checkpoints results/csv results/plots

echo ""
echo -e "${BLUE}Starting training at $(date +%H:%M:%S)${NC}"
echo "Logging to: $LOG_FILE"
echo ""

# --- Training variants ---
declare -a MODELS=()
declare -a TIMES=()
declare -a NAMES=()
TOTAL_START=$SECONDS

# Helper function
train_variant() {
    local name="$1"
    local algo="$2"
    local reward="$3"
    local script="$4"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Training: $name${NC}"
    echo -e "${BLUE}  Algorithm: $algo | Reward: $reward${NC}"
    echo -e "${BLUE}========================================${NC}"

    local start=$SECONDS

    python "$script" \
        --total-timesteps "$TIMESTEPS" \
        --seed "$SEED" \
        --reward "$reward" \
        --net-file "$NET_FILE" \
        --route-file "$ROUTE_FILE" \
        2>&1 | tee -a "$LOG_FILE"

    local elapsed=$(( SECONDS - start ))
    local minutes=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    echo -e "${GREEN}  Completed in ${minutes}m ${secs}s${NC}"
    echo ""

    NAMES+=("$name")
    TIMES+=("${minutes}m ${secs}s")
}

# 1. DQN + diff-waiting-time
train_variant \
    "DQN (diff-waiting-time)" \
    "DQN" \
    "diff-waiting-time" \
    "src/training/train_dqn.py"

# 2. DQN + pressure
train_variant \
    "DQN (pressure)" \
    "DQN" \
    "pressure" \
    "src/training/train_dqn.py"

# 3. PPO + diff-waiting-time
train_variant \
    "PPO (diff-waiting-time)" \
    "PPO" \
    "diff-waiting-time" \
    "src/training/train_ppo.py"

# 4. PPO + pressure
train_variant \
    "PPO (pressure)" \
    "PPO" \
    "pressure" \
    "src/training/train_ppo.py"

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SEC=$(( TOTAL_ELAPSED % 60 ))

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  All training complete${NC}"
echo -e "${BLUE}  Total time: ${TOTAL_MIN}m ${TOTAL_SEC}s${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# --- Evaluation ---
echo -e "${BLUE}--- Running evaluation ---${NC}"

# Find trained models
echo "Looking for trained models in models/checkpoints/..."
for model_file in models/checkpoints/*.zip; do
    if [ -f "$model_file" ]; then
        echo "  Evaluating: $model_file"
        python src/evaluation/evaluate.py \
            --model "$model_file" \
            --n-episodes 30 \
            2>&1 | tee -a "$LOG_FILE" || true
    fi
done

# Run baseline
echo ""
echo "Running fixed-time baseline..."
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
from pathlib import Path
from src.evaluation.baseline import run_fixed_time_baseline
results = run_fixed_time_baseline(
    net_file=Path('data/sumo_config/loerrach.net.xml'),
    route_file=Path('data/sumo_config/loerrach.rou.xml'),
    n_episodes=30,
)
results.to_csv('results/csv/baseline_fixed_time.csv', index=False)
print(f'Baseline saved: results/csv/baseline_fixed_time.csv')
print(results.describe())
" 2>&1 | tee -a "$LOG_FILE" || true

# --- Write TRAINING_SUMMARY.md ---
SUMMARY_FILE="results/TRAINING_SUMMARY.md"
{
    echo "# Training Summary"
    echo ""
    echo "- **Date:** $DATE"
    echo "- **Device:** $DEVICE"
    echo "- **Total timesteps:** $TIMESTEPS per variant"
    echo "- **Seed:** $SEED"
    echo "- **Total wall-clock time:** ${TOTAL_MIN}m ${TOTAL_SEC}s"
    echo ""
    echo "## Training Times"
    echo ""
    echo "| Variant | Wall-Clock Time |"
    echo "|---|---|"
    for i in "${!NAMES[@]}"; do
        echo "| ${NAMES[$i]} | ${TIMES[$i]} |"
    done
    echo ""
    echo "## Model Checkpoints"
    echo ""
    echo '```'
    ls -la models/checkpoints/*.zip 2>/dev/null || echo "No model files found"
    echo '```'
    echo ""
    echo "## Evaluation Results"
    echo ""
    echo '```'
    ls -la results/csv/*.csv 2>/dev/null || echo "No CSV files found"
    echo '```'
} > "$SUMMARY_FILE"

echo ""
echo -e "${GREEN}Training summary written to: $SUMMARY_FILE${NC}"

# --- Post-training analysis ---
echo ""
echo -e "${BLUE}--- Post-training analysis ---${NC}"

# Checkpoint evaluation (find best DQN model)
BEST_DQN=$(ls -t models/checkpoints/dqn_diff-waiting-time_*.zip 2>/dev/null | head -1)
if [ -n "$BEST_DQN" ]; then
    echo "Signal plan analysis: $BEST_DQN"
    python scripts/analyze_signal_plan.py --model "$BEST_DQN" \
        2>&1 | tee -a "$LOG_FILE" || true

    echo "Checkpoint evaluation..."
    python scripts/evaluate_checkpoints.py \
        --checkpoint-dir models/checkpoints/ \
        --prefix dqn_diff-waiting-time \
        2>&1 | tee -a "$LOG_FILE" || true
fi

# Generate verification report
echo "Generating verification report..."
python scripts/generate_report.py \
    --results-dir results/ \
    2>&1 | tee -a "$LOG_FILE" || true

echo ""
echo -e "${GREEN}=== All done ===${NC}"
echo -e "${GREEN}Training logs at: $LOG_FILE${NC}"
echo ""
echo "Outputs:"
echo "  Training summary: $SUMMARY_FILE"
echo "  Verification report: results/VERIFICATION_REPORT.md"
echo "  Signal plans: results/signal_plans/"
echo "  Training curves: tensorboard --logdir runs/"
