#!/usr/bin/env bash
# Setup script for Linux GPU training machine (Ubuntu/Debian)
# Idempotent: safe to run multiple times
# Usage: chmod +x scripts/setup_linux_training.sh && ./scripts/setup_linux_training.sh
set -euo pipefail

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Track pass/fail for summary
declare -A STATUS
PASS="${GREEN}PASS${NC}"
FAIL="${RED}FAIL${NC}"
WARN="${YELLOW}WARN${NC}"

echo -e "${BLUE}=== sumo-rl-loerrach Linux Training Setup ===${NC}"
echo "Project root: $PROJECT_ROOT"
echo ""

# ----------------------------------------------------------------
# 1. Python 3.11
# ----------------------------------------------------------------
echo -e "${BLUE}--- [1/8] Python 3.11 ---${NC}"
if command -v python3.11 &>/dev/null; then
    PY_VERSION=$(python3.11 --version)
    echo -e "  ${GREEN}Found: $PY_VERSION${NC}"
    STATUS[python]="$PASS"
else
    echo "  Python 3.11 not found. Installing via deadsnakes PPA..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    if command -v python3.11 &>/dev/null; then
        echo -e "  ${GREEN}Installed: $(python3.11 --version)${NC}"
        STATUS[python]="$PASS"
    else
        echo -e "  ${RED}Installation failed!${NC}"
        echo "  Fix: sudo apt install python3.11 python3.11-venv python3.11-dev"
        STATUS[python]="$FAIL"
    fi
fi

# ----------------------------------------------------------------
# 2. SUMO
# ----------------------------------------------------------------
echo ""
echo -e "${BLUE}--- [2/8] SUMO ---${NC}"
if command -v sumo &>/dev/null; then
    SUMO_VER=$(sumo --version 2>&1 | head -1)
    echo -e "  ${GREEN}Found: $SUMO_VER${NC}"
    STATUS[sumo]="$PASS"
else
    echo "  SUMO not found. Installing via PPA..."
    sudo add-apt-repository -y ppa:sumo/stable
    sudo apt-get update
    sudo apt-get install -y sumo sumo-tools sumo-doc
    if command -v sumo &>/dev/null; then
        echo -e "  ${GREEN}Installed: $(sumo --version 2>&1 | head -1)${NC}"
        STATUS[sumo]="$PASS"
    else
        echo -e "  ${RED}Installation failed!${NC}"
        echo "  Fix: sudo apt-get install sumo sumo-tools sumo-doc"
        STATUS[sumo]="$FAIL"
    fi
fi

# ----------------------------------------------------------------
# 3. SUMO_HOME
# ----------------------------------------------------------------
echo ""
echo -e "${BLUE}--- [3/8] SUMO_HOME ---${NC}"
SUMO_HOME_DEFAULT="/usr/share/sumo"
if [ -n "${SUMO_HOME:-}" ]; then
    echo -e "  ${GREEN}SUMO_HOME already set: $SUMO_HOME${NC}"
    STATUS[sumo_home]="$PASS"
elif [ -d "$SUMO_HOME_DEFAULT" ]; then
    export SUMO_HOME="$SUMO_HOME_DEFAULT"
    # Add to .bashrc if not already there
    if ! grep -q 'export SUMO_HOME=' ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# SUMO environment (added by sumo-rl-loerrach setup)" >> ~/.bashrc
        echo "export SUMO_HOME=\"$SUMO_HOME_DEFAULT\"" >> ~/.bashrc
        echo "  Added SUMO_HOME=$SUMO_HOME_DEFAULT to ~/.bashrc"
    fi
    echo -e "  ${GREEN}Set SUMO_HOME=$SUMO_HOME_DEFAULT${NC}"
    STATUS[sumo_home]="$PASS"
else
    echo -e "  ${RED}$SUMO_HOME_DEFAULT not found!${NC}"
    echo "  Fix: Install SUMO first, then set: export SUMO_HOME=/path/to/sumo"
    STATUS[sumo_home]="$FAIL"
fi

# ----------------------------------------------------------------
# 4. Virtual Environment + Dependencies
# ----------------------------------------------------------------
echo ""
echo -e "${BLUE}--- [4/8] Python venv + dependencies ---${NC}"
if [ ! -d ".venv" ]; then
    echo "  Creating .venv with Python 3.11..."
    python3.11 -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "  Activated .venv ($(python --version))"

echo "  Installing requirements.txt (runtime deps only)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "  ${GREEN}Dependencies installed${NC}"
STATUS[venv]="$PASS"

# ----------------------------------------------------------------
# 5. CUDA / GPU
# ----------------------------------------------------------------
echo ""
echo -e "${BLUE}--- [5/8] CUDA / GPU ---${NC}"
GPU_INFO=$(python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'{name} ({vram:.1f} GB VRAM)')
else:
    print('CPU only')
" 2>&1)
echo "  Device: $GPU_INFO"
if echo "$GPU_INFO" | grep -q "CPU only"; then
    echo -e "  ${YELLOW}No GPU detected. Training will be slow.${NC}"
    echo "  Fix: Install NVIDIA drivers + CUDA toolkit, then reinstall torch with CUDA support"
    STATUS[gpu]="$WARN"
else
    echo -e "  ${GREEN}GPU available${NC}"
    STATUS[gpu]="$PASS"
fi

# ----------------------------------------------------------------
# 6. Critical Imports
# ----------------------------------------------------------------
echo ""
echo -e "${BLUE}--- [6/8] Critical imports ---${NC}"
IMPORT_CHECK=$(python -c "
import sumo_rl
import stable_baselines3
import gymnasium
import scipy
import torch

print(f'sumo-rl:           {sumo_rl.__version__}')
print(f'stable-baselines3: {stable_baselines3.__version__}')
print(f'gymnasium:         {gymnasium.__version__}')
print(f'scipy:             {scipy.__version__}')
print(f'torch:             {torch.__version__}')

assert gymnasium.__version__.startswith('0.29'), \
    f'FATAL: gymnasium {gymnasium.__version__} - need 0.29.x!'
print('gymnasium version OK (0.29.x)')
" 2>&1) || true
echo "$IMPORT_CHECK" | sed 's/^/  /'
if echo "$IMPORT_CHECK" | grep -q "FATAL\|Error\|ModuleNotFoundError"; then
    echo -e "  ${RED}Import check failed!${NC}"
    echo "  Fix: pip install -r requirements.txt"
    STATUS[imports]="$FAIL"
else
    echo -e "  ${GREEN}All imports OK${NC}"
    STATUS[imports]="$PASS"
fi

# ----------------------------------------------------------------
# 7. libsumo
# ----------------------------------------------------------------
echo ""
echo -e "${BLUE}--- [7/8] libsumo (headless mode) ---${NC}"
LIBSUMO_CHECK=$(LIBSUMO_AS_TRACI=1 python -c "
import libsumo
print('libsumo imported OK')
" 2>&1) || true
if echo "$LIBSUMO_CHECK" | grep -q "OK"; then
    echo -e "  ${GREEN}libsumo available (LIBSUMO_AS_TRACI=1 will work)${NC}"
    STATUS[libsumo]="$PASS"
else
    echo -e "  ${YELLOW}libsumo not available - training will use TraCI (slower)${NC}"
    echo "  This is OK for training but ~8x slower than libsumo."
    echo "  Fix: pip install eclipse-sumo (or install SUMO from PPA)"
    STATUS[libsumo]="$WARN"
fi

# ----------------------------------------------------------------
# 8. Smoke Test
# ----------------------------------------------------------------
echo ""
echo -e "${BLUE}--- [8/8] Smoke test (300s random-agent episode) ---${NC}"
NET_FILE="$PROJECT_ROOT/data/sumo_config/loerrach.net.xml"
ROUTE_FILE="$PROJECT_ROOT/data/sumo_config/loerrach.rou.xml"

if [ ! -f "$NET_FILE" ] || [ ! -f "$ROUTE_FILE" ]; then
    echo -e "  ${RED}SUMO files missing!${NC}"
    echo "  Expected: $NET_FILE"
    echo "  Expected: $ROUTE_FILE"
    STATUS[smoke]="$FAIL"
else
    SMOKE_RESULT=$(LIBSUMO_AS_TRACI=1 python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from sumo_rl import SumoEnvironment

env = SumoEnvironment(
    net_file='$NET_FILE',
    route_file='$ROUTE_FILE',
    use_gui=False,
    num_seconds=300,
    delta_time=5,
    yellow_time=3,
    min_green=10,
    reward_fn='diff-waiting-time',
    sumo_seed=42,
    single_agent=True,
    time_to_teleport=-1,
)
obs, info = env.reset()
steps = 0
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1

n_metrics = len(env.metrics)
env.close()
print(f'OK: {steps} steps, {n_metrics} metric rows')
" 2>&1) || true

    if echo "$SMOKE_RESULT" | grep -q "^OK:"; then
        echo -e "  ${GREEN}$SMOKE_RESULT${NC}"
        STATUS[smoke]="$PASS"
    else
        echo -e "  ${RED}Smoke test failed!${NC}"
        echo "$SMOKE_RESULT" | tail -5 | sed 's/^/  /'
        echo "  Fix: Check SUMO installation and SUMO_HOME setting"
        STATUS[smoke]="$FAIL"
    fi
fi

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Setup Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  Python 3.11:    ${STATUS[python]}"
echo -e "  SUMO:           ${STATUS[sumo]}"
echo -e "  SUMO_HOME:      ${STATUS[sumo_home]}"
echo -e "  venv + deps:    ${STATUS[venv]}"
echo -e "  GPU/CUDA:       ${STATUS[gpu]}"
echo -e "  Imports:        ${STATUS[imports]}"
echo -e "  libsumo:        ${STATUS[libsumo]}"
echo -e "  Smoke test:     ${STATUS[smoke]}"
echo -e "${BLUE}========================================${NC}"

# Check for any failures
HAS_FAIL=false
for key in "${!STATUS[@]}"; do
    if echo -e "${STATUS[$key]}" | grep -q "FAIL"; then
        HAS_FAIL=true
    fi
done

if $HAS_FAIL; then
    echo -e "${RED}Some checks failed. Fix the issues above before training.${NC}"
    exit 1
else
    echo -e "${GREEN}Ready for training!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Implement train_dqn.py and train_ppo.py (Week 5)"
    echo "  2. Run: ./scripts/run_training.sh --timesteps 500000"
    echo "  3. Monitor: tensorboard --logdir runs/"
fi
