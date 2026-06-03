#!/usr/bin/env bash
# Setup script for sumo-rl-loerrach project
# Supports macOS (dev) and Linux/Windows-WSL (training)
set -euo pipefail

echo "=== sumo-rl-loerrach Environment Setup ==="

# Detect OS
OS="$(uname -s)"
echo "Detected OS: $OS"

# --- 0. Python Version Check ---
echo ""
echo "--- Checking Python version ---"
PY_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if echo "$PY_VERSION" | grep -qE "^3\.11$|^3\.12$"; then
    echo "Python $PY_VERSION OK"
else
    echo "WARNING: Python $PY_VERSION detected. Recommended: 3.11 or 3.12"
    echo "  Python 3.13+ has compatibility issues with SUMO/PyTorch/sumo-rl"
    echo "  Install 3.11: brew install python@3.11 (macOS) or pyenv install 3.11"
fi

# --- 1. Conda Detection ---
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo ""
    echo "WARNING: Conda environment '$CONDA_DEFAULT_ENV' detected."
    echo "  This project uses venv, not conda. Deactivate conda first:"
    echo "  conda deactivate"
    echo "  Then re-run this script."
fi

# --- 2. Python Virtual Environment ---
echo ""
echo "--- Setting up Python virtual environment ---"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists, skipping"
fi

source .venv/bin/activate
echo "Activated .venv (Python: $(python --version))"

# --- 3. Python Dependencies ---
echo ""
echo "--- Installing Python dependencies ---"
pip install --upgrade pip
pip install -r requirements.txt
echo "Python dependencies installed"

# --- 4. SUMO Check ---
echo ""
echo "--- Checking SUMO installation ---"
if command -v sumo &> /dev/null; then
    echo "SUMO found: $(sumo --version | head -1)"
else
    echo "WARNING: SUMO not found in PATH!"
    echo ""
    if [ "$OS" = "Darwin" ]; then
        echo "Install on macOS:  brew install sumo"
    elif [ "$OS" = "Linux" ]; then
        echo "Install on Linux:  sudo apt-get install sumo sumo-tools sumo-doc"
    else
        echo "Install on Windows: Download from https://sumo.dlr.de/docs/Downloads.php"
    fi
    echo ""
    echo "After installation, set SUMO_HOME:"
    echo "  export SUMO_HOME=/path/to/sumo"
fi

# --- 5. SUMO_HOME Check ---
echo ""
echo "--- Checking SUMO_HOME ---"
if [ -n "${SUMO_HOME:-}" ]; then
    echo "SUMO_HOME is set: $SUMO_HOME"
    if [ -f "$SUMO_HOME/tools/osmWebWizard.py" ]; then
        echo "osmWebWizard.py found"
    else
        echo "WARNING: osmWebWizard.py not found at $SUMO_HOME/tools/"
    fi
else
    echo "WARNING: SUMO_HOME is not set!"
    echo ""
    echo "Add to your shell profile (~/.zshrc or ~/.bashrc):"
    if [ "$OS" = "Darwin" ]; then
        echo '  export SUMO_HOME="$(brew --prefix sumo)/share/sumo"'
    else
        echo '  export SUMO_HOME="/usr/share/sumo"'
    fi
fi

# --- 6. GPU Check (for training machine) ---
echo ""
echo "--- Checking GPU availability ---"
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Apple MPS (Metal) available')
else:
    print('No GPU detected - CPU training only')
"

# --- 7. Critical Dependency Checks ---
echo ""
echo "--- Verifying critical dependencies ---"
python -c "
import sumo_rl
import stable_baselines3
import gymnasium
import scipy
import tensorboard
import torch

print(f'sumo-rl:           {sumo_rl.__version__}')
print(f'stable-baselines3: {stable_baselines3.__version__}')
print(f'gymnasium:         {gymnasium.__version__}')
print(f'scipy:             {scipy.__version__}')
print(f'torch:             {torch.__version__}')

# CRITICAL: gymnasium version check
assert gymnasium.__version__.startswith('0.29'), \
    f'FATAL: gymnasium {gymnasium.__version__} detected! sumo-rl requires gymnasium<1.0. Run: pip install gymnasium==0.29.1'
print()
print('All imports OK, gymnasium version compatible')
"

# --- 8. Create directories ---
echo ""
echo "--- Ensuring directories exist ---"
mkdir -p logs data/raw/bast data/processed/demand_profiles models/checkpoints results/plots results/csv results/evaluation
echo "Directories OK"

# --- 9. .env Setup ---
echo ""
echo "--- Environment file ---"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example - edit with your paths"
    fi
else
    echo ".env already exists"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: source .venv/bin/activate"
