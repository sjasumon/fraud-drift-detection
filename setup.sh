#!/usr/bin/env bash
# ==============================================================================
# One-Command Setup & Verification Script for Linux / macOS / WSL
# Clones, sets up virtualenv, installs dependencies, and runs test suite in < 2 mins
# ==============================================================================

set -e

echo "================================================================================"
echo "Concept-Drift-Aware Fraud Detection Platform: Automated Setup"
echo "Target Platform: Linux / macOS / WSL (Python 3.10+)"
echo "================================================================================"

# 1. Check Python version
if command -v python3.12 &>/dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    echo "ERROR: Python 3 is not installed or not in PATH."
    exit 1
fi

echo "[1/4] Using Python interpreter: $($PYTHON_CMD --version)"

# 2. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment (.venv)..."
    $PYTHON_CMD -m venv .venv
else
    echo "[2/4] Virtual environment (.venv) already exists."
fi

# Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install package with all test dependencies
echo "[3/4] Installing dependencies from pyproject.toml..."
pip install --upgrade pip setuptools wheel
pip install -e ".[test]"

# 4. Run automated test suite
echo "[4/4] Executing 51 automated invariant tests..."
pytest -v

echo "================================================================================"
echo "SETUP COMPLETE: All 51 automated tests passed!"
echo "To run the benchmark pipeline:  python -m src.pipeline"
echo "To run multi-seed stats rigor: python -m src.benchmark_statistical_rigor --seeds 42 101"
echo "To run non-stationary bandit:  python -m src.bandit.benchmark --seeds 42 --steps 2000"
echo "To start FastAPI microservice: uvicorn src.api:app --host 0.0.0.0 --port 8000"
echo "================================================================================"
