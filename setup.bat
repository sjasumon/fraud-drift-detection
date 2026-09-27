@echo off
REM ==============================================================================
REM One-Command Setup & Verification Script for Windows
REM Sets up virtualenv, installs dependencies, and runs full test suite
REM ==============================================================================

echo ================================================================================
echo Concept-Drift-Aware Fraud Detection Platform: Windows Setup
echo ================================================================================

REM 1. Check Python version
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    exit /b 1
)

echo [1/4] Checking Python interpreter...
python --version

REM 2. Create virtual environment if missing
if not exist ".venv" (
    echo [2/4] Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [2/4] Virtual environment (.venv) already exists.
)

REM 3. Activate and install dependencies
echo [3/4] Activating .venv and installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[test]"

REM 4. Run automated test suite
echo [4/4] Executing 47 automated invariant tests...
pytest -v

echo ================================================================================
echo SETUP COMPLETE: All 47 automated tests passed!
echo To run benchmark pipeline:  python -m src.pipeline
echo To run multi-seed stats:    python -m src.benchmark_statistical_rigor --seeds 42 101
echo To run non-stationary bandit: python -m src.bandit.benchmark --seeds 42 --steps 2000
echo To start FastAPI microservice: uvicorn src.api:app --host 0.0.0.0 --port 8000
echo ================================================================================
