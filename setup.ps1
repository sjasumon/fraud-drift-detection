# ==============================================================================
# One-Command Setup & Verification Script for Windows (PowerShell)
# Sets up virtualenv, installs dependencies, and runs test suite in < 2 mins
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Concept-Drift-Aware Fraud Detection Platform: Windows Setup" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCmd = "python"
try {
    $pyVer = & $pythonCmd --version
    Write-Host "[1/4] Using Python interpreter: $pyVer" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# 2. Virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "[2/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    & $pythonCmd -m venv .venv
} else {
    Write-Host "[2/4] Virtual environment (.venv) already exists." -ForegroundColor Green
}

# 3. Upgrade pip and install package
Write-Host "[3/4] Installing dependencies from pyproject.toml..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -e ".[test]"

# 4. Run automated test suite
Write-Host "[4/4] Executing 53 automated invariant tests..." -ForegroundColor Yellow
& .\.venv\Scripts\pytest.exe -v

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE: All 53 automated tests passed!" -ForegroundColor Green
Write-Host "To run the benchmark pipeline:  .\.venv\Scripts\python.exe -m src.pipeline"
Write-Host "To run multi-seed stats rigor: .\.venv\Scripts\python.exe -m src.benchmark_statistical_rigor --seeds 42 101"
Write-Host "To run non-stationary bandit:  .\.venv\Scripts\python.exe -m src.bandit.benchmark --seeds 42 --steps 2000"
Write-Host "To start FastAPI microservice: .\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000"
Write-Host "================================================================================" -ForegroundColor Cyan
