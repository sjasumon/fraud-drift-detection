# End-to-End Reproducibility Guide

**Project**: Concept-Drift-Aware Fraud Detection (`fraud-drift-detection` v0.5.0)  
**Target Release**: `V3`  
**Audit Date**: 17 October 2026  
**Deterministic Guarantee**: Master seed $S=42$ (fixed random states across splits, bootstrap, and estimators)

---

## 1. Fresh Checkout & Environment Setup

### 1.1 Clone Repository and Checkout Release Tag
```bash
git clone https://github.com/sjasumon/fraud-drift-detection.git
cd fraud-drift-detection
git checkout V3
```

### 1.2 Python Virtual Environment (Python 3.12 Recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```
Or install the pre-built wheel:
```powershell
pip install dist/fraud_drift_detection-0.5.0-py3-none-any.whl
```

---

## 2. Dataset Acquisition & Verification

Per Section 2 of the specification, all primary research results require the genuine IEEE-CIS dataset.

### 2.1 Download Official IEEE-CIS Files
1. Create an authorized account on Kaggle: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
2. Accept the competition research terms and download `train_transaction.csv` and `train_identity.csv`.
3. Place them into `data/ieee/`:
   ```
   data/
   └── ieee/
       ├── train_transaction.csv
       └── train_identity.csv
   ```

### 2.2 Verify Cryptographic Checksums
Run the following PowerShell command to verify data integrity:
```powershell
Get-FileHash data\ieee\train_transaction.csv -Algorithm SHA256
Get-FileHash data\ieee\train_identity.csv -Algorithm SHA256
```
Expected SHA-256 Hashes:
- `train_transaction.csv`: `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642`
- `train_identity.csv`: `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c`

---

## 3. Running Automated Tests

Run the complete 127-test regression suite:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected output:
```
127 passed, 4 warnings in ~34s
```

---

## 4. Reproducing Empirical Research Benchmarks

### 4.1 Primary Research Benchmark (5 Strategies Across 10 Out-of-Sample Weeks)
Run the frozen chronological research benchmark (Training: Weeks 0–7; Out-of-sample evaluation: Weeks 8–17):
```powershell
.\.venv\Scripts\python.exe -m src.pipeline `
  --dataset data/ieee/train_transaction.csv `
  --train-slices 8 `
  --evaluation-slices 10 `
  --output-dir results/benchmark_runs `
  --seed 42
```
Expected generated outputs in the run directory (`results/benchmark_runs/RUN_<ID>/`):
- `manifest.json`: Full environment, git digest, and provenance metadata.
- `metrics.csv`: Precision, Recall, F1, Average Precision, and Brier scores.
- `alert_budgets.csv`: Daily review capacity metrics at 100 and 500 alerts/day.
- `compute_cost.csv`: Wall-clock training seconds and retrain event counts.
- `confidence_intervals.json`: Paired circular moving-block bootstrap intervals.
- `overview.png`: 4-panel publication figure (PR curves, weekly F1, calibration, compute cost).
- `report.html`: Standalone self-contained HTML audit report.

### 4.2 Feature Ablations
1. **Base Transaction + Identity Join**:
   ```powershell
   .\.venv\Scripts\python.exe -m src.pipeline `
     --dataset data/ieee/train_transaction.csv `
     --identity data/ieee/train_identity.csv `
     --train-slices 8 `
     --evaluation-slices 10 `
     --output-dir results/benchmark_runs
   ```
2. **Base Transaction + History Features**:
   ```powershell
   .\.venv\Scripts\python.exe -m src.pipeline `
     --dataset data/ieee/train_transaction.csv `
     --history-features `
     --train-slices 8 `
     --evaluation-slices 10 `
     --output-dir results/benchmark_runs
   ```
3. **Operational Review Budget Thresholding (100 alerts/day)**:
   ```powershell
   .\.venv\Scripts\python.exe -m src.pipeline `
     --dataset data/ieee/train_transaction.csv `
     --threshold-objective review_budget_recall `
     --operational-daily-budget 100 `
     --train-slices 8 `
     --evaluation-slices 10 `
     --output-dir results/benchmark_runs
   ```

---

## 5. Starting the Operational Microservice

Start the microservice with the active real-data model bundle:
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```
Verify the service health:
```powershell
curl http://127.0.0.1:8000/health
```
