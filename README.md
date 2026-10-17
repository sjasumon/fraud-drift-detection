# Fraud Drift Detection — v0.4.1

A research framework for chronological fraud detection, delayed verification, and
comparison of static, scheduled, sliding-window, and drift-triggered retraining.

**Start here:** [CHANGELOG.md](CHANGELOG.md), [REVIEW_CORRECTIONS.md](REVIEW_CORRECTIONS.md),
[TASK_COMPLETION_AUDIT.md](TASK_COMPLETION_AUDIT.md), and [VALIDATION.md](VALIDATION.md). Existing real-data scores are historical; the corrected
protocol must be rerun before claiming an accuracy, latency, or compute improvement.

## Install (Python 3.12)

Windows PowerShell:
```powershell
.\setup.ps1
```
Linux/macOS:
```bash
bash setup.sh
```
Or create a virtual environment and run:
```bash
python -m pip install -e ".[test]"
python -m pytest -q
```
CPU is the default. Set `USE_GPU=true` for optional XGBoost CUDA; record the actual
reported device. GPU and CPU results are not assumed bit-for-bit identical.

## Run the included synthetic example

```bash
python -m src.demo --output-dir results/demo
```
Open the printed `report.html`. This small run verifies behavior; it is not evidence
of real-world fraud accuracy. Synthetic model packages cannot be served by the API.

## Run a real-data benchmark

Raw datasets are not redistributed. Download IEEE-CIS from the official competition:
https://www.kaggle.com/c/ieee-fraud-detection/data

```bash
python -m src.pipeline --dataset data/ieee/train_transaction.csv --train-slices 5 --output-dir results/runs
```
Optional feature and memory experiments:
```bash
python -m src.pipeline --dataset data/ieee/train_transaction.csv --identity data/ieee/train_identity.csv --all-features --history-features --adaptive-window 4 --explain
python -m src.pipeline --dataset data/paysim/paysim_full.csv --pre-transaction-only
```
Every run has its own unique ID and directory:
- `manifest.json`: dataset/identity/source hashes, seed, configuration, environment, actual device.
- `predictions.csv`: OOS decisions, probabilities, thresholds, labels, model generations.
- `summary.json`, `metrics.csv`, `weekly_metrics.csv`: reconciled OOS metrics.
- `alert_budgets.csv`: precision/recall at 100 and 500 daily review slots.
- `confidence_intervals.json`: paired paired moving-block bootstrap with shared time draws across seeds F1 differences.
- `retrain_events.json`, `compute_cost.csv`: actual measured retraining durations.
- `drift_events.json`, `attributions.json`: detector evidence and optional diagnostic rankings.
- `error_examples.csv`: sampled false alarms and missed frauds, with transaction references.
- `overview.png`, `report.html`: PR, calibration, weekly performance and compute trade-offs.
- `models/<strategy>/`: model package, preprocessing, calibration, threshold and schema.

Regenerate a report using exactly its saved ledger:
```bash
python -m src.generate_key_figures --run-dir results/runs/RUN_YOUR_ID
```

## Compare seeds, memory windows, label delays and compute budgets

```bash
python -m src.study --dataset data/ieee/train_transaction.csv --output-dir results/study --seeds 42 101 202 303 404 --train-windows 5 10 15 --test-slices 5 --windows 0 2 4 8
python -m src.study --dataset data/ieee/train_transaction.csv --output-dir results/budget_study --train-windows 5 10 15 --test-slices 5 --windows 0 4 --budgets 2 4 --half-lives 0 4 --delays 0 4 8
python -m src.study --dataset data/paysim/paysim_full.csv --output-dir results/paysim_ablation --train-windows 2 --test-slices 3 --windows 0 2 --paysim-ablation
```
`--windows 0` means all available history; `--half-lives 0` disables age weighting.
Each configuration evaluates all five strategies, including every-slice retraining.
A retrain-count cap applies to all adaptive/scheduled strategies; measured seconds are
also reported because equal counts do not imply equal cost. Evaluation periods must
not overlap. The configuration sweep is exploratory: select settings on development
periods and confirm once on an untouched future period. More seeds do not replace
independent time periods or datasets.

## Serve an exported real-data model

PowerShell:
```powershell
$env:MODEL_BUNDLE_PATH="results/runs/RUN_YOUR_ID/models/adaptive"
$env:API_KEY="replace-with-a-long-random-secret"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```
Linux/macOS:
```bash
MODEL_BUNDLE_PATH=results/runs/RUN_YOUR_ID/models/adaptive API_KEY=your-long-secret python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```
Call `/v1/schema` with `X-API-Key` to obtain required raw/input features. Send every
required field, using JSON null for an observed missing value. The package applies
its fitted preprocessing. If historical features are enabled, supply those values
from a causal feature store. `/v1/score` returns the threshold and model generation.
No default random model is created. Missing or synthetic model packages yield 503.
Only load trusted local model packages: joblib deserialization is not a safe format
for arbitrary third-party files; checksums detect corruption, not malicious authors.

Accepted serving revisions are restored on restart via an atomic active-model pointer.
Model packages contain fitted state and schema, without training rows.
SQLite stores original predictions/features and idempotent labels. `/v1/feedback`
uses the original decision, never a new prediction. Duplicate conflicting feedback
is rejected. Feedback from older generations remains eligible for training but does
not pollute the current generation's detector. Background candidates require >=100
verified rows and must improve F1 on a separate chronological promotion holdout.
This is a research promotion rule, not a complete production acceptance policy.
Run one worker; distributed coordination is outside this release.

The optional MySQL experiment adapter remains available through the Python
`run_pipeline(..., db=...)` interface. SQLite serving persistence is independent of it.

## Protocol and limitations

All strategies clone one common initial estimator. The fitting partition precedes
its calibration/threshold partition; both precede OOS evaluation. Detectors start on
OOS predictions only. Labels become available at the end of their maturity slice.
Only matured labels enter updates. Recent windows end at the latest verified transaction arrival slice; this remains
usable when label delay is longer than the window. Label staleness is logged.
Detector updates use original decisions from the active generation only. Small validation sets may disable calibration; the manifest
states this explicitly. F1 threshold optimization is a development policy; daily
alert budgets expose the resulting operational trade-off.

Daily budgets are retrospective batch-review rankings, not a promise of live routing.
PaySim `step` supplies relative hours. Feedzai has month-level time; its within-month
steps are simulated, so daily budget/time-of-day interpretation is synthetic there.
Identity keys are proxies, not proven unique people. SHAP/PSI rankings are diagnostic,
not causal proof. Average precision is used consistently for the main PR metric.

The original documents and results remain in `docs/archive_v0.3/` and
`results/legacy_v0.3/`. The bandit simulator is a separate research experiment; it is
not integrated into transaction scoring and does not establish fraud-model gains.

## Audits and latency commands

```bash
fraud-audit --dataset data/ieee/train_transaction.csv --mode feature-blind --output-dir results/cohorts
fraud-latency-audit --dataset data/ieee/train_transaction.csv --output-dir results/delays --delays 0 4 8
fraud-stats-rigor --benchmark paysim --paysim-dataset data/paysim/paysim_full.csv --output-dir results/statistics --seeds 42 101
fraud-latency --features transaction_features.json --requests 1000
```

Use `--synthetic` on experimental commands when using generated data. The API
requires `MODEL_BUNDLE_PATH` for a real-data package. Latency commands measure
successful authenticated in-process requests, including the prediction ledger;
production network/concurrency latency remains a separate measurement.

The model package can be installed as a wheel. Runtime outputs and `.env` are
resolved from the working directory, or `FRAUD_WORKSPACE` if explicitly set.
A study also accepts `--identity`, `--all-features`, `--history-features`,
`--exclude-identifiers`, `--latency-mode`, `--lognormal-mu`, and `--lognormal-sigma`.
