# v0.4.1 — follow-up audit, 2026-10-16

The v0.4.0 delivery had gaps despite its passing tests. This corrective release:

- Anchors recent windows to the newest verified arrival and logs label staleness.
- Prevents unchanged training data from consuming retrain budgets.
- Filters delayed detector updates by original model generation.
- Removes training rows from immutable, atomically published model packages.
- Persists/restores accepted serving revisions and closes SQLite connections.
- Uses one experiment pipeline for main, delayed, cohort and statistical runs.
- Includes precision, average precision, review budgets and measured compute in studies.
- Uses shared temporal bootstrap draws across seeds and rejects inconsistent labels.
- Preserves zero-fraud cohort counts and checks all three identity components.
- Provides installed CLI entrypoints for both delayed audits and latency measurement.
- Uses the working directory/FRAUD_WORKSPACE for data and .env, including wheel installs.
- Adds failure-oriented regression tests, formats the code and removes unused imports.

See TASK_COMPLETION_AUDIT.md and VALIDATION.md for exact evidence and remaining
real-data/environment work. The earlier release's test result is historical only.

# v0.4.0 — initial implementation, 2026-10-16

## Model and evaluation
- Centralized estimator parameters, chronological validation, sigmoid calibration,
  and F1 threshold tuning in BaseFraudModel.
- Cloned the initial estimator across five strategies; added every-slice baseline.
- Removed adaptive warmup updates; seeded/deduplicated history exactly once.
- Added recent-window and age-weighted adaptive retraining and shared retrain caps.
- Added fixed/lognormal label delays to the main benchmark using original decisions.
- Corrected legacy delayed audit and first-arrival cohort calculations.
- Added seed/time-window studies and paired block-bootstrap F1 intervals.

## Features and diagnostics
- Optional full IEEE transaction columns and one-to-one identity-file join.
- Automatic categorical handling, email domains and missingness indicators.
- Strictly past-only entity history features.
- Pre-transaction PaySim feature-availability ablation.
- Fixed constant-feature PSI and changed attribution wording to diagnostic evidence.

## Serving
- Explicit model package loading, readiness enforcement, strict input schema.
- Bundled preprocessing, calibrator, threshold, metadata and integrity checksum.
- Atomic bundle/generation snapshot, SQLite prediction persistence and idempotent labels.
- Old-generation feedback excluded from current detector statistics.
- Background retraining uses verified data and a separate chronological promotion gate.
- Authenticated latency benchmark verifies success; single-worker deployment default.

## Outputs and documentation
- Immutable per-run manifests, prediction ledgers and model packages.
- OOS-only reports, PR/calibration/weekly plots, review budgets, measured compute,
  paired confidence intervals, and false-alarm/missed-fraud examples.
- Preserved original results/documents under explicitly historical directories.
- Replaced unsupported empirical claims with a reproducible protocol.
- Updated install scripts, dependency specification, container and CI configuration.
- Added regression coverage for the corrected behavior; see VALIDATION.md.

## Migration
The API now expects the complete input schema from `/v1/schema` and returns 503
until a real-data package is loaded. Duplicate score IDs return 409. Prediction
probability overrides are no longer accepted for unknown feedback IDs. Output
paths are run-specific. Old unversioned JSON files are not current results.
