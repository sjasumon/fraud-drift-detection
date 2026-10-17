# v0.5.0 — Research Benchmark & Production Hardening, 2026-10-17

This release delivers the reproducible research benchmarks, genuine data provenance enforcement, and production microservice hardening specified in the Next-Phase engineering brief:

## Provenance and Genuine Data Enforcement
- Implemented `src/provenance.py` defining strict classification: `GENUINE_OBSERVED`, `SIMULATOR_GENERATED`, `SYNTHETIC`, `UNVERIFIED`.
- Classified IEEE-CIS as the sole verified genuine benchmark (`data/ieee/train_transaction.csv`); strictly excluded PaySim and Feedzai-BAF from primary PhD proposal claims.
- Defaulted provenance to `UNVERIFIED`; `ModelBundle.ready` rejects non-genuine models from live scoring traffic.
- Disclosed the absence of genuine dispute confirmation timestamps in IEEE-CIS, treating simulated delay as exploratory sensitivity analysis only.

## Research Protocol and Empirical Benchmarks
- Authored and timestamped `RESEARCH_PROTOCOL.md` grounded in Carminati et al. (2015) and Dal Pozzolo et al. (2015, 2018).
- Executed the frozen chronological benchmark across 5 strategies on 221,727 pure out-of-sample transactions (Weeks 8–17).
- Demonstrated statistically significant outperformance of drift-adaptive retraining over static baseline ($\Delta F1 = +0.0123$, 95% CI $[+0.0052, +0.0188]$) and superior fraud recall under fixed daily review budgets (100 and 500 alerts/day).

## Leakage Prevention and Operational Policies
- Hardened `src/history_features.py` against intra-second causal leakage on tied timestamps ($t < T$).
- Added operational review budget threshold selection (`threshold_objective="review_budget_recall"`) in `BaseFraudModel`.
- Enhanced `BackgroundRetrainer` in `src/api.py` with an atomic improvement margin check ($\ge 0.005$), timeout guard ($\le 60$s), and structured `promotion_log`.

## Performance Optimization and Infrastructure
- Vectorized entity history feature computation (0.138s for 50,000 transactions).
- Added SQLite compound index `idx_predictions_family_label_time` on `(family_id, label, event_time)`.
- Verified live FastAPI microservice on `http://127.0.0.1:8000` with authentication, schema discovery, duplicate rejection, and recovery.
- Added comprehensive unit tests in `tests/test_v050_improvements.py`, bringing automated suite to 127 passing tests.

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
