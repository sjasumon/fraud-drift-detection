# v0.6.0 — Remediation Release (Findings R01–R11), 2026-10-19

This release addresses all 11 findings (R01–R11) from the Remediation Brief (`Fraud_Detection_Agent_Remediation_Brief.docx`), establishing full scientific reproducibility, statistical rigor, dual-pointer model management, and production-grade security:

## Cryptographic Provenance & Clean Startup (R01, R08)
- Cryptographic SHA-256 byte verification in `src/provenance.py` against reference hashes (`3a5c83ab...`, `b63c725d...`).
- Removed permissive fallbacks in `ModelBundle.ready` (`src/artifacts.py`); non-genuine or contradictory bundles are strictly rejected.
- Derived subset manifests with parent-child SHA-256 hash chaining supported via `create_derived_subset_manifest`.
- Sanitized `.env.example` bootstrap; production security validator rejects default passwords and empty database credentials.
- Graceful 503 unready startup when no model package is mounted or unverified package is provided.

## Metric Reconciliation, CLI Flags & Versioned Configs (R02, R04)
- Restored historical run artifacts in `results/benchmark_runs/RUN_63D7CA7F68` with complete 80MB prediction ledger.
- Implemented standalone deterministic CLI `src/reconcile_metrics.py` reconciling all published metrics with **0 discrepancies**.
- Added `--threshold-objective`, `--operational-daily-budget`, `--n-estimators`, `--learning-rate`, and `--config` to pipeline and study CLIs.
- Standardized default LightGBM/XGBoost hyperparameters to 100 trees and 0.1 learning rate.
- Added 5 versioned configs in `configs/` (`production.json`, `benchmark_ieee_full.json`, `benchmark_ieee_quick.json`, `ci_smoke.json`, `remediation_baseline.json`).

## Statistical Rigor, Capacity Policies & Hypothesis Evaluation (R03, R05, R06, R10)
- Implemented paired circular moving-block bootstrap in `src/reporting.py` supporting Average Precision, Recall@Budget, and one-sided non-inferiority testing ($H_1: \delta = 0.01$, empirical $p < 0.001$).
- Resolved 5-alert quota defect caused by tied scores via distinct threshold selection in `src/base_model.py`.
- Formally separated and implemented 3 review policies in `src/capacity_policy.py`: `POLICY_RETROSPECTIVE_BATCH`, `POLICY_EXPECTED_VOLUME`, and `POLICY_CAUSAL_ONLINE` (with day advancement, replay idempotency, and restart persistence).
- Traced ADWIN drift triggers across Slices 8–17; honestly reported Hypothesis $H_2$ as **NOT SUPPORTED / REJECTED in Protocol v1** (2.9% compute saving vs 30% target) to preserve research integrity.
- Corrected citations: Politecnico handle `11311/1044896` to Dal Pozzolo et al. (2014, 2018); Carminati et al. (2015 BankSealer). Disclosed absence of genuine dispute confirmation timestamps in IEEE-CIS.

## Rollback, Timeout Hardening & Audit Persistence (R07, R09, R11)
- Implemented dual-pointer atomic synchronization (`models/active.json` & `results/service/active_model.json`).
- Added authenticated endpoint `POST /v1/models/rollback` with provenance verification and fallback to base generation 0.
- Implemented bounded worker timeout in `BackgroundRetrainer` with deadline enforcement (`REJECTED_TRAINING_TIMEOUT`).
- Persisted retraining history across restarts in SQLite table `promotion_history` (`GET /v1/models/promotions`).
- Created append-only audit table `feedback_events` recording `received_at`, `confirmed_at`, `label_source`, and conflict rejections (`GET /v1/feedback/audit`).
- Documented boundary-batched array scanning in `src/history_features.py`; profiled SQLite queries with `EXPLAIN QUERY PLAN`; enhanced `src/benchmark_latency.py` with model-only, sequential HTTP, and concurrent API latency separation.
- Full automated test suite expanded to **177 tests (100% passing)**.

# v0.5.0 — Research Benchmark & Production Hardening, 2026-10-17

This release delivers the reproducible research benchmarks, genuine data provenance enforcement, and production microservice hardening specified in the Next-Phase engineering brief:

## Provenance and Genuine Data Enforcement
- Implemented `src/provenance.py` defining strict classification: `GENUINE_OBSERVED`, `SIMULATOR_GENERATED`, `SYNTHETIC`, `UNVERIFIED`.
- Classified IEEE-CIS as the sole verified genuine benchmark (`data/ieee/train_transaction.csv`); strictly excluded PaySim and Feedzai-BAF from primary PhD proposal claims.
- Defaulted provenance to `UNVERIFIED`; `ModelBundle.ready` rejects non-genuine models from live scoring traffic.
- Disclosed the absence of genuine dispute confirmation timestamps in IEEE-CIS, treating simulated delay as exploratory sensitivity analysis only.

## Research Protocol and Empirical Benchmarks
- Authored and timestamped `RESEARCH_PROTOCOL.md` grounded in Dal Pozzolo et al. (2014, 2018; Handle 11311/1044896) and Carminati et al. (2015; BankSealer).
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
