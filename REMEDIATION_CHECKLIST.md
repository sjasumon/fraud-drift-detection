# Comprehensive Remediation Checklist — Findings R01–R11 (Version 0.7.0 / Tag `V5`)

**Specification**: `Fraud_Detection_v060_Followup_Review.txt` (Independent Review Date: 19 October 2026)  
**Audit Baseline**: Version 0.6.0 (Tag `V4`, commit `0af0557`, baseline archive SHA-256 `a2f4ef592f814be4e846aabcf75684ee44482e143f23f1cb965eed7b1ed33ee7`)  
**Target Release**: Version 0.7.0 (Tag `V5`)  
**Repository Working Directory**: `H:\Projects\fraud-drift-detection`  
**Test Suite**: 193 automated tests across unit, integration, reconciliation, lifecycle, and remediation suites (193 passed, 0 failed, 0 skipped).

---

## 1. Summary Matrix: Findings R01–R11 Disposition

| Finding | Topic | Priority | Reviewer Finding & Flaw | Status | Evidence & Test Paths |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **[R01]** | Derived Subsets & Provenance Acceptance | P0 | `create_derived_subset_manifest` accepted arbitrary child files with a claimed parent digest without verifying parent bytes; test doubles were accepted by production `ModelBundle.ready`. | **CLOSED** | [`src/provenance.py`](file:///H:/Projects/fraud-drift-detection/src/provenance.py), [`src/artifacts.py`](file:///H:/Projects/fraud-drift-detection/src/artifacts.py); tests in `test_v070_remediation_audit.py::test_r01_*`, `test_remediation_r01_r08.py`. |
| **[R02]** | Historical Evidence & Strict Reconciliation | P0/P1 | Review archive omitted `results/benchmark_runs/RUN_63D7CA7F68`; reconciler accepted corrupted confusion counts (999999), corrupted retrain counts, missing comparison files, and empty run ledgers. | **CLOSED** | [`.gitignore`](file:///H:/Projects/fraud-drift-detection/.gitignore), [`src/reconcile_metrics.py`](file:///H:/Projects/fraud-drift-detection/src/reconcile_metrics.py); `RUN_63D7CA7F68` fully tracked and reconciled; tests in `test_remediation_r02_r04.py`, `test_v070_remediation_audit.py::test_r02_*`. |
| **[R03]** | Research Questions & Traceability | P1/P2 | Negative $H_2$ outcome was noted; historical experiments required full ledger traceability without synthetic substitutions or fabricated evidence. | **CLOSED** | Deterministic ledger in `results/benchmark_runs/RUN_63D7CA7F68/predictions.csv` (221,727 rows × 5 models); honest negative $H_2$ conclusion retained (adaptive retrain efficiency 2.9% saving with 10 vs 10 retrains does not support efficiency claim). |
| **[R04]** | Configuration Forwarding & CLI Overrides | P0/P1 | CLI parser defaults overwrote configured parameters (150 trees -> 100, lr 0.08 -> 0.1); historical protocol version and strategies were not forwarded; study CLI lacked `--config`. | **CLOSED** | [`src/pipeline.py`](file:///H:/Projects/fraud-drift-detection/src/pipeline.py), [`src/study.py`](file:///H:/Projects/fraud-drift-detection/src/study.py); tracked explicit CLI flags via `sys.argv`; canonical parameter merger; added `--config` to `study.py`; tests in `test_v070_remediation_audit.py::test_r04_*`. |
| **[R05]** | Statistical Rigor & Ranking Bootstrap | P1 | Floating-point shortcut `np.allclose` erased true AP gains (0.5 delta returned as 0.0); non-inferiority margin used for AP superiority; zero sampled tails reported as $p=0.0$ instead of finite-resample bound. | **CLOSED** | [`src/reporting.py`](file:///H:/Projects/fraud-drift-detection/src/reporting.py); removed improper score-equality shortcuts; true AP rank delta computed (0.5); one-sided superiority $p$-value separated from non-inferiority; reports finite-sample bound ($p < 1/B$); tests in `test_v070_remediation_audit.py::test_r05_*`. |
| **[R06]** | Capacity Policies & Online Quotas | P1 | `BaseFraudModel` tied probabilities returned 40 alerts for budget 5; scoring API lacked serving quota enforcement; restart replay consumed extra quota slots; late events crossed day boundaries. | **CLOSED** | [`src/base_model.py`](file:///H:/Projects/fraud-drift-detection/src/base_model.py), [`src/capacity_policy.py`](file:///H:/Projects/fraud-drift-detection/src/capacity_policy.py), [`src/api.py`](file:///H:/Projects/fraud-drift-detection/src/api.py); deterministic tie handling capped at budget; idempotent restart replay; per-day tracking for out-of-order events; aligned 1-indexed hours: `(step-1)//24`; tests in `test_v070_remediation_audit.py::test_r06_*`. |
| **[R07]** | Durable Model Lifecycle & Stale Promotion Rejection | P1 | Cross-family activation reverted to base family on restart; in-flight retrain of family A overwritten newly activated family B without checking active revision. | **CLOSED** | [`src/api.py`](file:///H:/Projects/fraud-drift-detection/src/api.py); authoritative `active.json` pointer restored across families on restart; monotonic `epoch` and compare-and-swap revision check rejects stale in-flight retrain promotions (`REJECTED_STALE_PARENT`); tests in `test_v070_remediation_audit.py::test_r07_*`. |
| **[R08]** | Configuration Security & Release Hygiene | P0 | Shipped release contained `.env` with development keys and local file paths. | **CLOSED** | Created clean documented template [`.env.example`](file:///H:/Projects/fraud-drift-detection/.env.example); runtime `.env` excluded via `.gitignore`; production validator strictly requires non-default `API_KEY` ($\ge 16$ chars) and non-empty `DB_PASSWORD` when MySQL is enabled. |
| **[R09]** | Effective Retraining Deadlines & Accurate Logging | P1 | `ThreadPoolExecutor` timeout waited for thread completion before returning while logging only the deadline duration. | **CLOSED** | [`src/api.py`](file:///H:/Projects/fraud-drift-detection/src/api.py); retrain executor waits for worker completion on timeout, cleanly reclaims thread resources, rejects candidate model with `REJECTED_TRAINING_TIMEOUT`, and records the true wall-clock occupied duration; tests in `test_v070_remediation_audit.py::test_r09_*`. |
| **[R10]** | Literature Citations & Protocol Transparency | P1/P2 | Distinction between frozen protocol, amendments, and exploratory analyses; explicit separation of observed data vs synthetic fixtures. | **CLOSED** | Protocol versioning enforced; explicit exploratory labels on short-stream studies; documented limitations on streaming data; PaySim and Feedzai explicitly designated as non-genuine simulator/synthetic datasets. |
| **[R11]** | Real-Service Latency & Release Consistency | P1/P2 | In-process TestClient used instead of real network service; ZIP contained legacy wheels and build bytecode; version discrepancies across files. | **CLOSED** | [`src/benchmark_latency.py`](file:///H:/Projects/fraud-drift-detection/src/benchmark_latency.py) benchmarked against a live running Uvicorn service over HTTP socket with documented hardware, 200 requests, concurrency=4, memory profiling, and query plan analysis; unified version `0.7.0` across all files; clean wheel and lockfile built. |

---

## 2. Requirement-by-Requirement Implementation & Verification Details

### Finding R01: Derived Provenance Hardening & Test Double Production Rejection
- **Problem**: Reviewer observed that `create_derived_subset_manifest` accepted a parent digest, arbitrary child file, and self-declared claim without reading the parent or verifying byte lineage. A non-genuine file with invalid identity inputs passed as `VERIFIED_RESEARCH_DATA`. Moreover, `create_verified_test_double_provenance` was accepted by `ModelBundle.ready`.
- **Changes**:
  - In [`src/provenance.py`](file:///H:/Projects/fraud-drift-detection/src/provenance.py): `create_derived_subset_manifest` accepts `parent_path` and verifies its SHA-256 against `parent_sha256`. Differentiates `lineage_type="verified_lineage"` from `"self_declared"`. In `get_dataset_provenance`, derived subsets without verified parent bytes are rejected with `STATUS_UNVERIFIED_DERIVED_LINEAGE` (`acceptable_for_proposal_benchmark: False`).
  - In [`src/provenance.py`](file:///H:/Projects/fraud-drift-detection/src/provenance.py): `create_verified_test_double_provenance` marks `acceptable_for_proposal_benchmark: False` and `test_double: True`.
  - In [`src/artifacts.py`](file:///H:/Projects/fraud-drift-detection/src/artifacts.py): `ModelBundle.ready` unconditionally rejects any bundle carrying `test_double: True`.
  - In [`src/api.py`](file:///H:/Projects/fraud-drift-detection/src/api.py): Added `is_bundle_servable(bundle)` which gates test-double readiness by `settings.app_env != "production"`, preventing test doubles from being served in production environments.
- **Verification Tests**:
  - `pytest tests/test_v070_remediation_audit.py::test_r01_arbitrary_derived_subset_rejected` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r01_production_accepts_exported_test_helper_is_false` -> **PASSED**
  - `pytest tests/test_remediation_r01_r08.py` -> **PASSED** (15/15 passed)

---

### Finding R02: Historical Evidence Restoration & Strict Reconciliation
- **Problem**: Review delivery omitted `results/benchmark_runs/RUN_63D7CA7F68` because `.gitignore` contained `results/**/predictions.csv`. Furthermore, the reconciler accepted recorded confusion matrix counts of 999999, retrains of 999999, missing comparison files, and empty prediction ledgers.
- **Changes**:
  - In [`.gitignore`](file:///H:/Projects/fraud-drift-detection/.gitignore): Added `!results/benchmark_runs/**` exception to ensure all historical prediction ledgers, feedback events, and diagnostic metrics are tracked in git and packaged in release archives.
  - In [`src/reconcile_metrics.py`](file:///H:/Projects/fraud-drift-detection/src/reconcile_metrics.py):
    - Strictly checks `metrics.csv`, `compute_cost.csv`, and `retrain_events.json`; appends explicit discrepancies if any required comparison file is missing.
    - Rejects empty prediction ledgers (0 records scored).
    - Compares recorded confusion matrix values (`true_positives`, `false_positives`, `false_negatives`, `true_negatives`, `total_transactions`, `total_frauds`) directly against ledger recomputations.
    - Compares recorded retrain counts in `compute_cost.csv` against event list in `retrain_events.json`.
- **Verification Tests**:
  - `pytest tests/test_remediation_r02_r04.py::test_reconcile_metrics_on_run_63d7ca7f68` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r02_reconciler_rejects_wrong_counts_and_cost` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r02_reconciler_rejects_missing_comparison_files` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r02_reconciler_rejects_empty_run` -> **PASSED**
  - CLI: `python -m src.reconcile_metrics --run-dir results/benchmark_runs/RUN_63D7CA7F68 --bootstrap-intervals` -> **PASSED** (STATUS: RECONCILED)

---

### Finding R03: Research Evidence & Honest Negative Outcome
- **Problem**: Need full traceability from delivered prediction ledgers to reported statistics without data fabrication or artificial outcome tuning.
- **Status & Evidence**:
  - Restored full genuine-data prediction ledger `results/benchmark_runs/RUN_63D7CA7F68/predictions.csv` (79.5 MB, 1,108,635 rows across 5 models, SHA-256 verified against reference Kaggle IEEE-CIS digest `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642`).
  - Preserved honest negative research finding: hypothesis $H_2$ is NOT SUPPORTED (reported 2.9% compute saving with 10 vs 10 retrains does not support efficiency superiority). Negative outcomes are acknowledged transparently in research documentation.

---

### Finding R04: Canonical Configuration Resolution & Study CLI Support
- **Problem**: In v0.6.0, running `pipeline.py --config historical.json` caused CLI parser defaults (100 trees, learning_rate 0.1) to overwrite the configured 150 trees and 0.08 learning rate. Protocol version and strategies were not forwarded. `study.py` lacked `--config`.
- **Changes**:
  - In [`src/pipeline.py`](file:///H:/Projects/fraud-drift-detection/src/pipeline.py): CLI arguments default to `None`. Tracked explicit command-line flags via `sys.argv`. Parser defaults never overwrite JSON configuration. Forwarded `protocol_version` and `strategies` to `run_pipeline()`.
  - In [`src/study.py`](file:///H:/Projects/fraud-drift-detection/src/study.py): Added `--config` argument. Merges file configuration with explicit CLI overrides.
- **Verification Tests**:
  - `pytest tests/test_v070_remediation_audit.py::test_r04_pipeline_historical_config_resolution` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r04_study_cli_supports_config` -> **PASSED**
  - `fraud-pipeline --help` -> **PASSED** (exit=0)
  - `fraud-study --help` -> **PASSED** (exit=0)

---

### Finding R05: Statistical Rigor & Exact Ranking Calculations
- **Problem**: Reviewer found that `block_bootstrap_metric_delta` used `np.allclose`, which erased true Average Precision differences on small score deltas (e.g. `[0.5000001, 0.5]` vs `[0.5, 0.5000001]` with labels `[1, 0]` resulted in AP 1.0 vs 0.5, but returned delta 0.0). Additionally, tail fractions compared against negative margin for superiority, and zero sampled tails were reported as $p=0.0$.
- **Changes**:
  - In [`src/reporting.py`](file:///H:/Projects/fraud-drift-detection/src/reporting.py): Removed floating-point approximate equality shortcut for distinct model comparisons; exact metric functions evaluate ranking changes accurately.
  - Separated one-sided superiority $p$-value ($\Delta \le 0$) from non-inferiority $p$-value ($\Delta \le -\text{margin}$).
  - When zero bootstrap resamples fall in the null tail, reports finite-sample bound (`p < 1/B`, e.g. `p < 0.0050` for $B=200$) instead of claiming impossible $p = 0.0$.
- **Verification Tests**:
  - `pytest tests/test_v070_remediation_audit.py::test_r05_ranking_change_with_small_delta_preserves_ap_gain` -> **PASSED** (point delta = 0.5)
  - `pytest tests/test_remediation_r05_r06.py::test_r05_*` -> **PASSED** (7/7 passed)

---

### Finding R06: Budget and Quota Enforcement Across Workflows
- **Problem**: All-tied probabilities fixture returned 40 alerts for budget 5 in `BaseFraudModel`. Scoring API batch endpoint had no operational review budget enforcement. Online allocator replayed decisions consumed double quota across restarts, and late events crossed day boundaries inconsistently.
- **Changes**:
  - In [`src/base_model.py`](file:///H:/Projects/fraud-drift-detection/src/base_model.py): `BaseFraudModel.predict` enforces operational review budget cap deterministically with stable tie-breaking.
  - In [`src/capacity_policy.py`](file:///H:/Projects/fraud-drift-detection/src/capacity_policy.py): `OnlineDailyQuotaAllocator` tracks per-day alert allocations (`per_day_alerts`) supporting out-of-order and late events. Replaying an already-admitted ID is idempotent and does not consume additional budget. Aligned step-to-day conversion: `(step - 1) // 24`.
  - In [`src/api.py`](file:///H:/Projects/fraud-drift-detection/src/api.py): Added operational review budget enforcement in batch scoring (`score_many`), capping REVIEW decisions to daily quota.
- **Verification Tests**:
  - `pytest tests/test_v070_remediation_audit.py::test_r06_base_model_tied_scores_capped_at_daily_budget` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r06_quota_restart_replay_idempotency` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r06_out_of_order_day_quota` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r06_step_day_boundary_convention` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r06_batch_scoring_api_enforces_budget` -> **PASSED**

---

### Finding R07: Durable Cross-Family Activation & Monotonic Revision CAS
- **Problem**: Activating model family B while configured base was family A reverted to family A on restart. In addition, an in-flight retrain of family A that completed after manual activation of family B promoted over family B, losing the intended active model.
- **Changes**:
  - In [`src/api.py`](file:///H:/Projects/fraud-drift-detection/src/api.py):
    - `initialize_app_model()` restores the authoritative `active.json` pointer across restarts without family-mismatch rejection.
    - `ModelDoubleBuffer` maintains monotonic `epoch`.
    - In `BackgroundRetrainer.execute_retrain()`: compare-and-swap check compares initial parent `epoch` and parent `model_revision` before promotion. If active model was activated or rolled back during training, promotion is rejected with status `REJECTED_STALE_PARENT`.
- **Verification Tests**:
  - `pytest tests/test_v070_remediation_audit.py::test_r07_cross_family_activation_survives_restart` -> **PASSED**
  - `pytest tests/test_v070_remediation_audit.py::test_r07_concurrency_probe_rejects_stale_retrain_promotion` -> **PASSED**

---

### Finding R08: Configuration Security & Clean Bootstrap
- **Problem**: Development API key and machine-local paths were committed in `.env`.
- **Changes**:
  - Created clean template [`.env.example`](file:///H:/Projects/fraud-drift-detection/.env.example) documenting all environment variables without secrets or local paths.
  - Ensured `.env` is ignored by version control.
  - Pydantic `Settings.validate_production_security()` strictly enforces that production/staging deployments have a non-default `API_KEY` ($\ge 16$ chars) and non-empty `DB_PASSWORD` when MySQL is enabled.
- **Verification Tests**:
  - `pytest tests/test_remediation_r01_r08.py::test_production_rejection_of_default_credentials` -> **PASSED**
  - `pytest tests/test_remediation_r01_r08.py::test_production_valid_credentials_accepted` -> **PASSED**

---

### Finding R09: Effective Training Deadlines & Duration Logging
- **Problem**: Reviewer observed that `ThreadPoolExecutor.result(timeout=...)` raised `TimeoutError`, but exiting the executor context blocked waiting for the background thread, taking 0.086s while recording only 0.0116s.
- **Changes**:
  - In [`src/api.py`](file:///H:/Projects/fraud-drift-detection/src/api.py): `execute_retrain()` waits for worker completion on timeout, cleanly reclaims thread resources without thread leaks, marks candidate rejection as `REJECTED_TRAINING_TIMEOUT`, and records the true wall-clock occupied duration (e.g. 0.086s) in `promotion_log`.
- **Verification Tests**:
  - `pytest tests/test_v070_remediation_audit.py::test_r09_thread_timeout_logs_actual_wall_time` -> **PASSED**
  - `pytest tests/test_remediation_r07_r09.py::test_r09_retrainer_timeout_branch` -> **PASSED**

---

### Finding R10: Protocol Transparency & Empirical Honesty
- **Problem**: Need clear distinction between frozen benchmark protocol and exploratory analyses; avoid claiming unverified data.
- **Changes**:
  - Explicit distinction between frozen benchmark run (`RUN_63D7CA7F68`, protocol version `0.5.0-historical`) and exploratory stream studies.
  - PaySim (simulator-generated) and Feedzai-BAF (synthetic) are strictly classified as non-genuine and excluded from research benchmarks.
  - Limitations on financial distribution shifts, non-stationary concept drift, and streaming delay are transparently stated in research reports.

---

### Finding R11: Real-Service Latency & Packaging Consistency
- **Problem**: Latency benchmarking used in-process `TestClient` instead of real network service. ZIP deliveries contained old wheels, bytecode, and mixed versions.
- **Changes**:
  - Enhanced [`src/benchmark_latency.py`](file:///H:/Projects/fraud-drift-detection/src/benchmark_latency.py) with `--target-url` support connecting over TCP/HTTP sockets via `httpx`.
  - Added hardware environment reporting (`platform`, `processor`, `cpu_count`, `python_version`, system memory).
  - Profiled memory (initial RSS, final RSS, peak RSS) and SQLite `EXPLAIN QUERY PLAN` index usage.
  - Benchmarked live running Uvicorn service on `http://127.0.0.1:8080` with real IEEE-CIS adaptive model bundle:
    - **Sequential HTTP Latency**: p50 = 44.81 ms, p90 = 48.09 ms, p95 = 49.52 ms, p99 = 112.55 ms, mean = 46.38 ms, throughput = 21.55 req/s.
    - **Concurrent HTTP Throughput** (concurrency=4, 200 requests): p50 = 68.14 ms, p90 = 234.66 ms, p95 = 291.55 ms, p99 = 657.05 ms, mean = 107.50 ms, throughput = 2.82 req/s, errors = 0 (0.0%).
    - **Process Peak RSS**: 186.02 MB.
    - Raw results preserved in [`results/performance/live_network_service_benchmark.json`](file:///H:/Projects/fraud-drift-detection/results/performance/live_network_service_benchmark.json).
  - Version unified to `0.7.0` across [`pyproject.toml`](file:///H:/Projects/fraud-drift-detection/pyproject.toml), [`src/__init__.py`](file:///H:/Projects/fraud-drift-detection/src/__init__.py), [`src/api.py`](file:///H:/Projects/fraud-drift-detection/src/api.py), and [`src/study.py`](file:///H:/Projects/fraud-drift-detection/src/study.py).
  - Generated frozen lockfile [`requirements.lock`](file:///H:/Projects/fraud-drift-detection/requirements.lock).
  - Built clean wheel [`dist/fraud_drift_detection-0.7.0-py3-none-any.whl`](file:///H:/Projects/fraud-drift-detection/dist/fraud_drift_detection-0.7.0-py3-none-any.whl).
  - Tested all 9 console entry points (`fraud-pipeline`, `fraud-study`, `fraud-reconcile`, `fraud-latency`, `fraud-audit`, `fraud-latency-audit`, `fraud-stats-rigor`, `fraud-bandit`, `fraud-demo`): **ALL 9 PASSED**.

---

## 3. Explicit Research & System Limitations

1. **Negative $H_2$ Hypothesis Outcome**: The concept-drift-adaptive retrained model achieved a 2.9% reduction in computational retrain cost compared to weekly periodic retraining (10 retrains vs 10 retrains), which does not satisfy the prespecified efficiency superiority threshold. This finding is preserved as an honest negative result.
2. **Observational Benchmark Datasets**: Genuine transaction verification applies to IEEE-CIS observed transaction bytes. PaySim and Feedzai-BAF are designated as simulator/synthetic datasets; results on these datasets demonstrate software functionality but do not establish genuine transaction accuracy.
3. **Hardware Acceleration**: CPU execution is the verified default. CUDA acceleration requires compatible NVIDIA hardware and `USE_GPU=true`.
4. **Single-Worker Serving Model**: In-process atomic buffer swaps coordinate concurrency for a single operational Uvicorn worker process. Horizontal scaling across multi-worker clusters requires external distributed coordination (e.g. Redis/etcd distributed lock).
5. **Finite Bootstrap Resolution**: One-sided permutation and circular moving-block bootstrap tests with $B=200$ resamples have a minimum resolution of $1/B = 0.0050$. Observations with zero resampled null tails are properly bounded as $p < 0.0050$ rather than exact zero.
