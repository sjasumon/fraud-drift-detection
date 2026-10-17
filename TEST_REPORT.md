# Automated Test Suite and Verification Report

**Release**: v0.5.0 (`fraud-drift-detection`)  
**Audit Date**: 17 October 2026  
**Test Framework**: Pytest 9.1.1, Pluggy 1.6.0  
**Python Runtime**: Python 3.12.10 (64-bit, Windows 11)  
**Execution Command**: `.\.venv\Scripts\python.exe -m pytest -q`  
**Overall Status**: **PASS** (127 passed, 0 failed, 4 non-fatal warnings in 34.42s)

---

## 1. Test Suite Summary by Module

| Module / Test File | Purpose & Invariants Verified | Collected | Passed | Failed |
| :--- | :--- | :---: | :---: | :---: |
| `tests/test_api.py` | Microservice endpoints, authentication, input validation, schemas | 16 | 16 | 0 |
| `tests/test_bandit.py` | Contextual multi-armed bandit routing logic | 5 | 5 | 0 |
| `tests/test_bandit_benchmark.py` | Bandit benchmarking and policy performance | 2 | 2 | 0 |
| `tests/test_baselines.py` | Static, periodic, and sliding-window estimators | 2 | 2 | 0 |
| `tests/test_config.py` | Environment variable parsing, Pydantic type safety, directory paths | 9 | 9 | 0 |
| `tests/test_data_generator.py` | Synthetic stream generation for software unit tests | 3 | 3 | 0 |
| `tests/test_db_persistence.py` | MySQL and SQLite prediction ledger persistence, idempotency | 7 | 7 | 0 |
| `tests/test_deduplication.py` | Deduplication of transaction IDs and replayed events | 3 | 3 | 0 |
| `tests/test_delayed_feedback.py` | Label queueing, fixed and log-normal delay maturational checks | 6 | 6 | 0 |
| `tests/test_drift_explainer.py` | Feature attribution engines and fallback diagnostic text | 5 | 5 | 0 |
| `tests/test_integration_orchestrators.py` | End-to-end integration across preprocessing and pipelines | 4 | 4 | 0 |
| `tests/test_key_figures.py` | Publication-ready figure and visualization generation | 3 | 3 | 0 |
| `tests/test_leakage_invariants.py` | Chronological split boundaries and target leakage traps | 4 | 4 | 0 |
| `tests/test_metrics_reconciliation.py` | Confusion matrix integer reconciliation and metric formulas | 4 | 4 | 0 |
| `tests/test_multi_dataset.py` | Cross-dataset column mappings and compatibility | 8 | 8 | 0 |
| `tests/test_reaudit.py` | Historical regression checks from v0.4.0 audit | 16 | 16 | 0 |
| `tests/test_revision_regressions.py` | Model bundle immutability, atomic promotion, rollback logic | 10 | 10 | 0 |
| `tests/test_statistical_rigor.py` | Bootstrap confidence intervals, Student's t, Wilcoxon tests | 10 | 10 | 0 |
| `tests/test_uid_collision.py` | Strict stream-unseen customer card entity isolation | 5 | 5 | 0 |
| `tests/test_v050_improvements.py` | Provenance classification, tied-timestamp leakage safety, budgets | 5 | 5 | 0 |
| **Total** | **Comprehensive Full Regression Suite** | **127** | **127** | **0** |

---

## 2. Invariants and Critical Failure Paths Verified

1. **Provenance Classification and Rejection (`test_v050_improvements.py`)**:
   - Verified that `IEEE-CIS` is classified as `GENUINE_OBSERVED` and accepted.
   - Verified that `PaySim` (`SIMULATOR_GENERATED`) and `Feedzai-BAF` (`SYNTHETIC`) are strictly classified as non-genuine and rejected by `ModelBundle.ready`.
   - Verified that unverified datasets default to `ready = False`.
2. **Tied-Timestamp Intra-Second Leakage Safety (`test_v050_improvements.py`)**:
   - Tested multiple transactions with identical timestamps ($t_i = 10$).
   - Verified that prior transaction counts and rolling averages for all tied events at $t = 10$ evaluate strictly to $0.0$, observing only state from $t < 10$.
   - Verified that subsequent transactions at $t = 12$ observe all prior tied transactions collectively.
3. **Operational Review Budget Threshold Selection (`test_v050_improvements.py`)**:
   - Verified that `BaseFraudModel(threshold_objective="review_budget_recall")` selects an operating threshold calibrated to human review capacity without looking ahead to future test labels.
4. **Promotion Acceptance Rule & Margin (`test_v050_improvements.py`)**:
   - Verified that `BackgroundRetrainer` rejects candidates offering $< 0.005$ F1 gain over the active model.
   - Verified that candidates exceeding 60-second training timeouts are aborted and logged with explicit rejection reasons.
5. **Model Immutability and State Protection (`test_reaudit.py`)**:
   - Verified that exported model packages are stripped of training records and cannot be overwritten.
   - Verified that identical temporal bootstrap draws are applied across seeds, preventing cloned seeds from manufacturing false statistical significance.

---

## 3. Warning Audit and Diagnostic Evaluation

Pytest reported 4 non-fatal runtime warnings:
1. `StarletteDeprecationWarning`: `fastapi.testclient` imports `httpx` (standard deprecation notice in current Starlette; scheduled for migration in httpx2).
2. `UserWarning`: `plt.tight_layout()` warning in `test_bandit_benchmark.py` due to compact axes width during headless rendering.
3. `RuntimeWarning` (2 occurrences): `scipy.stats._axis_nan_policy.py:431` precision loss warning during paired Student's t moment calculation when two vectors are identical. Expected mathematical behavior under identical baseline inputs.

None of the warnings affect evaluation soundness, model scoring, or service stability.

---

## 4. Environment & Hardware Boundaries

- **Verified Native Environment**: Windows 11, x86_64, 16 logical cores, 95.3 GB RAM.
- **Python Version**: Python 3.12.10 in virtual environment `.venv`.
- **Hardware Acceleration Note**: Tested on CPU using optimized histogram-based tree learning (`tree_method="hist"`).
- **Explicit Limitations**:
  - `Docker Daemon`: **NOT TESTED / HOST NATIVE** (Docker daemon not active on local host; native Python virtual environment verified).
  - `CUDA eGPU`: **RUNNING ON CPU / FALLBACK VERIFIED** (CUDA hardware not attached; CPU hist fallback active and validated).
