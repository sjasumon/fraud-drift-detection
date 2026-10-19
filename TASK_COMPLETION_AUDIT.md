# Completion audit of the promised enhancements — v0.4.1

The first revised ZIP (v0.4.0) was incomplete. Its 104 passing tests did not exercise
several important combinations, and some tests checked only that output keys existed.
This follow-up started from that exact delivered ZIP, reproduced defects, corrected
them and tested the installed package away from the source directory.

## What the second review found

| Defect in the delivered v0.4.0 ZIP | Reproduction or consequence | v0.4.1 correction |
|---|---|---|
| Recent window anchored to wall-clock slice | A 2-slice window with a 3-slice label delay made zero recent-model retrains despite 310 matured rows | Window ends at latest verified arrival; logs verification staleness |
| Delayed errors mixed model generations | Predictions made by an older classifier could enter detectors reset for a newer classifier | Store prediction generation; monitor only matching active generation; retain older labels for training |
| Serving artifact retained training data | Included adaptive example package contained 606 training-history rows | Serialize fitted serving state without history/retrain buffers; test prediction parity |
| Model save could overwrite an existing package | Reusing an output directory changed the artifact in place | Stage and atomically publish immutable packages; reject overwrites |
| Accepted revision did not resume on restart | Startup loaded the original configured model instead of the accepted candidate | Persist/restore an atomic active-model pointer with stable revision identity |
| Bootstrap trusted inconsistent paired labels | A changed baseline label was accepted and produced a numerical interval | Reject mismatches and missing pairs |
| Seeds resampled temporal data independently | Repeated seeds could masquerade as additional independent streams | Use shared moving-block time draws across seeds; cloned-seed invariance test |
| Study summary omitted promised dimensions | Only F1, recall and paired F1 deltas were summarized | Include precision, AP, Brier score, FP/FN counts, review budgets, retrains, seconds and run IDs |
| Legacy audit/statistical paths diverged | Different initial preprocessing, duplicate seed handling, stale summaries and fixed paths remained | Main, delayed, cohort and statistical workflows use the same pipeline |
| Zero-fraud cohorts disappeared | Population and false-alarm counts were discarded | Keep counts; mark recall/AP as undefined where appropriate |
| Incomplete identity validation | Missing card1 could reach integer conversion | Validate card1/card2/addr1 jointly; qualify entity keys as proxies |
| BAF labels cast before validation | Fractional labels could silently become integers | Validate original labels before conversion |
| Installed command/path assumptions | Delayed CLI ignored explicit arguments; latency command depended on tests; runtime paths pointed inside the package | Explicit installed CLI entrypoints and working-directory/FRAUD_WORKSPACE paths |
| SQLite lifecycle and score timing | Connections were not explicitly closed; total timing omitted persistence | Close connections and include ledger writes in total timing |
| Attribution fallback could look like SHAP | Fallback values used the same labels without a persistent method flag | Record importance_method; use diagnostic wording; do not fabricate uniform importance |
| PSI collapsed binary features into one bin | A change from 5% to 95% missingness could remain invisible | Separate low-cardinality values with midpoint boundaries; regression test |
| PaySim availability ablation discarded a valid ratio | Removing post-balances also prevented amount/old-balance ratio generation | Preserve features computable before the transaction |

The original reproduction output is in `results/validation/v0.4.0-reproduction.txt`.

## Every enhancement from the original review

| Promised enhancement | Implementation | Validation evidence | Status |
|---|---|---|---|
| Serve the actual model, preprocessing, threshold and schema | artifacts.py and api.py; readiness, auth, checksums, atomic snapshots, verified-feedback promotion | API parity/readiness/schema/checksum/restart tests; installed latency command rejects unready state | Implemented and fixture-tested |
| Fair strategy comparison | Clone one fitted initial model across static, periodic, every-slice, sliding and adaptive strategies; no warmup adaptation | First OOS predictions match; OOS rows and retrain indices checked | Implemented and synthetic-tested |
| Original-decision delayed verification | One maturity queue; original decisions and generations; deduplicated labels; same rules for all strategies | Queue and end-to-end delayed-generation regression tests; delay CLI runs | Implemented and synthetic-tested |
| Threshold tuning and calibration | Prior chronological validation, optional sigmoid calibration, F1 threshold; daily review budgets; threshold diagnostics | Model/serving parity tests and report/study artifacts | Implemented; real-data operating point pending |
| Recent-memory/age-weighted adaptation and fair costs | Configurable windows/half-lives/caps; measured seconds and staleness; shared initial-fit policy | Recent-window/lag tests and installed study over seeds/time periods | Implemented; empirical configuration selection pending |
| Better features with availability controls | Full IEEE columns, identity join, categories/email, missingness, past-only entity history, PaySim pre-transaction ablation | Identity, causal-feature, missingness and availability tests; installed feature run | Implemented; real-data feature gains pending |
| Correct statistical claims and reproducible outputs | Per-run fingerprints, OOS prediction ledger, shared-time paired bootstrap, corrected cohorts/PSI; archived old claims | Pairing/cloned-seed tests, cohort reconciliation, strict JSON, installed statistical run | Implemented; real-data conclusions pending |
| Useful final experiment report | HTML, PR/calibration/time/cost plots, metrics, review budgets, error examples, thresholds, models, manifests | Report tests and included fresh synthetic report | Implemented and synthetic-tested |

## Work that remains genuinely unperformed

- Corrected IEEE-CIS, PaySim and Feedzai empirical reruns: their raw files were absent.
  This prevents a justified claim that accuracy, F1, recall, calibration or compute improved.
- Real-data threshold/window/feature selection on development periods and confirmation
  on an untouched future period. A sweep implementation is not an experimental result.
- Production network/concurrency latency and load under background retraining.
- CUDA execution, Windows setup, Docker startup and external MySQL execution. CPU
  tests exercise the database adapter with SQLite; they do not validate those environments.

These items are **pending**, not completed by synthetic fixtures. The active manuscript
remains a protocol draft. Historical benchmark values are retained only as archived evidence.
See VALIDATION.md for exact test/command results and the scope of the clean-wheel check.

---

# Completion Audit of Remediation Brief Findings R01–R11 (Release v0.6.0)

**Remediation Date**: 2026-10-19  
**Specification**: `Fraud_Detection_Agent_Remediation_Brief.docx`  
**Automated Tests**: 177 passed; 0 failed; 0 skipped  
**Status**: **ALL 11 FINDINGS R01–R11 SATISFIED AND SUPPORTED BY CODE AND TESTS**

| Remediation Finding | Problem Found in Audit | Code Action & Implementation | Verification Evidence | Status |
|---|---|---|---|:---:|
| **[R01] Provenance & Cryptographic Verification** | Metadata matching assigned genuine status without verifying file bytes; permissive fallback allowed synthetic models to serve. | Cryptographic SHA-256 byte comparison implemented in `src/provenance.py` against reference hashes (`3a5c83ab...`, `b63c725d...`); removed fallbacks in `src/artifacts.py`; added derived subset parent-child hash manifests. | `tests/test_remediation_r01_r08.py` (10/10 PASS): arbitrary transaction file fails, missing file fails, IEEE fixture fails, missing provenance fails, synthetic fails, approved genuine passes. | **PASS** |
| **[R02] Metric Reconciliation & Ledger Verification** | Benchmark report cited historical numbers without reproducible run artifacts or deterministic reconciliation script. | Recovered `results/benchmark_runs/RUN_63D7CA7F68` artifacts (80MB prediction ledger); created standalone `src/reconcile_metrics.py` CLI. | Reconciles all 5 strategies with **0 discrepancies** across TP, FP, TN, FN cells, AP, budgets, and costs. `tests/test_remediation_r02_r04.py` (2/2 PASS). | **PASS** |
| **[R03] Research Question & Efficiency Assessment** | Adaptive compute saving was only 2.9% vs the $\ge 30\%$ target promised for Hypothesis $H_2$. | Honestly reported Hypothesis $H_2$ as **NOT SUPPORTED / REJECTED in Protocol v1** across all project documentation; traced ADWIN continuous triggers in Slices 8–17; preserved negative result. | Confirmed $H_1$ Supported ($\Delta = 0.0000$, non-inferiority $p < 0.001$); confirmed $H_3$ Supported ($\Delta \text{AP} = +0.0276$, $p < 0.001$). Documented in `BENCHMARK_REPORT.md` and `PROPOSAL_RESULTS.md`. | **PASS** |
| **[R04] Executable Commands & Versioned Configs** | Missing CLI flags for hyperparameters, threshold objectives, and budgets; unversioned experiment configs. | Exposed `--threshold-objective`, `--operational-daily-budget`, `--n-estimators`, `--learning-rate`, and `--config` in `src/pipeline.py` and `src/study.py`; aligned default hyperparameters to protocol; created 5 versioned configs in `configs/`. | `tests/test_remediation_r02_r04.py` (6/6 PASS): CLI help output, parameter validation, versioned config validation. | **PASS** |
| **[R05] Paired Statistics & Bootstrap for AP/Quota** | Published CIs for Average Precision and Recall@Budget lacked bootstrap implementation; spurious p-values. | Implemented paired circular moving-block bootstrap in `src/reporting.py` supporting AP, Recall@Budget, and one-sided non-inferiority tests ($H_1: \delta=0.01$); added `--bootstrap-intervals` to `src/reconcile_metrics.py`. | `tests/test_remediation_r05_r06.py` (7/7 PASS): independent AP match, identical models yield 0 delta, label mismatch rejection, H1 non-inferiority test. | **PASS** |
| **[R06] Defined Review Capacity Policy** | Score ties inflated 5-alert quota to 40; API lacked online quota state. | Implemented `src/capacity_policy.py` formally separating batch ranking, expected volume, and causal online quota allocation; fixed score ties in `src/base_model.py` using distinct threshold selection. | `tests/test_remediation_r05_r06.py` (5/5 PASS): 5-alert quota regression with 50 tied probabilities produces exactly 5 alerts; causal online allocator enforces daily budget and survives restarts. | **PASS** |
| **[R07] Auditable Model Rollback & State Revert** | Inconsistent pointer path (`results/service/active_model.json` vs `models/active.json`); claimed file watching without implementation. | Implemented dual-pointer atomic synchronization; authenticated rollback endpoint `POST /v1/models/rollback` supporting previous generation or explicit package; hardened pointer loader in `initialize_app_model()`. | `tests/test_remediation_r07_r09.py` (3/3 PASS): dual-pointer synchronization, previous/explicit/base-0 rollback, corrupted candidate rejection, clean restart fallback. | **PASS** |
| **[R08] Distribute Clean Configuration & Usable Bootstrap** | Default test credentials in configuration; unhandled model load crashes. | Sanitized `.env.example`; production security validator in `src/config.py`; clean 503 unready startup in `src/api.py`. | `tests/test_remediation_r01_r08.py` (5/5 PASS): default password rejection, empty DB password rejection, clean unready startup. | **PASS** |
| **[R09] Hardened Timeout, Promotion & Feedback History** | Volatile in-memory promotion log; post-fit duration check called a timeout; ledger lacked feedback receipt timestamps. | Bounded worker timeout using `ThreadPoolExecutor` with deadline enforcement; persistent SQLite table `promotion_history`; append-only `feedback_events` audit table with `received_at`, `confirmed_at`, `label_source`, and conflict handling; schema migration for `predictions`. | `tests/test_remediation_r07_r09.py` (8/8 PASS): all 7 retrainer execution branches executed; promotion history persists across restarts; feedback audit trail records timestamps, replays, and 409 conflicts. | **PASS** |
| **[R10] Correct References, Novelty & Interpretation** | Politecnico handle 11311/1044896 confused with Dal Pozzolo; idealized IEEE-CIS feedback conflated with measured operational feedback. | Corrected citations to Dal Pozzolo et al. (2014, 2018) and Carminati et al. (2015 BankSealer); added primary literature comparison table in `RESEARCH_PROTOCOL.md`; explicitly disclosed lack of genuine dispute confirmation timestamps in IEEE-CIS. | Documented in `RESEARCH_PROTOCOL.md`, `BENCHMARK_REPORT.md`, `PROPOSAL_RESULTS.md`, `CHANGELOG.md`. | **PASS** |
| **[R11] Measure the Scope You Claim** | History aggregation claimed "vectorized" but used array iteration; query plans and latency breakdowns lacked separation. | Accurately documented causal boundary-batched NumPy scanning in `src/history_features.py`; added compound indexes in `src/api.py`; profiled queries with `EXPLAIN QUERY PLAN`; enhanced `src/benchmark_latency.py` with isolated model, sequential HTTP, and concurrent API timings. | `tests/test_remediation_r11.py` (4/4 PASS): tied timestamp causality, missing identity isolation, index query plan verification, latency separation. Full pytest suite: 177/177 PASS. | **PASS** |
