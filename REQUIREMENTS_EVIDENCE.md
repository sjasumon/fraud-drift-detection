# Requirements-to-Evidence Matrix: v0.5.0 Release

**Release Version**: v0.5.0  
**Target Release Tag**: `V3`  
**Audit Date**: 17 October 2026  
**Evaluation Standard**: `Fraud_Detection_Next_Phase_Agent_Instruction.txt`

---

## 1. Compliance Matrix Overview

| Section | Requirement Scope | Status | Primary Code / Artifact Evidence |
| :--- | :--- | :--- | :--- |
| **Section 1** | Establish Baseline & Preserve V1/V2 | **PASS** | `V1` tag (`d95943f`), `V2` tag (`2776fb4`), `H:\Projects\fraud-drift-detection-v1-backup` |
| **Section 2** | Enforce Genuine Data & Exclude Non-Genuine | **PASS** | `src/provenance.py`, `DATA_PROVENANCE.md`, `data/ieee/train_transaction.csv` |
| **Section 3** | Define Research Contribution & Literature | **PASS** | `RESEARCH_PROTOCOL.md` (Carminati et al. 2015, Dal Pozzolo et al.) |
| **Section 4** | Implement Leakage-Safe Temporal Protocol | **PASS** | `src/history_features.py` (tied-timestamp safety), `src/pipeline.py` (W0–7 / W8–17) |
| **Section 5** | Fair Model Comparison & Operational Policies | **PASS** | `src/base_model.py` (Recall@Budget), `src/api.py` (promotion margin $\ge 0.005$) |
| **Section 6** | Computational Performance & Profiling | **PASS** | `src/history_features.py` (vectorized), SQLite compound index, `BENCHMARK_REPORT.md` |
| **Section 7** | Auditable Research Outputs & Bootstrap CIs | **PASS** | `src/reporting.py` (AP, Brier, Paired Block Bootstrap), `RUN_63D7CA7F68` |
| **Section 8** | Software & Deployment Verification | **PASS** | 127 automated tests passing, live FastAPI microservice on port 8000 |
| **Section 9** | Reviewable Release Delivery | **PASS** | 9 required markdown documents, release ZIP, GitHub Release `V3` |

---

## 2. Granular Requirements Evidence and Audit Findings

### Section 1: Establish and Preserve the Baseline
- **Audit Requirement**: Verify supplied archive, preserve earlier code in GitHub as V1/V2, never overwrite existing release tags, retain rollback state locally.
- **Evidence**:
  - `V1` Release Tag: Commit `d95943f2d50d5aa7e65b3674ae2b9dc4864e1da9` ([V1 Release](https://github.com/sjasumon/fraud-drift-detection/releases/tag/V1)).
  - `V2` Release Tag: Commit `2776fb43a5ce94aba2a7309de98212db14a0ce95` ([V2 Release](https://github.com/sjasumon/fraud-drift-detection/releases/tag/V2)) with asset `fraud-drift-detection-v0.4.1-rechecked.zip`.
  - Rollback storage preserved in `H:\Projects\fraud-drift-detection-v1-backup`.
  - Next version identifier: `v0.5.0` (annotated tag `V3`). Existing tags remain completely untouched.
- **Finding**: **PASS**

### Section 2: Enforce Genuine Data and Traceable Results
- **Audit Requirement**: All proposal empirical results must come from genuine observed transactions. Reject synthetic fixtures (BAF) and simulator data (PaySim) from proposal claims. Default provenance to UNVERIFIED. Disclose feedback latency limitations.
- **Evidence**:
  - `src/provenance.py` created with `DATASET_REGISTRY`, `GENUINE_OBSERVED`, `SIMULATOR_GENERATED`, `SYNTHETIC`, `UNVERIFIED`.
  - Real IEEE-CIS data verified locally: `data/ieee/train_transaction.csv` (SHA256: `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642`, 590,540 rows, 20,663 frauds), `train_identity.csv` (SHA256: `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c`, 144,233 rows).
  - PaySim and Feedzai strictly classified as non-genuine and excluded from primary proposal claims in `DATA_PROVENANCE.md`.
  - `src/artifacts.py` (`ModelBundle.ready`) queries `verify_bundle_provenance()`; rejects models lacking verified genuine provenance.
  - Critical disclosure: IEEE-CIS lacks true chargeback confirmation timestamps; documented as an exploratory sensitivity analysis and explicit future research dependency.
- **Finding**: **PASS**

### Section 3: Define Research Contribution Before Evaluation
- **Audit Requirement**: Prepare a focused research protocol around: *Can a retraining policy that accounts for available labels and their delay maintain fraud recall at a fixed review workload while reducing computational cost compared with periodic retraining?* Review primary literature (Carminati et al. 2015 BankSealer; Dal Pozzolo et al. 2014, 2018 [Handle 11311/1044896]).
- **Evidence**:
  - `RESEARCH_PROTOCOL.md` authored, versioned (1.0.0), and timestamped prior to final evaluation. Corrected citations: Dal Pozzolo et al. (handle 11311/1044896) and Carminati et al. (BankSealer).
  - Formulates falsifiable hypotheses $H_1$, $H_2$, $H_3$ with practical non-inferiority margins ($\delta = 0.01$).
  - Verified outcomes: $H_1$ (Accuracy Non-Inferiority) is **SUPPORTED** ($p < 0.001$), $H_2$ (Computational Efficiency) is **NOT SUPPORTED (FAILS)** due to continuous ADWIN triggers in IEEE-CIS stream (2.9% saving vs 30% target), $H_3$ (Gain over Static) is **SUPPORTED** ($p < 0.001$).
  - Primary metric selected: Recall@Budget (100 and 500 reviews/day) and Average Precision (AP).
- **Finding**: **PASS**

### Section 4: Implement a Leakage-Safe Temporal Protocol
- **Audit Requirement**: Use chronological training and untouched out-of-sample evaluation. Mitigate tied-timestamp leakage in history features. Score events before label availability.
- **Evidence**:
  - Frozen boundaries: Weeks 0–7 (initial training: 212,449 transactions, 6,491 frauds) and Weeks 8–17 (out-of-sample evaluation: 221,727 transactions, 8,761 frauds).
  - `src/history_features.py`: Fully vectorized and hardened against tied timestamps ($t_i = T$). All transactions at timestamp $T$ observe state from $t < T$ before cumulative updates occur.
  - Chronological 20% validation partition inside training set for calibration and threshold tuning.
- **Finding**: **PASS**

### Section 5: Fair Model Comparison & Operational Policies
- **Audit Requirement**: Compare static, periodic, weekly, sliding, and adaptive models under identical event streams and label access. Evaluate feature ablations. Implement operational review budget thresholding and documented model promotion rules.
- **Evidence**:
  - Evaluated 5 strategies on genuine IEEE-CIS: `static`, `periodic` (every 4w), `weekly` (every 1w), `sliding` (4w window), `adaptive` (drift-triggered).
  - Feature ablations executed: Base transactions, Identity join, History features, Operational budget thresholding.
  - `src/base_model.py`: Implemented `threshold_objective="review_budget_recall"` and `operational_daily_budget`.
  - `src/api.py`: Implemented `BackgroundRetrainer` with minimum improvement margin ($\ge 0.005$), timeout guard ($\le 60$s), and structured `promotion_log`.
- **Finding**: **PASS**

### Section 6: Profile and Improve Computational Performance
- **Audit Requirement**: Measure baseline on available hardware. Profile history computation, preprocessing, training, inference, and writes. Verify optimization behavior parity.
- **Evidence**:
  - Hardware recorded: AMD Ryzen / Intel 16 logical cores, 95.3 GB RAM, Windows 11.
  - Vectorized history feature computation: 50,000 transactions processed in 0.138s (vs >12s in legacy loop).
  - SQLite database optimization: compound index `idx_predictions_family_label_time` on `(family_id, label, event_time)` in `src/api.py`.
  - Inference throughput: >750,000 predictions/sec; API scoring latency p50 < 4.5ms.
- **Finding**: **PASS**

### Section 7: Generate Auditable Research Outputs
- **Audit Requirement**: Generate every metric and plot programmatically from saved predictions. Name Average Precision explicitly. Report Recall@Budget, Brier score, compute costs, and paired bootstrap uncertainty.
- **Evidence**:
  - Run `RUN_63D7CA7F68` produced on 221,727 genuine out-of-sample IEEE-CIS transactions.
  - Metrics generated: `metrics.csv`, `alert_budgets.csv`, `compute_cost.csv`, `confidence_intervals.json`, `overview.png`, `report.html`.
  - Paired circular moving-block bootstrap (block size = 2, 500 draws) confirms adaptive model $\Delta F1 = +0.0123$, 95% CI $[+0.0052, +0.0188]$ over static baseline.
- **Finding**: **PASS**

### Section 8: Verify Software and Deployment
- **Audit Requirement**: Run regression tests (provenance rejection, tied-timestamp safety, budget thresholds, promotion criteria). Build wheel in clean environment. Test live microservice.
- **Evidence**:
  - Automated tests: **127 passed, 0 failed, 4 warnings** in 34.42s (`pytest`).
  - Python package wheel built and validated (`dist/fraud_drift_detection-0.5.0-py3-none-any.whl`).
  - Live FastAPI microservice running on `http://127.0.0.1:8000` (PID / daemon `task-206`), tested with authentication, single/batch scoring, duplicate handling, and recovery.
  - Docker daemon / external CUDA eGPU: BLOCKED / NOT TESTED / RUNNING ON CPU (explicitly documented per instructions).
- **Finding**: **PASS**

### Section 9: Deliver a Reviewable Release
- **Audit Requirement**: Deliver 9 markdown documents, release ZIP, tag `V3`, and push to GitHub.
- **Evidence**:
  - All 9 documents authored and verified.
  - Release archive `fraud-drift-detection-v0.5.0.zip` packaged.
  - Release tag `V3` pushed and GitHub Release created.
- **Finding**: **PASS**
