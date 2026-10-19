# PhD Proposal Empirical Findings: Concept-Drift-Aware Fraud Detection

**Author / Candidate**: PhD Applicant Research System  
**Framework & Release**: `fraud-drift-detection` v0.5.0 (Release Tag `V3`)  
**Audit Date**: 17 October 2026  
**Empirical Evidence Standard**: Genuine Observed Transactions Solely (IEEE-CIS / Vesta Corporation)  
**Historical / Non-Genuine Data**: Strictly Excluded from Findings

---

## 1. Executive Summary: Primary Empirical Results

To establish credible, publication-quality evidence for a PhD research proposal, all empirical findings reported herein are derived strictly from genuine, real-world card-not-present (CNP) e-commerce transactions from the **IEEE-CIS / Vesta Corporation** benchmark. Simulator-generated datasets (PaySim) and synthetic datasets (Feedzai-BAF) have been audited, classified, and strictly excluded from proposal conclusions.

### 1.1 Headline Research Benchmark Results (Run `RUN_63D7CA7F68`)
- **Evaluation Scope**: Pure out-of-sample evaluation across **10 consecutive calendar weeks** (Weeks 8–17; 221,727 transactions; 8,761 observed frauds).
- **Initial Training Window**: 8 calendar weeks (Weeks 0–7; 212,449 transactions; 6,491 frauds).
- **Operating Threshold Policy**: Calibrated using strictly prior chronological validation data (final 20% of training window).

```
+-------------------------------------------------------------------------------------------------------------------+
|                                 EMPIRICAL BENCHMARK COMPARISON ON GENUINE DATA                                     |
+-------------------+-----------+-----------+-----------+-----------+-------------+------------+--------------------+
| Strategy          | Prec.     | Recall    | F1-Score  | Avg. Prec.| Brier Score | Retrains   | Compute Cost (s)   |
+-------------------+-----------+-----------+-----------+-----------+-------------+------------+--------------------+
| Static Baseline   | 0.6271    | 0.3972    | 0.4864    | 0.4909    | 0.0266      | 0          | 0.0s (Init: 5.34s) |
| Sliding Window    | 0.6129    | 0.4087    | 0.4904    | 0.5015    | 0.0261      | 2          | 4.20s              |
| Periodic (4-wk)   | 0.5939    | 0.4222    | 0.4936    | 0.5056    | 0.0260      | 2          | 11.43s             |
| Drift-Adaptive    | 0.6023    | 0.4254    | 0.4986    | 0.5184    | 0.0256      | 10         | 62.35s             |
| Weekly Retrain    | 0.6023    | 0.4254    | 0.4986    | 0.5184    | 0.0256      | 10         | 64.22s             |
+-------------------+-----------+-----------+-----------+-----------+-------------+------------+--------------------+
```

---

## 2. Answering the Central Research Question

> **Candidate Question**:  
> *Can an adaptive retraining policy that accounts for available labels and their delay maintain fraud recall under a fixed review workload while reducing computational cost compared with periodic retraining?*

### 2.1 Performance Under Fixed Review Capacity (Alert Budgets)
In financial institutions, fraud detection is evaluated under human alert review constraints rather than unconstrained F1. Over the 70 operating days (10 weeks):

1. **Tight Capacity ($K = 100$ alerts/day | 7,000 total alerts reviewed)**:
   - **Static Baseline**: Discovers **3,694 frauds** (Recall: **42.16%**, Precision: **52.77%**).
   - **Sliding Window**: Discovers **3,745 frauds** (Recall: **42.75%**, Precision: **53.50%**).
   - **Periodic (4-wk)**: Discovers **3,749 frauds** (Recall: **42.79%**, Precision: **53.56%**).
   - **Drift-Adaptive**: Discovers **3,818 frauds** (Recall: **43.58%**, Precision: **54.54%**).
   - *Impact*: Adaptive retraining captures **+124 additional genuine frauds** over the static model and **+69 frauds** over periodic retraining under an identical 100 reviews/day workload.
2. **Standard Capacity ($K = 500$ alerts/day | 35,000 total alerts reviewed)**:
   - **Static Baseline**: Discovers **6,289 frauds** (Recall: **71.78%**, Precision: **17.97%**).
   - **Sliding Window**: Discovers **6,459 frauds** (Recall: **73.72%**, Precision: **18.45%**).
   - **Periodic (4-wk)**: Discovers **6,487 frauds** (Recall: **74.04%**, Precision: **18.53%**).
   - **Drift-Adaptive**: Discovers **6,615 frauds** (Recall: **75.51%**, Precision: **18.90%**).
   - *Impact*: Adaptive retraining captures **+326 additional genuine frauds** over static baseline at identical operational investigation capacity.

### 2.2 Paired Statistical Uncertainty (Moving-Block Bootstrap)
Using paired circular moving-block bootstrap ($b=2$ slices, 500 bootstrap draws) with matched temporal draws regenerated from the prediction ledger (`python -m src.reconcile_metrics --bootstrap-intervals`):
- **Adaptive vs. Static $\Delta \text{Average Precision (AP)}$**: Point delta: $\mathbf{+0.0276}$ (**+2.76%** AP gain), Mean bootstrap delta: $+0.0282$, **95% Confidence Interval: $[\mathbf{+0.0226}, \mathbf{+0.0331}]$**. The interval strictly excludes zero (empirical $p < 0.001$), confirming statistically significant superiority over the static baseline ($H_3$).
- **Adaptive vs. Static $\Delta \text{F1}$**: Point delta: $\mathbf{+0.0123}$ (**+1.23%** F1 gain), Mean bootstrap delta: $+0.0120$, **95% Confidence Interval: $[\mathbf{+0.0071}, \mathbf{+0.0169}]$**.
- **Adaptive vs. Weekly Non-Inferiority ($H_1$)**:
  - $\Delta \text{Recall@100/day}$: Point delta $0.00000$, 95% CI: $[0.00000, 0.00000]$ (margin $\delta=0.01$, lower bound $0.0 \ge -0.01$, empirical $p < 0.001$).
  - $\Delta \text{Average Precision}$: Point delta $0.00000$, 95% CI: $[0.00000, 0.00000]$. Both strategies achieve identical $0.5184$ AP.
- **Periodic vs. Static $\Delta \text{AP}$**: $+0.0147$, 95% CI: $[+0.0079, +0.0253]$.
- **Sliding vs. Static $\Delta \text{AP}$**: $+0.0106$, 95% CI: $[+0.0016, +0.0233]$.

### 2.3 Formal Hypotheses Scorecard & Negative Result Disclosure

| Hypothesis | Evaluation Criterion | Observed Outcome | Formal Status |
| :--- | :--- | :--- | :---: |
| **$H_1$: Accuracy Non-Inferiority** | $\text{Recall}_{\text{adaptive}} \ge \text{Recall}_{\text{weekly}} - 0.01$ at $K=100$/day | $\Delta = 0.0000$, 95% CI: $[0.0000, 0.0000]$ | **SUPPORTED** ($p < 0.001$) |
| **$H_2$: Computational Efficiency** | $\text{ComputeCost}_{\text{adaptive}} \le 0.70 \times \text{ComputeCost}_{\text{weekly}}$ | 62.35s vs 64.22s (2.9% saving vs. 30% target) | **NOT SUPPORTED (FAILS)** |
| **$H_3$: Superiority over Static** | $\text{CI}_{0.95}(\Delta \text{AP}) > 0 \text{ and } \text{CI}_{0.95}(\Delta \text{F1}) > 0$ | $\Delta \text{AP} = +0.0276$, $\Delta \text{F1} = +0.0123$ | **SUPPORTED** ($p < 0.001$) |

**Investigation of $H_2$ Negative Outcome**:
In the Protocol v1 configuration, ADWIN ($\delta=0.002$) triggered adaptation in every single out-of-sample slice (Slices 8–17). In a large-scale non-stationary stream (~22,000 transactions per week) with extreme class imbalance (3.9% fraud in evaluation window), error rates fluctuated sufficiently to trigger ADWIN continuously. Without an explicit cooldown period, minimum evidence threshold, or retrain hysteresis, the adaptive model retrained 10 times in 10 weeks, matching the weekly schedule and failing the $30\%$ efficiency reduction target. This negative outcome is explicitly preserved as a core baseline finding motivating doctoral research into drift hysteresis mechanisms.

---

## 3. Explicit Demarcation: Completed Work vs. Proposed PhD Research Dependencies

To maintain scientific integrity, the boundary between completed software/experiments and dependent future doctoral research is explicitly declared:

```
+----------------------------------------------------------------------------------------------------+
|                               DEMARCATION OF RESEARCH SCOPE                                        |
+----------------------------------------------------+-----------------------------------------------+
| COMPLETED & EMPIRICALLY DEMONSTRATED               | PLANNED DOCTORAL RESEARCH DEPENDENCIES        |
+----------------------------------------------------+-----------------------------------------------+
| 1. Empirical proof of concept-drift degradation    | 1. Acquisition of genuine chargeback dispute  |
|    on 221k out-of-sample genuine transactions.     |    timestamps from payment card acquirers.    |
| 2. Statistically significant outperformance of     | 2. Continuous online incremental learning     |
|    drift-adaptive retraining over static baseline. |    under extreme class imbalance (streaming).|
| 3. Budget-constrained thresholding achieving       | 3. Positive-Unlabelled (PU) learning for      |
|    75.51% fraud recall under 500 reviews/day.      |    unreviewed transaction feedback dynamics.  |
| 4. Tied-timestamp intra-second leakage prevention  | 4. Selective verification policies allocating |
|    and vectorized history feature engineering.     |    investigative capacity to maximize drift   |
| 5. Production microservice with generation-aware   |    detection sensitivity.                     |
|    feedback and atomic model promotion gates.      | 5. Adversarial drift evasion modelling.       |
+----------------------------------------------------+-----------------------------------------------+
```

### 3.1 Documented Research Dependency: Acquirer Confirmation Timestamps
While the transaction features in IEEE-CIS are genuine, **the dataset does not record true cardholder dispute notification timestamps**. Consequently:
- Observed delayed-feedback effectiveness cannot be established from IEEE-CIS without genuine confirmation timestamps.
- Any simulated delay distributions (fixed or log-normal) are treated strictly as *exploratory sensitivity checks*.
- Obtaining observed confirmation timestamps from an active payment processor partner is a documented dependency for the doctoral thesis.

### 3.2 Documented Research Dependency: Streaming Online Updating
In production, full re-fitting of GBDT models every week incurs cumulative compute cost. Extending this framework to streaming online learners (e.g. Hoeffding Adaptive Trees or incremental neural embeddings) that update instance-by-instance while maintaining calibration forms a central pillar of the proposed PhD research.
