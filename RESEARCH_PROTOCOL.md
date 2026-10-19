# Research Protocol: Concept-Drift-Aware Fraud Detection

**Document Version**: 1.0.0 (Frozen Protocol)  
**Date & Timestamp**: 2026-10-17T17:30:00Z  
**Principal Objective**: PhD Research Proposal Empirical Methodology  
**Grounded Literature Reference**: Carminati et al. (2015); Dal Pozzolo et al. (2015, 2018)  
**Integrity Status**: FROZEN BEFORE EVALUATION

---

## 1. Candidate Research Question & Motivation

In production fraud management, fraud analysts operate under fixed human capacity constraints (e.g. 100 to 500 high-priority alerts reviewed per operating day). Periodic retraining (e.g. retraining a gradient boosted tree every week) incurs substantial computational overhead, risks model thrashing on transient fluctuations, and ignores the operational delay inherent in fraud discovery (chargebacks).

This study formulates and tests the following falsifiable research question:

> **Core Research Question**:  
> *Can an adaptive retraining policy that triggers model updates based on statistical concept drift maintain fraud recall under a fixed daily review capacity while significantly reducing computational training cost compared with scheduled periodic retraining?*

---

## 2. Theoretical Grounding & Literature Review

### 2.1 Foundational Literature & Closest Methods
1. **Dal Pozzolo et al. (2014, 2015, 2018)** (*Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy*, IEEE TNNLS 29(8):3784–3797, 2018; *Learned lessons in credit card fraud detection from a practitioner perspective*, Expert Systems with Applications 41(10):4915–4928, 2014; Politecnico di Milano Repository [Handle: 11311/1044896](https://re.public.polimi.it/handle/11311/1044896)):
   - *Assumptions*: Formulated card-not-present fraud detection as an evolving, non-stationary data stream characterized by extreme class imbalance (fraud prevalence $\approx 0.2\%$) and delayed verification latency (feedback delays from chargebacks).
   - *Limitations*: Investigated fixed sliding-window and ensemble strategies with static re-weighting, but did not formalize adaptive statistical drift triggering coupled with calibrated review budget thresholds or paired moving-block bootstrap uncertainty.
2. **Carminati et al. (2015)** (*BankSealer: A Decision Support System for Online Banking Fraud Analysis and Investigation*, Computers & Security, 53:175–186, DOI: [10.1016/j.cose.2015.04.002](https://doi.org/10.1016/j.cose.2015.04.002)):
   - *Assumptions*: Online banking transactions evaluated via multi-profile anomaly detection (user profiles, group profiles, temporal profiles) with active analyst feedback loops.
   - *Limitations*: Relied on heuristic anomaly scores without strict operational daily quota allocation, formal concept drift hypothesis testing, or out-of-sample temporal block bootstrap estimation.

### 2.2 Primary Literature Comparison Table

| Attribute | Dal Pozzolo et al. (2014, 2018) | Carminati et al. (2015) | Proposed Research Protocol (v1.0.0) |
| :--- | :--- | :--- | :--- |
| **Domain & Data** | Commercial Credit Card (Europe) | Online Banking Transactions | Card-Not-Present E-Commerce (IEEE-CIS) |
| **Drift Adaptation** | Fixed sliding windows, time-decay ensembles | Heuristic multi-profile behavioral update | Adaptive drift triggers (ADWIN / DDM on error stream) |
| **Decision Policy** | Unconstrained ranking / top-$K$ per day | Anomaly score thresholding | Calibrated operational review budget ($K=100, 500$/day) |
| **Feedback Model** | Verification latency (delayed feedback window) | Iterative analyst dispute review | Idealized end-of-slice release (Analytical baseline) |
| **Statistical Rigor** | Average Precision, point estimates | Detection rate, false alarm rate | Paired moving-block bootstrap ($b=2$, 500 resamples), empirical non-inferiority ($H_1$) |
| **Generational Isolation**| Not formalized in stream architecture | Single-model profile adjustment | Generational prediction store with atomic promotion gate |

### 2.3 Proposed Scientific Contribution & Exact Difference
Novelty is **not** claimed merely from combining standard off-the-shelf drift detectors (ADWIN/DDM) with GBDT models or bookkeeping. Rather, our scientific contribution addresses the specific intersection of:
1. **Budget-Constrained Operational Thresholding**: Solving the operating point under human capacity constraints ($K \in \{100, 500\}$ alerts/day) on strictly prior chronological validation data, avoiding threshold inflation from score ties.
2. **Empirical Non-Inferiority Testing under Concept Drift**: Formalizing statistical testing ($H_1$) to prove that an adaptive policy achieves non-inferior fraud recall ($\delta = 0.01$) relative to blind exhaustive retraining.
3. **Generational Microservice Architecture**: Decoupling online scoring, asynchronous drift monitoring, and atomic model promotion across generation boundaries.

### 2.4 Label Availability & Temporal Assumptions
- **Idealized Availability Assumption**: In IEEE-CIS, transactions are timestamped relative to the dataset start (`step` in seconds). Crucially, **the dataset does not record genuine chargeback confirmation timestamps**.
- **Operational Reality vs. Model Assumption**: The default protocol assumes ground-truth labels mature at the end of each weekly slice. This is an **idealized availability assumption**, not measured operational dispute feedback (which in real banking networks incurs a 30-to-90 day delay). This limitation is explicitly acknowledged as a research dependency for future work.

---

## 3. Falsifiable Hypotheses & Verified Outcomes

### 3.1 Formal Hypotheses & Empirical Verification Status
- **Hypothesis $H_1$ (Accuracy Non-Inferiority)**: **SUPPORTED**  
  Under a fixed review capacity $K=100$ alerts/day on genuine IEEE-CIS transactions, the adaptive retraining policy achieves fraud recall non-inferior to weekly retraining within $\delta = 0.01$:
  $$\text{Recall}_{\text{adaptive}} \ge \text{Recall}_{\text{weekly}} - 0.01$$
  - *Observed Evidence*: Point delta is $0.0000$ (both detected 3,818 / 8,761 frauds = 43.58% recall). Paired moving-block bootstrap 95% CI is $[0.0000, 0.0000]$, and empirical p-value for $H_0: \Delta < -0.01$ is $p < 0.001$.
- **Hypothesis $H_2$ (Computational Efficiency)**: **NOT SUPPORTED (REJECTED in Protocol v1)**  
  The adaptive retraining policy was hypothesized to achieve non-inferior recall while reducing retraining events and training time by $\ge 30\%$ relative to weekly retraining:
  $$\text{ComputeCost}_{\text{adaptive}} \le 0.70 \times \text{ComputeCost}_{\text{weekly}}$$
  - *Observed Evidence*: In benchmark run `RUN_63D7CA7F68`, adaptive retrained 10 times in 62.35s vs weekly 10 times in 64.22s (a saving of only $2.9\%$, failing the $30\%$ target).
  - *Root Cause Analysis*: In the 22,000 transactions per weekly slice of the genuine IEEE-CIS stream, continuous feature drift and error fluctuations triggered ADWIN ($\delta=0.002$) in every single slice (slices 8–17). In the absence of a minimum evidence threshold or retrain cooldown period, adaptation occurred every week.
- **Hypothesis $H_3$ (Outperformance Over Static Baseline)**: **SUPPORTED**  
  In the presence of non-stationary drift over 10 consecutive weeks, adaptive retraining achieves a statistically significant gain in AP and F1 over the static baseline:
  $$\text{CI}_{0.95}(\Delta \text{AP}_{\text{adaptive} - \text{static}}) > 0$$
  - *Observed Evidence*: $\Delta \text{AP} = \mathbf{+0.0276}$ (+2.76 percentage points), 95% CI: $[\mathbf{+0.0226}, \mathbf{+0.0331}]$. $\Delta \text{F1} = \mathbf{+0.0123}$ (+1.23 percentage points), 95% CI: $[\mathbf{+0.0071}, \mathbf{+0.0169}]$. Both intervals strictly exclude zero ($p < 0.001$).

---

## 4. Leakage-Safe Temporal Experimental Protocol

```
+----------------------------------------------------------------------------------------------------+
|                                    CHRONOLOGICAL DATA PARTITIONING                                  |
+------------------------------------+------------------------------------+--------------------------+
| Part 1: Initial Training (W0-W7)   | Part 2: Pure Out-of-Sample (W8-W17)| Part 3: Future Reserved  |
| 212,449 transactions               | 221,727 transactions               | 156,364 transactions     |
| [Train: 80% | Chrono-Val: 20%]     | 10 Calendar Weeks                  | (Untouched Horizon)      |
+------------------------------------+------------------------------------+--------------------------+
```

### 4.1 Temporal Splitting Rules
1. **Partition Boundaries**:
   - Initial Training: Week 0 through Week 7 (212,449 transactions; 6,491 frauds).
   - Out-of-Sample Evaluation: Week 8 through Week 17 (221,727 transactions; 8,761 frauds).
   - The boundary between Week 7 and Week 8 is frozen. No transactions from Week $\ge 8$ are observed during preprocessing, feature selection, or initial hyperparameter fitting.
2. **Inner Validation Protocol**:
   - The initial training set is split chronologically: the first 80% is used for fitting model trees; the final 20% (strictly chronological holdout) is reserved exclusively for Platt sigmoid probability calibration and review budget threshold selection.

### 4.2 Feature Availability & Intra-Second Leakage Traps
- **Strict Prior Observable State**: Behavioral history features (`hist_tx_count`, `hist_tx_24h`, `hist_amount_mean`) are computed using transactions where $t_{\text{history}} < t_{\text{current}}$.
- **Tied-Timestamp Handling**: When transactions share identical timestamps ($t_A = t_B = T$), both transactions are scored against history from $t < T$ before the history accumulator is updated. Transaction IDs are never used as a proxy for causal ordering.

---

## 5. Decision Policies & Comparators

### 5.1 Competing Strategies
All strategies share identical initial estimators (XGBoost with histogram tree method, depth 6, learning rate 0.1, 100 estimators), identical features, and identical chronological data streams:
1. **Static Baseline (`static`)**: Trained once on Weeks 0–7 and frozen. Never updates.
2. **Periodic Retraining (`periodic`)**: Retrained every 4 calendar weeks on all accumulated verified data.
3. **Weekly Retraining (`weekly`)**: Retrained every single calendar week (10 retraining cycles). Represents maximum compute expenditure.
4. **Sliding Window (`sliding`)**: Retrained every 4 weeks on the most recent 4 weeks of verified data, dropping older history.
5. **Drift-Adaptive (`adaptive`)**: Retrained only when the drift monitoring subsystem (ADWIN / DDM on delayed feedback error rates) detects a statistically significant distribution shift.

### 5.2 Operating Decision Policies
1. **Unconstrained Validation-F1 Thresholding**: Optimal threshold $\theta^*$ maximizing F1 on prior chronological validation partition.
2. **Operational Review Budget Thresholding**: Threshold $\theta_K$ calibrated on prior chronological validation data such that daily alert volume matches human review bandwidth ($K \in \{100, 500\}$ reviews/day).

---

## 6. Primary and Secondary Evaluation Metrics

### 6.1 Performance Metrics
- **Average Precision (AP)**: Prespecified primary precision-recall metric:
  $$\text{AP} = \sum_{k} (R_k - R_{k-1}) P_k$$
  Distinguished explicitly from trapezoidal AUC-PR.
- **Recall@Budget ($K=100, 500$)**: Fraction of total out-of-sample frauds identified under fixed alert budgets.
- **Brier Score**: Evaluates probability calibration quality:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
- **Precision, Recall, F1**: Measured on binary predictions at the operating threshold.

### 6.2 Paired Uncertainty Estimation
- **Circular Moving-Block Bootstrap**: Paired block resampling (block size = 2 slices, 500 bootstrap iterations) across matched predictions on identical transactions.
- Produces 95% confidence intervals on $\Delta \text{AP}$ and $\Delta \text{F1}$ relative to baseline.

### 6.3 Measured Computational Cost
- Total wall-clock training seconds on standardized hardware (16-core CPU).
- Retrain event frequency (number of re-fits executed during the evaluation horizon).
