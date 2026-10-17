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
1. **Carminati et al. (2015)** (*BankSealer: A Decision Support System for Online Banking Fraud Analysis and Investigation*, Politecnico di Milano, [Handle: 11311/1044896](https://re.public.polimi.it/handle/11311/1044896)):
   - *Assumptions*: Assumed analyst feedback is available incrementally and that anomalies indicate emerging fraud trends. Introduced multi-profile behavioral analysis.
   - *Limitation*: Relies on heuristic anomaly scoring without explicit budget-constrained decision policies or temporal block-bootstrap statistical rigor.
2. **Dal Pozzolo et al. (2015, 2018)** (*Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy*, IEEE TNNLS):
   - *Assumptions*: Formulated credit card fraud as an unbalanced, non-stationary data stream characterized by delayed feedback (verification latency).
   - *Limitation*: Examined sliding-window and ensemble strategies with fixed time horizons, but did not formalize adaptive statistical drift triggering coupled with calibrated review budget thresholds.

### 2.2 Proposed Scientific Contribution & Exact Difference
Unlike naive streaming methods that either retrain blindly on every temporal slice (weekly) or maintain a static model indefinitely, our proposed methodology introduces:
1. **Drift-Triggered State Transitions**: Incorporating adaptive drift monitoring (ADWIN + DDM error-rate tracking) that retrains only when the incoming transaction error distribution diverges significantly from baseline ($p < 0.002$).
2. **Operational Review Budget Thresholding**: Fitting decision thresholds on historical chronological validation sets to target an operational daily alert capacity ($K=100$ or $K=500$ reviews/day) rather than maximizing unconstrained F1.
3. **Generational Label Isolation**: Guaranteeing that delayed dispute outcomes update only the detector state corresponding to their prediction generation, avoiding retroactive contamination.

---

## 3. Falsifiable Hypotheses & Success Criteria

### 3.1 Formal Hypotheses
- **Hypothesis $H_1$ (Accuracy Non-Inferiority)**:  
  Under a fixed review capacity $K=100$ alerts/day on genuine IEEE-CIS transactions, an adaptive retraining policy achieves a fraud recall non-inferior to weekly scheduled retraining within a practical equivalence margin $\delta = 0.01$ (1.0% recall):
  $$\text{Recall}_{\text{adaptive}} \ge \text{Recall}_{\text{weekly}} - 0.01$$
- **Hypothesis $H_2$ (Computational Efficiency)**:  
  The adaptive retraining policy will achieve this non-inferior recall while reducing cumulative retraining events and training time (CPU seconds) by at least 30% relative to weekly retraining:
  $$\text{ComputeCost}_{\text{adaptive}} \le 0.70 \times \text{ComputeCost}_{\text{weekly}}$$
- **Hypothesis $H_3$ (Outperformance Over Static Baseline)**:  
  In the presence of authentic non-stationary drift over 10 consecutive weeks, adaptive retraining achieves a statistically significant gain in Average Precision (AP) and F1-score over the static baseline, verified by a strictly positive 95% block-bootstrap confidence interval:
  $$\text{CI}_{0.95}(\Delta \text{AP}_{\text{adaptive} - \text{static}}) > 0$$

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
