# Data Provenance and Feature Availability Audit

**Document Version**: 1.0.0  
**Date**: 17 October 2026  
**Project**: Concept-Drift-Aware Fraud Detection (`fraud-drift-detection` v0.5.0)  
**System Integrity Status**: ENFORCED (`src/provenance.py`, `src/artifacts.py`, `src/ingestion.py`)

---

## 1. Executive Summary & Provenance Hierarchy

In accordance with Section 2 of the Next-Phase Engineering & Research Specification, this project enforces strict provenance classification across all ingested datasets. Synthetic fixtures, simulator-generated transactions, and unverified data are rigorously excluded from PhD proposal claims, primary research conclusions, and operational production serving.

```
+----------------------------------------------------------------------------------------------------+
|                                    PROVENANCE CLASSIFICATION                                        |
+------------------------------------+-----------------------+---------------------+-----------------+
| Dataset                            | Source Type           | Classification      | Research Status |
+------------------------------------+-----------------------+---------------------+-----------------+
| IEEE-CIS / Vesta E-Commerce        | Real Production CNP   | GENUINE_OBSERVED    | ACCEPTED        |
| PaySim Mobile Money                | Multi-Agent Simulator | SIMULATOR_GENERATED | EXCLUDED        |
| Feedzai Bank Account Fraud (BAF)   | Synthesized Anonymized| SYNTHETIC           | EXCLUDED        |
| Unknown / Unregistered Sources     | Third-Party Mirror    | UNVERIFIED          | REJECTED        |
+------------------------------------+-----------------------+---------------------+-----------------+
```

---

## 2. Benchmark Dataset Audit: IEEE-CIS Fraud Detection

The **IEEE-CIS / Vesta Corporation** dataset serves as the sole genuine real-world benchmark supporting the research findings and PhD proposal claims.

### 2.1 Provider & Access Terms
- **Provider**: Vesta Corporation & IEEE Computational Intelligence Society (CIS)
- **Competition Reference**: [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/overview)
- **Official Data Repository**: [IEEE-CIS Data Access](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- **License / Terms**: IEEE-CIS Competition Data Agreement; non-commercial academic research and benchmarking.
- **Verification Method**: Cryptographic SHA-256 digest validation against official archives.

### 2.2 Dataset Profiles & File Checksums
| File Name | SHA-256 Digest | File Size | Row Count | Column Count |
| :--- | :--- | :--- | :--- | :--- |
| `train_transaction.csv` | `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642` | 683,353,498 bytes | 590,540 | 394 |
| `train_identity.csv` | `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c` | 26,524,341 bytes | 144,233 | 41 |

### 2.3 Temporal Structure & Class Distribution
- **Observed Duration**: 182 calendar days (~26 calendar weeks).
- **Time Offset Key**: `TransactionDT` in seconds ($86,400 \le \text{TransactionDT} \le 15,811,131$).
- **Total Fraudulent Transactions**: 20,663 out of 590,540 (overall prevalence: **3.50%**).
- **Temporal Partitions (Frozen Research Split)**:
  - **Initial Training Partition (Weeks 0–7 / Slices 0–7)**: 212,449 transactions; 6,491 frauds (**3.06%** prevalence).
  - **Out-of-Sample Evaluation Partition (Weeks 8–17 / Slices 8–17)**: 221,727 transactions; 8,761 frauds (**3.95%** prevalence).
  - **Reserved Future Horizon (Weeks 18–25 / Slices 18–25)**: 156,364 transactions; 5,411 frauds (**3.46%** prevalence).
- **Observed Natural Drift**: Fraud prevalence exhibits authentic temporal evolution across the evaluation stream, ranging from a low of 2.07% in Week 3 to a peak of 5.07% in Week 16.

### 2.4 Critical Research Disclosure: Feedback Latency Nature
> [!IMPORTANT]
> **Transaction Authenticity vs. Feedback Authenticity**  
> While transaction attributes and events in IEEE-CIS are genuine observed records from Vesta's payment processing network, **true dispute confirmation timestamps (e.g. chargeback notification dates from card issuing banks) are absent from the dataset**. The ground-truth `isFraud` label is an offline, retrospective outcome.  
> 
> Therefore, **observed delayed-feedback dynamics cannot be established from IEEE-CIS without genuine dispute confirmation timestamps**. Any simulated delay functions (fixed delays or log-normal distributions) must be reported strictly as *exploratory sensitivity analyses* and do not establish real-world feedback operational performance. Acquirers' operational confirmation timestamps remain a documented research dependency for future work.

---

## 3. Excluded Datasets & Justification

### 3.1 PaySim Mobile Money Simulator
- **Classification**: `SIMULATOR_GENERATED`
- **Origin**: Multi-agent financial simulator developed by Lopez-Rojas et al.
- **Repository**: [https://github.com/EdgarLopezPhD/PaySim](https://github.com/EdgarLopezPhD/PaySim)
- **Scientific Limitation**: Agent-based generative models follow programmed behavioural rules that lack the adversarial complexity, subtle feature correlations, and unmodelled drift mechanisms present in human financial crime.
- **Enforcement**: Excluded from PhD proposal primary empirical benchmarks. Retained only in historical test scripts (`tests/test_multi_dataset.py`).

### 3.2 Feedzai Bank Account Fraud (BAF)
- **Classification**: `SYNTHETIC`
- **Origin**: Synthetic tabular dataset generated by Feedzai (NeurIPS 2022 Benchmark Suite).
- **Repository**: [https://github.com/feedzai/bank-account-fraud](https://github.com/feedzai/bank-account-fraud)
- **Scientific Limitation**: BAF uses differential privacy and generative models conditioned on banking data. While realistic, synthetic marginals do not exhibit genuine production-grade sensor noise or true adversarial drift.
- **Enforcement**: Excluded from primary empirical claims.

---

## 4. Feature Availability & Leakage Prevention Protocol

To guarantee that no model evaluates with access to future or counterfactual information, the following mathematical invariants are enforced in `src/history_features.py` and `src/preprocessing.py`:

### 4.1 Strict Chronological Availability
- All aggregate statistics (mean, count, standard deviation) computed over historical transactions must satisfy:
  $$\tau(\text{historical transaction}) < \tau(\text{current transaction})$$
- Future transactions ($\tau \ge T$) and future labels ($Y_{t \ge T}$) are strictly inaccessible during scoring.

### 4.2 Tied-Timestamp Leakage Trap Mitigation
When multiple transactions share the exact same timestamp ($t = T$):
- **Vulnerability**: If transactions at $t = T$ are processed sequentially in arbitrary dataframe order, transaction $B$ at $t = T$ observes features influenced by transaction $A$ at $t = T$. This creates artificial intra-second causal ordering.
- **Mathematical Invariant**: In `src/history_features.py`, grouping is executed such that all events at timestamp $T$ are scored against state derived strictly from $t < T$:
  $$\text{State}(T) = f(\{x_i \mid t_i < T\})$$
  The shared cumulative state is updated only after all events in the tied equivalence class $\{x \mid t = T\}$ have been scored.

### 4.3 Missing Identity Handling
- Unauthenticated transactions in IEEE-CIS (`train_identity.csv`) lack device/browser metadata.
- Missing identity values are never imputed by merging them into an artificial customer ID or synthetic aggregate profile. Missing indicators (`is_null`) preserve observed sparsity patterns without creating synthetic entity collusions.

---

## 5. Automated Provenance Verification System

The codebase enforces dataset integrity at multiple checkpoints:

1. **Ingestion Level (`src/ingestion.py`)**: `DatasetIngestion.load_and_validate()` assigns cryptographic provenance records and flags non-genuine datasets as `EXCLUDED_NON_GENUINE`.
2. **Package Level (`src/artifacts.py`)**: `ModelBundle.ready` queries `verify_bundle_provenance()`. Models trained on synthetic, simulator, or unverified data return `ready = False`.
3. **API Level (`src/api.py`)**: The `/v1/score` microservice returns `HTTP 503 Service Unavailable` if an unverified bundle attempts to serve live scoring traffic.
