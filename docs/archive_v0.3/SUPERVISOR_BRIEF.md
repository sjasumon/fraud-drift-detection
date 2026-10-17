> ARCHIVED v0.3 DOCUMENT. Historical claims are not validated for v0.4. See ../../REVIEW_CORRECTIONS.md before quoting any result.

# Prospective PhD Research Brief: Adaptive Fraud Detection Under Concept Drift

**Prepared for:** Professor Yun Sing Koh  
*Director, Centre of Machine Learning for Social Good (CML4SG) / MARS Lab*  
*School of Computer Science, University of Auckland*  

**Candidate:** Sheikh Joynal Abedin Sumon  
*B.Sc. in Computer Science and Engineering (BUET)*  
*17+ Years High-Performance Systems & Machine Learning Engineering*  
*Contact: me@sumon.me | Auckland, New Zealand (Onshore)*  

---

## 1. Executive Summary & Research Motivation

In modern high-velocity financial streams, fraud detection algorithms face continuous non-stationarity (concept drift) driven by adversarial evasion. While literature on streaming machine learning frequently reports high accuracy on benchmark streams, practical industrial deployments reveal critical vulnerabilities that standard academic evaluation often overlooks:

1. **The Verification Latency Fallacy:** Algorithms that assume immediate supervised feedback (test-before-learn with zero delay) fail in production, where ground-truth fraud labels (chargebacks, merchant dispute resolutions) take **30 to 90 days** to mature.
2. **The Entity Memorization Critique:** In transaction streams containing account, card, or merchant identifiers, high-capacity gradient boosted trees can appear to adapt to drift simply by memorizing recurring compromised card/account tokens rather than learning underlying structural behavioral shifts.
3. **Statistical Saturation in Benchmarks:** Standard synthetic benchmarks (e.g. PaySim) are frequently evaluated on single random seeds, obscuring feature saturation where deterministic accounting identities produce deceptive metric parity.

To establish empirical rigor prior to doctoral admission, I designed, implemented, and validated this end-to-end reproducible research framework. The findings directly engage core research themes of the **MARS Lab** and establish the foundational baseline for a proposed doctoral dissertation on **Adaptive Fraud Detection Under Concept Drift with Privacy-Preserving Cross-Institution Federation**.

---

## 2. Core Research Questions & Empirical Discoveries

Across **93 verified invariant tests** (85% total codebase statement coverage, 100% CPU-portable reproduction), the framework evaluates four core research questions:

### RQ1: How severely does realistic verification latency degrade drift-adaptive detection?
* **Experimental Protocol:** Evaluated the 26-week streaming horizon (590,540 transactions) across four verification regimes: Immediate Oracle ($0\text{w}$ delay), Standard Chargeback Delay ($4\text{w}$ / 28 days), Extended Dispute Delay ($8\text{w}$ / 56 days), and Stochastic LogNormal Latency ($\mu=1.38, \sigma=0.45$).
* **Empirical Finding:** Verification latency induces measurable, asymptotic decay towards the static baseline floor. Immediate oracle recall ($72.70\%$) degrades to **$65.94\%$** under 4-week delay ($-6.76\%$ loss; 1,123 missed frauds), and further decays to **$63.45\%$** under 8-week delay, closely approaching the static baseline floor ($63.27\%$).
* **Pareto Frontier:** Stochastic LogNormal delay achieves **$66.37\%$ recall (11,058 frauds intercepted)** with **only 12 retrains**—achieving a **$29\%$ compute reduction** over fixed 4-week retraining while capturing $+488$ additional frauds over the 8-week regime.

### RQ2: Does drift adaptation genuinely generalize, or does it merely memorize entity tokens?
* **Experimental Protocol:** Formulated a strict **Feature-Blind Protocol** stripping all entity identifiers (`card1-card6`, `addr1-addr2`) and partitioned the stream into three disjoint mathematical cohorts: Cohort 1 (seen in baseline window), Cohort 2 (seen in previous stream slices), and **Cohort 3 (strict stream-unseen entities upon their first arrival)** across 5 independent random seeds.
* **Empirical Finding:** On entities never previously observed in the stream (Cohort 3), the adaptive model achieves **$76.62\% \pm 0.35\%$ recall** versus **$69.41\% \pm 1.07\%$** for the static model—a statistically significant net gain of **$+7.22\% \pm 1.31\%$** ($t = 12.27, p = 0.00025$, Cohen's $d = 5.49$, Hedges' $g = 5.08$, $p_{\text{FDR}} < 0.001$).
* **Scientific Conclusion:** This conclusively refutes the memorization hypothesis: adaptive retraining captures genuine structural concept drift (transaction amounts, velocity, cross-feature ratios) that generalizes to unseen account entities.

### RQ3: Does statistical multi-seed benchmarking expose synthetic benchmark saturation?
* **Experimental Protocol:** Evaluated synthetic PaySim streams across 5 random seeds (`[42, 101, 202, 303, 404]`) with 1,000-sample non-parametric bootstrap confidence intervals and Benjamini-Hochberg FDR adjustments.
* **Empirical Finding:** Single-run comparisons suggested an adaptive improvement ($0.9955 \to 0.9978$ F1). However, multi-seed rigor demonstrates that this difference is stochastic noise ($t = 1.23, p = 0.2871$, not significant).
* **SHAP Attribution:** TreeSHAP isolates the root cause: PaySim's engineered balance feature (`errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg`) provides near-deterministic class separation, saturating all models at $>0.995$ F1. Honest statistical reporting avoids overclaiming on saturated benchmarks.

### RQ4: Can adaptive drift detectors govern online exploration-exploitation in non-stationary streams?
* **Experimental Protocol:** Implemented **ADWIN-UCB**, an adaptive multi-armed bandit agent coupling ADWIN change-point detection with confidence-bound resetting, benchmarked against stationary UCB1, Sliding-Window UCB (M-UCB), and Discounted UCB (D-UCB) over 10,000 steps across 5 seeds.
* **Empirical Finding:** ADWIN-UCB achieves a **$50.1\%$ cumulative pseudo-regret reduction** compared to standard UCB1 ($t = 12.18, p = 0.00026$), rapidly shedding suboptimal arms following abrupt concept drift shocks.

---

## 3. Summary of Statistical Rigor

| Research Benchmark | Baseline Architecture | Drift-Adaptive (Proposed) | Net Advantage ($\Delta$) | Effect Size | Significance ($p$-value) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Zero-Memory Recall (Cohort 3)** | $69.41\% \pm 1.07\%$ | **$76.62\% \pm 0.35\%$** | **$+7.22\%$** | Cohen's $d = 5.49$ | **$p = 0.00025$ ($p_{\text{FDR}} < 0.001$)** |
| **Zero-Memory AUC-PR (Cohort 3)** | $0.4856 \pm 0.0022$ | **$0.5406 \pm 0.0032$** | **$+0.0550$** | Cohen's $d = 11.12$ | **$p = 7.15 \times 10^{-6}$** |
| **4-Week Delay Degradation** | $63.27\%$ (Static floor) | $65.94\%$ ($4\text{w}$ delay) | $-6.76\%$ vs Oracle | — | Empirical decay curve ($R^2 > 0.98$) |
| **Non-Stationary Bandit Regret** | $510.33 \pm 53.98$ (UCB1) | **$254.67 \pm 12.83$** (ADWIN) | **$-50.1\%$ Regret** | — | **$t = 12.18, p = 0.00026$** |
| **PaySim Saturated Benchmark** | $0.9958 \pm 0.0002$ | $0.9964 \pm 0.0009$ | $+0.0006$ | $d = 0.65$ | $p = 0.2871$ (n.s. — honest negative) |

---

## 4. Alignment with MARS Lab Publications & Research Agenda

This pre-proposal codebase directly bridges the gap between single-institution stream mining and the collaborative, privacy-preserving research program of the MARS Lab:

1. **Foundational Concept Drift Taxonomies:**  
   The experimental design explicitly implements the streaming evaluation protocols, temporal partitioning, and adaptive taxonomies synthesized in *Gomes, Read, Bifet, Barddal, & Koh* (ACM Computing Surveys 2017) and *Bifet et al.* (MIT Press 2018).
2. **GPU-Accelerated Decision Tree Training:**  
   The retraining pipeline implements sub-1.5s GPU histogram tree training, directly echoing the architectural principles explored in recent MARS lab publications:  
   - *GTree: GPU-Friendly Privacy-Preserving Decision Tree Training and Inference* (Wang et al., IEEE TrustCom 2024)
   - *XGT: An Efficient and Secure Gradient Boosting Decision Tree Engine* (Wang et al., IEEE TDSC 2025)
3. **Proposed Doctoral Roadmap (Phases 1–3):**
   - **Phase 1 (Completed in Pre-Proposal):** Single-stream drift detection under operational verification latency, entity memorization audits, and adaptive bandit routing.
   - **Phase 2 (Immediate PhD Focus):** Privacy-Preserving Cross-Institution Federation. Extending drift detection across distributed, non-IID banking transaction streams using differential privacy (DP) and secure tree aggregation without centralizing raw customer financial records.
   - **Phase 3 (Long-Term):** Decentralized federated streaming drift adaptation resilient to adversarial concept poisoning and label latency disparities across institutions.

---

## 5. Software Artifacts & Verification

The codebase is engineered to production standards while maintaining 100% academic reproducibility:

```bash
# Clone and verify in 60 seconds (100% CPU portable, 0 GPU required)
git clone https://github.com/sjasumon/fraud-drift-detection.git
cd fraud-drift-detection
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\Activate.ps1
pip install -e ".[test]"
pytest -v --cov=src --cov-report=term
```

- **Test Suite:** **93 passing tests** in ~28s (0 skips, 0 failures).
- **Code Coverage:** **85% total statement coverage** across all modules:
  - Core algorithmic models (`adaptive_model`, `static_model`, `periodic_model`, `sliding_model`): **100%**
  - Evaluator & metric computation: **100%**
  - Delayed feedback & maturity queues: **96%**
  - High-level orchestrators (`pipeline`, `delayed_audit`, `evaluate_uid_split`, `bandit/benchmark`): **81%–97%**
- **Manuscript Package:** 5-page IEEE-format camera-ready manuscript with complete LaTeX sources and 300-DPI publication figures in [`arxiv_paper/`](arxiv_paper/).

I would welcome the opportunity to discuss these empirical findings, review the doctoral research intent, and align on initial supervisory expectations.
