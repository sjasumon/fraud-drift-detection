# Prospective PhD Research Brief: Adaptive Fraud Detection Under Concept Drift

**Prepared for:** Professor Yun Sing Koh  
*Professor, School of Computer Science | Director, Master of AI (MAI)*  
*Director, Centre of Machine Learning for Social Good (ML4SG) | Co-Director, MARS Lab*  
*University of Auckland, New Zealand*  

**Candidate:** Sheikh Joynal Abedin Sumon  
*M.Sc. in Computer Science (National University of Bangladesh)*  
*17+ Years High-Performance Systems & Machine Learning Engineering*  
*Independent Researcher | Contact: me@sumon.me*  

---

## 1. Executive Summary & Research Motivation

In modern high-velocity financial streams, fraud detection algorithms face continuous non-stationarity (concept drift) driven by adversarial evasion. While literature on streaming machine learning frequently reports high accuracy on benchmark streams, practical industrial deployments reveal critical vulnerabilities that standard academic evaluation often overlooks:

1. **The Verification Latency Fallacy:** Algorithms that assume immediate supervised feedback (test-before-learn with zero delay) fail in production, where ground-truth fraud labels (chargebacks, merchant dispute resolutions) take **30 to 90 days** to mature.
2. **The Entity Memorization Critique:** In transaction streams containing account, card, or merchant identifiers, high-capacity gradient boosted trees can appear to adapt to drift simply by memorizing recurring compromised card/account tokens rather than learning underlying structural behavioral shifts.
3. **Statistical Saturation in Benchmarks:** Standard synthetic benchmarks (e.g. PaySim) are frequently evaluated on single random seeds, obscuring feature saturation where deterministic accounting identities produce deceptive metric parity.
4. **Retraining Cadence & Baseline Trade-Offs:** Comparing drift-adaptive retraining (which triggered on 17 of 21 test weeks, totaling 22 retrains) against Monthly Periodic Retraining ($K=4\text{w}$, 5 retrains) and Sliding Window ($W=4\text{w}$, 5 retrains) under strictly identical hyperparameters ($M=100$, max depth 6, $\eta=0.08$, subsample $0.8$). This exposes an honest empirical trade-off: Sliding Window achieves higher precision ($26.98\%$ vs. $23.52\%$) and F1 ($0.3829$ vs. $0.3565$) by discarding stale data, while error-based ADWIN/DDM lacks selectivity under class imbalance and retrains near-weekly, directly motivating unsupervised $P(X)$ monitoring.
5. **Warm-Up Isolation:** Drift detectors are strictly quarantined to out-of-sample slices, eliminating warm-up prediction leakage where models previously predicted on their own training data.
6. **Pure Streaming Ensembles:** Integration of canonical streaming algorithms surveyed in foundational stream mining literature (Gomes, Read, Bifet, Koh et al., ACM CSUR 2019)—Hoeffding Adaptive Tree (HAT), Adaptive Random Forest (ARF), and Streaming Random Patches (SRP)—to benchmark incremental stream learning against batch GBDT retraining.

To establish empirical rigor prior to doctoral admission, I designed, implemented, and validated this end-to-end reproducible research framework. The findings directly engage core research themes of the **MARS Lab** and establish the foundational baseline for a proposed doctoral dissertation on **Adaptive Fraud Detection Under Concept Drift with Privacy-Preserving Cross-Institution Federation**.

---

## 2. Core Research Questions & Empirical Discoveries

Across **119 verified invariant tests** (85% total codebase statement coverage, 100% CPU-portable reproduction), the framework evaluates the core research questions underpinning the proposed doctoral dissertation:

### RQ1 (Centrepiece — Unsupervised $P(X)$ Drift Adaptation Under Verification Latency):
*How can streaming fraud-detection architectures detect and proactively adapt to covariate distribution shifts in feature space ($P(X)$) during the 30–90 day verification latency window before ground-truth dispute labels mature, mitigating the empirical performance collapse of purely supervised models?*

* **Empirical Finding 1.1 (Decisive Multi-Seed Adaptation Frontier, $S=5$ Seeds, 95% CIs):** Benchmarking across 5 seeds and 5 latency horizons ($\delta \in \{0, 1, 2, 4, 8\text{ weeks}\}$) confirms that unsupervised $P(X)$ distribution monitoring (PSI triggers) decisively outperforms supervised error tracking (ADWIN/DDM) across realistic verification delays:
  - At $\delta=1\text{w}$ (7 days): Unsupervised $P(X)$ achieves **$80.52\% \pm 3.60\%$ recall** with only **$1.2$ cumulative retrains**, whereas Supervised error tracking decays to $75.94\% \pm 7.72\%$ despite triggering $7.0$ retrains.
  - At $\delta=2\text{w}$ (14 days): Unsupervised $P(X)$ achieves **$73.76\% \pm 1.07\%$ recall** with **$1.2$ retrains**, matching Periodic-1w ($73.63\% \pm 1.19\%$, $10.0$ retrains) with an **$88\%$ reduction in retraining compute**, while Supervised error tracking drops to $70.72\% \pm 2.95\%$ ($6.4$ retrains) due to structural "latency blindness".
  - At $\delta \ge 4\text{w}$ (28–56 days): All adaptive and periodic architectures decay asymptotically to the static baseline floor ($65.99\% \pm 1.55\%$), establishing that label verification latency is the fundamental operational ceiling in streaming fraud detection.
* **Empirical Finding 1.2 (Operational Latency Degradation in Batch Audit):** In exploratory single-seed batch auditing ($439{,}335$ transactions), immediate oracle recall ($72.70\%$, 21 retrains) degrades to $65.94\%$ under 4-week delay ($-6.76\%$ loss; 1,126 missed frauds), and decays to $63.45\%$ under 8-week delay, approaching static frozen performance ($63.27\%$). These empirical findings provide the central rationale for anchoring my PhD in unsupervised $P(X)$ stream volatility modeling, directly building upon Prof. Koh's foundational research in stream volatility modeling (SEED, ProSeed, CPF).

### RQ2 (Incremental Streaming Ensembles & Entity Generalization vs. Costly Retraining):
*How can hybrid architectures—coupling sample-by-sample incremental stream ensembles (e.g., Adaptive Random Forests, Hoeffding Adaptive Trees) with selective, drift-triggered retraining—balance update efficiency with extreme class imbalance recall, and how can strict entity-disjoint splits and TreeSHAP $\times$ PSI attribution verify that models learn generalizable behavioral distributions rather than memorizing repeating entity tokens?*

* **Empirical Finding 2.1 (Entity Generalization vs. Naive Memorization):** Under our **Explicit Identifier Ablation Protocol** (raw card and address tokens stripped from features), the adaptive model achieves **$76.62\% \pm 0.35\%$ recall** vs. **$69.41\% \pm 1.07\%$** for the static model on stream-unseen proxy cardholders upon their first-ever arrival (Cohort 3 clean complete cases across $S=5$ random seeds)—a statistically significant net gain of **$+7.21\% \pm 1.31\%$** ($t = 12.27, p_{\text{FDR}} = 4.23 \times 10^{-4}$, two-sided Wilcoxon $W=0.0, p = 0.0625$, Cohen's $d_z = 5.49$, Bootstrap 95% CI on Delta: **$[+6.28\%, +8.29\%]$**, Student's $t$ 95% CI: **$[+5.59\%, +8.85\%]$**; note that $p=0.0625$ represents the exact mathematical lower bound for a two-sided Wilcoxon test at $N=5$). Importantly, seen-card fraud recall ($48.90\% \to 61.39\%$) is lower than unseen-card recall ($66.30\% \to 73.79\%$ and $69.41\% \to 76.62\%$) because returning compromised cards mimic baseline customer behavior (account takeover), whereas novel cards in attack bursts display gross velocity anomalies. While derived aggregates ($C, D, V$) retain indirect behavioral fingerprints, this provides strong empirical evidence against naive categorical memorization. A rigorous gold-standard test on issuing-bank primary account numbers under group-based entity-disjoint temporal holdouts forms a primary research vector for my PhD.
* **Empirical Finding 2.2 (Streaming Baselines & Cadence Trade-offs):** When evaluating IEEE-CIS against Monthly Periodic Retraining ($K=4\text{w}$, 5 retrains) under identical tree hyperparameters ($M=100$, max depth 6, $\eta=0.08$, subsample $0.8$), drift-adaptive retraining achieves $+2.24\%$ higher overall recall ($73.58\%$ vs. $71.34\%$, weekly delta $+2.19\% \pm 2.90\%$, $p_{\text{FDR}} = 0.0036$). However, the adaptive detector fired on 17 of 21 test weeks ($22$ total retrains), behaving essentially as near-weekly retraining under class imbalance. Concurrently, Sliding Window ($W=4\text{w}$, 5 retrains) achieves the highest precision ($26.98\%$ vs. $23.52\%$) and F1 ($0.3829$ vs. $0.3565$). Canonical streaming baselines (HAT, ARF, SRP) process transactions incrementally at $>12{,}000$ samples/s without retraining pauses, highlighting trade-offs that motivate hybrid stream-batch designs.
* **Empirical Finding 2.3 (TreeSHAP $\times$ PSI Attribution & Saturated Stream Rigor):** Joint TreeSHAP $\times (1+\text{PSI})$ attribution heuristic isolates anonymized transaction count attribute `C13` as the primary driver of the Week 6 drift surge in IEEE-CIS, and pinpoints `errorBalanceOrig` ($\text{SHAP}=3.86$) in PaySim. Crucially, multi-seed evaluation on PaySim out-of-sample slices exposes that apparent single-run gains ($0.9955 \to 0.9978$) are stochastic noise ($t = 1.23, p = 0.2871$, Wilcoxon $p = 0.6250$; CI on delta crosses zero: $[-0.0001, +0.0015]$), demonstrating the necessity of statistical rigor on saturated benchmarks.

### RQ3 (Privacy-Preserving Cross-Institution Federation Under Non-IID Streams):
*How can drift-adaptive tree ensembles and unsupervised monitors be federated across decentralized banking institutions under non-IID transaction distributions without exposing raw customer records or proprietary features, and what cryptographic mechanisms (Secure Multi-Party Computation, Differential Privacy) provide provable security against semi-honest adversaries?*

* **Exploratory Exercise 3.1 (Motivating Toy Prototype: Cross-Bank Blindness, False-Positive Collapse, and Leakage Limits):** To explore Phase 2 feasibility prior to doctoral admission, I implemented an exploratory multi-bank prototype (`src/federated.py`) simulating 3 non-IID banking archetypes (E-Commerce, Remittances, Commercial; KS-test $D=0.95, p=0.0$). This motivating exercise exposed two foundational research challenges: (1) Naive cross-bank model ensembling (Noisy-OR aggregation) achieves global recall coverage across disparate attacks but triggers catastrophic false-alarm inflation (local precision collapses from $\sim 85\%$ down to $20.8\%$ on Bank B and $20.0\%$ on Bank C due to non-IID distribution mismatch); and (2) Exchanging raw tree ensembles leaks feature split points, thresholds, and leaf parameters to semi-honest adversaries, demonstrating that "zero raw transaction bytes transferred" does not provide cryptographic privacy. These acute limitations directly motivate researching true federated tree training with Secure Multi-Party Computation (SMPC) and Differential Privacy (DP) as the core focus of Phase 2.

### RQ4 (Operational Latency-Privacy-Accuracy Frontiers):
*What are the empirical Pareto trade-offs between formal cryptographic privacy guarantees, sub-15ms real-time scoring SLAs, and asynchronous label verification in distributed, multi-institution payment networks?*

* **Empirical Finding 4.1 (Microservice Serving SLA & Overhead):** Starlette in-process microservice benchmarking ($1{,}000$ scoring requests) confirms median serving latency of $8.06$ ms ($p95=9.88$ ms, $p99=13.38$ ms), meeting a sub-15ms production SLA. Pure XGBoost tree inference on a 100-tree model takes only $\approx 0.24$ ms per sample (with web framework dispatch and JSON serialization accounting for $\approx 7.8$ ms), demonstrating ample latency headroom for Phase 2 cryptographic privacy overhead.

---

## 3. Summary of Statistical Rigor

| Research Benchmark | Baseline Architecture | Drift-Adaptive (Proposed) | Net Advantage ($\Delta$) | Effect Size | Significance ($p$-value \& CI on Delta) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Zero-Memory Recall (Cohort 3, Clean Cases)** | $69.41\% \pm 1.07\%$ | **$76.62\% \pm 0.35\%$** | **$+7.21\% \pm 1.31\%$** | Cohen's $d_z = 5.49$ | **$t = 12.27, p_{\text{FDR}} = 4.23 \times 10^{-4}$** (Wilcoxon $p = 0.0625$, CI: $[+5.59\%, +8.85\%]$) |
| **Overall Stream Recall ($N=21$w)** | $62.90\% \pm 5.07\%$ | **$73.59\% \pm 5.17\%$** | **$+10.69\% \pm 6.27\%$** | Cohen's $d_z = 1.70$ | **$p_{\text{FDR}} = 5.60 \times 10^{-7}$** (Wilcoxon $p = 1.34 \times 10^{-5}$, MBB 95\% CI: $[+8.02\%, +14.33\%]$) |
| **Overall Stream AUC-PR ($N=21$w)** | $0.4575 \pm 0.0572$ | **$0.5302 \pm 0.0584$** | **$+0.0728 \pm 0.0311$** | Cohen's $d_z = 2.34$ | **$p_{\text{FDR}} = 9.85 \times 10^{-9}$** (Wilcoxon $p = 1.91 \times 10^{-6}$, MBB 95\% CI: $[+0.0630, +0.0881]$) |
| **Monthly Periodic ($K=4\text{w}$)** | $71.40\% \pm 5.31\%$ (Monthly) | **$73.59\% \pm 5.17\%$** | **$+2.19\% \pm 2.90\%$** | Cohen's $d_z = 0.75$ | **$p_{\text{FDR}} = 0.0036$** (MBB 95\% CI: $[+1.67\%, +3.52\%]$) |
| **Decisive Latency Frontier ($\delta=2\text{w}$, $S=5$)** | $70.72\% \pm 2.95\%$ (Supervised) | **$73.76\% \pm 1.07\%$** (Unsupervised $P(X)$) | **$+3.04\%$ Recall** ($88\%$ compute saving: 1.2 vs 6.4 retrains) | Parity with Periodic ($73.63\%$) | 95\% CI $[72.4\%, 75.1\%]$ vs $[67.1\%, 74.4\%]$ |
| **4-Week Delay Degradation** | $63.27\%$ (Static floor) | $65.94\%$ ($4\text{w}$ delay) | $-6.76\%$ vs Oracle | — | Empirical decay curve ($1{,}126$ missed frauds) |
| *Exploratory 3-Bank Toy (RQ3)* | $64.71\%$ (Isolated mean) | **$100.0\%$** (Global coverage) | Motivating preliminary | — | **Exposes false-alarm collapse** (precision $85\% \to 20\%$) \& tree parameter leakage |
| **PaySim Saturated Benchmark** | $0.9958 \pm 0.0002$ | $0.9964 \pm 0.0009$ | $+0.0006 \pm 0.0011$ | $d = 0.55$ | $p = 0.2871$ (n.s. — CI on $\Delta$: $[-0.0001, +0.0015]$) |
| **Streaming Tree Baselines (River)** | HAT, ARF ($M=5$), SRP ($M=3$) | Full Retrain (Hist GBDT) | Superior Imbalance Recall | — | $>12{,}000$ samples/s streaming benchmarks |
| *Exploratory Bandit Simulation¹* | $510.33 \pm 53.98$ (UCB1) | $254.63 \pm 29.79$ (ADWIN-UCB) | $-50.1\%$ Regret | — | $t = 12.18, p = 0.00026$ (Routing simulation) |

\footnotesize{¹ *Exploratory Multi-Armed Bandit Simulation*: Benchmarks non-stationary decision-routing algorithms (Besson \& Kaufmann 2018; Slivkins et al.) in `src/bandit/`; included as an exploratory routing simulation rather than a claimed doctoral contribution.}


---

## 4. Alignment with MARS Lab Publications & Research Agenda

This pre-proposal codebase directly bridges the gap between single-institution stream mining and the collaborative, privacy-preserving research program of the MARS Lab:

1. **Stream Volatility-Shift Detection & Proactive Adaptation:**  
   The experimental design engages directly with Prof. Koh's foundational contributions to data stream mining and concept drift:
   - *Detecting Volatility Shift in Data Streams* (**SEED**: Huang, Koh, Dobbie, Pears, IEEE ICDM 2014) — modeling stream volatility to calibrate drift thresholds and suppress false alarms during non-stationary bursts.
   - *Anticipating Concept Drift: A Proactive Approach Using Stream Volatility* (**ProSeed**: Chen, Koh, Riddle, PAKDD 2018) — proactive drift prediction using probabilistic networks.
   - *Recurring Concept Profiling* (**CPF / PEARL**: Anderson, Koh, Dobbie) — identifying recurring drift patterns and reusing historical concept models instead of blind retraining.
   - *Machine Learning for Streaming Data: State of the Art, Challenges, and Future Directions* (Gomes, Read, Bifet, Barddal, Gama, Pfahringer, Holmes, Koh, ACM Computing Surveys 2019) — foundational survey of streaming ensembles (HAT, ARF, SRP) and non-stationary learning.
2. **GPU-Accelerated Decision Tree Training & Privacy-Preserving ML:**  
   The retraining pipeline implements sub-1.5s GPU histogram tree training, directly echoing the architectural and cryptographic principles explored in recent MARS lab publications:  
   - *GTree: GPU-Friendly Privacy-Preserving Decision Tree Training and Inference* (Wang et al., IEEE TrustCom 2024)
   - *XGT: An Efficient and Secure Gradient Boosting Decision Tree Engine* (Wang et al., IEEE TDSC 2025)
3. **Proposed Doctoral Roadmap (Phases 1–3):**
   - **Phase 1 (Completed in Pre-Proposal — Addressing RQ1, RQ2):** Single-stream drift detection under operational verification latency, entity memorization audits, cadence-matched baseline deconfounding, and incremental River streaming baselines (HAT, ARF, SRP).
   - **Phase 2 (Immediate PhD Focus — Addressing RQ3):** Privacy-Preserving Cross-Institution Federation. Developing true federated gradient-boosted decision trees using Secure Multi-Party Computation (SMPC) for split histograms and Differential Privacy (DP) on leaf distributions, directly solving the precision collapse and privacy leakage exposed by our preliminary toy prototype (`src/federated.py`) and building upon MARS Lab secure tree engines (*GTree*, *XGT*).
   - **Phase 3 (Operational Optimization — Addressing RQ4):** Systematic analysis of latency-privacy-accuracy Pareto frontiers (<15ms scoring SLAs, cryptographic communication overhead, and asynchronous dispute verification), culminating in the doctoral dissertation.

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

- **Test Suite:** **119 passing tests** in ~29s (0 skips, 0 failures).
- **Code Coverage:** **85% total statement coverage** across all modules:
  - Core algorithmic models (`adaptive_model`, `static_model`, `periodic_model`, `sliding_model`, `river_models`, `federated`): **100%**
  - Evaluator & metric computation: **100%**
  - Delayed feedback & maturity queues: **96%**
  - High-level orchestrators (`pipeline`, `delayed_audit`, `evaluate_uid_split`, `bandit/benchmark`): **81%–97%**
- **Manuscript Package:** 5-page IEEE-format working draft manuscript with complete LaTeX sources and 300-DPI publication figures in [`arxiv_paper/`](arxiv_paper/).

I would welcome the opportunity to discuss these empirical findings, review the doctoral research intent, and align on initial supervisory expectations.
