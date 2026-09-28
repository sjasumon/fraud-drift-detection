# Statement of Research Intent

**Working Title:** Unsupervised Concept Drift Adaptation in Financial Streams Under Asynchronous Verification Latency  
*(Secondary Vector: Towards Privacy-Preserving Cross-Institution Extension)*

**Applicant:** Sheikh Joynal Abedin Sumon  
**Proposed Supervisor:** Professor Yun Sing Koh, School of Computer Science | Director, Centre of Machine Learning for Social Good (ML4SG) / Co-Director, MARS Lab, University of Auckland  
**Target Programme:** PhD in Computer Science, University of Auckland (Scholarship Round: 1 November 2026)  

---

## 1. Motivation & The Verification Latency Dilemma

Modern financial fraud detection systems operate on non-stationary, adversarial data streams: perpetrators continuously modify tactics to bypass deployed machine learning models—a phenomenon formalized in data stream mining as **concept drift**. However, existing academic literature routinely relies upon an unrealistic operational assumption: that ground-truth labels arrive instantaneously.

In industrial banking and payment networks, fraud labels arrive with substantial asynchronous verification latency (**30 to 90 days**), as unauthorized transactions undergo customer reporting, bank investigation, and chargeback adjudication. Our preliminary empirical findings reveal that under standard verification latency (4 to 8 weeks), conventional supervised drift adaptation collapses: out-of-sample recall degrades from an oracle immediate level of $72.70\%$ down to $63.45\%$, virtually converging to the unadapted static baseline floor ($63.27\%$). 

Because supervised error signals ($P(Y \mid X)$) arrive weeks after an attack surge has occurred, streaming architectures cannot rely on error counters to prompt adaptation. To protect payment networks in production, systems must detect distribution shifts directly in the **unsupervised feature space ($P(X)$)**, initiating adaptation *before* dispute labels mature. This proposal centers squarely on solving this fundamental bottleneck.

---

## 2. Primary Research Question

**Primary Research Question (RQ1):**  
*How can streaming fraud-detection architectures detect and proactively adapt to concept drift in unsupervised feature space ($P(X)$) during the 30–90 day verification latency window before ground-truth dispute labels mature, mitigating the empirical performance collapse of purely supervised models?*

**Investigative Sub-Questions:**
1. **$P(X)$ Drift Sensitivity & False-Alarm Suppression:** How can multivariate feature-space monitors (such as volatility-shift windows and Population Stability Index divergence) reliably detect subtle and abrupt fraud pattern shifts under extreme class imbalance without being triggered by benign non-stationary transaction volume?
2. **Deconfounding Adaptation from Retrain Cadence:** How can drift-triggered retraining be deconfounded from fixed-cadence periodic retraining (e.g., Weekly Periodic), achieving optimal trade-offs between recall and compute overhead under label delay?
3. **Recurring Concept Memory vs. Costly Retraining:** How can recurring attack patterns be recognized and re-activated using concept profiling rather than repeatedly executing costly batch retraining from scratch?

---

## 3. Proposed Methodology & Preliminary Empirical Evidence

The proposed research will proceed across three integrated phases, building directly upon a reproducible codebase and preliminary empirical benchmarks:

```
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                           PHASE 1: UNSUPERVISED P(X) DRIFT ADAPTATION                  │
  │   Unsupervised Feature Monitoring (SEED / PSI / MMD) ──► Early Retrain / Recalibration │
  └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                              ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                           PHASE 2: RECURRING CONCEPT POOLING                           │
  │   Concept Profiling Framework (CPF) ──► Reservoir Memory ──► Zero-Retrain Re-Activation │
  └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                              ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                           PHASE 3: FEDERATED CROSS-INSTITUTION EXTENSION               │
  │   Multi-Bank Non-IID Streams ──► Secure MPC / DP ──► Privacy-Preserving Collaboration  │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Pillar 1: Unsupervised $P(X)$ Shift Monitoring
Drawing directly upon Professor Koh’s foundational work in stream volatility-shift detection (**SEED**, *Huang, Koh, Dobbie, Pears, IEEE ICDM 2014*) and proactive drift prediction (**ProSeed**, *Chen, Koh, Riddle, PAKDD 2018*), this phase investigates continuous monitoring of incoming feature distributions $P(X_t)$ relative to model training reference baselines. By detecting when the current feature manifold diverges from the historical training distribution, the system triggers adaptation immediately at arrival time $t$, rather than waiting $\delta$ weeks for chargebacks.

### 3.2 Preliminary Empirical Benchmark ($S=5$ Seeds, 95% CIs)
To evaluate the viability of unsupervised $P(X)$ adaptation under label delay prior to doctoral admission, I designed and executed a multi-seed streaming benchmark across latency regimes $\delta \in \{0, 1, 2, 4, 8\}$ weeks on multi-phase drift streams:

| Label Delay ($\delta$) | Cadence-Matched Periodic-1w | Proposed Unsupervised $P(X)$ | Supervised (ADWIN/DDM) | Static (Frozen) |
|:---:|:---:|:---:|:---:|:---:|
| **$\delta = 0\text{w}$ (Immediate)** | $91.33\% \pm 1.47\%$ (12 retrains) | **$86.21\% \pm 7.48\%$** (**1.2 retrains**) | $81.47\% \pm 12.82\%$ (7.4 retrains) | $65.99\%$ (0 retrains) |
| **$\delta = 1\text{w}$ (7 Days)** | $82.95\% \pm 1.93\%$ (11 retrains) | **$80.52\% \pm 3.60\%$** (**1.2 retrains**) | $75.94\% \pm 7.72\%$ (7.0 retrains) | $65.99\%$ (0 retrains) |
| **$\delta = 2\text{w}$ (14 Days)** | $73.63\% \pm 1.19\%$ (10 retrains) | **$73.76\% \pm 1.07\%$** (**1.2 retrains**) | $70.72\% \pm 2.95\%$ (6.4 retrains) | $65.99\%$ (0 retrains) |
| **$\delta = 4\text{w}$ (28 Days)** | $65.79\% \pm 1.84\%$ (8 retrains) | **$65.99\% \pm 1.53\%$** (**1.0 retrains**) | $65.79\% \pm 1.84\%$ (5.6 retrains) | $65.99\%$ (0 retrains) |
| **$\delta = 8\text{w}$ (56 Days)** | $65.99\% \pm 1.53\%$ (4 retrains) | **$65.99\% \pm 1.53\%$** (**1.0 retrains**) | $65.99\% \pm 1.53\%$ (2.6 retrains) | $65.99\%$ (0 retrains) |

*Key Findings & Self-Critique:*
1. **$P(X)$ Preserves Recall with 88% Fewer Retrains:** At $\delta \in \{1\text{w}, 2\text{w}\}$, unsupervised $P(X)$ matching or outperforms supervised error-monitoring ($80.52\%$ vs. $75.94\%$ and $73.76\%$ vs. $70.72\%$), while matching blind Periodic-1w with an **88% reduction in retraining compute** (1.2 vs. 10.0 retrains).
2. **Delayed Supervised Monitoring Suffers Latency Blindness:** Supervised ADWIN/DDM collapses under delay because error signals are postponed by $\delta$ weeks, leaving the model unadapted during the attack surge.
3. **The 4-Week Horizon Bottleneck:** Beyond $\delta \ge 4\text{w}$, all architectures decay towards the static floor ($\approx 66.0\%$), demonstrating that label delay is the fundamental bottleneck in streaming fraud detection and motivating hybrid semi-supervised representations.

### 3.3 Pillar 2: Recurring Concept Profiling (CPF)
Drawing on Professor Koh’s Concept Profiling Framework (**CPF / PEARL**, *Anderson, Koh, Dobbie*), Phase 2 will construct a memory reservoir of historical fraud models. When a $P(X)$ shift matches a known historical attack vector (e.g. card-testing velocity), the system instantaneously re-activates the corresponding historical model with zero retraining pause.

### 3.4 Future Extension: Privacy-Preserving Cross-Institution Federation
In the final stage of the research, we will investigate extending the single-institution architecture to a decentralized banking consortium. Our exploratory prototype (`src/federated.py`) confirmed that naive model ensembling across non-IID banks collapses local precision (from $\sim 85\%$ to $20\%$) and leaks feature splits to semi-honest adversaries. We plan to explore true federated tree learning combining Secure Multi-Party Computation (SMPC) and Differential Privacy, directly advancing the MARS Lab’s research on secure GPU tree training (**GTree**, *Wang et al., IEEE TrustCom 2024*; and **XGT**, *IEEE TDSC 2025*).

---

## 4. Academic Alignment with Prof. Yun Sing Koh's Research

This research directly continues and deepens Professor Koh’s established contributions:
1. **Stream Volatility & Drift Detection:** The theoretical foundation rests upon Professor Koh's seminal works: **SEED** (*IEEE ICDM 2014*), **ProSeed** (*PAKDD 2018*), recurring concepts (**CPF**), and her foundational survey (*Gomes, Read, Bifet, Koh et al., ACM Computing Surveys 2019*).
2. **Secure Machine Learning:** The Phase 3 federation vector builds upon the MARS Lab’s work in secure decision trees (**GTree**, **XGT**), extending those cryptographic primitives to non-stationary financial data streams.

---

## 5. Candidate Background & Research Readiness

I bring 17+ years of hands-on software engineering, distributed systems, and ML infrastructure experience, alongside a Master of Science in Computer Science (First Class, including a supervised research thesis). To demonstrate research preparedness prior to application, I developed a 100% reproducible benchmark platform featuring **119 verified invariant tests**, automated Docker orchestration, and strict statistical rigor (Student's $t$ and block bootstrap CIs). This provides a solid operational foundation to translate theoretical stream mining concepts into robust empirical contributions.

---

## 6. Application Timeline & Next Steps

I intend to apply for the **University of Auckland Doctoral Scholarship round closing 1 November 2026** (for a 2027 start). I would be deeply grateful for the opportunity to discuss this proposed research intent with Professor Koh, receive her critical feedback, and refine the scope to align closely with the ongoing priorities of the MARS Lab.
