# Concept-Drift-Aware Fraud Detection Architecture

[![CI Status](https://github.com/OWNER/fraud-drift-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/fraud-drift-detection/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-RTX%205080%20eGPU-green.svg)](https://developer.nvidia.com/cuda-zone)
[![arXiv Preprint](https://img.shields.io/badge/arXiv-Preprint%20Ready-b31b1b.svg)](arxiv_paper/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-54%20passed-brightgreen.svg)](tests/)

A reproducible, production-grade empirical research framework and preprint package developed in support of doctoral research under **Professor Yun Sing Koh** at the **MARS Lab / Centre of Machine Learning for Social Good, University of Auckland**.

This repository provides rigorous empirical validation, statistical confidence intervals, streaming multi-armed bandit extensions, and production deployment artifacts for:
> **"Adaptive Fraud Detection Under Concept Drift, Towards Privacy-Preserving Cross-Institution Extension"**

---

## ⚡ 5-Minute Stranger Quickstart

To reproduce the environment, run the 47-test verification suite, and inspect benchmarks within 5 minutes:

### Linux / macOS
```bash
git clone https://github.com/OWNER/fraud-drift-detection.git
cd fraud-drift-detection
chmod +x setup.sh
./setup.sh
```

### Windows (PowerShell)
```powershell
git clone https://github.com/OWNER/fraud-drift-detection.git
cd "fraud-drift-detection"
.\setup.bat
```

### Run Test Suite
```bash
pytest -v
# Output: ======================= 47 passed in 10.2s =======================
```

---

## 🏛️ System Architecture

The framework operates under a strict **test-before-learn** discipline, combining streaming statistical drift instrumentation with GPU-accelerated gradient boosting and lock-free model hot-swapping:

```mermaid
flowchart TD
    subgraph Ingestion["1. Temporal Stream Ingestion"]
        A["Continuous Transaction Stream<br>(IEEE-CIS 590K / PaySim 6.3M)"] --> B["Temporal Sequential Slicer<br>(Non-Overlapping Weekly Windows)"]
    end

    subgraph Preprocessing["2. Leakage-Free Preprocessing"]
        B --> C["Baseline Numerical Imputation<br>& Categorical Vocabulary Extraction"]
        C --> D["Feature-Blind UID Partitioning<br>(Disproving Memorization Cohorts)"]
    end

    subgraph Models["3. Test-Before-Learn Evaluation Engine"]
        D --> E["Active XGBoost Model<br>(CUDA hist Acceleration)"]
        D --> F["Static Baseline Model<br>(Frozen Early Window)"]
        E --> G["Streaming Inference & Error Stream"]
    end

    subgraph DriftAttribution["4. Real-Time Drift Intelligence"]
        G --> H{"Statistical Drift Monitor<br>(ADWIN & DDM on Error Stream)"}
        H -- "No Drift" --> G
        H -- "Drift Confirmed" --> I["TreeSHAP x PSI Explainer<br>(Isolate Divergent Feature Drivers)"]
        I --> J["GPU Full Retraining Trigger<br>(Accumulated Historical Window)"]
        J --> K["Lock-Free Atomic Pointer Swap<br>(Sub-10ms Serving SLA)"]
        K --> E
    end

    subgraph Latency["5. Verification Latency Buffer"]
        G --> L["DelayedFeedbackQueue<br>(30-90 Day Chargeback Arrival Lag)"]
        L --> J
    end
```

---

## 📄 arXiv Preprint Package (`arxiv_paper/`)

A camera-ready 5-page IEEE-format research manuscript is included in [`arxiv_paper/`](arxiv_paper/):
- **Title:** *Empirical Evaluation of Drift-Adaptive Gradient Boosting and Bandit Adaptation in Continuous Financial Transaction Streams*
- **Source Files:** [`arxiv_paper/main.tex`](arxiv_paper/main.tex), [`arxiv_paper/references.bib`](arxiv_paper/references.bib), and 300-DPI publication figures.
- **One-Click Archive:** [`arxiv_paper/arxiv_preprint_package.zip`](arxiv_paper/arxiv_preprint_package.zip) ready for immediate upload to arXiv or Overleaf.
- **Detailed Build Instructions:** [`arxiv_paper/README.md`](arxiv_paper/README.md).

---

## 🔬 Key Empirical Discoveries & Publication Figures

### 1. 26-Week Streaming Trajectory Under Real-World Concept Drift
Across 590,540 sequential transactions, the static baseline's efficacy degrades substantially as perpetrators shift tactics (dropping as low as 0.2418 F1 in Week 23). In contrast, the drift-adaptive model monitors prediction errors via ADWIN and DDM, triggering 22 sub-1.5s GPU full retrains on NVIDIA RTX 5080 eGPU to sustain performance:

![Streaming F1 Drift Timeline](results/figures/stream_f1_drift_timeline.png)

- **Panel A (Trajectory):** Out-of-sample evaluation (Weeks 6–26) demonstrates that while periodic and sliding-window retrains fluctuate, adaptive retraining consistently recovers from drops.
- **Panel B (Net Delta):** The adaptive model achieves sustained net F1 gains ($\Delta \text{F1} > 0$) across virtually all out-of-sample weeks.
- **Mechanistic Root Causes:** TreeSHAP $\times (1 + \text{PSI})$ attribution isolates specific drivers behind each trigger, such as $C_{13}$ count divergence in Week 6 and $C_1$ velocity shifts in Week 13.

---

### 2. Disproving Memorization via Feature-Blind UID Zero-Memory Evaluation
A central critique in financial stream mining is that adaptive models merely "memorize" newly introduced card identifiers (`card1-card6`, `addr1-addr2`). 
To address this, we implemented a **Feature-Blind Protocol** stripping all entity identifiers and tracked **Cohort 3 (strict stream-unseen cards)** upon their first arrival in the stream across 5 random seeds:

| Evaluation Metric | Static Baseline (Frozen) | Drift-Adaptive (Full GPU Retrain) | Delta ($\Delta$) | Statistical Significance |
|---|:---:|:---:|:---:|:---:|
| **Cohort 3 Recall** | $69.41\% \pm 1.07\%$ | **$76.62\% \pm 0.35\%$** | **$+7.22\% \pm 1.31\%$** | **$t = 12.27, p = 0.00025$ ($p < 0.001$)** |
| **Cohort 3 AUC-PR** | $0.4856 \pm 0.0022$ | **$0.5406 \pm 0.0032$** | **$+0.0550 \pm 0.0049$** | **$t = 24.87, p = 7.15 \times 10^{-6}$** |
| **Cohort 1 Recall** | $48.90\% \pm 0.48\%$ | **$61.41\% \pm 0.50\%$** | **$+12.51\% \pm 0.34\%$** | $p < 0.0001$ |

> **Conclusion:** The $+7.22\%$ recall improvement on transactions from entities never previously observed confirms that adaptive retraining learns generalized structural drift (e.g. shifts in transaction velocity, amounts, and cross-feature distributions), conclusively refuting the memorization hypothesis.

---

### 3. Empirical Verification-Latency Degradation Curve
In real-world banking streams, ground truth chargeback and dispute confirmations arrive with a 30-to-90 day latency. Modeling this verification lag demonstrates how delayed labels erode adaptive advantages towards the static baseline:

![Verification Latency Degradation](results/figures/verification_latency_degradation.png)

- **Panel A (Degradation Horizon):** While an immediate oracle achieves $72.70\%$ recall, standard 4-week chargeback delays degrade recall to $65.94\%$ ($-6.76\%$ loss). Under extended 8-week delays, recall drops to $63.45\%$, asymptotically approaching the static baseline floor ($63.27\%$).
- **Panel B (Operational Pareto Frontier):** Modeling realistic stochastic latency via LogNormal distribution ($\mu=1.38, \sigma=0.45$) achieves **$66.37\%$ recall (11,058 frauds intercepted)** with **only 12 GPU retrains**—delivering $29\%$ compute savings over fixed 4-week retraining while capturing $+488$ additional frauds over the 8-week regime.

---

### 4. Multi-Seed Statistical Rigor & Saturated Feature Discovery
Single-run comparisons on synthetic PaySim streams ($0.9955 \to 0.9978$ F1) were re-evaluated across 5 random seeds (`[42, 101, 202, 303, 404]`):

| Model Regime | Mean F1-Score | Std Dev ($\sigma$) | 95% Bootstrap CI | Paired $t$-test vs. Static |
|---|:---:|:---:|:---:|:---:|
| **Static Baseline** | 0.9958 | $\pm 0.0002$ | $[0.9944, 0.9970]$ | — |
| **Periodic Retrain (4-Wk)** | 0.9966 | $\pm 0.0003$ | $[0.9947, 0.9974]$ | $t = 5.74, p = 0.0046$ |
| **Sliding Window (8-Wk)** | **0.9967** | $\pm 0.0003$ | $[0.9948, 0.9975]$ | $t = 5.86, p = 0.0042$ |
| **Drift-Adaptive (ADWIN)** | 0.9964 | $\pm 0.0009$ | $[0.9943, 0.9970]$ | **$t = 1.23, p = 0.2871$ (n.s.)** |

> **Finding:** On PaySim, the observed single-run difference between static and adaptive models was stochastic noise. Feature attribution (SHAP) reveals that PaySim's engineered accounting balance feature (`errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg`) provides near-deterministic separation, saturating all models at $>0.995$ F1. Honest statistical reporting prevents overclaiming on saturated synthetic benchmarks.

![Statistical Rigor Summary](results/figures/statistical_rigor_errorbars.png)

---

### 5. Non-Stationary Multi-Armed Bandit Adaptation (ADWIN-UCB)
To evaluate online exploration-exploitation under non-stationary fraud environments (e.g., dynamic risk routing or rule allocation), we designed a 4-phase non-stationary bandit environment (10,000 steps per seed, 5 seeds) comparing standard UCB1, Sliding-Window UCB (M-UCB), Discounted UCB, and our proposed **ADWIN-UCB**:

| Agent Algorithm | Cumulative Regret ($T=10,000$) | 95% Confidence Interval | Optimal Action % | Paired Regret Reduction vs. ADWIN |
|---|:---:|:---:|:---:|:---:|
| **Standard UCB1** | $510.33 \pm 53.98$ | $[443.32, 577.34]$ | $91.49\% \pm 0.87\%$ | $-50.1\%$ ($p = 0.000257$) |
| **Sliding-Window UCB** ($W=1000, c=0.6$) | $364.46 \pm 25.32$ | $[333.02, 395.90]$ | $93.83\% \pm 0.36\%$ | **$-30.1\%$ ($p = 0.00164$)** |
| **Discounted UCB** ($\gamma=0.999, c=0.5$) | $323.27 \pm 13.03$ | $[307.09, 339.45]$ | $94.60\% \pm 0.25\%$ | **$-21.2\%$ ($p = 0.0142$)** |
| **ADWIN-UCB (Ours)** | **$254.63 \pm 29.79$** | **$[217.65, 291.61]$** | **$95.65\% \pm 0.46\%$** | **Baseline Winner** |

![Bandit Cumulative Regret](results/figures/bandit_cumulative_regret.png)

> **Key Takeaway:** When properly tuned to the environment's drift scale, non-stationary baselines (Sliding-Window UCB and Discounted UCB) successfully adapt to distribution shifts and outperform stationary UCB1 (regret $364.46$ and $323.27$ vs. $510.33$). However, ADWIN-UCB achieves the lowest cumulative regret ($254.63$, a further $21.2\%$ to $30.1\%$ reduction) because it dynamically maintains full historical memory during stationary regimes (retaining $O(\log t)$ efficiency) and resets an arm's statistical state only when Hoeffding bounds verify genuine concept drift.

---

## 🛠️ CLI Reproduction Commands

All experiments can be executed via console script entrypoints or standard module invocations:

### 1. Generate Key Publication Figures
Re-renders 300-DPI publication figures for streaming trajectories and verification latency curves:
```bash
python -m src.generate_key_figures
```
*Outputs saved to [`results/figures/stream_f1_drift_timeline.png`](results/figures/stream_f1_drift_timeline.png) and [`results/figures/verification_latency_degradation.png`](results/figures/verification_latency_degradation.png).*

### 2. Statistical Rigor Multi-Seed Benchmark
Executes 5-seed evaluation on PaySim and IEEE-CIS Zero-Memory UID cohorts, computes paired hypothesis tests ($t$-test and Wilcoxon), and generates publication error bar plots:
```bash
fraud-stats-rigor
# Or:
python -m src.benchmark_statistical_rigor --seeds 42 101 202 303 404
```
*Outputs saved to [`results/metrics/statistical_rigor_summary.json`](results/metrics/statistical_rigor_summary.json) and [`results/figures/statistical_rigor_errorbars.png`](results/figures/statistical_rigor_errorbars.png).*

### 3. Non-Stationary Bandit Benchmark
Runs 10,000 steps across 5 seeds comparing Standard UCB, Sliding-Window UCB, Discounted UCB, and ADWIN-UCB:
```bash
fraud-bandit
# Or:
python -m src.bandit.benchmark --steps 10000 --seeds 42 101 202 303 404
```
*Outputs saved to [`results/metrics/bandit_benchmark_summary.json`](results/metrics/bandit_benchmark_summary.json) and [`results/figures/bandit_cumulative_regret.png`](results/figures/bandit_cumulative_regret.png).*

### 4. Full IEEE-CIS Stream Pipeline (26 Weeks)
Executes end-to-end drift detection and GPU-accelerated XGBoost retraining on 590,540 real transactions:
```bash
fraud-pipeline --dataset ieee/train_transaction.csv
# Or:
python -m src.pipeline --dataset ieee/train_transaction.csv
```

### 5. Production Microservice API
Starts the FastAPI microservice with atomic double-buffering and live ADWIN/DDM drift monitoring:
```bash
fraud-api --port 8000
# Or:
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
* **Interactive OpenAPI Docs:** Available at `http://localhost:8000/docs`.
* **Authentication:** Protected endpoints (`/v1/score`, `/v1/batch-score`, `/v1/feedback`, `/v1/drift-status`) require the `X-API-Key` header (`dev-secret-key` by default, configurable in `.env`). `/health` remains public for container liveness/readiness probes.
* **Concurrency & Multi-Worker Architecture:** In this prototype, `model_buffer`, `drift_monitor`, and `recent_predictions` are thread-safe in-process singletons suited for single-worker or edge deployments. When horizontally scaling across multiple Uvicorn/Gunicorn worker processes (`--workers N`), each worker has an independent memory space; in high-throughput distributed production, prediction caches should be backed by Redis and feedback events aggregated via a shared Kafka/Redis stream.
* **Label-Poisoning Attack Surface:** The `/v1/feedback` stream directly updates ADWIN and DDM drift statistics. Without authentication, adversaries could inject falsified labels to prematurely force retraining or conceal genuine concept drift. API key enforcement closes this attack surface, and designing Byzantine-resilient drift detectors against label poisoning represents a compelling future research vector.

---

## 🐳 Docker Deployment

The application includes a multi-stage Dockerfile supporting NVIDIA GPU acceleration and health probes:

```bash
# Build the Docker image
docker build -t fraud-drift-microservice .

# Run with GPU passthrough
docker run --gpus all -p 8000:8000 fraud-drift-microservice

# Or run complete stack with MySQL using Docker Compose
docker compose up -d
```

---

## 🏛️ Repository Layout

```
├── .github/workflows/ci.yml         <- GitHub Actions continuous integration workflow
├── arxiv_paper/                     <- arXiv camera-ready paper package
│   ├── main.tex                     <- IEEE-format camera-ready manuscript (5 pages)
│   ├── references.bib               <- Authentic academic bibliography
│   ├── arxiv_preprint_package.zip   <- 1-click submission package for arXiv/Overleaf
│   └── README.md                    <- LaTeX compilation & arXiv submission instructions
├── src/
│   ├── bandit/                      <- Non-stationary Multi-Armed Bandit suite
│   │   ├── agents.py                <- StandardUCB, SlidingWindowUCB, DiscountedUCB, AdwinUCB
│   │   ├── environment.py           <- 4-phase non-stationary Bernoulli bandit environment
│   │   └── benchmark.py             <- Multi-seed benchmark runner & regret plotter
│   ├── benchmark_statistical_rigor.py <- 5-seed statistical confidence runner
│   ├── generate_key_figures.py      <- Publication figure generator (timeline & latency)
│   ├── api.py                       <- FastAPI microservice with /predict & /drift/status
│   ├── drift_detector.py            <- Streaming ADWIN & DDM detectors (River)
│   ├── adaptive_model.py            <- GPU-accelerated drift-triggered retraining
│   ├── static_model.py              <- Frozen static baseline model
│   ├── baselines.py                 <- Periodic & Sliding-Window retrain models
│   ├── delayed_feedback.py          <- Chargeback verification latency simulator
│   ├── drift_explainer.py           <- TreeSHAP drift attribution
│   ├── preprocessing.py             <- Temporal feature engineering & UID extraction
│   └── db.py                        <- MySQL SQLAlchemy persistence layer
├── tests/                           <- Comprehensive test suite (47 tests)
│   ├── test_statistical_rigor.py    <- Tests for bootstrap CIs and paired hypothesis testing
│   ├── test_bandit.py               <- Tests for bandit agents and environment transitions
│   ├── test_api.py                  <- Microservice endpoint validation
│   ├── test_delayed_feedback.py     <- Verification lag queue invariants
│   ├── test_drift_explainer.py      <- SHAP attribution verification
│   └── test_leakage_invariants.py   <- Strict temporal sequence integrity checks
├── pyproject.toml                   <- Modern PEP 517/621 package configuration
├── requirements.txt                 <- Exact pinned dependencies for 100% reproducibility
├── setup.sh                         <- One-command setup for Linux / macOS
├── setup.bat                        <- One-command setup for Windows
├── Dockerfile                       <- Multi-stage container build with CUDA support
└── LICENSE                          <- MIT Open-Source License
```

---

## 🎓 Academic Alignment: Prof. Yun Sing Koh

This research directly aligns with the research agenda of **Prof. Yun Sing Koh** (MARS Lab / Centre of Machine Learning for Social Good, University of Auckland):
1. **Adaptive Stream Mining & Concept Drift:** Extends foundational online algorithms (ADWIN, DDM) to massive, non-stationary financial transaction streams.
2. **GPU Stream Acceleration:** Implements sub-1.5s model retraining on 500k+ transactions using CUDA histogram gradient boosting, echoing recent MARS lab publications (*GTree: GPU-Friendly Privacy-Preserving Decision Tree Training and Inference*, Wang et al., IEEE TrustCom 2024; and *XGT*, IEEE TDSC 2025).
3. **Phased Research Trajectory:** Forms the empirical baseline for **Phase 2** (extending adaptive models to privacy-preserving federated settings across financial institutions).

---

## 📜 Citation

If you use this codebase or benchmark suite in your research, please cite:

```bibtex
@article{sumon2026adaptive,
  title={Adaptive Fraud Detection Under Concept Drift: An Empirical Study on Verification Latency, Entity Generalization, and Statistical Rigor},
  author={Sumon, Sheikh Joynal Abedin},
  journal={arXiv preprint arXiv:2609.xxxxx},
  year={2026}
}
```

---

## ⚖️ License
This project is open-source under the [MIT License](LICENSE).
