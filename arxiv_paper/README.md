# Research Manuscript Package: Concept-Drift-Aware Fraud Detection

This package contains the working draft LaTeX source, bibliography, and figures for the preprint manuscript:
**"Adaptive Fraud Detection Under Concept Drift: An Empirical Study on Verification Latency, Entity Generalization, and Statistical Rigor"**

**Author:** Sheikh Joynal Abedin Sumon  
**Affiliation:** Independent Researcher  
**Target Submission:** arXiv (Computer Science > Machine Learning / Artificial Intelligence)

---

## Package Directory Structure

```
arxiv_paper/
├── main.tex                    # Master LaTeX manuscript (IEEE conference/compsoc format)
├── references.bib              # Complete authentic BibTeX citations
├── README.md                   # Build and submission instructions
└── figures/
    ├── statistical_rigor_errorbars.png  # Multi-seed error bars & significance plot (300 DPI)
    ├── delayed_feedback_impact.png      # Verification latency degradation curve
    ├── drift_attribution_week_06.png    # TreeSHAP x PSI waterfall attribution plot
    └── model_comparison_f1_aucpr.png    # Longitudinal F1 & AUC-PR comparison
```

---

## How to Build and Compile

### Option A: Overleaf (Fastest & Recommended)
1. Zip the `arxiv_paper` folder contents (`main.tex`, `references.bib`, and `figures/`).
2. Go to [Overleaf](https://www.overleaf.com), click **New Project** $\to$ **Upload Project**, and select the `.zip` file.
3. Select **pdfLaTeX** compiler in project settings and click **Recompile**.

### Option B: Local Compilation via Command Line (TeX Live / MiKTeX)
```bash
# Run pdflatex and bibtex sequence:
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option C: Upload to arXiv
1. Ensure all files (`main.tex`, `references.bib`, and figures) are packaged into a single `.tar.gz` or `.zip` file.
2. In the arXiv submission portal, upload the archive. arXiv automatically runs TeX Live with `IEEEtran` and compiles the PDF.

---

## Key Experimental Findings Documented in Paper

1. **Empirical Evidence Against Direct Categorical Memorization:**
   - On IEEE-CIS Strict Stream-Unseen Cardholders (Cohort 3, clean complete cases), under direct categorical identifier ablation (raw card/address IDs stripped), the drift-adaptive model achieves **$76.62\% \pm 0.35\%$ recall** vs. **$69.41\% \pm 1.07\%$** for the static baseline (**$+7.21\% \pm 1.31\%$ gain**, $t = 12.27$, $p = 2.54 \times 10^{-4}$, two-sided Wilcoxon $W=0.0, p = 0.0625$, bootstrap 95\% CI: $[+6.28\%, +8.29\%]$ and Student's $t$ 95\% CI: $[+5.59\%, +8.85\%]$). We present this as evidence against naive categorical memorization rather than proof, highlighting that processor-engineered $C, D, V$ features retain indirect entity signals and proxy UIDs (`card1_card2_addr1`) motivate entity-disjoint group holdouts.
2. **Deconstruction of Single-Run Micro-Gains on Saturated Streams:**
   - Multi-seed replication on PaySim ($K=5$) proves that the initial single-run difference ($0.9955 \to 0.9978$) is within random variance ($p = 0.2871$), demonstrating the imperative for statistical rigor.
3. **Quantified Cost of Verification Latency:**
   - In exploratory single-seed simulations, an 8-week dispute resolution lag degrades out-of-sample recall from $72.70\%$ down to $63.45\%$ (approaching the frozen static baseline of $63.27\%$), theoretically motivating future research into unsupervised feature-space monitoring while labels mature.
4. **Sub-15ms Real-Time Serving SLA Demonstration:**
   - In-process benchmark throughput of $122.9$ QPS with $p95$ latency of $9.88$ ms and $p99$ of $13.38$ ms (< 15 ms SLA) demonstrating zero-downtime double-buffering and API serialization overhead.
