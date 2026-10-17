# Prototype implementation report — v0.4.1

This project studies fraud detection under chronological distribution change and
verification latency. It compares static, periodic, every-slice, sliding-window,
and drift-triggered XGBoost using a common initial estimator and validation policy.

The revised implementation adds decision threshold tuning, calibration, realistic
original-decision feedback pairing, recent-memory experiments, feature availability
ablations, and traceable OOS reporting. These changes strengthen the experiment;
they do not establish a new empirical improvement until the real datasets are rerun.

Current evidence and test scope: [VALIDATION.md](VALIDATION.md).
Implemented changes: [CHANGELOG.md](CHANGELOG.md).
Research corrections: [REVIEW_CORRECTIONS.md](REVIEW_CORRECTIONS.md).
Reproduction commands and limitations: [README.md](README.md).

Historical results and the original version of this document are preserved under
`docs/archive_v0.3/` and `results/legacy_v0.3/`. They should not be quoted as v0.4
results. The research direction remains an evaluation of adaptation benefits under
matched compute, delayed labels and entity generalization, followed by a possible
privacy-preserving cross-institution extension.
