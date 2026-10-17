# Engineering and research plan — v0.4.1

Implemented changes and their migration implications are in CHANGELOG.md. This
plan lists the remaining empirical work, which requires the original datasets.

1. Reproduce IEEE-CIS with fixed preprocessing, shared estimator, matched initial
   state and OOS-only decisions. Inspect manifest and prediction ledger.
2. Compare all-history and 2/4/8-slice adaptive windows, optional age weighting,
   and fixed retrain caps. Use the same seeds/time periods for every strategy.
3. Evaluate fixed 0/4/8-slice and stochastic label delays. Preserve original
   predictions and record matured/pending label counts.
4. Compare precision/recall under review budgets; choose an operational objective
   on development periods. Confirm its threshold/policy on a later reserved period.
5. Run identity/history/missingness feature ablations. Run PaySim with and without
   post-transaction fields. Treat data availability as part of the task definition.
6. Check first-arrival cohorts, temporal dependence, calibration, and subgroup errors.
7. Benchmark the real serving package under representative concurrent traffic,
   including persistence and background retraining, before stating a latency SLA.
8. Refresh the manuscript only from recorded v0.4 run IDs, hashes and intervals.

Distributed serving, selective-label bias, missing labels, privacy-preserving
cross-institution learning, and a fraud-integrated bandit policy remain research
extensions. They are not implemented or claimed by this release.
