# Release validation — v0.4.0

Validation completed on 2026-10-16 using Python 3.12.14 and CPU execution.

- Full suite: **104 passed, 0 failed, 0 errors**, in 16.00 seconds.
- Evidence: `results/validation/pytest-results.xml` and `pytest-output.txt`.
- Dependency resolution: the complete editable installation with test extras resolved
  successfully in a pip dry run; see `dependency-resolution.txt`.
- End-to-end synthetic example: generated 8 weeks of 300 transactions/week, executed
  all five strategies, saved model packages, generated the prediction ledger,
  confidence intervals, budget metrics, error examples and report.
- The included fresh example uses the release source fingerprint recorded in its
  manifest. Its model packages remain explicitly synthetic and cannot be served.
- Additional synthetic study smoke check completed across 2 seeds, 2 disjoint
  evaluation periods and 2 adaptive windows with a retrain cap and delayed labels.
  Exploratory study outputs are not distributed as real-data results.

Regression coverage includes shared initial predictions; absence of warmup
retrains; delayed prediction preservation; seed deduplication; constant-feature
PSI; causal historical features; PaySim feature availability; first-arrival cohorts;
identity joins; model package parity/checksums; API schema/authentication/readiness;
feedback idempotency; revision handling across restart; and background promotion
on a separate holdout.

Three warnings remained in the full test run: an upstream Starlette TestClient
httpx deprecation, a tight-layout warning in a deliberately small bandit chart,
and SciPy precision loss for the constant-difference statistical test fixture.
No real-data accuracy or production latency conclusion is drawn from these tests.

## Scope that requires your data or environment

IEEE-CIS, PaySim and Feedzai raw benchmark datasets were not included in the upload.
Their corrected real-data experiments have **not** been rerun. The old scores remain
under `results/legacy_v0.3/` and must not be described as v0.4 performance. CUDA,
Windows installation, Docker startup, external MySQL and concurrent production
load were not exercised here. CPU tests cover the database adapter with SQLite.

Follow README.md to install and run the real-data experiments. The historical
manuscript and reports are preserved under `docs/archive_v0.3/`; the active
manuscript is a protocol draft pending new empirical results.
