# Validation — reaudited v0.4.1

Completed 2026-10-16 with Python 3.12.14 and CPU execution.

## Tests and checks actually completed

- **122 tests passed; 0 failed; 0 errors; 0 skipped**, approximately 24.30 seconds.
- Static checks for unused imports, undefined names and syntax errors: passed.
- Built a wheel and installed it into a separate virtual environment.
- Verified every Python source module in that wheel matches the delivered source bytes.
- Executed the wheel from a separate working directory with PYTHONPATH removed;
  confirmed imports came from the installed package, not the development source.
- Checked help/argument parsing for all eight installed entrypoints.
- Ran six end-to-end synthetic commands: demo; identity/history feature pipeline;
  standard and feature-blind cohort audit; delayed-label comparison; study over two
  seeds and two non-overlapping future periods; and multi-seed statistical reporting.
- Verified the latency CLI refuses to measure an unready service as successful inference.
- Preserved the command arguments, return codes, logs and JUnit results in
  `results/validation/`. Included a fresh synthetic HTML report and study example.

The separate wheel environment reused available dependency installations. This is
an installed-package/source-isolation check, not a fresh operating-system or
Windows installation test. No production latency SLA follows from it.

## What the new regression tests cover

The original delivery had 104 tests. The revised suite adds targeted cases for
training-data-free exports, immutable packages, shared temporal bootstrap draws,
inconsistent labels, delayed-generation pairing, recent verified memory,
unchanged-data budgets, complete study outputs, zero-fraud counts, fractional
labels, incomplete identity fields, restart restoration, input validation,
finite statistical JSON, explicit attribution fallbacks, binary PSI shifts and
preservation of pre-transaction ratio features.

One earlier integration fixture represented only 72 hours and produced no OOS
slice under weekly partitioning. It now spans actual weekly slices and asserts
nonempty OOS evaluation. The suite count alone is not the completion criterion;
TASK_COMPLETION_AUDIT.md maps each promised enhancement to code and behavior.

Four warnings remain: Starlette TestClient/httpx deprecation, a small bandit-plot
layout warning, and SciPy precision-loss warnings for two intentionally constant
paired-difference fixtures. Their serialized statistics contain no Infinity/NaN.

## Limits and pending empirical work

The raw IEEE-CIS, PaySim and Feedzai datasets were absent from the upload. Corrected
real-data benchmark runs, threshold/window selection, and held-out confirmation
remain pending. Synthetic outputs demonstrate execution and invariants, not
real-world model improvements. CUDA, Windows setup, Docker startup, external
MySQL and production network/concurrent load were not exercised. SQLite covers
the database adapter in CPU tests.

Historical v0.3 artifacts and the first v0.4.0 release notes/test evidence remain
clearly archived. The active manuscript remains a protocol draft pending new data.
