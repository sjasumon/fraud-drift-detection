# Completion audit of the promised enhancements — v0.4.1

The first revised ZIP (v0.4.0) was incomplete. Its 104 passing tests did not exercise
several important combinations, and some tests checked only that output keys existed.
This follow-up started from that exact delivered ZIP, reproduced defects, corrected
them and tested the installed package away from the source directory.

## What the second review found

| Defect in the delivered v0.4.0 ZIP | Reproduction or consequence | v0.4.1 correction |
|---|---|---|
| Recent window anchored to wall-clock slice | A 2-slice window with a 3-slice label delay made zero recent-model retrains despite 310 matured rows | Window ends at latest verified arrival; logs verification staleness |
| Delayed errors mixed model generations | Predictions made by an older classifier could enter detectors reset for a newer classifier | Store prediction generation; monitor only matching active generation; retain older labels for training |
| Serving artifact retained training data | Included adaptive example package contained 606 training-history rows | Serialize fitted serving state without history/retrain buffers; test prediction parity |
| Model save could overwrite an existing package | Reusing an output directory changed the artifact in place | Stage and atomically publish immutable packages; reject overwrites |
| Accepted revision did not resume on restart | Startup loaded the original configured model instead of the accepted candidate | Persist/restore an atomic active-model pointer with stable revision identity |
| Bootstrap trusted inconsistent paired labels | A changed baseline label was accepted and produced a numerical interval | Reject mismatches and missing pairs |
| Seeds resampled temporal data independently | Repeated seeds could masquerade as additional independent streams | Use shared moving-block time draws across seeds; cloned-seed invariance test |
| Study summary omitted promised dimensions | Only F1, recall and paired F1 deltas were summarized | Include precision, AP, Brier score, FP/FN counts, review budgets, retrains, seconds and run IDs |
| Legacy audit/statistical paths diverged | Different initial preprocessing, duplicate seed handling, stale summaries and fixed paths remained | Main, delayed, cohort and statistical workflows use the same pipeline |
| Zero-fraud cohorts disappeared | Population and false-alarm counts were discarded | Keep counts; mark recall/AP as undefined where appropriate |
| Incomplete identity validation | Missing card1 could reach integer conversion | Validate card1/card2/addr1 jointly; qualify entity keys as proxies |
| BAF labels cast before validation | Fractional labels could silently become integers | Validate original labels before conversion |
| Installed command/path assumptions | Delayed CLI ignored explicit arguments; latency command depended on tests; runtime paths pointed inside the package | Explicit installed CLI entrypoints and working-directory/FRAUD_WORKSPACE paths |
| SQLite lifecycle and score timing | Connections were not explicitly closed; total timing omitted persistence | Close connections and include ledger writes in total timing |
| Attribution fallback could look like SHAP | Fallback values used the same labels without a persistent method flag | Record importance_method; use diagnostic wording; do not fabricate uniform importance |
| PSI collapsed binary features into one bin | A change from 5% to 95% missingness could remain invisible | Separate low-cardinality values with midpoint boundaries; regression test |
| PaySim availability ablation discarded a valid ratio | Removing post-balances also prevented amount/old-balance ratio generation | Preserve features computable before the transaction |

The original reproduction output is in `results/validation/v0.4.0-reproduction.txt`.

## Every enhancement from the original review

| Promised enhancement | Implementation | Validation evidence | Status |
|---|---|---|---|
| Serve the actual model, preprocessing, threshold and schema | artifacts.py and api.py; readiness, auth, checksums, atomic snapshots, verified-feedback promotion | API parity/readiness/schema/checksum/restart tests; installed latency command rejects unready state | Implemented and fixture-tested |
| Fair strategy comparison | Clone one fitted initial model across static, periodic, every-slice, sliding and adaptive strategies; no warmup adaptation | First OOS predictions match; OOS rows and retrain indices checked | Implemented and synthetic-tested |
| Original-decision delayed verification | One maturity queue; original decisions and generations; deduplicated labels; same rules for all strategies | Queue and end-to-end delayed-generation regression tests; delay CLI runs | Implemented and synthetic-tested |
| Threshold tuning and calibration | Prior chronological validation, optional sigmoid calibration, F1 threshold; daily review budgets; threshold diagnostics | Model/serving parity tests and report/study artifacts | Implemented; real-data operating point pending |
| Recent-memory/age-weighted adaptation and fair costs | Configurable windows/half-lives/caps; measured seconds and staleness; shared initial-fit policy | Recent-window/lag tests and installed study over seeds/time periods | Implemented; empirical configuration selection pending |
| Better features with availability controls | Full IEEE columns, identity join, categories/email, missingness, past-only entity history, PaySim pre-transaction ablation | Identity, causal-feature, missingness and availability tests; installed feature run | Implemented; real-data feature gains pending |
| Correct statistical claims and reproducible outputs | Per-run fingerprints, OOS prediction ledger, shared-time paired bootstrap, corrected cohorts/PSI; archived old claims | Pairing/cloned-seed tests, cohort reconciliation, strict JSON, installed statistical run | Implemented; real-data conclusions pending |
| Useful final experiment report | HTML, PR/calibration/time/cost plots, metrics, review budgets, error examples, thresholds, models, manifests | Report tests and included fresh synthetic report | Implemented and synthetic-tested |

## Work that remains genuinely unperformed

- Corrected IEEE-CIS, PaySim and Feedzai empirical reruns: their raw files were absent.
  This prevents a justified claim that accuracy, F1, recall, calibration or compute improved.
- Real-data threshold/window/feature selection on development periods and confirmation
  on an untouched future period. A sweep implementation is not an experimental result.
- Production network/concurrency latency and load under background retraining.
- CUDA execution, Windows setup, Docker startup and external MySQL execution. CPU
  tests exercise the database adapter with SQLite; they do not validate those environments.

These items are **pending**, not completed by synthetic fixtures. The active manuscript
remains a protocol draft. Historical benchmark values are retained only as archived evidence.
See VALIDATION.md for exact test/command results and the scope of the clean-wheel check.
