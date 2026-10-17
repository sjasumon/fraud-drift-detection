# Corrections to the previous research claims

1. v0.3 static and adaptive models had different tree/subsampling settings, and
   adaptive warmup could overwrite the initial model. v0.4 uses the same initial
   fitted model and OOS-only monitoring. Previous deltas require rerunning.
2. Delayed-label monitoring previously rescored old transactions with the latest
   model and could duplicate initial training rows. Original predictions are now
   retained and training IDs deduplicated.
3. The archived IEEE first-arrival Wilcoxon p-value is 0.0625, not <0.05. A paired
   t-test and a Wilcoxon test are separate results. No new significance is claimed.
4. Five seeds reflect algorithmic variability on one dataset, not five independent
   populations. The primary study now resamples seeds and temporal blocks and
   reports uncertainty for paired F1 differences. Short streams remain exploratory.
5. First-arrival sets now update after each transaction. Missing-ID transactions
   remain a separately qualified proxy cohort; they cannot establish entity novelty.
6. PaySim accounting/post-balance features may be unavailable at decision time.
   `--pre-transaction-only` provides an explicit availability ablation.
7. PSI detects a marginal distribution change, not necessarily a change in P(y|x).
   The constant-feature blind spot is fixed. Attribution uses the pre-update model
   and does not prove the cause of an alert. The >90% concordance claim is withdrawn
   pending a separately recorded validation experiment.
8. The serving API previously bootstrapped random examples. v0.4 starts unready,
   loads an explicit model package, and does not promote synthetic training data.
9. The old latency script omitted authentication/status checks. Old sub-15ms SLA
   figures are not validated by v0.4. The revised script requires successful scores
   and identifies its in-process measurement scope.
10. Retrain count is not GPU compute cost. Reports include measured training seconds.
    No constant speed, GPU/CPU equivalence or accuracy improvement is promised.
11. Current figures use an explicit run directory. Legacy plots/logs are not inputs
    to current report generation. No smooth asymptotic latency curve is inferred
    from three observations.

The current work is an improved implementation and evaluation protocol. Real-data
accuracy, generalization, latency and publication claims require new experiments.

## Follow-up audit of v0.4.0

The first revised ZIP still had implementation gaps. They are documented in
TASK_COMPLETION_AUDIT.md and corrected in v0.4.1. In particular, completed code
and passing synthetic tests do not establish real-data performance improvements.
