"""
Multi-Seed Statistical Rigor & Bootstrap Confidence Interval Benchmarking Suite.

Addresses:
1. PaySim Mobile Money Out-of-Sample Stream F1 Rigor (0.9955 -> 0.9978):
   Evaluates Static, Periodic, Sliding, and Drift-Adaptive across K random seeds.
2. IEEE-CIS Zero-Memory UID Generalization Recall Rigor (67.82% -> 77.09%):
   Evaluates Static vs Drift-Adaptive across K random seeds on Strict Stream-Unseen Cards (Cohort 3).
3. For the unified five-strategy and chronological-window benchmark, use src.study.

Calculates:
- Mean and Standard Deviation (mean ± std)
- 95% Student's t Confidence Interval
- 1,000-sample Non-Parametric Bootstrap 95% Confidence Interval [CI_low, CI_high]
- Paired Student's t-test (t-stat, p-value) and Wilcoxon Signed-Rank test
- Publication-quality error-bar visualization with significance annotations
"""

import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import METRICS_DIR, FIGURES_DIR, USE_GPU, GPU_DEVICE, RANDOM_SEED
from src.ingestion import DatasetIngestion
from src.preprocessing import Preprocessor
from src.evaluate_uid_split import run_single_audit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fraud_drift.statistical_rigor")

DEFAULT_SEEDS = [42, 101, 202, 303, 404]


def bootstrap_all_seed_f1(predictions, name, n_bootstrap):
    from src.reporting import block_bootstrap_delta

    model = predictions[predictions.model == name].copy()
    zero = model.copy()
    zero["model"] = "zero"
    zero["prediction"] = 0
    result = block_bootstrap_delta(
        pd.concat([model, zero]), name, baseline="zero", n_bootstrap=n_bootstrap
    )
    return result["ci_95"]


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    metric_type: str = "recall",
    random_seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """
    Legacy IID utility for independent observations only. Streaming reports use seed/time-block bootstrap.
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    boot_scores = []

    # Stratified or direct resampling
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) == 0:
        return 0.0, 0.0

    for _ in range(n_bootstraps):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        b_idx = np.concatenate([b_pos, b_neg])

        yb = y_true[b_idx]
        pb = y_pred[b_idx]

        tp = int(((yb == 1) & (pb == 1)).sum())
        fp = int(((yb == 0) & (pb == 1)).sum())
        fn = int(((yb == 1) & (pb == 0)).sum())

        if metric_type == "recall":
            denom = tp + fn
            score = (tp / denom) * 100 if denom > 0 else 0.0
        elif metric_type == "precision":
            denom = tp + fp
            score = (tp / denom) * 100 if denom > 0 else 0.0
        elif metric_type == "f1":
            p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            score = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        else:
            score = 0.0
        boot_scores.append(score)

    low = float(np.percentile(boot_scores, 100 * (alpha / 2)))
    high = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return round(low, 4), round(high, 4)


def compute_sample_stats(values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Computes sample mean, standard deviation, and Student's t 95% confidence interval.
    """
    arr = np.array(values, dtype=float)
    k = len(arr)
    mean_val = float(np.mean(arr))
    if k > 1:
        std_val = float(np.std(arr, ddof=1))
        se = std_val / np.sqrt(k)
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=k - 1))
        ci_low = mean_val - t_crit * se
        ci_high = mean_val + t_crit * se
    else:
        std_val = 0.0
        ci_low = mean_val
        ci_high = mean_val

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "n_seeds": k,
    }


def adjust_p_values_fdr_bh(p_values: List[float]) -> List[float]:
    """
    Benjamini-Hochberg False Discovery Rate (FDR) multiple-comparisons correction.
    Controls the expected proportion of false positive discoveries across family of tests.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n <= 1:
        return [float(x) for x in p]
    order = np.argsort(p)
    ranked_p = p[order]
    factors = np.arange(1, n + 1) / float(n)
    adjusted = ranked_p / factors
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    rev_order = np.empty(n, dtype=int)
    rev_order[order] = np.arange(n)
    return [float(f"{x:.6e}") for x in adjusted[rev_order]]


def adjust_p_values_bonferroni(p_values: List[float]) -> List[float]:
    """
    Bonferroni multiple-comparisons correction.
    Controls Family-Wise Error Rate (FWER) by adjusting p_adj = min(1.0, p * m).
    """
    m = len(p_values)
    return [float(f"{min(1.0, float(p * m)):.6e}") for p in p_values]


def compute_paired_tests(
    a_values: List[float], b_values: List[float]
) -> Dict[str, Any]:
    """
    Runs paired Student's t-test and Wilcoxon signed-rank test.
    Computes effect size (Cohen's d_z) and small-sample bias-corrected Hedges' g.
    Provides explicit documentation of sample size limitations (n=5 seeds).
    """
    a = np.array(a_values, dtype=float)
    b = np.array(b_values, dtype=float)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or not len(a)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError(
            "Paired samples must be finite, nonempty, one-dimensional and equal length"
        )
    diff = a - b
    n = len(diff)

    if np.all(diff == 0) or n < 2:
        return {
            "mean_diff": float(np.mean(diff)),
            "std_diff": 0.0,
            "t_stat": 0.0,
            "p_value_t": 1.0,
            "cohens_d": 0.0,
            "hedges_g": 0.0,
            "effect_size": "negligible",
            "wilcoxon_stat": 0.0,
            "p_value_wilcoxon": 1.0,
            "significant_05": False,
            "significant_01": False,
            "significant_001": False,
            "n_seeds": n,
            "test_status": "insufficient_pairs" if n < 2 else "identical_pairs",
            "sample_size_note": f"n={n}: insufficient sample size"
            if n < 2
            else f"n={n}: paired differences are all zero",
        }

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    # Paired Cohen's d (d_z = mean_diff / std_diff)
    if std_diff > 1e-12:
        cohens_d = round(mean_diff / std_diff, 4)
        # Hedges' g small-sample bias correction factor (n < 20)
        j_factor = 1.0 - (3.0 / (4 * n - 5)) if n > 1 else 1.0
        hedges_g = round(cohens_d * j_factor, 4)
    else:
        cohens_d = float("inf") if mean_diff > 0 else float("-inf")
        hedges_g = cohens_d

    abs_d = abs(cohens_d) if not np.isinf(cohens_d) else float("inf")
    if abs_d < 0.2:
        effect_desc = "negligible"
    elif abs_d < 0.5:
        effect_desc = "small"
    elif abs_d < 0.8:
        effect_desc = "medium"
    elif abs_d < 1.2:
        effect_desc = "large"
    else:
        effect_desc = "very large (huge)"

    t_res = stats.ttest_rel(a, b)
    try:
        w_res = stats.wilcoxon(a, b)
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")

    p_t = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 1.0
    t_val = float(t_res.statistic)
    t_stat_rounded = round(t_val, 4) if not np.isinf(t_val) else float("inf")

    return {
        "mean_diff": round(mean_diff, 4),
        "std_diff": round(std_diff, 4),
        "t_stat": t_stat_rounded if np.isfinite(t_stat_rounded) else None,
        "p_value_t": float(f"{p_t:.6e}"),
        "cohens_d": cohens_d if np.isfinite(cohens_d) else None,
        "hedges_g": hedges_g if np.isfinite(hedges_g) else None,
        "effect_size": effect_desc,
        "wilcoxon_stat": round(w_stat, 4) if not np.isnan(w_stat) else None,
        "p_value_wilcoxon": float(f"{w_p:.6e}") if not np.isnan(w_p) else None,
        "significant_05": bool(p_t < 0.05),
        "significant_01": bool(p_t < 0.01),
        "significant_001": bool(p_t < 0.001),
        "n_seeds": n,
        "sample_size_note": (
            f"n={n} seeds measures algorithmic variation on this dataset, not independent population replication. "
            "Treat tests as exploratory; inspect Wilcoxon separately and repeat across chronological windows."
        ),
    }


# =========================================================================
# BENCHMARK 1: PAYSIM MOBILE MONEY MULTI-SEED RIGOR
# =========================================================================
def run_paysim_statistical_rigor(
    seeds, n_bootstrap=1000, dataset_path=None, output_dir=None, data_kind="unverified"
):
    from src.pipeline import run_pipeline

    path = Path(dataset_path or "data/paysim/paysim_full.csv")
    ledgers = []
    values = {m: [] for m in ["static", "periodic", "weekly", "sliding", "adaptive"]}
    run_ids = []
    for seed in seeds:
        run = run_pipeline(path, seed=seed, output_dir=output_dir, data_kind=data_kind)
        ledgers.append(
            pd.read_csv(run["run_dir"] / "predictions.csv", dtype={"trans_id": str})
        )
        run_ids.append(str(run["run_dir"]))
        for model in values:
            m = run["overall_summary"]["out_of_sample_stream"][model + "_model"]
            values[model].append(m)
    ledger = pd.concat(ledgers, ignore_index=True)
    summary = {"models": {}, "paired_tests": {}}
    for model, runs in values.items():
        summary["models"][model] = {
            metric: compute_sample_stats([r[key] for r in runs])
            for metric, key in [
                ("f1", "overall_f1"),
                ("recall", "overall_recall"),
                ("precision", "overall_precision"),
                ("auc_pr", "overall_auc_pr"),
            ]
        }
        summary["models"][model]["bootstrap_f1_ci"] = bootstrap_all_seed_f1(
            ledger, model, n_bootstrap
        )
        summary["models"][model]["raw_seed_f1"] = [r["overall_f1"] for r in runs]
    for baseline in ["static", "periodic", "weekly", "sliding"]:
        summary["paired_tests"]["adaptive_vs_" + baseline + "_f1"] = (
            compute_paired_tests(
                summary["models"]["adaptive"]["raw_seed_f1"],
                summary["models"][baseline]["raw_seed_f1"],
            )
        )
    tests = list(summary["paired_tests"].values())
    adjusted = adjust_p_values_fdr_bh([r["p_value_t"] for r in tests])
    for test, pvalue in zip(tests, adjusted):
        test["p_value_t_fdr_bh"] = pvalue
    return {
        "summary": summary,
        "run_directories": run_ids,
        "protocol": "shared chronological pipeline; recall and precision on 0–1 scale",
    }


# =========================================================================
# BENCHMARK 2: IEEE-CIS ZERO-MEMORY UID GENERALIZATION MULTI-SEED RIGOR
# =========================================================================
def run_ieee_uid_statistical_rigor(
    seeds: List[int],
    n_bootstrap: int = 1000,
    dataset_path=None,
    output_dir=None,
    data_kind="unverified",
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(
        f"STARTING IEEE-CIS ZERO-MEMORY UID GENERALIZATION BENCHMARK (Seeds: {seeds})"
    )
    logger.info(
        "Target: estimate first-arrival recall differences across supplied seeds"
    )
    logger.info("=" * 80)

    dataset_path = Path(dataset_path or "data/ieee/train_transaction.csv")
    if not dataset_path.exists():
        fallback = Path("ieee/train_transaction.csv")
        if fallback.exists():
            dataset_path = fallback
        else:
            raise FileNotFoundError(f"IEEE-CIS dataset not found at {dataset_path}")

    ingestor = DatasetIngestion(dataset_path)
    df_raw, _ = ingestor.load_and_validate()

    # Precompute collision-free UID annotations
    valid_mask = df_raw[["card1", "card2", "addr1"]].notna().all(axis=1)
    uid_clean = pd.Series(index=df_raw.index, dtype=object)
    uid_clean[valid_mask] = (
        df_raw.loc[valid_mask, "card1"].astype(int).astype(str)
        + "_"
        + df_raw.loc[valid_mask, "card2"].astype(int).astype(str)
        + "_"
        + df_raw.loc[valid_mask, "addr1"].astype(int).astype(str)
    )
    uid_cf = uid_clean.copy()
    missing_mask = ~valid_mask
    uid_cf[missing_mask] = "MISSING_" + df_raw.loc[
        missing_mask, "TransactionID"
    ].astype(str)

    df_raw["uid_clean"] = uid_clean.values
    df_raw["uid_cf"] = uid_cf.values
    df_raw["is_clean"] = valid_mask.values

    # Pre-partition slices once for efficiency
    preprocessor = Preprocessor(exclude_card_identifiers=True)
    raw_slices = preprocessor.partition_time_slices(df_raw)

    cohort3_results: Dict[str, List[float]] = {
        "static_recall": [],
        "adaptive_recall": [],
        "recall_gain": [],
        "static_auc": [],
        "adaptive_auc": [],
        "auc_gain": [],
    }

    cohort1_results: Dict[str, List[float]] = {
        "static_recall": [],
        "adaptive_recall": [],
        "recall_gain": [],
    }
    cohort2_results: Dict[str, List[float]] = {
        "static_recall": [],
        "adaptive_recall": [],
        "recall_gain": [],
    }

    audit_runs = []

    for seed_idx, seed in enumerate(seeds):
        logger.info(
            f"\n--- [UID Audit Run {seed_idx + 1}/{len(seeds)}] Seed={seed} (Feature-Blind Mode) ---"
        )
        t0 = time.time()
        audit_res = run_single_audit(
            df_raw=df_raw,
            exclude_card_identifiers=True,
            title=f"FEATURE-BLIND AUDIT (Seed {seed})",
            random_seed=seed,
            preprocessor=preprocessor,
            raw_slices=raw_slices,
            output_dir=output_dir,
            data_kind=data_kind,
        )
        audit_runs.append(audit_res["run_dir"])

        # Cohort 3: Strict Stream-Unseen (Zero-Memory)
        c3 = audit_res["clean"]["strict"]
        if not c3 or not c3["fraud_tx"]:
            raise ValueError(
                "First-arrival cohort has no fraud labels; recall inference is undefined"
            )
        cohort3_results["static_recall"].append(float(c3["stat_recall"]))
        cohort3_results["adaptive_recall"].append(float(c3["adapt_recall"]))
        cohort3_results["recall_gain"].append(float(c3["recall_gain"]))
        cohort3_results["static_auc"].append(float(c3["stat_auc"]))
        cohort3_results["adaptive_auc"].append(float(c3["adapt_auc"]))
        cohort3_results["auc_gain"].append(float(c3["auc_gain"]))

        # Cohort 1 & 2
        c1 = audit_res["clean"]["seen"]
        cohort1_results["static_recall"].append(float(c1["stat_recall"]))
        cohort1_results["adaptive_recall"].append(float(c1["adapt_recall"]))
        cohort1_results["recall_gain"].append(float(c1["recall_gain"]))

        c2 = audit_res["clean"]["unseen"]
        cohort2_results["static_recall"].append(float(c2["stat_recall"]))
        cohort2_results["adaptive_recall"].append(float(c2["adapt_recall"]))
        cohort2_results["recall_gain"].append(float(c2["recall_gain"]))

        duration = time.time() - t0
        logger.info(
            f"Seed {seed} finished in {duration:.1f}s | "
            f"Zero-Mem Static Recall = {c3['stat_recall']:.2f}% | "
            f"Adaptive Recall = {c3['adapt_recall']:.2f}% | "
            f"Net Gain = +{c3['recall_gain']:.2f}%"
        )

    # Paired hypothesis tests for Cohort 3 (Zero-Memory)
    pt_recall = compute_paired_tests(
        cohort3_results["adaptive_recall"], cohort3_results["static_recall"]
    )
    pt_auc = compute_paired_tests(
        cohort3_results["adaptive_auc"], cohort3_results["static_auc"]
    )

    # Multiple comparisons adjustment across Cohort 3 primary endpoints
    uid_p_list = [pt_recall["p_value_t"], pt_auc["p_value_t"]]
    uid_fdr = adjust_p_values_fdr_bh(uid_p_list)
    uid_bonf = adjust_p_values_bonferroni(uid_p_list)

    pt_recall["p_value_t_fdr_bh"] = uid_fdr[0]
    pt_recall["p_value_t_bonferroni"] = uid_bonf[0]
    pt_recall["significant_fdr_05"] = bool(uid_fdr[0] < 0.05)

    pt_auc["p_value_t_fdr_bh"] = uid_fdr[1]
    pt_auc["p_value_t_bonferroni"] = uid_bonf[1]
    pt_auc["significant_fdr_05"] = bool(uid_fdr[1] < 0.05)

    # Compute Statistical Summaries
    summary: Dict[str, Any] = {
        "cohort3_zero_memory": {
            "static_recall": compute_sample_stats(cohort3_results["static_recall"]),
            "adaptive_recall": compute_sample_stats(cohort3_results["adaptive_recall"]),
            "recall_gain": compute_sample_stats(cohort3_results["recall_gain"]),
            "static_auc": compute_sample_stats(cohort3_results["static_auc"]),
            "adaptive_auc": compute_sample_stats(cohort3_results["adaptive_auc"]),
            "auc_gain": compute_sample_stats(cohort3_results["auc_gain"]),
            "paired_ttest_recall": pt_recall,
            "paired_ttest_auc": pt_auc,
        },
        "cohort1_seen": {
            "static_recall": compute_sample_stats(cohort1_results["static_recall"]),
            "adaptive_recall": compute_sample_stats(cohort1_results["adaptive_recall"]),
            "recall_gain": compute_sample_stats(cohort1_results["recall_gain"]),
        },
        "cohort2_unseen": {
            "static_recall": compute_sample_stats(cohort2_results["static_recall"]),
            "adaptive_recall": compute_sample_stats(cohort2_results["adaptive_recall"]),
            "recall_gain": compute_sample_stats(cohort2_results["recall_gain"]),
        },
        "multiple_testing_correction": {
            "method": "Benjamini-Hochberg (FDR) and Bonferroni",
            "hypotheses_tested": 2,
            "sample_size_per_group": len(seeds),
        },
    }

    logger.info("\n" + "=" * 80)
    logger.info("IEEE-CIS ZERO-MEMORY UID GENERALIZATION STATISTICAL RIGOR SUMMARY:")
    logger.info("=" * 80)
    c3_s = summary["cohort3_zero_memory"]
    logger.info(
        f"   Static Recall   : {c3_s['static_recall']['mean']:.2f}% ± {c3_s['static_recall']['std']:.2f}%  "
        f"(95% CI: [{c3_s['static_recall']['ci_95'][0]:.2f}%, {c3_s['static_recall']['ci_95'][1]:.2f}%])"
    )
    logger.info(
        f"   Adaptive Recall : {c3_s['adaptive_recall']['mean']:.2f}% ± {c3_s['adaptive_recall']['std']:.2f}%  "
        f"(95% CI: [{c3_s['adaptive_recall']['ci_95'][0]:.2f}%, {c3_s['adaptive_recall']['ci_95'][1]:.2f}%])"
    )
    logger.info(
        f"   Net Recall Gain : +{c3_s['recall_gain']['mean']:.2f}% ± {c3_s['recall_gain']['std']:.2f}%  "
        f"(95% CI: [+{c3_s['recall_gain']['ci_95'][0]:.2f}%, +{c3_s['recall_gain']['ci_95'][1]:.2f}%])"
    )
    p_t = c3_s["paired_ttest_recall"]
    logger.info(
        f"   Paired t-test   : t={p_t['t_stat']}, p={p_t['p_value_t']:.6e} "
        f"(p_FDR={p_t['p_value_t_fdr_bh']:.6e}) | Cohen's d={p_t['cohens_d']} ({p_t['effect_size']}) | "
        f"Significant (p < 0.001): {p_t['significant_001']}"
    )

    return {
        "run_directories": audit_runs,
        "raw_seeds": {
            "cohort3": cohort3_results,
            "cohort1": cohort1_results,
            "cohort2": cohort2_results,
        },
        "summary": summary,
    }


# =========================================================================
# PUBLICATION-QUALITY ERROR BAR VISUALIZATION
# =========================================================================
def generate_statistical_errorbar_charts(
    paysim_res: Dict[str, Any],
    ieee_uid_res: Dict[str, Any],
    output_filename: str = "statistical_rigor_errorbars.png",
) -> Path:
    logger.info(
        "Generating publication-quality error-bar comparison plots with statistical significance annotations..."
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. PaySim Out-of-Sample F1 Comparison (Left Subplot)
    models = ["Static Baseline", "Blind Periodic", "Sliding Window", "Drift-Adaptive"]
    keys = ["static", "periodic", "sliding", "adaptive"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]

    means = [paysim_res["summary"]["models"][k]["f1"]["mean"] for k in keys]
    stds = [paysim_res["summary"]["models"][k]["f1"]["std"] for k in keys]

    bars1 = ax1.bar(
        models,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        alpha=0.88,
        edgecolor="black",
        width=0.55,
    )
    ax1.set_title(
        "PaySim Out-of-Sample Stream F1-Score\n(Mean ± Std across supplied seeds)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel("Out-of-Sample F1-Score", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.985, 1.002)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bar values
    for bar, m, s in zip(bars1, means, stds):
        h = bar.get_height()
        ax1.annotate(
            f"{m:.4f}\n(±{s:.4f})",
            xy=(bar.get_x() + bar.get_width() / 2, h + 0.0005),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Significance bracket between Static and Adaptive
    p_val_paysim = paysim_res["summary"]["paired_tests"]["adaptive_vs_static_f1"][
        "p_value_t"
    ]
    sig_text = "*** p < 0.001" if p_val_paysim < 0.001 else f"p = {p_val_paysim:.4f}"
    ax1.plot([0, 0, 3, 3], [0.9998, 1.0008, 1.0008, 0.9998], lw=1.3, c="k")
    ax1.text(
        1.5,
        1.0010,
        sig_text,
        ha="center",
        va="bottom",
        color="#d62728",
        fontsize=10,
        fontweight="bold",
    )

    # 2. IEEE-CIS Zero-Memory UID Cohorts Recall (Right Subplot)
    cohort_names = [
        "Seen Cards\n(Cohort 1)",
        "Unseen Cards\n(Cohort 2)",
        "Zero-Memory Arrival\n(Cohort 3)",
    ]
    c_keys = ["cohort1_seen", "cohort2_unseen", "cohort3_zero_memory"]

    x = np.arange(len(cohort_names))
    width = 0.35

    stat_means = [ieee_uid_res["summary"][k]["static_recall"]["mean"] for k in c_keys]
    stat_stds = [ieee_uid_res["summary"][k]["static_recall"]["std"] for k in c_keys]

    adp_means = [ieee_uid_res["summary"][k]["adaptive_recall"]["mean"] for k in c_keys]
    adp_stds = [ieee_uid_res["summary"][k]["adaptive_recall"]["std"] for k in c_keys]

    b_stat = ax2.bar(
        x - width / 2,
        stat_means,
        width,
        yerr=stat_stds,
        capsize=5,
        label="Static Baseline (Frozen)",
        color="#7f7f7f",
        alpha=0.88,
        edgecolor="black",
    )
    b_adp = ax2.bar(
        x + width / 2,
        adp_means,
        width,
        yerr=adp_stds,
        capsize=5,
        label="Drift-Adaptive (Proposed)",
        color="#2ca02c",
        alpha=0.88,
        edgecolor="black",
    )

    ax2.set_title(
        "IEEE-CIS Feature-Blind Recall by Card Cohort\n(Mean ± Std across supplied seeds)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax2.set_ylabel("Out-of-Sample Recall (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cohort_names, fontsize=10, fontweight="bold")
    ax2.set_ylim(40, 90)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate Cohort 3 Gain & Significance
    c3_gain = ieee_uid_res["summary"]["cohort3_zero_memory"]["recall_gain"]["mean"]
    c3_gain_std = ieee_uid_res["summary"]["cohort3_zero_memory"]["recall_gain"]["std"]
    p_val_uid = ieee_uid_res["summary"]["cohort3_zero_memory"]["paired_ttest_recall"][
        "p_value_t"
    ]
    sig_uid_text = (
        f"+{c3_gain:.2f}% (±{c3_gain_std:.2f}%)\n*** p < 0.001"
        if p_val_uid < 0.001
        else f"+{c3_gain:.2f}% (p={p_val_uid:.4f})"
    )

    # Bracket over Cohort 3
    ax2.plot(
        [2 - width / 2, 2 - width / 2, 2 + width / 2, 2 + width / 2],
        [70, 81, 81, 79],
        lw=1.3,
        c="k",
    )
    ax2.text(
        2,
        81.8,
        sig_uid_text,
        ha="center",
        va="bottom",
        color="#d62728",
        fontsize=9.5,
        fontweight="bold",
    )

    plt.tight_layout()
    chart_path = FIGURES_DIR / output_filename
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Statistical error-bar charts successfully saved to {chart_path}")
    return chart_path


def export_summary_csv(results_all: Dict[str, Any], output_path: Path):
    """Exports structured metrics to CSV for LaTeX table rendering."""
    import csv

    rows = []

    # PaySim results
    if "paysim_oos_f1" in results_all:
        p_res = results_all["paysim_oos_f1"]["summary"]
        for m, v in p_res["models"].items():
            p_val = "-"
            cohens_d = "-"
            hedges_g = "-"
            p_val_fdr = "-"
            if (
                m == "adaptive"
                and "paired_tests" in p_res
                and "adaptive_vs_static_f1" in p_res["paired_tests"]
            ):
                pt = p_res["paired_tests"]["adaptive_vs_static_f1"]
                p_val = f"{pt['p_value_t']:.6e}"
                cohens_d = str(pt.get("cohens_d", "-"))
                hedges_g = str(pt.get("hedges_g", "-"))
                p_val_fdr = f"{pt.get('p_value_t_fdr_bh', pt['p_value_t']):.6e}"

            rows.append(
                {
                    "benchmark": "PaySim Out-of-Sample Stream",
                    "architecture_or_cohort": m.capitalize(),
                    "metric": "F1-Score",
                    "mean": v["f1"]["mean"],
                    "std": v["f1"]["std"],
                    "ci_95_low": v["f1"]["ci_95"][0],
                    "ci_95_high": v["f1"]["ci_95"][1],
                    "bootstrap_ci_95": f"[{v['bootstrap_f1_ci'][0]}, {v['bootstrap_f1_ci'][1]}]",
                    "cohens_d": cohens_d,
                    "hedges_g": hedges_g,
                    "p_value_vs_static": p_val,
                    "p_value_fdr_bh": p_val_fdr,
                }
            )

    # IEEE-CIS UID Generalization results
    if "ieee_zero_memory_uid" in results_all:
        uid_s = results_all["ieee_zero_memory_uid"]["summary"]
        cohorts = [
            ("Cohort 1: Repeat/Seen Cards", "cohort1_seen"),
            ("Cohort 2: Baseline-Unseen Cards", "cohort2_unseen"),
            ("Cohort 3: Zero-Memory Cards", "cohort3_zero_memory"),
        ]
        for c_label, c_key in cohorts:
            c_data = uid_s[c_key]
            p_val = "-"
            cohens_d = "-"
            hedges_g = "-"
            p_val_fdr = "-"
            if c_key == "cohort3_zero_memory" and "paired_ttest_recall" in c_data:
                pt = c_data["paired_ttest_recall"]
                p_val = f"{pt['p_value_t']:.6e}"
                cohens_d = str(pt.get("cohens_d", "-"))
                hedges_g = str(pt.get("hedges_g", "-"))
                p_val_fdr = f"{pt.get('p_value_t_fdr_bh', pt['p_value_t']):.6e}"

            rows.append(
                {
                    "benchmark": "IEEE-CIS Feature-Blind Audit",
                    "architecture_or_cohort": f"{c_label} (Static)",
                    "metric": "Recall (%)",
                    "mean": c_data["static_recall"]["mean"],
                    "std": c_data["static_recall"]["std"],
                    "ci_95_low": c_data["static_recall"]["ci_95"][0],
                    "ci_95_high": c_data["static_recall"]["ci_95"][1],
                    "bootstrap_ci_95": "-",
                    "cohens_d": "-",
                    "hedges_g": "-",
                    "p_value_vs_static": "-",
                    "p_value_fdr_bh": "-",
                }
            )
            rows.append(
                {
                    "benchmark": "IEEE-CIS Feature-Blind Audit",
                    "architecture_or_cohort": f"{c_label} (Adaptive)",
                    "metric": "Recall (%)",
                    "mean": c_data["adaptive_recall"]["mean"],
                    "std": c_data["adaptive_recall"]["std"],
                    "ci_95_low": c_data["adaptive_recall"]["ci_95"][0],
                    "ci_95_high": c_data["adaptive_recall"]["ci_95"][1],
                    "bootstrap_ci_95": "-",
                    "cohens_d": cohens_d,
                    "hedges_g": hedges_g,
                    "p_value_vs_static": p_val,
                    "p_value_fdr_bh": p_val_fdr,
                }
            )

    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# =========================================================================
# MAIN EXECUTION ORCHESTRATOR
# =========================================================================
def run():
    parser = argparse.ArgumentParser(
        description="Multi-Seed Statistical Rigor & Bootstrap Benchmarking Suite"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="List of random seeds to evaluate",
    )
    parser.add_argument(
        "--extended-seeds",
        action="store_true",
        help="Run expanded 10-seed evaluation ([42, 101, 202, 303, 404, 505, 606, 707, 808, 909])",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["all", "paysim", "ieee_uid"],
        default="all",
        help="Benchmark suite to execute",
    )
    parser.add_argument("--paysim-dataset")
    parser.add_argument("--ieee-dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for empirical CI",
    )
    args = parser.parse_args()

    if args.extended_seeds:
        seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    else:
        seeds = args.seeds
    logger.info(f"Running Statistical Rigor Suite across {len(seeds)} seeds: {seeds}")

    import uuid

    run_root = Path(args.output_dir or METRICS_DIR) / ("STATS_" + uuid.uuid4().hex[:10])
    run_root.mkdir(parents=True, exist_ok=False)
    summary_path = run_root / "statistical_rigor_summary.json"
    results_all: Dict[str, Any] = {}

    results_all.setdefault("metadata", {})
    results_all["metadata"].update(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "requested_device": GPU_DEVICE if USE_GPU else "cpu",
            "protocol": "shared v0.4.1 pipeline; inspect referenced run manifests for actual devices",
        }
    )

    if args.benchmark in ("all", "ieee_uid"):
        uid_res = run_ieee_uid_statistical_rigor(
            seeds,
            n_bootstrap=args.n_bootstrap,
            dataset_path=args.ieee_dataset,
            output_dir=run_root / "runs",
            data_kind="synthetic" if args.synthetic else "real",
        )
        results_all["ieee_zero_memory_uid"] = uid_res

    if args.benchmark in ("all", "paysim"):
        paysim_res = run_paysim_statistical_rigor(
            seeds,
            n_bootstrap=args.n_bootstrap,
            dataset_path=args.paysim_dataset,
            output_dir=run_root / "runs",
            data_kind="synthetic" if args.synthetic else "real",
        )
        results_all["paysim_oos_f1"] = paysim_res

    # Save to metrics directory
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=4, allow_nan=False)
    logger.info(f"Summary JSON saved to {summary_path}")

    # Export to CSV for paper table generation
    csv_path = run_root / "statistical_rigor_summary.csv"
    export_summary_csv(results_all, csv_path)
    logger.info(f"Summary CSV saved to {csv_path}")

    # Generate Visualization if both completed
    if "paysim_oos_f1" in results_all and "ieee_zero_memory_uid" in results_all:
        chart_path = generate_statistical_errorbar_charts(
            paysim_res=results_all["paysim_oos_f1"],
            ieee_uid_res=results_all["ieee_zero_memory_uid"],
            output_filename=str(
                (run_root / "statistical_rigor_errorbars.png").resolve()
            ),
        )
        logger.info(f"Visualization generated: {chart_path}")

    logger.info("\n" + "=" * 92)
    logger.info("STATISTICAL RIGOR BENCHMARK SUITE SUCCESSFULLY COMPLETED")
    logger.info("=" * 92)


if __name__ == "__main__":
    run()
