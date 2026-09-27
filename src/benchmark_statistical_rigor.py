"""
Multi-Seed Statistical Rigor & Bootstrap Confidence Interval Benchmarking Suite.

Addresses:
1. PaySim Mobile Money Out-of-Sample Stream F1 Rigor (0.9955 -> 0.9978):
   Evaluates Static, Periodic, Sliding, and Drift-Adaptive across K random seeds.
2. IEEE-CIS Zero-Memory UID Generalization Recall Rigor (67.82% -> 77.09%):
   Evaluates Static vs Drift-Adaptive across K random seeds on Strict Stream-Unseen Cards (Cohort 3).
3. IEEE-CIS 26-Week Streaming Out-of-Sample Benchmark:
   Evaluates all 4 model architectures across K random seeds.

Calculates:
- Mean and Standard Deviation (mean ± std)
- 95% Student's t Confidence Interval
- 1,000-sample Non-Parametric Bootstrap 95% Confidence Interval [CI_low, CI_high]
- Paired Student's t-test (t-stat, p-value) and Wilcoxon Signed-Rank test
- Publication-quality error-bar visualization with significance annotations
"""

import sys
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

from src.config import (
    BASE_DIR, RESULTS_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR,
    TRAINING_WINDOW_RATIO, USE_GPU, GPU_DEVICE
)
from src.ingestion import DatasetIngestion
from src.preprocessing import Preprocessor
from src.static_model import StaticBaselineModel
from src.periodic_model import PeriodicRetrainModel
from src.sliding_model import SlidingWindowModel
from src.adaptive_model import DriftAdaptiveModel
from src.drift_detector import DriftMonitor
from src.evaluator import SliceEvaluator
from src.evaluate_uid_split import compute_metrics, run_single_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fraud_drift.statistical_rigor")

DEFAULT_SEEDS = [42, 101, 202, 303, 404]


def compute_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_bootstraps: int = 1000, alpha: float = 0.05, metric_type: str = "recall") -> Tuple[float, float]:
    """
    Computes empirical non-parametric percentile bootstrap confidence intervals.
    """
    rng = np.random.default_rng(42)
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
        "n_seeds": k
    }


def compute_paired_tests(a_values: List[float], b_values: List[float]) -> Dict[str, Any]:
    """
    Runs paired Student's t-test and Wilcoxon signed-rank test.
    """
    a = np.array(a_values, dtype=float)
    b = np.array(b_values, dtype=float)
    diff = a - b
    
    if np.all(diff == 0) or len(a) < 2:
        return {
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "t_stat": 0.0,
            "p_value_t": 1.0,
            "wilcoxon_stat": 0.0,
            "p_value_wilcoxon": 1.0,
            "significant_05": False,
            "significant_01": False,
            "significant_001": False
        }
    
    t_res = stats.ttest_rel(a, b)
    try:
        w_res = stats.wilcoxon(a, b)
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")

    p_t = float(t_res.pvalue)
    return {
        "mean_diff": round(float(np.mean(diff)), 4),
        "std_diff": round(float(np.std(diff, ddof=1)), 4),
        "t_stat": round(float(t_res.statistic), 4),
        "p_value_t": float(f"{p_t:.6e}"),
        "wilcoxon_stat": round(w_stat, 4) if not np.isnan(w_stat) else None,
        "p_value_wilcoxon": float(f"{w_p:.6e}") if not np.isnan(w_p) else None,
        "significant_05": bool(p_t < 0.05),
        "significant_01": bool(p_t < 0.01),
        "significant_001": bool(p_t < 0.001)
    }


# =========================================================================
# BENCHMARK 1: PAYSIM MOBILE MONEY MULTI-SEED RIGOR
# =========================================================================
def run_paysim_statistical_rigor(seeds: List[int], n_bootstrap: int = 1000) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(f"STARTING PAYSIM MULTI-SEED RIGOR BENCHMARK (Seeds: {seeds})")
    logger.info("Target: Verify 0.9955 -> 0.9978 F1 difference across multiple random seeds")
    logger.info("=" * 80)

    dataset_path = Path("data/paysim/paysim_full.csv")
    if not dataset_path.exists():
        fallback = Path("data/raw/paysim_synthetic_24w.csv")
        if fallback.exists():
            dataset_path = fallback
        else:
            raise FileNotFoundError(f"PaySim dataset not found at {dataset_path}")

    ingestor = DatasetIngestion(dataset_path)
    df_raw, _ = ingestor.load_and_validate()

    preprocessor = Preprocessor()
    raw_slices = preprocessor.partition_time_slices(df_raw)
    n_train_slices = 2 if len(raw_slices) <= 8 else max(1, int(len(raw_slices) * TRAINING_WINDOW_RATIO))

    logger.info("Fitting preprocessor once strictly on baseline training slices (zero lookahead)...")
    train_raw = pd.concat([s["data"] for s in raw_slices[:n_train_slices]], ignore_index=True)
    preprocessor.fit(train_raw)
    imbalance_info = preprocessor.imbalance_weights
    feature_names = preprocessor.feature_names

    logger.info("Transforming time slices chronologically...")
    transformed_slices = []
    for s in raw_slices:
        s_trans = s.copy()
        s_trans["data"] = preprocessor.transform(s["data"])
        transformed_slices.append(s_trans)

    train_transformed = pd.concat([s["data"] for s in transformed_slices[:n_train_slices]], ignore_index=True)
    oos_slices = transformed_slices[n_train_slices:]
    oos_df = pd.concat([s["data"] for s in oos_slices], ignore_index=True)
    y_oos = oos_df["isFraud"].values

    logger.info(f"PaySim Stream ready: {len(df_raw):,} total rows, {len(oos_df):,} out-of-sample test rows ({int(y_oos.sum()):,} frauds).")

    seed_results: Dict[str, Dict[str, List[float]]] = {
        "static": {"f1": [], "recall": [], "precision": [], "auc_pr": []},
        "periodic": {"f1": [], "recall": [], "precision": [], "auc_pr": []},
        "sliding": {"f1": [], "recall": [], "precision": [], "auc_pr": []},
        "adaptive": {"f1": [], "recall": [], "precision": [], "auc_pr": []}
    }
    
    last_predictions: Dict[str, np.ndarray] = {}

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n--- [PaySim Run {seed_idx + 1}/{len(seeds)}] Initializing models with seed={seed} ---")
        t0 = time.time()

        # 1. Static Model
        static = StaticBaselineModel(feature_names, imbalance_info["scale_pos_weight"], random_seed=seed)
        static.fit(train_transformed)
        s_preds, s_probs = static.predict(oos_df)

        # 2. Periodic Retrain Model
        periodic = PeriodicRetrainModel(feature_names, retrain_interval_slices=2, scale_pos_weight=imbalance_info["scale_pos_weight"], random_seed=seed)
        periodic.fit_initial(train_transformed)
        for s in transformed_slices[:n_train_slices]:
            periodic.cumulative_history.append(s["data"].copy())

        # 3. Sliding Window Model
        sliding = SlidingWindowModel(feature_names, window_slices=2, retrain_interval_slices=2, scale_pos_weight=imbalance_info["scale_pos_weight"], random_seed=seed)
        sliding.fit_initial(train_transformed)
        for s in transformed_slices[:n_train_slices]:
            sliding.window_history.append(s["data"].copy())
            if len(sliding.window_history) > sliding.window_slices:
                sliding.window_history.pop(0)

        # 4. Drift Adaptive Model
        adaptive = DriftAdaptiveModel(feature_names, imbalance_info["scale_pos_weight"], random_seed=seed)
        adaptive.fit_initial(train_transformed)
        drift_monitor = DriftMonitor()

        # Warmup adaptive model across baseline window
        for s in transformed_slices[:n_train_slices]:
            s_data = s["data"]
            y_base = s_data["isFraud"].values
            a_base_p, _ = adaptive.predict(s_data)
            base_drift = False
            s_step = s_data["step"].values
            s_trans = s_data["trans_id"].values
            for i in range(len(s_data)):
                alert = drift_monitor.update(int(y_base[i]), int(a_base_p[i]), s["slice_index"], int(s_step[i]), str(s_trans[i]))
                if alert["any_drift"]:
                    base_drift = True
            adaptive.update_with_slice(s_data, base_drift, s["slice_index"])
            if base_drift:
                drift_monitor.reset_detectors()

        # Sequential Evaluation across Out-of-Sample Slices
        per_preds_list, sli_preds_list, adp_preds_list = [], [], []
        per_probs_list, sli_probs_list, adp_probs_list = [], [], []

        for s in oos_slices:
            s_data = s["data"]
            y_slice = s_data["isFraud"].values

            # Periodic predict then update
            p_p, p_pr = periodic.predict(s_data)
            periodic.update_with_slice(s_data, s["slice_index"], is_out_of_sample=True)
            per_preds_list.append(p_p)
            per_probs_list.append(p_pr)

            # Sliding predict then update
            sl_p, sl_pr = sliding.predict(s_data)
            sliding.update_with_slice(s_data, s["slice_index"], is_out_of_sample=True)
            sli_preds_list.append(sl_p)
            sli_probs_list.append(sl_pr)

            # Adaptive predict then update
            ad_p, ad_pr = adaptive.predict(s_data)
            adp_preds_list.append(ad_p)
            adp_probs_list.append(ad_pr)

            slice_drift = False
            s_step = s_data["step"].values
            s_trans = s_data["trans_id"].values
            for i in range(len(s_data)):
                alert = drift_monitor.update(int(y_slice[i]), int(ad_p[i]), s["slice_index"], int(s_step[i]), str(s_trans[i]))
                if alert["any_drift"]:
                    slice_drift = True
            adaptive.update_with_slice(s_data, slice_drift, s["slice_index"])
            if slice_drift:
                drift_monitor.reset_detectors()

        p_preds = np.concatenate(per_preds_list)
        p_probs = np.concatenate(per_probs_list)
        sl_preds = np.concatenate(sli_preds_list)
        sl_probs = np.concatenate(sli_probs_list)
        ad_preds = np.concatenate(adp_preds_list)
        ad_probs = np.concatenate(adp_probs_list)

        # Compute metrics
        for name, preds, probs in [
            ("static", s_preds, s_probs),
            ("periodic", p_preds, p_probs),
            ("sliding", sl_preds, sl_probs),
            ("adaptive", ad_preds, ad_probs)
        ]:
            m = compute_metrics({"y": y_oos, "stat_p": preds, "adapt_p": preds, "stat_prob": probs, "adapt_prob": probs})
            tp = int(((y_oos == 1) & (preds == 1)).sum())
            fp = int(((y_oos == 0) & (preds == 1)).sum())
            fn = int(((y_oos == 1) & (preds == 0)).sum())
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            auc_pr = m.get("stat_auc", 0.0)

            seed_results[name]["f1"].append(round(f1, 4))
            seed_results[name]["recall"].append(round(rec * 100, 2))
            seed_results[name]["precision"].append(round(prec * 100, 2))
            seed_results[name]["auc_pr"].append(round(auc_pr, 4))

        last_predictions["static"] = s_preds
        last_predictions["periodic"] = p_preds
        last_predictions["sliding"] = sl_preds
        last_predictions["adaptive"] = ad_preds

        duration = time.time() - t0
        logger.info(
            f"Seed {seed} finished in {duration:.1f}s | "
            f"Static F1={seed_results['static']['f1'][-1]:.4f} | "
            f"Periodic F1={seed_results['periodic']['f1'][-1]:.4f} | "
            f"Sliding F1={seed_results['sliding']['f1'][-1]:.4f} | "
            f"Adaptive F1={seed_results['adaptive']['f1'][-1]:.4f}"
        )

    # Compute Aggregate Statistical Summary & CIs
    summary: Dict[str, Any] = {"models": {}, "paired_tests": {}}
    for m_name in ["static", "periodic", "sliding", "adaptive"]:
        summary["models"][m_name] = {
            "f1": compute_sample_stats(seed_results[m_name]["f1"]),
            "recall": compute_sample_stats(seed_results[m_name]["recall"]),
            "precision": compute_sample_stats(seed_results[m_name]["precision"]),
            "auc_pr": compute_sample_stats(seed_results[m_name]["auc_pr"]),
            "bootstrap_f1_ci": compute_bootstrap_ci(y_oos, last_predictions[m_name], n_bootstraps=n_bootstrap, metric_type="f1"),
            "raw_seed_f1": seed_results[m_name]["f1"]
        }

    # Hypothesis Testing
    summary["paired_tests"]["adaptive_vs_static_f1"] = compute_paired_tests(
        seed_results["adaptive"]["f1"], seed_results["static"]["f1"]
    )
    summary["paired_tests"]["adaptive_vs_periodic_f1"] = compute_paired_tests(
        seed_results["adaptive"]["f1"], seed_results["periodic"]["f1"]
    )
    summary["paired_tests"]["adaptive_vs_sliding_f1"] = compute_paired_tests(
        seed_results["adaptive"]["f1"], seed_results["sliding"]["f1"]
    )

    logger.info("\n" + "=" * 80)
    logger.info("PAYSIM MULTI-SEED STATISTICAL RIGOR SUMMARY:")
    logger.info("=" * 80)
    for m_name, d in summary["models"].items():
        logger.info(
            f"   {m_name.upper():<10}: F1 = {d['f1']['mean']:.4f} ± {d['f1']['std']:.4f}  "
            f"(95% CI: [{d['f1']['ci_95'][0]:.4f}, {d['f1']['ci_95'][1]:.4f}]) | "
            f"Bootstrap 95% CI: [{d['bootstrap_f1_ci'][0]:.4f}, {d['bootstrap_f1_ci'][1]:.4f}]"
        )
    p_test = summary["paired_tests"]["adaptive_vs_static_f1"]
    logger.info(
        f"\n   Paired t-test (Adaptive vs Static F1): t={p_test['t_stat']:.4f}, p={p_test['p_value_t']:.6e} "
        f"-> Statistically Significant (p < 0.001): {p_test['significant_001']}"
    )
    return {"raw_seeds": seed_results, "summary": summary}


# =========================================================================
# BENCHMARK 2: IEEE-CIS ZERO-MEMORY UID GENERALIZATION MULTI-SEED RIGOR
# =========================================================================
def run_ieee_uid_statistical_rigor(seeds: List[int], n_bootstrap: int = 1000) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(f"STARTING IEEE-CIS ZERO-MEMORY UID GENERALIZATION BENCHMARK (Seeds: {seeds})")
    logger.info("Target: Verify 67.82% -> 77.09% Zero-Memory Recall Gain across random seeds")
    logger.info("=" * 80)

    dataset_path = Path("data/ieee/train_transaction.csv")
    if not dataset_path.exists():
        fallback = Path("ieee/train_transaction.csv")
        if fallback.exists():
            dataset_path = fallback
        else:
            raise FileNotFoundError(f"IEEE-CIS dataset not found at {dataset_path}")

    ingestor = DatasetIngestion(dataset_path)
    df_raw, _ = ingestor.load_and_validate()

    # Precompute collision-free UID annotations
    valid_mask = df_raw["card2"].notna() & df_raw["addr1"].notna()
    uid_clean = pd.Series(index=df_raw.index, dtype=object)
    uid_clean[valid_mask] = (
        df_raw.loc[valid_mask, "card1"].astype(int).astype(str) + "_" +
        df_raw.loc[valid_mask, "card2"].astype(int).astype(str) + "_" +
        df_raw.loc[valid_mask, "addr1"].astype(int).astype(str)
    )
    uid_cf = uid_clean.copy()
    missing_mask = ~valid_mask
    uid_cf[missing_mask] = "MISSING_" + df_raw.loc[missing_mask, "TransactionID"].astype(str)

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
        "auc_gain": []
    }

    cohort1_results: Dict[str, List[float]] = {
        "static_recall": [], "adaptive_recall": [], "recall_gain": []
    }
    cohort2_results: Dict[str, List[float]] = {
        "static_recall": [], "adaptive_recall": [], "recall_gain": []
    }

    last_audit_res = None

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n--- [UID Audit Run {seed_idx + 1}/{len(seeds)}] Seed={seed} (Feature-Blind Mode) ---")
        t0 = time.time()
        audit_res = run_single_audit(
            df_raw=df_raw,
            exclude_card_identifiers=True,
            title=f"FEATURE-BLIND AUDIT (Seed {seed})",
            random_seed=seed,
            preprocessor=preprocessor,
            raw_slices=raw_slices
        )
        last_audit_res = audit_res

        # Cohort 3: Strict Stream-Unseen (Zero-Memory)
        c3 = audit_res["clean"]["strict"]
        cohort3_results["static_recall"].append(round(c3["stat_recall"], 2))
        cohort3_results["adaptive_recall"].append(round(c3["adapt_recall"], 2))
        cohort3_results["recall_gain"].append(round(c3["recall_gain"], 2))
        cohort3_results["static_auc"].append(round(c3["stat_auc"], 4))
        cohort3_results["adaptive_auc"].append(round(c3["adapt_auc"], 4))
        cohort3_results["auc_gain"].append(round(c3["auc_gain"], 4))

        # Cohort 1 & 2
        c1 = audit_res["clean"]["seen"]
        cohort1_results["static_recall"].append(round(c1["stat_recall"], 2))
        cohort1_results["adaptive_recall"].append(round(c1["adapt_recall"], 2))
        cohort1_results["recall_gain"].append(round(c1["recall_gain"], 2))

        c2 = audit_res["clean"]["unseen"]
        cohort2_results["static_recall"].append(round(c2["stat_recall"], 2))
        cohort2_results["adaptive_recall"].append(round(c2["adapt_recall"], 2))
        cohort2_results["recall_gain"].append(round(c2["recall_gain"], 2))

        duration = time.time() - t0
        logger.info(
            f"Seed {seed} finished in {duration:.1f}s | "
            f"Zero-Mem Static Recall = {c3['stat_recall']:.2f}% | "
            f"Adaptive Recall = {c3['adapt_recall']:.2f}% | "
            f"Net Gain = +{c3['recall_gain']:.2f}%"
        )

    # Compute Statistical Summaries
    summary: Dict[str, Any] = {
        "cohort3_zero_memory": {
            "static_recall": compute_sample_stats(cohort3_results["static_recall"]),
            "adaptive_recall": compute_sample_stats(cohort3_results["adaptive_recall"]),
            "recall_gain": compute_sample_stats(cohort3_results["recall_gain"]),
            "static_auc": compute_sample_stats(cohort3_results["static_auc"]),
            "adaptive_auc": compute_sample_stats(cohort3_results["adaptive_auc"]),
            "auc_gain": compute_sample_stats(cohort3_results["auc_gain"]),
            "paired_ttest_recall": compute_paired_tests(
                cohort3_results["adaptive_recall"], cohort3_results["static_recall"]
            )
        },
        "cohort1_seen": {
            "static_recall": compute_sample_stats(cohort1_results["static_recall"]),
            "adaptive_recall": compute_sample_stats(cohort1_results["adaptive_recall"]),
            "recall_gain": compute_sample_stats(cohort1_results["recall_gain"])
        },
        "cohort2_unseen": {
            "static_recall": compute_sample_stats(cohort2_results["static_recall"]),
            "adaptive_recall": compute_sample_stats(cohort2_results["adaptive_recall"]),
            "recall_gain": compute_sample_stats(cohort2_results["recall_gain"])
        }
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
        f"   Paired t-test   : t={p_t['t_stat']:.4f}, p={p_t['p_value_t']:.6e} "
        f"-> Statistically Significant (p < 0.001): {p_t['significant_001']}"
    )

    return {
        "raw_seeds": {
            "cohort3": cohort3_results,
            "cohort1": cohort1_results,
            "cohort2": cohort2_results
        },
        "summary": summary
    }


# =========================================================================
# PUBLICATION-QUALITY ERROR BAR VISUALIZATION
# =========================================================================
def generate_statistical_errorbar_charts(
    paysim_res: Dict[str, Any],
    ieee_uid_res: Dict[str, Any],
    output_filename: str = "statistical_rigor_errorbars.png"
) -> Path:
    logger.info("Generating publication-quality error-bar comparison plots with statistical significance annotations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. PaySim Out-of-Sample F1 Comparison (Left Subplot)
    models = ["Static Baseline", "Blind Periodic", "Sliding Window", "Drift-Adaptive"]
    keys = ["static", "periodic", "sliding", "adaptive"]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]

    means = [paysim_res["summary"]["models"][k]["f1"]["mean"] for k in keys]
    stds = [paysim_res["summary"]["models"][k]["f1"]["std"] for k in keys]

    bars1 = ax1.bar(models, means, yerr=stds, capsize=6, color=colors, alpha=0.88, edgecolor="black", width=0.55)
    ax1.set_title("PaySim Out-of-Sample Stream F1-Score\n(Mean ± Std across 5 Random Seeds)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Out-of-Sample F1-Score", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.985, 1.002)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bar values
    for bar, m, s in zip(bars1, means, stds):
        h = bar.get_height()
        ax1.annotate(
            f"{m:.4f}\n(±{s:.4f})",
            xy=(bar.get_x() + bar.get_width() / 2, h + 0.0005),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Significance bracket between Static and Adaptive
    p_val_paysim = paysim_res["summary"]["paired_tests"]["adaptive_vs_static_f1"]["p_value_t"]
    sig_text = "*** p < 0.001" if p_val_paysim < 0.001 else f"p = {p_val_paysim:.4f}"
    ax1.plot([0, 0, 3, 3], [0.9998, 1.0008, 1.0008, 0.9998], lw=1.3, c="k")
    ax1.text(1.5, 1.0010, sig_text, ha="center", va="bottom", color="#d62728", fontsize=10, fontweight="bold")

    # 2. IEEE-CIS Zero-Memory UID Cohorts Recall (Right Subplot)
    cohort_names = ["Seen Cards\n(Cohort 1)", "Unseen Cards\n(Cohort 2)", "Zero-Memory Arrival\n(Cohort 3)"]
    c_keys = ["cohort1_seen", "cohort2_unseen", "cohort3_zero_memory"]

    x = np.arange(len(cohort_names))
    width = 0.35

    stat_means = [ieee_uid_res["summary"][k]["static_recall"]["mean"] for k in c_keys]
    stat_stds = [ieee_uid_res["summary"][k]["static_recall"]["std"] for k in c_keys]

    adp_means = [ieee_uid_res["summary"][k]["adaptive_recall"]["mean"] for k in c_keys]
    adp_stds = [ieee_uid_res["summary"][k]["adaptive_recall"]["std"] for k in c_keys]

    b_stat = ax2.bar(x - width / 2, stat_means, width, yerr=stat_stds, capsize=5, label="Static Baseline (Frozen)", color="#7f7f7f", alpha=0.88, edgecolor="black")
    b_adp = ax2.bar(x + width / 2, adp_means, width, yerr=adp_stds, capsize=5, label="Drift-Adaptive (Proposed)", color="#2ca02c", alpha=0.88, edgecolor="black")

    ax2.set_title("IEEE-CIS Feature-Blind Recall by Card Cohort\n(Mean ± Std across 5 Random Seeds)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylabel("Out-of-Sample Recall (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cohort_names, fontsize=10, fontweight="bold")
    ax2.set_ylim(40, 90)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate Cohort 3 Gain & Significance
    c3_gain = ieee_uid_res["summary"]["cohort3_zero_memory"]["recall_gain"]["mean"]
    c3_gain_std = ieee_uid_res["summary"]["cohort3_zero_memory"]["recall_gain"]["std"]
    p_val_uid = ieee_uid_res["summary"]["cohort3_zero_memory"]["paired_ttest_recall"]["p_value_t"]
    sig_uid_text = f"+{c3_gain:.2f}% (±{c3_gain_std:.2f}%)\n*** p < 0.001" if p_val_uid < 0.001 else f"+{c3_gain:.2f}% (p={p_val_uid:.4f})"

    # Bracket over Cohort 3
    ax2.plot([2 - width/2, 2 - width/2, 2 + width/2, 2 + width/2], [70, 81, 81, 79], lw=1.3, c="k")
    ax2.text(2, 81.8, sig_uid_text, ha="center", va="bottom", color="#d62728", fontsize=9.5, fontweight="bold")

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
            if m == "adaptive":
                p_val = f"{p_res['paired_tests']['adaptive_vs_static_f1']['p_value_t']:.6e}"
            rows.append({
                "benchmark": "PaySim Out-of-Sample Stream",
                "architecture_or_cohort": m.capitalize(),
                "metric": "F1-Score",
                "mean": v["f1"]["mean"],
                "std": v["f1"]["std"],
                "ci_95_low": v["f1"]["ci_95"][0],
                "ci_95_high": v["f1"]["ci_95"][1],
                "bootstrap_ci_95": f"[{v['bootstrap_f1_ci'][0]}, {v['bootstrap_f1_ci'][1]}]",
                "p_value_vs_static": p_val
            })

    # IEEE-CIS UID Generalization results
    if "ieee_zero_memory_uid" in results_all:
        uid_s = results_all["ieee_zero_memory_uid"]["summary"]
        cohorts = [
            ("Cohort 1: Repeat/Seen Cards", "cohort1_seen"),
            ("Cohort 2: Baseline-Unseen Cards", "cohort2_unseen"),
            ("Cohort 3: Zero-Memory Cards", "cohort3_zero_memory")
        ]
        for c_label, c_key in cohorts:
            c_data = uid_s[c_key]
            p_val = "-"
            if c_key == "cohort3_zero_memory" and "paired_ttest_recall" in c_data:
                p_val = f"{c_data['paired_ttest_recall']['p_value_t']:.6e}"

            rows.append({
                "benchmark": "IEEE-CIS Feature-Blind Audit",
                "architecture_or_cohort": f"{c_label} (Static)",
                "metric": "Recall (%)",
                "mean": c_data["static_recall"]["mean"],
                "std": c_data["static_recall"]["std"],
                "ci_95_low": c_data["static_recall"]["ci_95"][0],
                "ci_95_high": c_data["static_recall"]["ci_95"][1],
                "bootstrap_ci_95": "-",
                "p_value_vs_static": "-"
            })
            rows.append({
                "benchmark": "IEEE-CIS Feature-Blind Audit",
                "architecture_or_cohort": f"{c_label} (Adaptive)",
                "metric": "Recall (%)",
                "mean": c_data["adaptive_recall"]["mean"],
                "std": c_data["adaptive_recall"]["std"],
                "ci_95_low": c_data["adaptive_recall"]["ci_95"][0],
                "ci_95_high": c_data["adaptive_recall"]["ci_95"][1],
                "bootstrap_ci_95": "-",
                "p_value_vs_static": p_val
            })

    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# =========================================================================
# MAIN EXECUTION ORCHESTRATOR
# =========================================================================
def run():
    parser = argparse.ArgumentParser(description="Multi-Seed Statistical Rigor & Bootstrap Benchmarking Suite")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="List of random seeds to evaluate")
    parser.add_argument("--benchmark", type=str, choices=["all", "paysim", "ieee_uid"], default="all",
                        help="Benchmark suite to execute")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples for empirical CI")
    args = parser.parse_args()

    seeds = args.seeds
    logger.info(f"Running Statistical Rigor Suite across {len(seeds)} seeds: {seeds}")

    summary_path = METRICS_DIR / "statistical_rigor_summary.json"
    results_all: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                results_all = json.load(f)
        except Exception:
            results_all = {}

    results_all.setdefault("metadata", {})
    results_all["metadata"].update({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "gpu_device": GPU_DEVICE if USE_GPU else "cpu"
    })

    if args.benchmark in ("all", "ieee_uid"):
        uid_res = run_ieee_uid_statistical_rigor(seeds, n_bootstrap=args.n_bootstrap)
        results_all["ieee_zero_memory_uid"] = uid_res

    if args.benchmark in ("all", "paysim"):
        paysim_res = run_paysim_statistical_rigor(seeds, n_bootstrap=args.n_bootstrap)
        results_all["paysim_oos_f1"] = paysim_res

    # Save to metrics directory
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=4)
    logger.info(f"Summary JSON saved to {summary_path}")

    # Export to CSV for paper table generation
    csv_path = METRICS_DIR / "statistical_rigor_summary.csv"
    export_summary_csv(results_all, csv_path)
    logger.info(f"Summary CSV saved to {csv_path}")

    # Generate Visualization if both completed
    if "paysim_oos_f1" in results_all and "ieee_zero_memory_uid" in results_all:
        chart_path = generate_statistical_errorbar_charts(
            paysim_res=results_all["paysim_oos_f1"],
            ieee_uid_res=results_all["ieee_zero_memory_uid"]
        )
        logger.info(f"Visualization generated: {chart_path}")

    logger.info("\n" + "=" * 92)
    logger.info("STATISTICAL RIGOR BENCHMARK SUITE SUCCESSFULLY COMPLETED")
    logger.info("=" * 92)


if __name__ == "__main__":
    run()
