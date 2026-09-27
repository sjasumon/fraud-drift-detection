"""
Empirical Delayed Feedback & Verification Latency Audit Module.

Evaluates concept-drift-adaptive fraud detection under realistic operational label delays:
1. Immediate Feedback (0 Weeks / Idealized baseline)
2. Standard Operational Delay (4 Weeks / 28 Days)
3. Extended Dispute Resolution (8 Weeks / 56 Days)
4. Stochastic LogNormal Latency (Median ~4 Weeks)

Measures the impact of verification latency on:
- Out-of-Sample Recall & AUC-PR
- Drift Detection Latency & Retrain Timing
- Quantifiable Degradation relative to the Immediate Verification Assumption
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from src.config import (
    BASE_DIR, DATA_DIR, RESULTS_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR,
    TRAINING_WINDOW_RATIO, RANDOM_SEED, USE_GPU, GPU_DEVICE
)
from src.ingestion import DatasetIngestion
from src.preprocessing import Preprocessor
from src.adaptive_model import DriftAdaptiveModel
from src.drift_detector import DriftMonitor
from src.delayed_feedback import DelayedFeedbackQueue

logger = logging.getLogger("fraud_drift.delayed_audit")


def run_latency_simulation_regime(
    regime_name: str,
    time_slices: List[Dict[str, Any]],
    n_train_slices: int,
    train_df: pd.DataFrame,
    feature_names: List[str],
    scale_pos_weight: float,
    delay_slices: int = 4,
    latency_mode: str = "fixed",
    lognormal_mu: float = 1.38,
    lognormal_sigma: float = 0.45
) -> Dict[str, Any]:
    """
    Executes a single 26-week streaming simulation under a specific verification latency regime.
    """
    logger.info(f"\n{'='*70}\n[REGIME: {regime_name}] Starting 26-week stream (Delay={delay_slices}w, Mode={latency_mode})...\n{'='*70}")

    queue = DelayedFeedbackQueue(
        delay_slices=delay_slices,
        latency_mode=latency_mode,
        lognormal_mu=lognormal_mu,
        lognormal_sigma=lognormal_sigma,
        random_seed=RANDOM_SEED
    )

    adaptive_model = DriftAdaptiveModel(
        feature_names=feature_names,
        scale_pos_weight=scale_pos_weight
    )
    adaptive_model.fit_initial(train_df)

    drift_monitor = DriftMonitor()

    slice_metrics: List[Dict[str, Any]] = []
    all_preds: List[int] = []
    all_probs: List[float] = []
    all_y_true: List[int] = []

    start_regime_time = time.time()

    for s_idx, slice_dict in enumerate(time_slices):
        slice_data = slice_dict["data"]
        slice_name = slice_dict["slice_name"]
        y_slice = slice_data["isFraud"].values
        is_oos = s_idx >= n_train_slices

        # 1. Immediate Inference (Test-Before-Learn)
        preds, probs = adaptive_model.predict(slice_data)

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_y_true.extend(y_slice)

        tp = int(np.sum((preds == 1) & (y_slice == 1)))
        fp = int(np.sum((preds == 1) & (y_slice == 0)))
        fn = int(np.sum((preds == 0) & (y_slice == 1)))
        tn = int(np.sum((preds == 0) & (y_slice == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        slice_metrics.append({
            "slice_index": s_idx,
            "slice_name": slice_name,
            "is_out_of_sample": is_oos,
            "total_transactions": len(slice_data),
            "actual_frauds": int(np.sum(y_slice)),
            "detected_frauds": tp,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })

        # 2. Enqueue unverified transactions
        queue.enqueue(slice_data, current_slice=s_idx)

        # 3. Dequeue matured transactions whose dispute/chargeback window has elapsed
        matured_df = queue.dequeue_matured(current_slice=s_idx)

        # 4. Supervised Drift Monitoring on newly matured transactions
        slice_drift_triggered = False
        if len(matured_df) > 0:
            matured_preds, _ = adaptive_model.predict(matured_df)
            m_true = matured_df["isFraud"].values

            for m_i in range(len(matured_df)):
                alert = drift_monitor.update(
                    y_true=int(m_true[m_i]),
                    y_pred=int(matured_preds[m_i]),
                    slice_index=s_idx,
                    step_number=int(matured_df["step"].iloc[m_i]),
                    trans_id=matured_df["trans_id"].iloc[m_i]
                )
                if alert["any_drift"]:
                    slice_drift_triggered = True

        # 5. Model Adaptation strictly on verified matured data
        matured_train_data = queue.get_all_matured_training_data(initial_train_df=train_df)
        adaptive_model.update_with_slice(
            slice_df=slice_data,
            drift_triggered=slice_drift_triggered,
            slice_idx=s_idx,
            training_df_override=matured_train_data
        )
        if slice_drift_triggered:
            drift_monitor.reset_detectors()

        drift_flag = " [RETRAIN]" if slice_drift_triggered else ""
        logger.info(
            f"[{regime_name}] Slice {slice_name} ({s_idx+1:02d}/26): "
            f"F1={f1:.4f} | Recall={rec*100:.1f}% | Matured={len(matured_df):,} | Pending={queue.get_pending_count():,}{drift_flag}"
        )

    # Calculate overall Out-Of-Sample (OOS) performance
    all_y_true = np.array(all_y_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    oos_mask = np.array([s["slice_index"] >= n_train_slices for s in time_slices for _ in range(len(s["data"]))])

    oos_y = all_y_true[oos_mask]
    oos_p = all_preds[oos_mask]
    oos_prob = all_probs[oos_mask]

    oos_tp = int(np.sum((oos_p == 1) & (oos_y == 1)))
    oos_fp = int(np.sum((oos_p == 1) & (oos_y == 0)))
    oos_fn = int(np.sum((oos_p == 0) & (oos_y == 1)))
    oos_tn = int(np.sum((oos_p == 0) & (oos_y == 0)))

    p_curve, r_curve, _ = precision_recall_curve(oos_y, oos_prob)
    oos_aucpr = float(auc(r_curve, p_curve))

    oos_prec = oos_tp / (oos_tp + oos_fp) if (oos_tp + oos_fp) > 0 else 0.0
    oos_rec = oos_tp / (oos_tp + oos_fn) if (oos_tp + oos_fn) > 0 else 0.0
    oos_f1 = (2 * oos_prec * oos_rec) / (oos_prec + oos_rec) if (oos_prec + oos_rec) > 0 else 0.0

    duration = time.time() - start_regime_time
    logger.info(
        f"[{regime_name} COMPLETE] OOS Recall: {oos_rec*100:.2f}% ({oos_tp:,}/{len(oos_y):,}) | "
        f"AUC-PR: {oos_aucpr:.4f} | F1: {oos_f1:.4f} | Retrains: {adaptive_model.adaptation_count} | Runtime: {duration:.1f}s"
    )

    return {
        "regime_name": regime_name,
        "delay_slices": delay_slices,
        "latency_mode": latency_mode,
        "retrains_count": adaptive_model.adaptation_count,
        "drift_alerts_count": len(drift_monitor.drift_events),
        "total_matured_transactions": queue.get_matured_count(),
        "total_pending_transactions": queue.get_pending_count(),
        "runtime_seconds": duration,
        "oos_tp": oos_tp,
        "oos_fp": oos_fp,
        "oos_fn": oos_fn,
        "oos_tn": oos_tn,
        "oos_precision": oos_prec,
        "oos_recall": oos_rec,
        "oos_f1": oos_f1,
        "oos_aucpr": oos_aucpr,
        "slice_metrics": slice_metrics
    }


def generate_latency_audit_plots(
    results: List[Dict[str, Any]],
    output_filename: str = "delayed_feedback_impact.png"
) -> Path:
    """
    Renders a publication-grade figure illustrating the empirical impact
    of verification latency on streaming fraud detection performance.
    """
    output_path = FIGURES_DIR / output_filename
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)

    colors = {
        "Immediate Feedback (0w)": "#1f77b4",
        "Standard Delay (4w)": "#2ca02c",
        "Extended Delay (8w)": "#d62728",
        "Stochastic LogNormal Delay": "#9467bd"
    }

    # Panel 1: Out-of-Sample F1 Score Across Slices
    ax1 = axes[0, 0]
    for res in results:
        name = res["regime_name"]
        color = colors.get(name, "#333333")
        slices = [m["slice_index"] for m in res["slice_metrics"] if m["is_out_of_sample"]]
        f1_vals = [m["f1_score"] for m in res["slice_metrics"] if m["is_out_of_sample"]]
        ax1.plot(slices, f1_vals, marker="o", markersize=4, linewidth=2, label=name, color=color)

    ax1.set_title("A. Out-of-Sample F1-Score Trajectory under Verification Latency", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time Slice (Weeks)", fontsize=10)
    ax1.set_ylabel("F1 Score", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Out-of-Sample Recall Across Slices
    ax2 = axes[0, 1]
    for res in results:
        name = res["regime_name"]
        color = colors.get(name, "#333333")
        slices = [m["slice_index"] for m in res["slice_metrics"] if m["is_out_of_sample"]]
        rec_vals = [m["recall"] * 100 for m in res["slice_metrics"] if m["is_out_of_sample"]]
        ax2.plot(slices, rec_vals, marker="s", markersize=4, linewidth=2, label=name, color=color)

    ax2.set_title("B. Out-of-Sample Recall (%) Trajectory under Verification Latency", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time Slice (Weeks)", fontsize=10)
    ax2.set_ylabel("Recall (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9)

    # Panel 3: Aggregate OOS Recall & AUC-PR Comparison (Bar Chart)
    ax3 = axes[1, 0]
    regime_names = [r["regime_name"] for r in results]
    rec_values = [r["oos_recall"] * 100 for r in results]
    auc_values = [r["oos_aucpr"] * 100 for r in results]

    x = np.arange(len(regime_names))
    width = 0.35

    rects1 = ax3.bar(x - width/2, rec_values, width, label="OOS Recall (%)", color="#1f77b4")
    rects2 = ax3.bar(x + width/2, auc_values, width, label="OOS AUC-PR (x100)", color="#ff7f0e")

    ax3.set_title("C. Out-of-Sample Recall and AUC-PR Comparison", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.replace(" ", "\n") for n in regime_names], fontsize=9)
    ax3.set_ylabel("Score (%)", fontsize=10)
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for rect in rects2:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Panel 4: Retraining Count vs. Fraud Catch Trade-Off
    ax4 = axes[1, 1]
    frauds_caught = [r["oos_tp"] for r in results]
    retrain_counts = [r["retrains_count"] for r in results]

    scatter_colors = [colors.get(n, "#333333") for n in regime_names]
    scatter = ax4.scatter(retrain_counts, frauds_caught, c=scatter_colors, s=150, zorder=3)

    for i, txt in enumerate(regime_names):
        ax4.annotate(
            f" {txt}\n ({frauds_caught[i]:,} frauds, {retrain_counts[i]} retrains)",
            (retrain_counts[i], frauds_caught[i]),
            fontsize=8,
            ha="left",
            va="center"
        )

    ax4.set_title("D. Operational Trade-Off: Retrain Frequency vs. Frauds Intercepted", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Number of GPU Retrains Triggered", fontsize=10)
    ax4.set_ylabel("Out-of-Sample Frauds Intercepted (TP)", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Latency audit plot saved cleanly to {output_path}")
    return output_path


def main(data_file_path: Optional[str] = None):
    """Entry point for running the delayed feedback audit."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if data_file_path is not None:
        csv_path = Path(data_file_path)
    else:
        candidate_paths = [
            Path("data/ieee/train_transaction.csv"),
            Path("ieee/train_transaction.csv"),
            DATA_DIR / "ieee" / "train_transaction.csv"
        ]
        csv_path = None
        for p in candidate_paths:
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            raise FileNotFoundError(f"IEEE-CIS dataset not found at {[str(p) for p in candidate_paths]}")

    logger.info(f"Loading IEEE-CIS dataset for latency audit from: {csv_path}")
    ingestion = DatasetIngestion(csv_path)
    df_raw, validation_report = ingestion.load_and_validate()

    preprocessor = Preprocessor()
    raw_slices = preprocessor.partition_time_slices(df_raw)

    n_train_slices = max(1, int(len(raw_slices) * TRAINING_WINDOW_RATIO))
    train_raw_df = pd.concat([s["data"] for s in raw_slices[:n_train_slices]], ignore_index=True)

    preprocessor.fit(train_raw_df)
    imbalance_info = preprocessor.imbalance_weights

    time_slices = []
    for s in raw_slices:
        s_transformed = s.copy()
        s_transformed["data"] = preprocessor.transform(s["data"])
        time_slices.append(s_transformed)

    train_slices = time_slices[:n_train_slices]
    train_df = pd.concat([s["data"] for s in train_slices], ignore_index=True)
    feature_names = preprocessor.feature_names
    scale_pos_weight = imbalance_info["scale_pos_weight"]

    # Define the 4 experimental latency regimes
    regimes = [
        {
            "name": "Immediate Feedback (0w)",
            "delay_slices": 0,
            "mode": "fixed"
        },
        {
            "name": "Standard Delay (4w)",
            "delay_slices": 4,
            "mode": "fixed"
        },
        {
            "name": "Extended Delay (8w)",
            "delay_slices": 8,
            "mode": "fixed"
        },
        {
            "name": "Stochastic LogNormal Delay",
            "delay_slices": 4,
            "mode": "lognormal",
            "lognormal_mu": 1.38,
            "lognormal_sigma": 0.45
        }
    ]

    all_results = []
    for reg in regimes:
        res = run_latency_simulation_regime(
            regime_name=reg["name"],
            time_slices=time_slices,
            n_train_slices=n_train_slices,
            train_df=train_df,
            feature_names=feature_names,
            scale_pos_weight=scale_pos_weight,
            delay_slices=reg["delay_slices"],
            latency_mode=reg["mode"],
            lognormal_mu=reg.get("lognormal_mu", 1.38),
            lognormal_sigma=reg.get("lognormal_sigma", 0.45)
        )
        all_results.append(res)

    # Save summary metrics
    summary_path = METRICS_DIR / "delayed_feedback_benchmark.json"
    clean_results = []
    for r in all_results:
        clean_r = {k: v for k, v in r.items() if k != "slice_metrics"}
        clean_results.append(clean_r)

    with open(summary_path, "w") as f:
        json.dump(clean_results, f, indent=4)
    logger.info(f"Summary metrics saved to {summary_path}")

    # Generate visualization
    plot_path = generate_latency_audit_plots(all_results, "delayed_feedback_impact.png")

    # Print publication-ready comparison table
    logger.info("\n" + "=" * 96)
    logger.info("EMPIRICAL VERIFICATION LATENCY BENCHMARK COMPLETE — FINAL SUMMARY")
    logger.info("=" * 96)
    logger.info(f"{'Verification Regime':<28} | {'Retrains':<8} | {'OOS Recall':<12} | {'OOS AUC-PR':<11} | {'OOS F1':<8} | {'Frauds Caught':<14}")
    logger.info("-" * 96)
    for r in all_results:
        rec_str = f"{r['oos_recall']*100:.2f}%"
        frauds_str = f"{r['oos_tp']:,} / {r['oos_tp']+r['oos_fn']:,}"
        logger.info(
            f"{r['regime_name']:<28} | {r['retrains_count']:<8} | {rec_str:<12} | "
            f"{r['oos_aucpr']:<11.4f} | {r['oos_f1']:<8.4f} | {frauds_str:<14}"
        )
    logger.info("=" * 96)


if __name__ == "__main__":
    cli_data_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(cli_data_path)
