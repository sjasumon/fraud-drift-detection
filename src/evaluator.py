"""
FR-6: Evaluation and Comparison Module.
Computes Precision, Recall, F1-score, and AUC-PR per time slice for both static
and adaptive models, produces tabular summaries, and renders publication-ready
visual comparison charts.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score
)

from src.config import FIGURES_DIR, METRICS_DIR

logger = logging.getLogger("fraud_drift.evaluator")


def compute_recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.01) -> Dict[str, float]:
    """
    Computes recall, precision, and operational decision threshold at a fixed target False Positive Rate (FPR).
    FPR = FP / (FP + TN).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    if total_pos == 0 or total_neg == 0:
        return {
            "target_fpr": float(target_fpr),
            "actual_fpr": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "threshold": 1.0
        }

    # Sort descending by predicted probability
    desc_indices = np.argsort(y_prob)[::-1]
    y_sorted = y_true[desc_indices]
    prob_sorted = y_prob[desc_indices]

    fps = np.cumsum(y_sorted == 0)
    tps = np.cumsum(y_sorted == 1)
    fprs = fps / float(total_neg)

    valid_indices = np.where(fprs <= target_fpr)[0]
    if len(valid_indices) == 0:
        return {
            "target_fpr": float(target_fpr),
            "actual_fpr": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "threshold": 1.0
        }

    best_idx = valid_indices[-1]
    rec = float(tps[best_idx] / total_pos)
    prec = float(tps[best_idx] / (best_idx + 1))
    actual_fpr = float(fprs[best_idx])
    threshold = float(prob_sorted[best_idx])

    return {
        "target_fpr": float(target_fpr),
        "actual_fpr": round(actual_fpr, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "threshold": round(threshold, 4)
    }


def compute_recall_at_alert_budget(y_true: np.ndarray, y_prob: np.ndarray, budget_ratio: float = 0.01) -> Dict[str, float]:
    """
    Computes fraud recall and precision under a fixed alert capacity budget (e.g. top 1% or 2% transactions).
    Simulates operational triage capacity in production fraud departments.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_samples = len(y_true)
    total_pos = int(np.sum(y_true == 1))

    if n_samples == 0 or total_pos == 0:
        return {
            "budget_ratio": float(budget_ratio),
            "alert_count": 0,
            "recall": 0.0,
            "precision": 0.0,
            "threshold": 1.0
        }

    k_alerts = max(1, int(np.ceil(budget_ratio * n_samples)))
    k_alerts = min(k_alerts, n_samples)

    desc_indices = np.argsort(y_prob)[::-1][:k_alerts]
    tps_budget = int(np.sum(y_true[desc_indices] == 1))

    rec = float(tps_budget / total_pos)
    prec = float(tps_budget / k_alerts)
    threshold = float(y_prob[desc_indices[-1]])

    return {
        "budget_ratio": float(budget_ratio),
        "alert_count": k_alerts,
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "threshold": round(threshold, 4)
    }


def compute_cost_weighted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    cost_per_investigation: float = 5.0
) -> Dict[str, float]:
    """
    Computes dollar-weighted fraud recall and net financial savings using transaction amounts.
    Addresses the limitation that standard unweighted recall treats a $5 fraud identically to a $50,000 fraud.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    amounts = np.asarray(amounts, dtype=float)

    total_fraud_dollars = float(np.sum(amounts[y_true == 1]))
    intercepted_dollars = float(np.sum(amounts[(y_true == 1) & (y_pred == 1)]))
    missed_dollars = float(np.sum(amounts[(y_true == 1) & (y_pred == 0)]))
    total_alerts = int(np.sum(y_pred == 1))
    investigation_cost = float(total_alerts * cost_per_investigation)
    net_savings = intercepted_dollars - investigation_cost

    dollar_recall = float(intercepted_dollars / total_fraud_dollars) if total_fraud_dollars > 0 else 0.0

    return {
        "total_fraud_dollars": round(total_fraud_dollars, 2),
        "intercepted_fraud_dollars": round(intercepted_dollars, 2),
        "missed_fraud_dollars": round(missed_dollars, 2),
        "dollar_recall": round(dollar_recall, 4),
        "total_alerts": total_alerts,
        "investigation_cost": round(investigation_cost, 2),
        "net_financial_savings": round(net_savings, 2)
    }


class SliceEvaluator:
    """Computes time-sliced fraud metrics and generates visual comparisons."""

    def __init__(self):
        self.metrics_log: List[Dict[str, Any]] = []

    def evaluate_slice(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        slice_index: int,
        slice_name: str,
        model_type: str,
        amounts: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        FR-6.1: Computes Precision, Recall, F1-score, AUC-PR, Alert-Budget Recall, and Cost Metrics.
        Handles zero-division edge cases safely.
        """
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Average precision score (AUC-PR)
        if np.sum(y_true) > 0 and len(np.unique(y_true)) > 1:
            auc_pr = average_precision_score(y_true, y_prob)
        else:
            auc_pr = 0.0

        # Operational metrics: alert budgets and fixed FPRs
        budget_01 = compute_recall_at_alert_budget(y_true, y_prob, budget_ratio=0.01)
        budget_02 = compute_recall_at_alert_budget(y_true, y_prob, budget_ratio=0.02)
        fpr_01 = compute_recall_at_fpr(y_true, y_prob, target_fpr=0.01)
        fpr_05 = compute_recall_at_fpr(y_true, y_prob, target_fpr=0.05)

        metric_entry = {
            "slice_index": slice_index,
            "slice_name": slice_name,
            "model_type": model_type,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "auc_pr": round(float(auc_pr), 4),
            "recall_at_alert_budget_1pct": budget_01["recall"],
            "precision_at_alert_budget_1pct": budget_01["precision"],
            "recall_at_alert_budget_2pct": budget_02["recall"],
            "precision_at_alert_budget_2pct": budget_02["precision"],
            "recall_at_fpr_1pct": fpr_01["recall"],
            "recall_at_fpr_5pct": fpr_05["recall"],
            "fraud_count": int(np.sum(y_true)),
            "total_count": len(y_true)
        }

        if amounts is not None and len(amounts) == len(y_true):
            cost_metrics = compute_cost_weighted_metrics(y_true, y_pred, amounts)
            metric_entry["dollar_recall"] = cost_metrics["dollar_recall"]
            metric_entry["intercepted_fraud_dollars"] = cost_metrics["intercepted_fraud_dollars"]
            metric_entry["net_financial_savings"] = cost_metrics["net_financial_savings"]

        self.metrics_log.append(metric_entry)
        return metric_entry

    def compute_overall_summary(
        self,
        static_preds: np.ndarray,
        static_probs: np.ndarray,
        adaptive_preds: np.ndarray,
        adaptive_probs: np.ndarray,
        y_all: np.ndarray,
        periodic_preds: Optional[np.ndarray] = None,
        periodic_probs: Optional[np.ndarray] = None,
        sliding_preds: Optional[np.ndarray] = None,
        sliding_probs: Optional[np.ndarray] = None,
        oos_mask: Optional[np.ndarray] = None,
        periodic_weekly_preds: Optional[np.ndarray] = None,
        periodic_weekly_probs: Optional[np.ndarray] = None,
        extra_models: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        amounts_all: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        FR-6.3: Computes overall whole-dataset and out-of-sample metrics for all active models,
        including fixed alert budgets, fixed FPRs, and cost-weighted dollar metrics.
        """
        def _calc_model_metrics(
            preds: np.ndarray,
            probs: np.ndarray,
            y: np.ndarray,
            amounts: Optional[np.ndarray] = None,
            scope_desc: str = "Combined (In-Sample Warmup + Out-of-Sample)"
        ) -> Dict[str, Any]:
            preds = np.asarray(preds)
            probs = np.asarray(probs)
            y = np.asarray(y)
            tp = int(((y == 1) & (preds == 1)).sum())
            fp = int(((y == 0) & (preds == 1)).sum())
            fn = int(((y == 1) & (preds == 0)).sum())
            tn = int(((y == 0) & (preds == 0)).sum())
            total_fraud = int(np.sum(y))

            if total_fraud > 0 and len(np.unique(y)) > 1:
                auc_pr = average_precision_score(y, probs)
            else:
                auc_pr = 0.0

            prec = round(float(precision_score(y, preds, zero_division=0)), 4)
            rec = round(float(recall_score(y, preds, zero_division=0)), 4)
            f1 = round(float(f1_score(y, preds, zero_division=0)), 4)
            auc = round(float(auc_pr), 4)
            is_pure_oos = "OOS Only" in scope_desc or "Pure Out-of-Sample" in scope_desc

            budget_01 = compute_recall_at_alert_budget(y, probs, budget_ratio=0.01)
            budget_02 = compute_recall_at_alert_budget(y, probs, budget_ratio=0.02)
            fpr_01 = compute_recall_at_fpr(y, probs, target_fpr=0.01)
            fpr_05 = compute_recall_at_fpr(y, probs, target_fpr=0.05)

            metrics = {
                "evaluation_scope": scope_desc,
                "is_pure_out_of_sample": is_pure_oos,
                "overall_precision": prec,
                "overall_recall": rec,
                "overall_f1": f1,
                "overall_auc_pr": auc,
                "recall_at_alert_budget_1pct": budget_01["recall"],
                "precision_at_alert_budget_1pct": budget_01["precision"],
                "recall_at_alert_budget_2pct": budget_02["recall"],
                "precision_at_alert_budget_2pct": budget_02["precision"],
                "recall_at_fpr_1pct": fpr_01["recall"],
                "recall_at_fpr_5pct": fpr_05["recall"],
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "total_frauds": total_fraud
            }

            if amounts is not None and len(amounts) == len(y):
                cost_m = compute_cost_weighted_metrics(y, preds, amounts)
                metrics["dollar_recall"] = cost_m["dollar_recall"]
                metrics["total_fraud_dollars"] = cost_m["total_fraud_dollars"]
                metrics["intercepted_fraud_dollars"] = cost_m["intercepted_fraud_dollars"]
                metrics["net_financial_savings"] = cost_m["net_financial_savings"]

            if not is_pure_oos:
                metrics["overall_f1_including_training_window"] = f1
                metrics["overall_auc_pr_including_training_window"] = auc
                metrics["metric_disambiguation_note"] = (
                    "DISAMBIGUATION: This metric is evaluated across all stream slices, including the initial training "
                    "window (in-sample + out-of-sample combined). Do not quote as pure out-of-sample generalization. "
                    "Refer to summary['out_of_sample_stream'] for strictly leak-free generalization results."
                )
            else:
                metrics["pure_oos_f1"] = f1
                metrics["pure_oos_auc_pr"] = auc
                metrics["metric_disambiguation_note"] = "Strictly out-of-sample test stream only (zero training slices included)."

            return metrics

        summary = {
            "static_model": _calc_model_metrics(static_preds, static_probs, y_all, amounts=amounts_all, scope_desc="Combined (In-Sample Warmup Slices + OOS)"),
            "adaptive_model": _calc_model_metrics(adaptive_preds, adaptive_probs, y_all, amounts=amounts_all, scope_desc="Combined (In-Sample Warmup Slices + OOS)")
        }

        if periodic_preds is not None and periodic_probs is not None:
            summary["periodic_model"] = _calc_model_metrics(periodic_preds, periodic_probs, y_all, amounts=amounts_all, scope_desc="Combined (In-Sample Warmup Slices + OOS)")

        if periodic_weekly_preds is not None and periodic_weekly_probs is not None:
            summary["periodic_weekly_model"] = _calc_model_metrics(periodic_weekly_preds, periodic_weekly_probs, y_all, amounts=amounts_all, scope_desc="Combined (In-Sample Warmup Slices + OOS)")

        if sliding_preds is not None and sliding_probs is not None:
            summary["sliding_model"] = _calc_model_metrics(sliding_preds, sliding_probs, y_all, amounts=amounts_all, scope_desc="Combined (In-Sample Warmup Slices + OOS)")

        if extra_models:
            for em_name, (em_preds, em_probs) in extra_models.items():
                summary[em_name] = _calc_model_metrics(em_preds, em_probs, y_all, amounts=amounts_all, scope_desc="Combined (In-Sample Warmup Slices + OOS)")

        # Pure out-of-sample evaluation if mask provided
        if oos_mask is not None and np.sum(oos_mask) > 0:
            oos_mask = np.asarray(oos_mask, dtype=bool)
            y_oos = np.asarray(y_all)[oos_mask]
            amounts_oos = np.asarray(amounts_all)[oos_mask] if amounts_all is not None else None
            summary["out_of_sample_stream"] = {
                "static_model": _calc_model_metrics(np.asarray(static_preds)[oos_mask], np.asarray(static_probs)[oos_mask], y_oos, amounts=amounts_oos, scope_desc="Pure Out-of-Sample (OOS Only)"),
                "adaptive_model": _calc_model_metrics(np.asarray(adaptive_preds)[oos_mask], np.asarray(adaptive_probs)[oos_mask], y_oos, amounts=amounts_oos, scope_desc="Pure Out-of-Sample (OOS Only)"),
            }
            if periodic_preds is not None and periodic_probs is not None:
                summary["out_of_sample_stream"]["periodic_model"] = _calc_model_metrics(
                    np.asarray(periodic_preds)[oos_mask], np.asarray(periodic_probs)[oos_mask], y_oos, amounts=amounts_oos,
                    scope_desc="Pure Out-of-Sample (OOS Only)"
                )
            if periodic_weekly_preds is not None and periodic_weekly_probs is not None:
                summary["out_of_sample_stream"]["periodic_weekly_model"] = _calc_model_metrics(
                    np.asarray(periodic_weekly_preds)[oos_mask], np.asarray(periodic_weekly_probs)[oos_mask], y_oos, amounts=amounts_oos,
                    scope_desc="Pure Out-of-Sample (OOS Only)"
                )
            if sliding_preds is not None and sliding_probs is not None:
                summary["out_of_sample_stream"]["sliding_model"] = _calc_model_metrics(
                    np.asarray(sliding_preds)[oos_mask], np.asarray(sliding_probs)[oos_mask], y_oos, amounts=amounts_oos,
                    scope_desc="Pure Out-of-Sample (OOS Only)"
                )
            if extra_models:
                for em_name, (em_preds, em_probs) in extra_models.items():
                    summary["out_of_sample_stream"][em_name] = _calc_model_metrics(
                        np.asarray(em_preds)[oos_mask], np.asarray(em_probs)[oos_mask], y_oos, amounts=amounts_oos,
                        scope_desc="Pure Out-of-Sample (OOS Only)"
                    )

        return summary

    def generate_comparison_plots(
        self,
        metrics_df: pd.DataFrame,
        drift_events: List[Dict[str, Any]],
        output_filename: str = "model_comparison_f1_aucpr.png",
        n_train_slices: int = 5
    ) -> Path:
        """
        FR-6.2: Generates a high-resolution time-series chart showing all evaluated models
        on the same axes for direct visual comparison, with detected concept drift markers.
        """
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

        model_styles = {
            "static": {
                "label": "Static Baseline (Frozen XGBoost)",
                "color": "#D9534F",
                "marker": "o",
                "linestyle": "--"
            },
            "periodic": {
                "label": "Periodic Retrain (4-Week)",
                "color": "#5CB85C",
                "marker": "D",
                "linestyle": "-."
            },
            "periodic_weekly": {
                "label": "Periodic Retrain (Weekly)",
                "color": "#E67E22",
                "marker": "v",
                "linestyle": ":"
            },
            "sliding": {
                "label": "Sliding Window Retrain (Last 4w)",
                "color": "#9B59B6",
                "marker": "^",
                "linestyle": ":"
            },
            "adaptive": {
                "label": "Drift-Adaptive Model (Proposed, ADWIN/DDM)",
                "color": "#0275D8",
                "marker": "s",
                "linestyle": "-"
            },
            "hat": {
                "label": "Hoeffding Adaptive Tree (HAT)",
                "color": "#17A2B8",
                "marker": "P",
                "linestyle": "-."
            },
            "arf": {
                "label": "Adaptive Random Forest (ARF)",
                "color": "#20C997",
                "marker": "*",
                "linestyle": "-"
            },
            "srp": {
                "label": "Streaming Random Patches (SRP)",
                "color": "#6F42C1",
                "marker": "X",
                "linestyle": "--"
            }
        }

        # Determine x-axis from any model present
        first_model = metrics_df["model_type"].iloc[0]
        ref_df = metrics_df[metrics_df["model_type"] == first_model].sort_values("slice_index")
        x = ref_df["slice_index"] + 1

        active_models = [m for m in model_styles.keys() if m in metrics_df["model_type"].values]

        for m in active_models:
            m_df = metrics_df[metrics_df["model_type"] == m].sort_values("slice_index")
            cfg = model_styles.get(m, {"label": m, "color": "gray", "marker": "x", "linestyle": "-"})
            # Panel 1: F1-score
            ax1.plot(
                x, m_df["f1_score"],
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                linestyle=cfg["linestyle"],
                linewidth=2.2,
                markersize=6
            )
            # Panel 2: AUC-PR
            ax2.plot(
                x, m_df["auc_pr"],
                label=f"{cfg['label']} AUC-PR",
                color=cfg["color"],
                marker=cfg["marker"],
                linestyle=cfg["linestyle"],
                linewidth=2.0,
                markersize=6
            )

        # Highlight baseline training window dynamically
        ax1.axvspan(1, n_train_slices, color="#EAEAEA", alpha=0.6, label=f"Baseline Window (Slices 1–{n_train_slices})")
        ax2.axvspan(1, n_train_slices, color="#EAEAEA", alpha=0.6)

        # Mark drift alerts
        drift_slices = set(e["slice_index"] + 1 for e in drift_events)
        for ds in drift_slices:
            ax1.axvline(x=ds, color="#F0AD4E", linestyle=":", alpha=0.7)
            ax2.axvline(x=ds, color="#F0AD4E", linestyle=":", alpha=0.7)

        if drift_slices:
            ax1.plot([], [], color="#F0AD4E", linestyle=":", label="Drift Alerts (ADWIN/DDM)")

        ax1.set_ylabel("F1-Score", fontsize=12, fontweight="bold")
        ax1.set_title("Multi-Regime Streaming Performance Comparison Under Concept Drift", fontsize=14, fontweight="bold")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc="lower left", frameon=True, fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.set_xlabel("Time Slices (Sequential Weeks)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("AUC-PR (Precision-Recall Area)", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"W{int(i)}" for i in x], rotation=45)
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc="lower left", frameon=True, fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_path = FIGURES_DIR / output_filename
        plt.savefig(out_path, dpi=300)
        plt.close()

        logger.info(f"FR-6.2: Multi-model comparison chart saved to {out_path}")
        return out_path
