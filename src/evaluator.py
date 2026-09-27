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
        model_type: str
    ) -> Dict[str, Any]:
        """
        FR-6.1: Computes Precision, Recall, F1-score, and AUC-PR for a given time slice.
        Handles zero-division edge cases safely.
        """
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Average precision score (AUC-PR)
        # If there are no positive samples in the slice, AUC-PR is undefined
        if np.sum(y_true) > 0 and len(np.unique(y_true)) > 1:
            auc_pr = average_precision_score(y_true, y_prob)
        else:
            auc_pr = 0.0

        metric_entry = {
            "slice_index": slice_index,
            "slice_name": slice_name,
            "model_type": model_type,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "auc_pr": round(float(auc_pr), 4),
            "fraud_count": int(np.sum(y_true)),
            "total_count": len(y_true)
        }
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
        oos_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        FR-6.3: Computes overall whole-dataset and out-of-sample metrics for all active models.
        """
        def _calc_model_metrics(preds: np.ndarray, probs: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
            tp = int(((y == 1) & (preds == 1)).sum())
            fp = int(((y == 0) & (preds == 1)).sum())
            fn = int(((y == 1) & (preds == 0)).sum())
            tn = int(((y == 0) & (preds == 0)).sum())
            total_fraud = int(np.sum(y))
            return {
                "overall_precision": round(float(precision_score(y, preds, zero_division=0)), 4),
                "overall_recall": round(float(recall_score(y, preds, zero_division=0)), 4),
                "overall_f1": round(float(f1_score(y, preds, zero_division=0)), 4),
                "overall_auc_pr": round(float(average_precision_score(y, probs)), 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "total_frauds": total_fraud
            }

        summary = {
            "static_model": _calc_model_metrics(static_preds, static_probs, y_all),
            "adaptive_model": _calc_model_metrics(adaptive_preds, adaptive_probs, y_all)
        }

        if periodic_preds is not None and periodic_probs is not None:
            summary["periodic_model"] = _calc_model_metrics(periodic_preds, periodic_probs, y_all)

        if sliding_preds is not None and sliding_probs is not None:
            summary["sliding_model"] = _calc_model_metrics(sliding_preds, sliding_probs, y_all)

        # Pure out-of-sample evaluation if mask provided
        if oos_mask is not None and np.sum(oos_mask) > 0:
            y_oos = y_all[oos_mask]
            summary["out_of_sample_stream"] = {
                "static_model": _calc_model_metrics(static_preds[oos_mask], static_probs[oos_mask], y_oos),
                "adaptive_model": _calc_model_metrics(adaptive_preds[oos_mask], adaptive_probs[oos_mask], y_oos),
            }
            if periodic_preds is not None and periodic_probs is not None:
                summary["out_of_sample_stream"]["periodic_model"] = _calc_model_metrics(
                    periodic_preds[oos_mask], periodic_probs[oos_mask], y_oos
                )
            if sliding_preds is not None and sliding_probs is not None:
                summary["out_of_sample_stream"]["sliding_model"] = _calc_model_metrics(
                    sliding_preds[oos_mask], sliding_probs[oos_mask], y_oos
                )

        return summary

    def generate_comparison_plots(
        self,
        metrics_df: pd.DataFrame,
        drift_events: List[Dict[str, Any]],
        output_filename: str = "model_comparison_f1_aucpr.png"
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
                "label": "Blind Periodic Retrain (Every 4w)",
                "color": "#5CB85C",
                "marker": "D",
                "linestyle": "-."
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
            }
        }

        # Determine x-axis from any model present
        first_model = metrics_df["model_type"].iloc[0]
        ref_df = metrics_df[metrics_df["model_type"] == first_model].sort_values("slice_index")
        x = ref_df["slice_index"] + 1

        active_models = [m for m in ["static", "periodic", "sliding", "adaptive"] if m in metrics_df["model_type"].values]

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

        # Highlight baseline training window (Weeks 1-5)
        ax1.axvspan(1, 5, color="#EAEAEA", alpha=0.6, label="Baseline Window (Weeks 1-5)")
        ax2.axvspan(1, 5, color="#EAEAEA", alpha=0.6)

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
