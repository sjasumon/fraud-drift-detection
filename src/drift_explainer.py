"""
SHAP Drift Attribution & Mechanistic Explainability Module (P4).
Computes TreeSHAP feature importances and Population Stability Index (PSI)
to explain WHAT features drove concept drift events flagged by ADWIN/DDM.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import FIGURES_DIR

logger = logging.getLogger("fraud_drift.explainer")


def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and current distributions.
    PSI < 0.1: No significant change.
    0.1 <= PSI < 0.25: Moderate drift.
    PSI >= 0.25: Significant concept drift.
    """
    baseline = baseline[~np.isnan(baseline)]
    target = target[~np.isnan(target)]

    if len(baseline) == 0 or len(target) == 0 or np.all(baseline == baseline[0]):
        return 0.0

    # Calculate quantile bin edges from baseline
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate bin counts
    b_counts, _ = np.histogram(baseline, bins=bin_edges)
    t_counts, _ = np.histogram(target, bins=bin_edges)

    # Convert to proportions with smoothing epsilon
    p_b = (b_counts / float(len(baseline))) + epsilon
    p_t = (t_counts / float(len(target))) + epsilon

    # Normalize back to 1
    p_b /= p_b.sum()
    p_t /= p_t.sum()

    # Calculate PSI formula
    psi_val = np.sum((p_t - p_b) * np.log(p_t / p_b))
    return float(max(0.0, psi_val))


class DriftAttributionEngine:
    """
    Mechanistic drift explainer using TreeSHAP and Population Stability Index (PSI).
    Ranks features by combined SHAP importance delta and empirical distribution divergence.
    """

    def __init__(self, feature_names: List[str], max_shap_samples: int = 500):
        self.feature_names = feature_names
        self.max_shap_samples = max_shap_samples

    def explain_drift_slice(
        self,
        model,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        slice_index: int,
        slice_name: str,
        detector_type: str = "ADWIN",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Explains concept drift at slice_index by isolating the top_k feature drivers.
        """
        logger.info(f"[Drift Attribution] Explaining drift in {slice_name} (Slice {slice_index + 1})...")

        # 1. Sample current data for efficient TreeSHAP
        sample_size = min(self.max_shap_samples, len(current_df))
        sample_indices = np.random.choice(len(current_df), size=sample_size, replace=False)
        X_sample = current_df[self.feature_names].iloc[sample_indices].values

        # 2. TreeSHAP computation
        try:
            # Native TreeExplainer on XGBoost Booster
            explainer = shap.TreeExplainer(model.model if hasattr(model, "model") else model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive (fraud) class
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        except Exception as e:
            logger.warning(f"TreeSHAP calculation warning ({e}). Falling back to feature importances.")
            if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
                mean_abs_shap = model.model.feature_importances_
            else:
                mean_abs_shap = np.ones(len(self.feature_names)) / len(self.feature_names)

        # 3. PSI Calculation across all numerical candidate features
        psi_scores = []
        for i, feat in enumerate(self.feature_names):
            if feat in baseline_df.columns and feat in current_df.columns:
                b_arr = baseline_df[feat].values.astype(float)
                c_arr = current_df[feat].values.astype(float)
                psi_score = calculate_psi(b_arr, c_arr)
            else:
                psi_score = 0.0
            psi_scores.append(psi_score)

        psi_scores = np.array(psi_scores)

        # 4. Composite Attribution Score = SHAP_Importance * (1 + PSI)
        attribution_scores = mean_abs_shap * (1.0 + psi_scores)

        # 5. Extract Top-K Features
        top_indices = np.argsort(attribution_scores)[::-1][:top_k]

        attributions = []
        for rank, idx in enumerate(top_indices, start=1):
            attributions.append({
                "slice_index": slice_index,
                "slice_name": slice_name,
                "detector_type": detector_type,
                "feature_name": self.feature_names[idx],
                "psi_score": round(float(psi_scores[idx]), 4),
                "shap_importance": round(float(mean_abs_shap[idx]), 4),
                "attribution_score": round(float(attribution_scores[idx]), 4),
                "rank_order": rank
            })

        logger.info(
            f"[Drift Attribution Top Driver] {slice_name}: "
            f"#{1} {attributions[0]['feature_name']} (PSI={attributions[0]['psi_score']:.3f}, SHAP={attributions[0]['shap_importance']:.3f})"
        )
        return attributions

    def render_attribution_plot(
        self,
        attributions: List[Dict[str, Any]],
        slice_name: str,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Renders a publication-ready horizontal waterfall attribution bar chart.
        """
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(10, 5))

        df_attr = pd.DataFrame(attributions).sort_values("attribution_score", ascending=True)
        y_pos = np.arange(len(df_attr))

        colors = ["#D9534F" if psi > 0.25 else "#0275D8" for psi in df_attr["psi_score"]]

        bars = ax.barh(y_pos, df_attr["attribution_score"], color=colors, alpha=0.85, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_attr["feature_name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Composite Drift Driver Score [SHAP Magnitude × (1 + PSI)]", fontsize=11, fontweight="bold")
        ax.set_title(f"Concept Drift Mechanistic Attribution: Top Drivers ({slice_name})", fontsize=13, fontweight="bold")

        # Annotate bars with PSI and SHAP values
        for bar, (_, row) in zip(bars, df_attr.iterrows()):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"PSI={row['psi_score']:.2f}, SHAP={row['shap_importance']:.3f}",
                va="center",
                fontsize=9,
                color="#333333"
            )

        plt.tight_layout()
        if output_filename is None:
            output_filename = f"drift_attribution_{slice_name.lower()}.png"
        out_path = FIGURES_DIR / output_filename
        plt.savefig(out_path, dpi=300)
        plt.close()

        logger.info(f"[Drift Attribution Plot] Chart saved to {out_path}")
        return out_path
