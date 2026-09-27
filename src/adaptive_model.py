"""
FR-5: Drift-Adaptive Full Retraining Module on NVIDIA GPU.
Implements the full retraining strategy upon concept drift alerts (ADWIN/DDM).
Re-trains the entire model on the accumulated historical and recent streaming data
using GPU (CUDA) with histogram-based gradient boosting.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.base_model import BaseFraudModel

logger = logging.getLogger("fraud_drift.adaptive_model")


class DriftAdaptiveModel(BaseFraudModel):
    """
    FR-5: Drift-Adaptive Classification System using Full Retraining on NVIDIA GPU.
    Evaluates each time slice sequentially (test-then-train paradigm).
    When concept drift is flagged by FR-4 (ADWIN/DDM), triggers a FULL RETRAIN
    on the accumulated data stream using CUDA GPU acceleration.
    """

    def __init__(self, feature_names: List[str], scale_pos_weight: float = 1.0, random_seed: int = RANDOM_SEED):
        super().__init__(feature_names, scale_pos_weight=scale_pos_weight, random_seed=random_seed)
        self.adaptation_count = 0
        self.cumulative_history: List[pd.DataFrame] = []
        self.retrain_log: List[Dict[str, Any]] = []

    def fit_initial(self, train_df: pd.DataFrame):
        """
        Initializes the model on the baseline training window (Weeks 1-5) on NVIDIA GPU.
        """
        X = train_df[self.feature_names].values
        y = train_df["isFraud"].values

        logger.info(
            f"FR-5 Initial Training on {len(X):,} transactions ({int(np.sum(y))} fraud) on GPU..."
        )

        xgb_params = {
            "n_estimators": 120,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_seed,
            "eval_metric": "logloss"
        }

        self._fit_xgb(X, y, xgb_params, model_name="FR-5 Initial Model")

    def update_with_slice(
        self,
        slice_df: pd.DataFrame,
        drift_triggered: bool,
        slice_idx: int,
        training_df_override: Optional[pd.DataFrame] = None
    ):
        """
        FR-5.2 & FR-5.3: Updates cumulative history with the evaluated slice.
        If drift was flagged by FR-4 (ADWIN/DDM), triggers a FULL RETRAIN
        on the entire accumulated stream (or strictly matured data under delayed feedback).
        """
        # Accumulate into internal history only when not using an external training dataset override
        # (e.g. DelayedFeedbackQueue maintains its own leak-free matured training buffer).
        if training_df_override is None:
            self.cumulative_history.append(slice_df.copy())

        if drift_triggered:
            self.adaptation_count += 1
            start_t = time.time()

            # Assemble full training data (or strictly matured data under verification latency)
            if training_df_override is not None:
                full_history_df = training_df_override
            else:
                full_history_df = pd.concat(self.cumulative_history, ignore_index=True)
            X_all = full_history_df[self.feature_names].values
            y_all = full_history_df["isFraud"].values

            # Recalculate class imbalance weight dynamically for full historical dataset
            pos = int(np.sum(y_all))
            neg = len(y_all) - pos
            current_scale_weight = float(neg) / max(1.0, float(pos))

            logger.info(
                f"[FR-5 FULL RETRAIN #{self.adaptation_count}] Drift alert triggered in slice {slice_idx + 1}. "
                f"Executing FULL RETRAIN on {len(X_all):,} accumulated transactions ({pos} fraud) on GPU..."
            )

            xgb_params = {
                "n_estimators": 150,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "scale_pos_weight": current_scale_weight,
                "random_state": self.random_seed,
                "eval_metric": "logloss"
            }

            self._fit_xgb(X_all, y_all, xgb_params, model_name=f"[FR-5 FULL RETRAIN #{self.adaptation_count}]")

            duration = time.time() - start_t
            retrain_info = {
                "adaptation_num": self.adaptation_count,
                "slice_idx": slice_idx,
                "accumulated_samples": len(X_all),
                "fraud_count": pos,
                "duration_seconds": round(duration, 2),
                "device": self.device_used
            }
            self.retrain_log.append(retrain_info)
            logger.info(
                f"[FR-5 FULL RETRAIN COMPLETED] #{self.adaptation_count} finished in {duration:.2f}s "
                f"on {self.device_used}."
            )
