"""
Periodic Retraining Baseline Model Module (Academic Baseline 2).
Retrains the classification model on all accumulated history on a fixed schedule
(every K weeks, e.g. K=4) regardless of whether concept drift occurred.
Used to demonstrate whether drift-informed retraining provides compute and performance
advantages over blind scheduled retraining.
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.base_model import BaseFraudModel, DEFAULT_XGB_HYPERPARAMS

logger = logging.getLogger("fraud_drift.periodic_model")


class PeriodicRetrainModel(BaseFraudModel):
    """
    Blind Periodic Retraining Baseline.
    Evaluates each time slice sequentially (test-before-learn).
    Retrains on the entire accumulated stream every `retrain_interval_slices`
    out-of-sample slices regardless of statistical drift signals.
    """

    def __init__(
        self,
        feature_names: List[str],
        retrain_interval_slices: int = 4,
        scale_pos_weight: float = 1.0,
        random_seed: int = RANDOM_SEED
    ):
        super().__init__(feature_names, scale_pos_weight=scale_pos_weight, random_seed=random_seed)
        self.retrain_interval_slices = retrain_interval_slices
        self.retrain_count = 0
        self.cumulative_history: List[pd.DataFrame] = []
        self.retrain_log: List[Dict[str, Any]] = []
        self.out_of_sample_slices_seen = 0

    def fit_initial(self, train_df: pd.DataFrame):
        """
        Initializes the model on the baseline training partition using NVIDIA GPU.
        """
        X = train_df[self.feature_names].values
        y = train_df["isFraud"].values

        logger.info(
            f"[Periodic Baseline] Initial Training on {len(X):,} transactions ({int(np.sum(y))} fraud) on GPU..."
        )

        xgb_params = {
            **DEFAULT_XGB_HYPERPARAMS,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_seed
        }

        self._fit_xgb(X, y, xgb_params, model_name="[Periodic Baseline] Initial Model")

    def update_with_slice(self, slice_df: pd.DataFrame, slice_idx: int, is_out_of_sample: bool = True):
        """
        Appends slice data to cumulative history.
        If out-of-sample and interval reached, retrains on all accumulated history.
        """
        self.cumulative_history.append(slice_df.copy())

        if is_out_of_sample:
            self.out_of_sample_slices_seen += 1
            if self.out_of_sample_slices_seen % self.retrain_interval_slices == 0:
                self._retrain(slice_idx)

    def _retrain(self, slice_idx: int):
        """Executes full retrain on all accumulated history."""
        self.retrain_count += 1
        start_t = time.time()

        full_history_df = pd.concat(self.cumulative_history, ignore_index=True)
        X_all = full_history_df[self.feature_names].values
        y_all = full_history_df["isFraud"].values

        pos = int(np.sum(y_all))
        neg = len(y_all) - pos
        current_scale_weight = float(neg) / max(1.0, float(pos))

        logger.info(
            f"[Periodic Retrain #{self.retrain_count}] Scheduled interval reached at slice {slice_idx + 1}. "
            f"Retraining on {len(X_all):,} accumulated transactions ({pos} fraud) on GPU..."
        )

        xgb_params = {
            **DEFAULT_XGB_HYPERPARAMS,
            "scale_pos_weight": current_scale_weight,
            "random_state": self.random_seed
        }

        self._fit_xgb(X_all, y_all, xgb_params, model_name=f"[Periodic Retrain #{self.retrain_count}]")
        duration = time.time() - start_t
        retrain_info = {
            "adaptation_num": self.retrain_count,
            "slice_idx": slice_idx,
            "accumulated_samples": len(X_all),
            "fraud_count": pos,
            "duration_seconds": round(duration, 2),
            "device": self.device_used
        }
        self.retrain_log.append(retrain_info)
        logger.info(
            f"[Periodic Retrain Completed] #{self.retrain_count} finished in {duration:.2f}s on {self.device_used}."
        )
