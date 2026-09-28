"""
Sliding Window Retraining Baseline Model Module (Academic Baseline 3).
Retrains the classification model on only the most recent W weeks of transactions
(e.g. W=4 weeks), discarding older historical data to simulate memory-bounded streaming.
Used to evaluate whether retaining full historical memory is superior to fixed-window forgetting
when non-stationary fraud patterns recur.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.base_model import BaseFraudModel, DEFAULT_XGB_HYPERPARAMS

logger = logging.getLogger("fraud_drift.sliding_model")


class SlidingWindowModel(BaseFraudModel):
    """
    Sliding Window Retraining Baseline.
    Evaluates each time slice sequentially (test-before-learn).
    Maintains a sliding queue of the most recent `window_slices` time slices.
    Retrains every `retrain_interval_slices` strictly on the data within the sliding window.
    """

    def __init__(
        self,
        feature_names: List[str],
        window_slices: int = 4,
        retrain_interval_slices: int = 4,
        scale_pos_weight: float = 1.0,
        random_seed: int = RANDOM_SEED
    ):
        super().__init__(feature_names, scale_pos_weight=scale_pos_weight, random_seed=random_seed)
        self.window_slices = window_slices
        self.retrain_interval_slices = retrain_interval_slices
        self.retrain_count = 0
        self.window_history: List[pd.DataFrame] = []
        self.retrain_log: List[Dict[str, Any]] = []
        self.out_of_sample_slices_seen = 0

    def fit_initial(self, train_df: pd.DataFrame):
        """
        Initializes the model on the baseline training partition using NVIDIA GPU.
        Seeds the sliding window with the baseline training data.
        """
        X = train_df[self.feature_names].values
        y = train_df["isFraud"].values

        logger.info(
            f"[Sliding Window Baseline] Initial Training on {len(X):,} transactions ({int(np.sum(y))} fraud) on GPU..."
        )

        xgb_params = {
            **DEFAULT_XGB_HYPERPARAMS,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_seed
        }

        self._fit_xgb(X, y, xgb_params, model_name="[Sliding Window Baseline] Initial Model")

    def update_with_slice(self, slice_df: pd.DataFrame, slice_idx: int, is_out_of_sample: bool = True):
        """
        Appends slice data to the sliding window, ejecting oldest slices beyond window_slices.
        If out-of-sample and interval reached, retrains strictly on window contents.
        """
        self.window_history.append(slice_df.copy())
        if len(self.window_history) > self.window_slices:
            self.window_history.pop(0)

        if is_out_of_sample:
            self.out_of_sample_slices_seen += 1
            if self.out_of_sample_slices_seen % self.retrain_interval_slices == 0:
                self._retrain(slice_idx)

    def _retrain(self, slice_idx: int):
        """Executes retrain strictly on recent window data."""
        self.retrain_count += 1
        start_t = time.time()

        window_df = pd.concat(self.window_history, ignore_index=True)
        X_win = window_df[self.feature_names].values
        y_win = window_df["isFraud"].values

        pos = int(np.sum(y_win))
        neg = len(y_win) - pos
        current_scale_weight = float(neg) / max(1.0, float(pos))

        logger.info(
            f"[Sliding Window Retrain #{self.retrain_count}] Retraining on {len(X_win):,} transactions "
            f"across last {len(self.window_history)} slices ({pos} fraud) on GPU..."
        )

        xgb_params = {
            **DEFAULT_XGB_HYPERPARAMS,
            "scale_pos_weight": current_scale_weight,
            "random_state": self.random_seed
        }

        self._fit_xgb(X_win, y_win, xgb_params, model_name=f"[Sliding Window Retrain #{self.retrain_count}]")

        duration = time.time() - start_t
        retrain_info = {
            "adaptation_num": self.retrain_count,
            "slice_idx": slice_idx,
            "accumulated_samples": len(X_win),
            "fraud_count": pos,
            "duration_seconds": round(duration, 2),
            "device": self.device_used
        }
        self.retrain_log.append(retrain_info)
        logger.info(
            f"[Sliding Window Retrain Completed] #{self.retrain_count} finished in {duration:.2f}s on {self.device_used}."
        )
