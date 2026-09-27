"""
Sliding Window Retraining Baseline Model Module (Academic Baseline 3).
Retrains the classification model on only the most recent W weeks of transactions
(e.g. W=4 weeks), discarding older historical data to simulate memory-bounded streaming.
Used to evaluate whether retaining full historical memory is superior to fixed-window forgetting
when non-stationary fraud patterns recur.
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import USE_GPU, GPU_DEVICE, RANDOM_SEED

logger = logging.getLogger("fraud_drift.sliding_model")


class SlidingWindowModel:
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
        self.feature_names = feature_names
        self.window_slices = window_slices
        self.retrain_interval_slices = retrain_interval_slices
        self.scale_pos_weight = scale_pos_weight
        self.random_seed = random_seed
        self.model: Optional[xgb.XGBClassifier] = None
        self.device_used = "cpu"
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
            "n_estimators": 120,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_seed,
            "eval_metric": "logloss"
        }

        if USE_GPU:
            try:
                self.model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device=GPU_DEVICE)
                start_t = time.time()
                self.model.fit(X, y)
                duration = time.time() - start_t
                self.device_used = f"NVIDIA GPU ({GPU_DEVICE})"
                logger.info(f"[Sliding Window Baseline] Initial Model fitted on NVIDIA GPU in {duration:.2f}s.")
            except Exception as e:
                logger.warning(f"GPU initial fit failed ({e}). Falling back to CPU.")
                self.model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device="cpu")
                self.model.fit(X, y)
                self.device_used = "cpu"
        else:
            self.model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device="cpu")
            self.model.fit(X, y)
            self.device_used = "cpu"

    def predict(self, slice_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates current model on incoming slice before updating with it (test-before-learn).
        """
        if self.model is None:
            raise RuntimeError("Sliding window model must be fitted before predict() is called.")

        X = slice_df[self.feature_names].values
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs

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
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": current_scale_weight,
            "random_state": self.random_seed,
            "eval_metric": "logloss"
        }

        if USE_GPU:
            try:
                new_model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device=GPU_DEVICE)
                new_model.fit(X_win, y_win)
                self.model = new_model
                self.device_used = f"NVIDIA GPU ({GPU_DEVICE})"
            except Exception as e:
                logger.warning(f"GPU sliding retrain failed ({e}). Falling back to CPU.")
                new_model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device="cpu")
                new_model.fit(X_win, y_win)
                self.model = new_model
                self.device_used = "cpu"
        else:
            new_model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device="cpu")
            new_model.fit(X_win, y_win)
            self.model = new_model
            self.device_used = "cpu"

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
