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
import xgboost as xgb

from src.config import USE_GPU, GPU_DEVICE, RANDOM_SEED

logger = logging.getLogger("fraud_drift.periodic_model")


class PeriodicRetrainModel:
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
        self.feature_names = feature_names
        self.retrain_interval_slices = retrain_interval_slices
        self.scale_pos_weight = scale_pos_weight
        self.random_seed = random_seed
        self.model: Optional[xgb.XGBClassifier] = None
        self.device_used = "cpu"
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
                logger.info(f"[Periodic Baseline] Initial Model fitted on NVIDIA GPU in {duration:.2f}s.")
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
            raise RuntimeError("Periodic model must be fitted before predict() is called.")

        X = slice_df[self.feature_names].values
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs

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
                new_model.fit(X_all, y_all)
                self.model = new_model
                self.device_used = f"NVIDIA GPU ({GPU_DEVICE})"
            except Exception as e:
                logger.warning(f"GPU periodic retrain failed ({e}). Falling back to CPU.")
                new_model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device="cpu")
                new_model.fit(X_all, y_all)
                self.model = new_model
                self.device_used = "cpu"
        else:
            new_model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device="cpu")
            new_model.fit(X_all, y_all)
            self.model = new_model
            self.device_used = "cpu"

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
