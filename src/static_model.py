"""
FR-3: Static Baseline Model Module.
Trains a frozen classification model on the initial temporal window (earliest 20%),
accelerated by NVIDIA eGPU (CUDA), and evaluates across all subsequent time slices
without retraining or parameter updates.
"""

import time
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import USE_GPU, GPU_DEVICE, RANDOM_SEED

logger = logging.getLogger("fraud_drift.static_model")


class StaticBaselineModel:
    """
    FR-3: Static baseline classification model.
    Trained once on the initial 20% temporal window (e.g. Weeks 1-5) and frozen.
    """

    def __init__(self, feature_names: List[str], scale_pos_weight: float = 1.0, random_seed: int = RANDOM_SEED):
        self.feature_names = feature_names
        self.scale_pos_weight = scale_pos_weight
        self.random_seed = random_seed
        self.model = None
        self.trained = False
        self.device_used = "cpu"
        self.train_duration_sec = 0.0

    def fit(self, train_df: pd.DataFrame):
        """
        FR-3.1: Trains model strictly on the initial training window.
        Leverages NVIDIA GPU acceleration if available.
        """
        X = train_df[self.feature_names].values
        y = train_df["isFraud"].values

        logger.info(f"Training FR-3 Static Model on {len(X):,} samples ({np.sum(y)} fraud instances)...")

        # Configure GPU parameters
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_seed,
            "eval_metric": "logloss"
        }

        if USE_GPU:
            try:
                # Test CUDA device on RTX 5080 eGPU
                self.model = xgb.XGBClassifier(
                    **xgb_params,
                    tree_method="hist",
                    device=GPU_DEVICE
                )
                start_t = time.time()
                self.model.fit(X, y)
                self.train_duration_sec = time.time() - start_t
                self.device_used = f"NVIDIA GPU ({GPU_DEVICE})"
                logger.info(f"Static Model fitted successfully on NVIDIA eGPU in {self.train_duration_sec:.2f}s.")
            except Exception as e:
                logger.warning(f"GPU training failed ({e}). Falling back to CPU.")
                self.model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device="cpu")
                start_t = time.time()
                self.model.fit(X, y)
                self.train_duration_sec = time.time() - start_t
                self.device_used = "cpu"
        else:
            self.model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device="cpu")
            start_t = time.time()
            self.model.fit(X, y)
            self.train_duration_sec = time.time() - start_t
            self.device_used = "cpu"

        self.trained = True

    def predict(self, slice_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        FR-3.3: Applies frozen model to an evaluation time slice.
        Returns predicted binary labels and predicted fraud probabilities.
        """
        if not self.trained or self.model is None:
            raise RuntimeError("Static model must be fitted before predict() can be called.")

        X = slice_df[self.feature_names].values
        # Probabilities for class 1 (fraud)
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        return preds, probs
