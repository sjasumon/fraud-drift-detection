"""
FR-3: Static Baseline Model Module.
Trains a frozen classification model on the initial temporal window (earliest 20%),
accelerated by NVIDIA eGPU (CUDA), and evaluates across all subsequent time slices
without retraining or parameter updates.
"""

import logging
from typing import List
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.base_model import BaseFraudModel

logger = logging.getLogger("fraud_drift.static_model")


class StaticBaselineModel(BaseFraudModel):
    """
    FR-3: Static baseline classification model.
    Trained once on the initial 20% temporal window (e.g. Weeks 1-5) and frozen.
    """

    def __init__(self, feature_names: List[str], scale_pos_weight: float = 1.0, random_seed: int = RANDOM_SEED):
        super().__init__(feature_names, scale_pos_weight=scale_pos_weight, random_seed=random_seed)

    def fit(self, train_df: pd.DataFrame):
        """
        FR-3.1: Trains model strictly on the initial training window.
        Leverages NVIDIA GPU acceleration with automatic CPU fallback.
        """
        X = train_df[self.feature_names].values
        y = train_df["isFraud"].values

        logger.info(f"Training FR-3 Static Model on {len(X):,} samples ({np.sum(y)} fraud instances)...")

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

        self._fit_xgb(X, y, xgb_params, model_name="Static Model")
