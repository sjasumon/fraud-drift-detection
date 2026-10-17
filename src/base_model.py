"""
Unified Base Model Architecture for Streaming Fraud Detection.
Provides standardized XGBoost GPU-acceleration with automatic CPU-fallback,
execution timing, and predict() contract for all benchmark models.
"""

import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import USE_GPU, GPU_DEVICE, RANDOM_SEED

logger = logging.getLogger("fraud_drift.model")

DEFAULT_XGB_HYPERPARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss"
}


def is_cuda_available() -> bool:
    """
    Probes whether a functional CUDA driver and at least one active CUDA GPU are present.
    Prevents XGBoost C++ fallback warnings and inaccurate device reporting when eGPU is disconnected.
    """
    try:
        import ctypes
        import sys
        if sys.platform == "win32":
            cuda = ctypes.CDLL("nvcuda.dll")
        else:
            cuda = ctypes.CDLL("libcuda.so")
        if cuda.cuInit(0) != 0:
            return False
        count = ctypes.c_int()
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value <= 0:
            return False
        return True
    except Exception:
        return False


def fit_xgb_with_fallback(
    X: np.ndarray,
    y: np.ndarray,
    xgb_params: Dict[str, Any],
    model_name: str = "Model"
) -> Tuple[xgb.XGBClassifier, str, float]:
    """
    Fits an XGBoost classifier using GPU acceleration (tree_method='hist') if enabled
    and a functional CUDA device is present, gracefully falling back to CPU histogram
    gradient boosting upon any failure or when running on CPU/Intel Arc environments.
    Returns:
        (fitted_model, device_used, duration_seconds)
    """
    if USE_GPU and is_cuda_available():
        try:
            model = xgb.XGBClassifier(
                **xgb_params,
                tree_method="hist",
                device=GPU_DEVICE
            )
            t0 = time.time()
            model.fit(X, y)
            duration = time.time() - t0
            device_used = f"NVIDIA GPU ({GPU_DEVICE})"
            logger.info(f"{model_name} fitted on {device_used} in {duration:.2f}s.")
            return model, device_used, duration
        except Exception as e:
            logger.warning(f"GPU fitting failed for {model_name} ({e}). Falling back to CPU.")
    elif USE_GPU:
        logger.debug(f"{model_name}: GPU acceleration enabled in config ({GPU_DEVICE}), but no active CUDA device is detected (eGPU disconnected). Running on CPU.")

    model = xgb.XGBClassifier(**xgb_params, tree_method="hist", device="cpu")
    t0 = time.time()
    model.fit(X, y)
    duration = time.time() - t0
    device_used = "cpu"
    logger.info(f"{model_name} fitted on CPU in {duration:.2f}s.")
    return model, device_used, duration


class BaseFraudModel:
    """
    Abstract base class providing shared GPU-accelerated fitting, fallback handling,
    and prediction contracts for all stream fraud models.
    """

    def __init__(
        self,
        feature_names: List[str],
        scale_pos_weight: float = 1.0,
        random_seed: int = RANDOM_SEED
    ):
        self.feature_names = feature_names
        self.scale_pos_weight = scale_pos_weight
        self.random_seed = random_seed
        self.model: Optional[xgb.XGBClassifier] = None
        self.trained = False
        self.device_used = "cpu"
        self.train_duration_sec = 0.0

    def _fit_xgb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> xgb.XGBClassifier:
        """Shared GPU-accelerated XGBoost fit with automatic CPU fallback."""
        name = model_name or self.__class__.__name__
        model, device, dur = fit_xgb_with_fallback(X, y, params, model_name=name)
        self.model = model
        self.device_used = device
        self.train_duration_sec = dur
        self.trained = True
        return model

    def predict(self, slice_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the current model on a time slice before learning from it.
        Returns:
            Tuple of (binary_predictions, predicted_probabilities).
        """
        if self.model is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before predict() is called.")
        X = slice_df[self.feature_names].values
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs
