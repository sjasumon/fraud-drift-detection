"""
River Streaming Baselines Module for Stream Mining Fraud Detection.
Implements canonical online stream mining algorithms co-authored by Prof. Yun Sing Koh:
1. Adaptive Random Forest (ARF) - river.forest.ARFClassifier
2. Hoeffding Adaptive Tree (HAT) - river.tree.HoeffdingAdaptiveTreeClassifier
3. Streaming Random Patches (SRP) - river.ensemble.SRPClassifier

All models implement the standardized BaseFraudModel contract:
- fit_initial(train_df): Initial warm-up across baseline training partition
- predict(slice_df): Test-before-learn evaluation returning (preds, probs)
- update_with_slice(slice_df): Incremental sample-by-sample online model update
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from river.tree import HoeffdingAdaptiveTreeClassifier
from river.forest import ARFClassifier
from river.ensemble import SRPClassifier

from src.config import RANDOM_SEED
from src.base_model import BaseFraudModel

logger = logging.getLogger("fraud_drift.river_models")


class RiverBaseStreamingModel(BaseFraudModel):
    """
    Abstract streaming baseline wrapper for River incremental classifiers.
    Processes data strictly sample-by-sample under the test-before-learn paradigm.
    """

    def __init__(
        self,
        feature_names: List[str],
        scale_pos_weight: float = 1.0,
        random_seed: int = RANDOM_SEED,
        decision_threshold: float = 0.5,
        model_name: str = "RiverStreamingModel"
    ):
        super().__init__(feature_names, scale_pos_weight=scale_pos_weight, random_seed=random_seed)
        self.decision_threshold = decision_threshold
        self.model_name = model_name
        self.river_classifier: Optional[Any] = None
        self.total_samples_learned = 0
        self.update_log: List[Dict[str, Any]] = []

    def fit_initial(self, train_df: pd.DataFrame):
        """
        Initializes and warms up the incremental model on the baseline training partition.
        """
        t0 = time.time()
        records = train_df[self.feature_names].to_dict(orient="records")
        labels = train_df["isFraud"].values

        logger.info(
            f"[{self.model_name}] Warming up on {len(records):,} baseline transactions..."
        )

        for x, y in zip(records, labels):
            self.river_classifier.learn_one(x, int(y))

        duration = time.time() - t0
        self.total_samples_learned += len(records)
        self.trained = True
        self.train_duration_sec = duration
        self.device_used = "CPU (River Streaming Engine)"

        logger.info(
            f"[{self.model_name}] Warm-up completed in {duration:.2f}s "
            f"({len(records) / max(0.001, duration):.1f} samples/s)."
        )

    def predict(self, slice_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the current online model on a streaming slice before updating (test-before-learn).
        Returns:
            Tuple of (binary_predictions, predicted_fraud_probabilities).
        """
        if not self.trained or self.river_classifier is None:
            raise RuntimeError(f"{self.model_name} must be initialized with fit_initial() before predict().")

        records = slice_df[self.feature_names].to_dict(orient="records")
        probs = np.zeros(len(records), dtype=float)

        for i, x in enumerate(records):
            p_dict = self.river_classifier.predict_proba_one(x)
            probs[i] = p_dict.get(1, 0.0) if p_dict else 0.0

        preds = (probs >= self.decision_threshold).astype(int)
        return preds, probs

    def update_with_slice(self, slice_df: pd.DataFrame, slice_idx: Optional[int] = None):
        """
        Incrementally updates the online model with the verified transactions in slice_df.
        """
        t0 = time.time()
        records = slice_df[self.feature_names].to_dict(orient="records")
        labels = slice_df["isFraud"].values

        for x, y in zip(records, labels):
            self.river_classifier.learn_one(x, int(y))

        duration = time.time() - t0
        self.total_samples_learned += len(records)

        log_entry = {
            "slice_idx": slice_idx,
            "samples_learned": len(records),
            "total_samples": self.total_samples_learned,
            "duration_seconds": round(duration, 3)
        }
        self.update_log.append(log_entry)


class HoeffdingAdaptiveTreeModel(RiverBaseStreamingModel):
    """
    Hoeffding Adaptive Tree (HAT) streaming baseline.
    Adapts to concept drift incrementally using ADWIN estimators embedded within tree nodes.
    """

    def __init__(
        self,
        feature_names: List[str],
        scale_pos_weight: float = 1.0,
        random_seed: int = RANDOM_SEED,
        decision_threshold: float = 0.5,
        grace_period: int = 200,
        delta: float = 1e-5
    ):
        super().__init__(
            feature_names=feature_names,
            scale_pos_weight=scale_pos_weight,
            random_seed=random_seed,
            decision_threshold=decision_threshold,
            model_name="Hoeffding Adaptive Tree (HAT)"
        )
        self.river_classifier = HoeffdingAdaptiveTreeClassifier(
            grace_period=grace_period,
            delta=delta,
            seed=random_seed
        )


class AdaptiveRandomForestModel(RiverBaseStreamingModel):
    """
    Adaptive Random Forest (ARF) streaming ensemble baseline (Gomes et al., ACM CSUR 2017).
    Deploys an online ensemble of Hoeffding Trees with warning and drift detectors per tree.
    """

    def __init__(
        self,
        feature_names: List[str],
        n_models: int = 10,
        scale_pos_weight: float = 1.0,
        random_seed: int = RANDOM_SEED,
        decision_threshold: float = 0.5,
        grace_period: int = 50
    ):
        super().__init__(
            feature_names=feature_names,
            scale_pos_weight=scale_pos_weight,
            random_seed=random_seed,
            decision_threshold=decision_threshold,
            model_name=f"Adaptive Random Forest (ARF, M={n_models})"
        )
        self.n_models = n_models
        self.river_classifier = ARFClassifier(
            n_models=n_models,
            grace_period=grace_period,
            seed=random_seed
        )


class StreamingRandomPatchesModel(RiverBaseStreamingModel):
    """
    Streaming Random Patches (SRP) ensemble baseline (Gomes, Bifet, Koh et al.).
    Combines online bagging and random subspace projection with drift detection.
    """

    def __init__(
        self,
        feature_names: List[str],
        n_models: int = 5,
        subspace_size: float = 0.6,
        scale_pos_weight: float = 1.0,
        random_seed: int = RANDOM_SEED,
        decision_threshold: float = 0.5
    ):
        super().__init__(
            feature_names=feature_names,
            scale_pos_weight=scale_pos_weight,
            random_seed=random_seed,
            decision_threshold=decision_threshold,
            model_name=f"Streaming Random Patches (SRP, M={n_models})"
        )
        self.n_models = n_models
        self.subspace_size = subspace_size
        self.river_classifier = SRPClassifier(
            n_models=n_models,
            subspace_size=subspace_size,
            seed=random_seed
        )
