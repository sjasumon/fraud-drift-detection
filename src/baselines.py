"""
Comparative Streaming Baselines Module for Stream Mining Fraud Detection.

This module consolidates and re-exports the comparative baselines evaluated in
the empirical study, providing a unified import interface for:
- PeriodicRetrainModel: Scheduled retraining on accumulated history (K-week interval)
- SlidingWindowModel: Bounded-memory retraining on recent W time slices
- StaticBaselineModel: Frozen baseline trained on warm-up data with zero retraining
- River streaming models (HAT, ARF, SRP) from src.river_models
"""

from src.periodic_model import PeriodicRetrainModel
from src.sliding_model import SlidingWindowModel
from src.static_model import StaticBaselineModel
from src.river_models import (
    RiverBaseStreamingModel,
    HoeffdingAdaptiveTreeModel,
    AdaptiveRandomForestModel,
    StreamingRandomPatchesModel,
)

__all__ = [
    "PeriodicRetrainModel",
    "SlidingWindowModel",
    "StaticBaselineModel",
    "RiverBaseStreamingModel",
    "HoeffdingAdaptiveTreeModel",
    "AdaptiveRandomForestModel",
    "StreamingRandomPatchesModel",
]
