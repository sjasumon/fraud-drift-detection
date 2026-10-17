"""Frozen baseline sharing exactly the same estimator and validation policy."""

from src.base_model import BaseFraudModel


class StaticBaselineModel(BaseFraudModel):
    pass
