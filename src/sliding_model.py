"""Scheduled retraining on a bounded number of recent slices."""

from src.periodic_model import PeriodicRetrainModel


class SlidingWindowModel(PeriodicRetrainModel):
    def __init__(self, *args, window_slices=4, **kwargs):
        super().__init__(*args, window_slices=window_slices, **kwargs)
