"""
Unit and Invariant Tests for River Streaming Baselines (src/river_models.py).
Tests HAT, ARF, and SRP models for:
1. Warmup training initialization
2. Sequential predict() contract (0/1 predictions, valid probabilities in [0, 1])
3. Incremental sample learning via update_with_slice()
4. Robustness against missing/NaN feature values
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import Preprocessor
from src.river_models import (
    HoeffdingAdaptiveTreeModel,
    AdaptiveRandomForestModel,
    StreamingRandomPatchesModel
)


@pytest.fixture
def river_test_data(synthetic_stream_df):
    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(synthetic_stream_df[synthetic_stream_df["step"] <= 24])
    df_processed = preprocessor.transform(synthetic_stream_df)
    slices = preprocessor.partition_time_slices(df_processed)
    return preprocessor.feature_names, slices


def test_river_models_warmup_and_predict(river_test_data):
    feature_names, slices = river_test_data
    warmup_df = slices[0]["data"]
    test_df = slices[1]["data"]

    models = [
        HoeffdingAdaptiveTreeModel(feature_names=feature_names),
        AdaptiveRandomForestModel(feature_names=feature_names, n_models=3),
        StreamingRandomPatchesModel(feature_names=feature_names, n_models=3)
    ]

    for model in models:
        # Before warmup, predict() must raise RuntimeError
        with pytest.raises(RuntimeError):
            model.predict(test_df)

        # Warmup on slice 0
        model.fit_initial(warmup_df)
        assert model.trained is True
        assert model.total_samples_learned == len(warmup_df)

        # Test-before-learn on slice 1
        preds, probs = model.predict(test_df)
        assert len(preds) == len(test_df)
        assert len(probs) == len(test_df)
        assert np.all(np.isin(preds, [0, 1]))
        assert np.all((probs >= 0.0) & (probs <= 1.0))

        # Incremental update on slice 1
        model.update_with_slice(test_df, slice_idx=1)
        assert model.total_samples_learned == len(warmup_df) + len(test_df)
        assert len(model.update_log) == 1


def test_river_models_nan_resilience(river_test_data):
    feature_names, slices = river_test_data
    slice_data = slices[0]["data"].copy()

    # Introduce intentional NaNs into features
    slice_data.iloc[0:10, 0] = np.nan

    hat = HoeffdingAdaptiveTreeModel(feature_names=feature_names)
    hat.fit_initial(slice_data)
    preds, probs = hat.predict(slice_data)
    assert len(preds) == len(slice_data)
    assert not np.any(np.isnan(probs))
