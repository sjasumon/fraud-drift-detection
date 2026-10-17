"""
Invariant Tests for Comparative Streaming Baselines (P2).
Asserts that:
1. PeriodicRetrainModel retrains strictly at the specified interval (e.g. every 4 slices).
2. SlidingWindowModel maintains a maximum queue depth of W slices and retrains only on that window.
"""

from src.preprocessing import Preprocessor
from src.periodic_model import PeriodicRetrainModel
from src.sliding_model import SlidingWindowModel


def test_periodic_model_scheduled_retrain(synthetic_stream_df):
    """
    Asserts that PeriodicRetrainModel triggers retrain strictly when
    out-of-sample slices seen reaches the interval (e.g. K=2).
    """
    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(synthetic_stream_df[synthetic_stream_df["step"] <= 24])
    df_processed = preprocessor.transform(synthetic_stream_df)
    slices = preprocessor.partition_time_slices(df_processed)

    periodic_model = PeriodicRetrainModel(
        feature_names=preprocessor.feature_names,
        retrain_interval_slices=2,
        scale_pos_weight=1.0,
    )
    periodic_model.fit_initial(slices[0]["data"])

    # Slice 0: Baseline (not out of sample)
    periodic_model.update_with_slice(
        slices[0]["data"], slice_idx=0, is_out_of_sample=False
    )
    assert periodic_model.retrain_count == 0

    # Slice 1: 1st out of sample slice -> interval (2) not reached
    periodic_model.update_with_slice(
        slices[1]["data"], slice_idx=1, is_out_of_sample=True
    )
    assert periodic_model.retrain_count == 0

    # Slice 2: 2nd out of sample slice -> interval (2) reached -> exactly 1 retrain!
    periodic_model.update_with_slice(
        slices[2]["data"], slice_idx=2, is_out_of_sample=True
    )
    assert periodic_model.retrain_count == 1
    assert len(periodic_model.retrain_log) == 1


def test_sliding_window_depth_and_retrain(synthetic_stream_df):
    """
    Asserts that SlidingWindowModel:
    1. Keeps at most window_slices in window_history.
    2. Retrains strictly on the data within window_history.
    """
    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(synthetic_stream_df[synthetic_stream_df["step"] <= 24])
    df_processed = preprocessor.transform(synthetic_stream_df)
    slices = preprocessor.partition_time_slices(df_processed)

    window_size = 2
    sliding_model = SlidingWindowModel(
        feature_names=preprocessor.feature_names,
        window_slices=window_size,
        retrain_interval_slices=2,
        scale_pos_weight=1.0,
    )
    sliding_model.fit_initial(slices[0]["data"])

    # Feed slices 0, 1, 2
    for s in slices:
        is_oos = s["slice_index"] > 0
        sliding_model.update_with_slice(
            s["data"], slice_idx=s["slice_index"], is_out_of_sample=is_oos
        )

    # Invariant: Window history length must not exceed window_slices
    assert len(sliding_model.window_history) <= window_size, (
        f"Window history length ({len(sliding_model.window_history)}) exceeded max {window_size}!"
    )

    # Invariant: Retrain occurred at interval=2
    assert sliding_model.retrain_count == 1
    # Check that retrained sample count equals the sum of the slices currently in window
    expected_samples = sum(len(w) for w in sliding_model.window_history)
    assert sliding_model.retrain_log[0]["accumulated_samples"] == expected_samples
