"""
Invariant Tests for Stream Accumulation Deduplication (FR-5).
Asserts that DriftAdaptiveModel accumulates streaming history without duplicate records,
and that retraining runs on exactly the unique historical observations.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import Preprocessor
from src.adaptive_model import DriftAdaptiveModel


def test_fit_initial_does_not_seed_history(synthetic_stream_df):
    """
    Asserts that fit_initial() does not add train_df to self.cumulative_history.
    Cumulative history must be populated strictly through sequential update_with_slice() calls.
    """
    preprocessor = Preprocessor(hours_per_slice=24)
    train_df = synthetic_stream_df[synthetic_stream_df["step"] <= 24].copy()
    preprocessor.fit(train_df)
    train_processed = preprocessor.transform(train_df)

    adaptive_model = DriftAdaptiveModel(
        feature_names=preprocessor.feature_names,
        scale_pos_weight=preprocessor.imbalance_weights["scale_pos_weight"]
    )
    adaptive_model.fit_initial(train_processed)

    # Invariant: cumulative_history must be empty before the stream loop starts
    assert len(adaptive_model.cumulative_history) == 0, (
        f"fit_initial() unexpectedly seeded cumulative_history with {len(adaptive_model.cumulative_history)} items!"
    )


def test_stream_accumulation_row_count(synthetic_stream_df):
    """
    Asserts that feeding K slices into update_with_slice results in
    cumulative_history having exactly the sum of row counts of all K slices,
    with zero duplicate TransactionIDs.
    """
    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(synthetic_stream_df[synthetic_stream_df["step"] <= 24])
    df_processed = preprocessor.transform(synthetic_stream_df)
    slices = preprocessor.partition_time_slices(df_processed)

    adaptive_model = DriftAdaptiveModel(
        feature_names=preprocessor.feature_names,
        scale_pos_weight=1.0
    )
    adaptive_model.fit_initial(slices[0]["data"])

    # Sequentially feed all slices into update_with_slice
    expected_total_rows = 0
    for s in slices:
        adaptive_model.update_with_slice(
            slice_df=s["data"],
            drift_triggered=False,
            slice_idx=s["slice_index"]
        )
        expected_total_rows += len(s["data"])

    # Check total rows in accumulated history
    accumulated_df = pd.concat(adaptive_model.cumulative_history, ignore_index=True)
    assert len(accumulated_df) == expected_total_rows, (
        f"Accumulated row count ({len(accumulated_df)}) != expected ({expected_total_rows})"
    )

    # Invariant: Zero duplicate TransactionIDs
    duplicate_count = accumulated_df["TransactionID"].duplicated().sum()
    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate TransactionIDs in cumulative history!"
    )


def test_retrain_data_has_zero_duplicates(synthetic_stream_df):
    """
    Asserts that when drift triggers a retrain, the dataset passed to model.fit()
    contains zero duplicate records.
    """
    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(synthetic_stream_df[synthetic_stream_df["step"] <= 24])
    df_processed = preprocessor.transform(synthetic_stream_df)
    slices = preprocessor.partition_time_slices(df_processed)

    adaptive_model = DriftAdaptiveModel(
        feature_names=preprocessor.feature_names,
        scale_pos_weight=1.0
    )
    adaptive_model.fit_initial(slices[0]["data"])

    # Trigger drift on slice 1 and slice 2
    for s in slices:
        # Predict first (test-before-train)
        preds, probs = adaptive_model.predict(s["data"])
        # Update with drift triggered on slices > 0
        drift_flag = (s["slice_index"] > 0)
        adaptive_model.update_with_slice(
            slice_df=s["data"],
            drift_triggered=drift_flag,
            slice_idx=s["slice_index"]
        )

    # Check retrain log
    assert len(adaptive_model.retrain_log) == 2, "Expected 2 retraining events."
    for log_entry in adaptive_model.retrain_log:
        # History at the time of retrain must equal sum of slices processed so far
        expected_rows_at_slice = sum(len(slices[i]["data"]) for i in range(log_entry["slice_idx"] + 1))
        assert log_entry["accumulated_samples"] == expected_rows_at_slice, (
            f"Retrained on {log_entry['accumulated_samples']} rows, but expected {expected_rows_at_slice}."
        )
