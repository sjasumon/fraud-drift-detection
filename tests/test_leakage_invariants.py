"""
Invariant Tests for Temporal Leakage Prevention (FR-2 & FR-3).
Asserts that preprocessing statistics (medians, one-hot category dictionaries, and scale_pos_weight)
are fit strictly on the initial training window and never leak future streaming data.
"""

import copy
import numpy as np

from src.preprocessing import Preprocessor


def test_preprocessor_temporal_isolation(synthetic_stream_df):
    """
    Asserts that Preprocessor.fit() extracts medians and imbalance weights
    strictly from the training partition without looking ahead.
    """
    df = synthetic_stream_df
    # Slice 0: step <= 24 (Training partition)
    train_mask = df["step"] <= 24
    train_df = df[train_mask].copy()
    stream_df = df[~train_mask].copy()

    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(train_df)

    # 1. Check numeric medians match train_df medians exactly
    engineered_train = preprocessor.extract_engineered_features(train_df)
    for col in preprocessor.numeric_cols:
        expected_median = (
            float(engineered_train[col[:-9]].isna().astype(float).median())
            if col.endswith("__missing")
            else float(engineered_train[col].median(skipna=True))
        )
        if np.isnan(expected_median):
            expected_median = 0.0
        assert np.isclose(preprocessor.imputation_values[col], expected_median), (
            f"Imputation median for {col} does not match training-only median."
        )

    # 2. Check scale_pos_weight is calculated strictly on training labels
    pos_train = int(train_df["isFraud"].sum())
    neg_train = len(train_df) - pos_train
    expected_scale_pos_weight = round(float(neg_train) / max(1.0, float(pos_train)), 2)
    assert (
        preprocessor.imbalance_weights["scale_pos_weight"] == expected_scale_pos_weight
    )
    assert preprocessor.imbalance_weights["positive_samples"] == pos_train


def test_transform_does_not_mutate_state(synthetic_stream_df):
    """
    Asserts that transforming streaming slices never alters the Preprocessor's
    internal imputation medians or categorical mappings.
    """
    df = synthetic_stream_df
    train_df = df[df["step"] <= 24].copy()
    stream_df = df[df["step"] > 24].copy()

    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(train_df)

    # Snapshot of internal state
    imputation_snapshot = copy.deepcopy(preprocessor.imputation_values)
    encoder_features_snapshot = list(preprocessor.feature_names)

    # Transform future stream
    transformed_stream = preprocessor.transform(stream_df)

    # State must remain bitwise identical
    assert preprocessor.imputation_values == imputation_snapshot, (
        "Preprocessor imputation values were mutated during transform()!"
    )
    assert preprocessor.feature_names == encoder_features_snapshot, (
        "Preprocessor feature names list was mutated during transform()!"
    )


def test_unseen_categorical_handling(synthetic_stream_df):
    """
    Asserts that new categorical values appearing in future slices (e.g. 'Z')
    are safely ignored (encoded as 0s) without causing dimension mismatch or errors.
    """
    df = synthetic_stream_df
    train_df = df[df["step"] <= 24].copy()
    # Slice 2 has 'Z' which was never in train_df
    stream_slice2 = df[df["step"] >= 49].copy()

    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(train_df)

    # Transform future slice containing unseen 'Z'
    transformed_df = preprocessor.transform(stream_slice2)

    # Feature columns must match preprocessor.feature_names exactly
    for feat in preprocessor.feature_names:
        assert feat in transformed_df.columns, (
            f"Feature {feat} missing from transformed slice."
        )

    # None of the generated columns should have NaN values
    assert not transformed_df[preprocessor.feature_names].isna().any().any(), (
        "Transformed features contain unexpected NaNs."
    )


def test_batch_vs_slice_transform_invariance(synthetic_stream_df):
    """
    Asserts that transforming a slice independently yields the exact same numeric
    values as slicing after transforming a full batch.
    """
    df = synthetic_stream_df
    train_df = df[df["step"] <= 24].copy()
    slice_1 = df[(df["step"] > 24) & (df["step"] <= 48)].copy()

    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(train_df)

    # Independent transform
    trans_indep = preprocessor.transform(slice_1)[preprocessor.feature_names].values

    # Batch transform then subset
    full_transformed = preprocessor.transform(df)
    trans_batch = full_transformed[
        (full_transformed["step"] > 24) & (full_transformed["step"] <= 48)
    ][preprocessor.feature_names].values

    np.testing.assert_allclose(
        trans_indep,
        trans_batch,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Independent slice transformation differs from batch transformation.",
    )
