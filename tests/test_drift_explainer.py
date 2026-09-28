"""
Invariant Tests for SHAP Drift Attribution and Explainability (P4).
Asserts that:
1. Population Stability Index (PSI) accurately reflects distribution shifts.
2. DriftAttributionEngine extracts and ranks Top-K feature drivers with valid scores.
3. Attribution plots render cleanly without errors.
"""

import numpy as np
import pandas as pd
import pytest

from src.drift_explainer import calculate_psi, DriftAttributionEngine, validate_attribution_on_injected_drift
from src.adaptive_model import DriftAdaptiveModel
from src.preprocessing import Preprocessor


def test_psi_identical_vs_shifted_distributions():
    """
    Asserts that PSI is near zero for identical distributions,
    and strongly positive (> 0.25) when distributions diverge.
    """
    np.random.seed(42)
    baseline = np.random.normal(loc=100.0, scale=15.0, size=1000)
    identical = np.random.normal(loc=100.0, scale=15.0, size=1000)
    shifted = np.random.normal(loc=140.0, scale=30.0, size=1000)

    psi_ident = calculate_psi(baseline, identical)
    psi_shift = calculate_psi(baseline, shifted)

    # Identical distributions should have minimal PSI (< 0.05)
    assert psi_ident < 0.05, f"PSI for identical distributions unexpectedly high: {psi_ident}"

    # Shifted distributions should trigger strong drift threshold (>= 0.25)
    assert psi_shift >= 0.25, f"PSI for shifted distributions failed to exceed 0.25: {psi_shift}"


def test_psi_edge_cases():
    """Asserts that calculate_psi handles edge cases gracefully."""
    # Constant array
    constant = np.ones(100)
    assert calculate_psi(constant, constant) == 0.0

    # Array with NaNs
    arr_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan])
    assert calculate_psi(arr_with_nan, arr_with_nan) >= 0.0

    # Empty array
    assert calculate_psi(np.array([]), np.array([])) == 0.0


def test_drift_attribution_ranking_and_schema(synthetic_stream_df, tmp_path):
    """
    Asserts that DriftAttributionEngine produces ranked Top-K driver records
    sorted monotonically by attribution_score.
    """
    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(synthetic_stream_df[synthetic_stream_df["step"] <= 24])
    df_processed = preprocessor.transform(synthetic_stream_df)
    slices = preprocessor.partition_time_slices(df_processed)

    model = DriftAdaptiveModel(
        feature_names=preprocessor.feature_names,
        scale_pos_weight=1.0
    )
    model.fit_initial(slices[0]["data"])

    explainer = DriftAttributionEngine(feature_names=preprocessor.feature_names, max_shap_samples=100)

    top_k = 4
    attributions = explainer.explain_drift_slice(
        model=model,
        baseline_df=slices[0]["data"],
        current_df=slices[1]["data"],
        slice_index=1,
        slice_name="Slice_02",
        detector_type="ADWIN",
        top_k=top_k
    )

    # Check length
    assert len(attributions) == top_k

    # Check schema and ranking monotonicity
    prev_score = float("inf")
    for i, attr in enumerate(attributions, start=1):
        assert attr["rank_order"] == i
        assert attr["slice_index"] == 1
        assert attr["detector_type"] == "ADWIN"
        assert attr["feature_name"] in preprocessor.feature_names
        assert attr["psi_score"] >= 0.0
        assert attr["shap_importance"] >= 0.0
        assert attr["attribution_score"] <= prev_score
        prev_score = attr["attribution_score"]

    # Test plot generation
    plot_file = tmp_path / "test_attribution_plot.png"
    out_path = explainer.render_attribution_plot(
        attributions=attributions,
        slice_name="Slice_02",
        output_filename=str(plot_file)
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_drift_attribution_reproducibility(synthetic_stream_df):
    """
    Asserts that DriftAttributionEngine with a fixed random_state produces
    identical attribution scores and rankings across repeated executions.
    """
    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(synthetic_stream_df[synthetic_stream_df["step"] <= 24])
    df_processed = preprocessor.transform(synthetic_stream_df)
    slices = preprocessor.partition_time_slices(df_processed)

    model = DriftAdaptiveModel(feature_names=preprocessor.feature_names, scale_pos_weight=1.0)
    model.fit_initial(slices[0]["data"])

    explainer1 = DriftAttributionEngine(feature_names=preprocessor.feature_names, max_shap_samples=100, random_state=42)
    explainer2 = DriftAttributionEngine(feature_names=preprocessor.feature_names, max_shap_samples=100, random_state=42)

    res1 = explainer1.explain_drift_slice(model, slices[0]["data"], slices[1]["data"], slice_index=1, slice_name="Slice_02")
    res2 = explainer2.explain_drift_slice(model, slices[0]["data"], slices[1]["data"], slice_index=1, slice_name="Slice_02")

    assert len(res1) == len(res2)
    for a1, a2 in zip(res1, res2):
        assert a1["feature_name"] == a2["feature_name"]
        assert np.isclose(a1["attribution_score"], a2["attribution_score"])
        assert np.isclose(a1["shap_importance"], a2["shap_importance"])


def test_drift_attribution_multidimensional_shap_shapes(synthetic_stream_df, monkeypatch):
    """
    Asserts that DriftAttributionEngine correctly parses both 3D ndarrays
    and list-of-arrays returned by different SHAP/XGBoost version conventions.
    """
    import shap
    preprocessor = Preprocessor(hours_per_slice=24)
    preprocessor.fit(synthetic_stream_df[synthetic_stream_df["step"] <= 24])
    df_processed = preprocessor.transform(synthetic_stream_df)
    slices = preprocessor.partition_time_slices(df_processed)

    model = DriftAdaptiveModel(feature_names=preprocessor.feature_names, scale_pos_weight=1.0)
    model.fit_initial(slices[0]["data"])

    n_feats = len(preprocessor.feature_names)
    explainer = DriftAttributionEngine(feature_names=preprocessor.feature_names, max_shap_samples=50, random_state=42)

    # 1. Simulate 3D array: (n_samples, n_features, 2)
    def mock_shap_values_3d(self, X):
        return np.ones((len(X), n_feats, 2)) * 0.42

    monkeypatch.setattr(shap.TreeExplainer, "shap_values", mock_shap_values_3d)
    res_3d = explainer.explain_drift_slice(model, slices[0]["data"], slices[1]["data"], slice_index=1, slice_name="Slice_02")
    assert len(res_3d) > 0
    assert all(r["shap_importance"] == pytest.approx(0.42) for r in res_3d)

    # 2. Simulate list of per-class arrays: [class_0, class_1]
    def mock_shap_values_list(self, X):
        return [np.ones((len(X), n_feats)) * 0.1, np.ones((len(X), n_feats)) * 0.75]

    monkeypatch.setattr(shap.TreeExplainer, "shap_values", mock_shap_values_list)
    res_list = explainer.explain_drift_slice(model, slices[0]["data"], slices[1]["data"], slice_index=1, slice_name="Slice_02")
    assert len(res_list) > 0
    # Positive fraud class (index 1) should be selected
    assert all(r["shap_importance"] == pytest.approx(0.75) for r in res_list)


def test_validate_attribution_on_injected_drift():
    """
    Asserts that TreeSHAP x (1 + PSI) correctly attributes drift to predictive features
    and suppresses pure noise features undergoing benign covariate shift across multiple seeds.
    """
    summary = validate_attribution_on_injected_drift(
        n_samples=500,
        n_features=8,
        drifted_predictive_indices=[0, 1],
        drifted_noise_index=7,
        seeds=[101, 102, 103],
        save_results=False
    )

    assert summary["predictive_detected_rate"] == 1.0
    assert summary["noise_suppressed_rate"] == 1.0
    assert summary["mean_noise_psi"] >= 0.25
    assert summary["mean_noise_shap"] < 0.20



