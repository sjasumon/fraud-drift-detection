"""Regressions reproduced by auditing the delivered v0.4.0 ZIP."""

import numpy as np
import pandas as pd
import pytest
from src.artifacts import ModelBundle, save_bundle, load_bundle
from src.preprocessing import Preprocessor
from src.adaptive_model import DriftAdaptiveModel
from src.reporting import block_bootstrap_delta, budget_metrics


def fitted_bundle(df):
    prep = Preprocessor().fit(df.iloc[:100])
    model = DriftAdaptiveModel(prep.feature_names, model_params={"n_estimators": 5})
    model.fit_initial(prep.transform(df.iloc[:100]))
    return ModelBundle(
        model, prep, {"run_id": "audit", "model": "adaptive", "data_kind": "synthetic"}
    )


def test_export_strips_training_rows_without_mutating_live_model(
    tmp_path, synthetic_stream_df
):
    bundle = fitted_bundle(synthetic_stream_df)
    original = bundle.model.predict(bundle.preprocessor.transform(synthetic_stream_df))[
        1
    ]
    save_bundle(bundle, tmp_path / "package")
    serving = load_bundle(tmp_path / "package")
    assert serving.model.cumulative_history == []
    assert len(bundle.model.cumulative_history[0]) == 100
    np.testing.assert_allclose(
        original,
        serving.model.predict(serving.preprocessor.transform(synthetic_stream_df))[1],
    )


def test_model_package_is_immutable(tmp_path, synthetic_stream_df):
    bundle = fitted_bundle(synthetic_stream_df)
    save_bundle(bundle, tmp_path / "package")
    before = (tmp_path / "package" / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        save_bundle(bundle, tmp_path / "package")
    assert (tmp_path / "package" / "manifest.json").read_bytes() == before


def bootstrap_frame(seeds=(42,)):
    rows = []
    for seed in seeds:
        for t in range(8):
            for i in range(6):
                for model in ["static", "adaptive"]:
                    y = int(i < 3)
                    pred = (
                        int(i < 2)
                        if model == "static"
                        else int(i < (1 if t < 4 else 4))
                    )
                    rows.append(
                        dict(
                            seed=seed,
                            window=2,
                            slice_index=t,
                            trans_id=f"{t}-{i}",
                            model=model,
                            y_true=y,
                            prediction=pred,
                        )
                    )
    return pd.DataFrame(rows)


def test_bootstrap_rejects_inconsistent_ground_truth():
    d = bootstrap_frame()
    d.loc[(d.model == "static") & (d.trans_id == "0-0"), "y_true"] = 0
    with pytest.raises(ValueError, match="ground-truth"):
        block_bootstrap_delta(d, "adaptive")


def test_cloned_seeds_do_not_invent_independent_temporal_evidence():
    a = block_bootstrap_delta(bootstrap_frame(), "adaptive", n_bootstrap=250)
    b = block_bootstrap_delta(
        bootstrap_frame(tuple(range(10))), "adaptive", n_bootstrap=250
    )
    assert a["ci_95"] == pytest.approx(b["ci_95"], abs=1e-12)
    assert a["temporal_blocks"] == b["temporal_blocks"] == 8


def test_recent_window_uses_latest_verified_arrival(tmp_path, synthetic_stream_df):
    bundle = fitted_bundle(synthetic_stream_df)
    model = bundle.model
    model.window_slices = 2
    known = bundle.preprocessor.transform(synthetic_stream_df.copy())
    known["_history_slice"] = np.repeat([2, 3, 4], 100)
    assert model._retrain(8, known)
    assert model.retrain_log[-1]["accumulated_samples"] == 200
    assert model.retrain_log[-1]["latest_training_arrival_slice"] == 4
    assert model.retrain_log[-1]["verification_staleness_slices"] == 4


def test_initial_dataset_replay_does_not_consume_retrain_budget(synthetic_stream_df):
    bundle = fitted_bundle(synthetic_stream_df)
    model = bundle.model
    assert not model._retrain(3, model.cumulative_history[0])
    assert model.retrain_count == 0


def test_delayed_detector_never_mixes_model_generations(tmp_path, monkeypatch):
    from src.pipeline import run_pipeline
    from src.data_generator import generate_paysim_stream

    class AlwaysDrift:
        def __init__(self):
            self.drift_events = []

        def update(self, *args, **kwargs):
            return {"any_drift": True}

        def reset_detectors(self):
            pass

    path = tmp_path / "input.csv"
    generate_paysim_stream(
        total_weeks=7, tx_per_week=80, fraud_rate=0.15, seed=42, output_path=path
    )
    monkeypatch.setattr("src.pipeline.DriftMonitor", AlwaysDrift)
    run = run_pipeline(
        path,
        n_train_slices=2,
        delay_slices=1,
        adaptive_window=2,
        data_kind="synthetic",
        output_dir=tmp_path / "runs",
        model_params={"n_estimators": 5},
    )
    feedback = pd.read_csv(run["run_dir"] / "feedback_events.csv")
    assert (~feedback.monitored).any()
    assert (
        feedback.loc[feedback.monitored, "prediction_generation"]
        == feedback.loc[feedback.monitored, "active_generation"]
    ).all()
    assert (
        feedback.loc[~feedback.monitored, "prediction_generation"]
        != feedback.loc[~feedback.monitored, "active_generation"]
    ).all()


def test_alert_budget_separates_repeated_seed_populations():
    rows = [
        dict(
            model="a",
            seed=s,
            window=2,
            day=0,
            trans_id=str(i),
            probability=0.9 - i * 0.1,
            y_true=int(i == 0),
        )
        for s in [1, 2]
        for i in range(3)
    ]
    result = budget_metrics(pd.DataFrame(rows), budgets=[1]).iloc[0]
    assert result.alerts == 2 and result.detected_fraud == 2 and result.precision == 1


def test_study_outputs_all_promised_comparison_dimensions(tmp_path):
    from src.study import run_study
    from src.data_generator import generate_paysim_stream

    path = tmp_path / "input.csv"
    generate_paysim_stream(
        total_weeks=4, tx_per_week=60, fraud_rate=0.15, seed=42, output_path=path
    )
    result = run_study(
        path,
        tmp_path / "study",
        seeds=[42],
        train_windows=[2],
        test_slices=2,
        windows=[2],
        budgets=[1],
        delays=[1],
        data_kind="synthetic",
    )
    assert len(result["comparisons"]) == 5
    for row in result["comparisons"]:
        assert {
            "mean_seed_precision",
            "mean_seed_recall",
            "mean_seed_f1",
            "mean_seed_auc_pr",
            "mean_seed_false_positives",
            "daily_review_budgets",
            "mean_training_seconds_per_run",
            "mean_retrains_per_run",
            "run_ids",
        }.issubset(row)
    assert result["data_kind"] == "synthetic"


def test_ingestion_rejects_fractional_labels_before_casting(tmp_path):
    from src.ingestion import DatasetIngestion, IngestionError

    path = tmp_path / "baf.csv"
    pd.DataFrame(
        {"month": [0, 1], "fraud_bool": [0.0, 0.5], "proposed_credit_limit": [100, 100]}
    ).to_csv(path, index=False)
    with pytest.raises(IngestionError, match="binary"):
        DatasetIngestion(path).load_and_validate()


def test_null_identity_is_not_cast_to_clean_entity(synthetic_stream_df):
    from src.evaluate_uid_split import annotate_entities

    df = synthetic_stream_df.copy()
    df.loc[df.index[0], "card1"] = np.nan
    result = annotate_entities(df)
    assert not result.iloc[0].is_clean and result.iloc[0].uid_cf.startswith("MISSING_")


def test_zero_fraud_cohort_preserves_population_counts():
    from src.evaluate_uid_split import compute_metrics

    r = compute_metrics(
        dict(
            y=[0] * 3,
            stat_p=[1, 0, 0],
            adapt_p=[0, 0, 0],
            stat_prob=[0.8, 0.1, 0.2],
            adapt_prob=[0.1, 0.1, 0.1],
        )
    )
    assert r["total_tx"] == 3 and r["fraud_tx"] == 0 and r["stat_fp"] == 1
    assert r["recall_status"] == "undefined_no_fraud" and r["stat_auc"] is None


def test_paired_summary_preserves_single_pair_difference_and_serializes_finitely():
    from src.benchmark_statistical_rigor import compute_paired_tests
    import json

    one = compute_paired_tests([0.8], [0.5])
    assert (
        one["mean_diff"] == pytest.approx(0.3)
        and one["test_status"] == "insufficient_pairs"
    )
    constant = compute_paired_tests([0.8, 0.8, 0.8], [0.5, 0.5, 0.5])
    json.dumps(constant, allow_nan=False)


def test_attribution_fallback_is_explicit(synthetic_stream_df, monkeypatch):
    from src.drift_explainer import DriftAttributionEngine

    bundle = fitted_bundle(synthetic_stream_df)

    def unavailable(*a, **kw):
        raise ValueError("simulate unsupported SHAP")

    monkeypatch.setattr("src.drift_explainer.shap.TreeExplainer", unavailable)
    frame = bundle.preprocessor.transform(synthetic_stream_df)
    rows = DriftAttributionEngine(bundle.model.feature_names).explain_drift_slice(
        bundle.model, frame.iloc[:100], frame.iloc[100:], 1, "test"
    )
    assert all(
        r["importance_method"] == "model_feature_importance_fallback" for r in rows
    )


def test_pretransaction_ablation_keeps_pretransaction_balance_ratio():
    df = pd.DataFrame(
        {
            "step": range(1, 21),
            "isFraud": [0, 1] * 10,
            "amount": 100.0,
            "oldbalanceOrg": 200.0,
            "newbalanceOrig": 100.0,
        }
    )
    p = Preprocessor(pre_transaction_only=True).fit(df)
    assert "amount_to_orig_ratio" in p.feature_names
    assert p.transform(df).amount_to_orig_ratio.iloc[0] == pytest.approx(100 / 201)


def test_psi_detects_binary_missingness_rate_change():
    from src.drift_explainer import calculate_psi

    baseline = np.r_[np.zeros(95), np.ones(5)]
    shifted = np.r_[np.zeros(5), np.ones(95)]
    assert calculate_psi(baseline, baseline) == 0
    assert calculate_psi(baseline, shifted) > 0.25
