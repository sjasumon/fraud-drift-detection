"""Behavioral regressions for the v0.4 evaluation corrections."""

import json
import numpy as np
import pandas as pd
import pytest
from src.delayed_feedback import DelayedFeedbackQueue
from src.drift_explainer import calculate_psi
from src.history_features import add_history_features
from src.preprocessing import Preprocessor
from src.adaptive_model import DriftAdaptiveModel
from src.static_model import StaticBaselineModel
from src.pipeline import run_pipeline
from src.data_generator import generate_paysim_stream


def test_seed_replay_cannot_duplicate_training_rows():
    seed = pd.DataFrame({"trans_id": ["a", "b"], "isFraud": [0, 1]})
    q = DelayedFeedbackQueue(delay_slices=0)
    q.enqueue(seed, 0)
    q.dequeue_matured(0)
    assert len(q.get_all_matured_training_data(seed)) == 2


def test_delayed_queue_preserves_original_prediction():
    q = DelayedFeedbackQueue(delay_slices=2)
    frame = pd.DataFrame(
        {
            "trans_id": ["a"],
            "isFraud": [1],
            "_prediction": [0],
            "_probability": [0.2],
            "_model_version": [3],
        }
    )
    q.enqueue(frame, 0)
    frame["_prediction"] = 1
    assert q.dequeue_matured(1).empty
    matured = q.dequeue_matured(2)
    assert (
        matured["_prediction"].iloc[0] == 0 and matured["_model_version"].iloc[0] == 3
    )


def test_psi_detects_constant_feature_change():
    assert calculate_psi(np.zeros(100), np.ones(100)) > 0.25
    assert calculate_psi(np.zeros(100), np.zeros(100)) == 0


def test_historical_features_do_not_see_future():
    df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "trans_id": ["a", "b", "c"],
            "nameOrig": ["x"] * 3,
            "amount": [10, 20, 900],
        }
    )
    first = add_history_features(df)
    df.loc[2, "amount"] = 999999
    second = add_history_features(df)
    columns = [c for c in first if c.startswith("hist_")]
    pd.testing.assert_frame_equal(first.loc[:1, columns], second.loc[:1, columns])
    assert first.hist_tx_count.tolist() == [0, 1, 2]
    assert first.hist_amount_mean.tolist() == [0, 10, 15]


def test_pretransaction_ablation_excludes_post_balances():
    df = pd.DataFrame(
        {
            "step": range(20),
            "trans_id": [str(i) for i in range(20)],
            "isFraud": [0, 1] * 10,
            "amount": 100.0,
            "oldbalanceOrg": 100.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 100.0,
            "errorBalanceOrig": 0.0,
        }
    )
    p = Preprocessor(pre_transaction_only=True).fit(df)
    assert not any(
        "newbalance" in c or "reconcile" in c or "errorBalance" in c
        for c in p.feature_names
    )


def test_shared_estimator_initial_predictions_identical(synthetic_stream_df):
    p = Preprocessor().fit(synthetic_stream_df)
    d = p.transform(synthetic_stream_df)
    a = StaticBaselineModel(p.feature_names, model_params={"n_estimators": 10}).fit(d)
    b = DriftAdaptiveModel(p.feature_names, model_params={"n_estimators": 10})
    b.fit_initial(d)
    assert a.threshold == b.threshold
    np.testing.assert_allclose(a.predict(d)[1], b.predict(d)[1])


def test_no_warmup_retrains_and_no_current_slice_training(tmp_path, monkeypatch):
    path = tmp_path / "paysim.csv"
    generate_paysim_stream(
        total_weeks=4, tx_per_week=100, fraud_rate=0.15, seed=42, output_path=path
    )

    class AlwaysDrift:
        drift_events = []

        def update(self, *a, **kw):
            return {"any_drift": True}

        def reset_detectors(self):
            pass

    monkeypatch.setattr("src.pipeline.DriftMonitor", AlwaysDrift)
    result = run_pipeline(
        path,
        output_dir=tmp_path / "runs",
        data_kind="synthetic",
        model_params={"n_estimators": 10},
    )
    d = pd.read_csv(result["run_dir"] / "predictions.csv")
    first = d[d.slice_index == 2].pivot(
        index="trans_id", columns="model", values="probability"
    )
    for c in first:
        np.testing.assert_allclose(first[c], first["static"])
    events = json.loads((result["run_dir"] / "retrain_events.json").read_text())
    assert all(e["slice_idx"] >= 2 for e in events)
    assert len([e for e in events if e["model"] == "adaptive"]) == 2
    assert d[d.slice_index == 2].model_generation.eq(0).all()


def test_retrain_budget_enforced(synthetic_stream_df):
    p = Preprocessor().fit(synthetic_stream_df)
    d = p.transform(synthetic_stream_df)
    m = DriftAdaptiveModel(
        p.feature_names, max_retrains=0, model_params={"n_estimators": 10}
    )
    m.fit_initial(d)
    assert not m.update_with_slice(d, True, 5)
    assert m.adaptation_count == 0


def test_first_arrival_cohort_has_one_row_per_new_entity(
    synthetic_stream_df, monkeypatch
):
    from src.evaluate_uid_split import run_single_audit

    df = synthetic_stream_df.copy()
    df["uid_clean"] = "A"
    df["uid_cf"] = "A"
    df["is_clean"] = True
    # Explicit slices: entity B appears twice in the same test slice.
    slices = []
    for i in range(5):
        part = df.iloc[i * 60 : (i + 1) * 60].copy()
        if i == 1:
            part.loc[part.index[:2], ["uid_clean", "uid_cf"]] = "B"
            part.loc[part.index[0], "isFraud"] = 1
        slices.append({"slice_index": i, "data": part})
    monkeypatch.setattr("src.evaluate_uid_split.print_audit_report", lambda *a: None)
    r = run_single_audit(
        df, True, "test", preprocessor=Preprocessor(), raw_slices=slices
    )
    assert r["clean"]["strict"]["total_tx"] == 1


def test_identity_join_and_additional_categorical_features(tmp_path):
    from src.ingestion import DatasetIngestion

    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [0, 600000, 1200000, 1800000],
            "TransactionAmt": [10, 20, 30, 40],
            "isFraud": [0, 1, 0, 1],
            "M1": ["T", "F", "T", "F"],
            "P_emaildomain": ["a", "b", "a", "c"],
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": [1, 3],
            "DeviceType": ["mobile", "desktop"],
            "id_30": ["osA", "osB"],
        }
    )
    transactions.to_csv(tmp_path / "tx.csv", index=False)
    identity.to_csv(tmp_path / "id.csv", index=False)
    df, _ = DatasetIngestion(
        tmp_path / "tx.csv", identity_path=tmp_path / "id.csv", all_features=True
    ).load_and_validate()
    prep = Preprocessor().fit(df)
    assert (
        "DeviceType" in prep.categorical_cols
        and "M1" in prep.categorical_cols
        and "P_emaildomain" in prep.categorical_cols
    )
    assert len(prep.transform(df)) == 4
    identity = pd.concat([identity, identity.iloc[:1]], ignore_index=True)
    identity.to_csv(tmp_path / "bad.csv", index=False)
    from src.ingestion import IngestionError

    with pytest.raises(IngestionError, match="unique"):
        DatasetIngestion(
            tmp_path / "tx.csv", identity_path=tmp_path / "bad.csv"
        ).load_and_validate()
