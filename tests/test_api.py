"""Serving regressions: no placeholder scoring, schema parity, durable original decisions."""

import copy
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src import api
from src.artifacts import ModelBundle, save_bundle, load_bundle
from src.static_model import StaticBaselineModel
from src.preprocessing import Preprocessor
from src.config import API_KEY


@pytest.fixture
def serving(tmp_path, monkeypatch, synthetic_stream_df):
    df = synthetic_stream_df.copy()
    prep = Preprocessor().fit(df.iloc[:200])
    model = StaticBaselineModel(
        prep.feature_names, model_params={"n_estimators": 10}
    ).fit(prep.transform(df.iloc[:200]))
    # Explicit test fixture double with verified mock provenance
    from src.provenance import create_verified_test_double_provenance
    bundle = ModelBundle(
        model,
        prep,
        {
            "run_id": "test",
            "model": "static",
            "data_kind": "real",
            "version": "0.6.0",
            "provenance": create_verified_test_double_provenance("test_api"),
        },
    )
    buffer = api.ModelDoubleBuffer()
    buffer.atomic_swap(bundle)
    monkeypatch.setattr(api, "model_buffer", buffer)
    monkeypatch.setattr(
        api, "ledger", api.PredictionStore(tmp_path / "predictions.sqlite")
    )
    from src.drift_detector import DriftMonitor

    monkeypatch.setattr(api, "drift_monitor", DriftMonitor())
    monkeypatch.setattr(api, "retrainer", api.BackgroundRetrainer())
    client = TestClient(api.app, headers={"X-API-Key": API_KEY})
    row = df.iloc[220][prep.required_input_columns].to_dict()
    row = {k: None if pd.isna(v) else v for k, v in row.items()}
    return client, bundle, row


def test_no_model_or_synthetic_model_cannot_score(serving, monkeypatch):
    client, bundle, row = serving
    monkeypatch.setattr(api, "model_buffer", api.ModelDoubleBuffer())
    assert client.get("/health").json()["is_ready"] is False
    assert (
        client.post("/v1/score", json={"trans_id": "a", "features": row}).status_code
        == 503
    )
    bundle.metadata["data_kind"] = "synthetic"
    api.model_buffer.atomic_swap(bundle)
    assert (
        client.post("/v1/score", json={"trans_id": "a", "features": row}).status_code
        == 503
    )


def test_authentication(serving):
    client, _, row = serving
    response = TestClient(api.app).post(
        "/v1/score", json={"trans_id": "x", "features": row}
    )
    assert response.status_code == 401
    assert client.get("/health").status_code == 200


def test_scoring_matches_bundle_and_roundtrip(serving, tmp_path):
    client, bundle, row = serving
    save_bundle(bundle, tmp_path / "model")
    loaded = load_bundle(tmp_path / "model")
    predicted, prob = loaded.predict([row])
    response = client.post("/v1/score", json={"trans_id": "a", "features": row})
    assert response.status_code == 200
    assert response.json()["fraud_probability"] == pytest.approx(prob[0])
    assert response.json()["is_fraud"] == bool(predicted[0])


def test_missing_features_rejected(serving):
    client, _, row = serving
    row.pop(next(iter(row)))
    assert (
        client.post("/v1/score", json={"trans_id": "a", "features": row}).status_code
        == 422
    )


def test_duplicate_score_preserves_original(serving):
    client, _, row = serving
    assert (
        client.post("/v1/score", json={"trans_id": "a", "features": row}).status_code
        == 200
    )
    assert (
        client.post("/v1/score", json={"trans_id": "a", "features": row}).status_code
        == 409
    )


def test_feedback_idempotent_and_conflicting_labels_rejected(serving):
    client, _, row = serving
    client.post("/v1/score", json={"trans_id": "a", "features": row})
    assert (
        client.post("/v1/feedback", json={"trans_id": "a", "is_fraud": 1}).json()[
            "status"
        ]
        == "RECORDED"
    )
    assert (
        client.post("/v1/feedback", json={"trans_id": "a", "is_fraud": 1}).json()[
            "status"
        ]
        == "ALREADY_RECORDED"
    )
    assert api.drift_monitor.total_samples_monitored == 1
    assert (
        client.post("/v1/feedback", json={"trans_id": "a", "is_fraud": 0}).status_code
        == 409
    )


def test_feedback_unknown_rejected(serving):
    client, _, _ = serving
    assert (
        client.post(
            "/v1/feedback", json={"trans_id": "unknown", "is_fraud": 1}
        ).status_code
        == 404
    )


def test_old_generation_feedback_does_not_pollute_current_detector(serving):
    client, bundle, row = serving
    client.post("/v1/score", json={"trans_id": "a", "features": row})
    api.model_buffer.atomic_swap(copy.deepcopy(bundle))
    response = client.post("/v1/feedback", json={"trans_id": "a", "is_fraud": 1})
    assert response.json()["monitored"] is False
    assert api.drift_monitor.total_samples_monitored == 0


def test_batch_is_atomic_and_reports_errors(serving):
    client, _, row = serving
    response = client.post(
        "/v1/batch-score",
        json={
            "transactions": [{"trans_id": str(i), "features": row} for i in range(3)]
        },
    )
    assert response.status_code == 200 and response.json()["total_processed"] == 3
    assert (
        client.post("/v1/batch-score", json={"transactions": []}).json()[
            "total_processed"
        ]
        == 0
    )


def test_checksum_detects_changed_bundle(serving, tmp_path):
    _, bundle, _ = serving
    save_bundle(bundle, tmp_path / "model")
    with (tmp_path / "model" / "bundle.joblib").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path / "model")


def test_retrain_needs_verified_data(serving):
    client, _, _ = serving
    assert (
        client.post("/v1/retrain?sync=true").json()["status"]
        == "SKIPPED_INSUFFICIENT_DATA"
    )


def test_ledger_survives_reopen(serving):
    client, _, row = serving
    client.post("/v1/score", json={"trans_id": "a", "features": row})
    reopened = api.PredictionStore(api.ledger.path)
    original, fresh = reopened.feedback("a", 1)
    assert fresh and original[2] == 1


def test_restart_cannot_treat_old_revision_as_current(serving, monkeypatch):
    client, bundle, row = serving
    bundle.metadata["model_revision"] = "old-revision"
    client.post("/v1/score", json={"trans_id": "a", "features": row})
    new = copy.deepcopy(bundle)
    new.metadata["model_revision"] = "new-revision"
    replacement = api.ModelDoubleBuffer()
    replacement.atomic_swap(new)  # generation resets after restart
    monkeypatch.setattr(api, "model_buffer", replacement)
    response = client.post("/v1/feedback", json={"trans_id": "a", "is_fraud": 1})
    assert not response.json()["monitored"]


def test_background_candidate_promotes_only_after_holdout_gain(tmp_path, monkeypatch):
    # Deliberately reversed relation creates a measurable adaptation requirement.
    amount = np.tile([0.0, 100.0], 100)
    original = pd.DataFrame(
        {
            "step": np.arange(1, 201),
            "amount": amount,
            "isFraud": (amount == 0).astype(int),
        }
    )
    prep = Preprocessor().fit(original.iloc[:160])
    model = StaticBaselineModel(
        prep.feature_names, model_params={"n_estimators": 20}
    ).fit(prep.transform(original))
    from src.provenance import create_verified_test_double_provenance
    bundle = ModelBundle(
        model,
        prep,
        {
            "run_id": "promotion-test",
            "model": "adaptive",
            "data_kind": "real",
            "model_revision": "original",
            "provenance": create_verified_test_double_provenance("promotion_test"),
        },
    )
    buffer = api.ModelDoubleBuffer()
    buffer.atomic_swap(bundle)
    monkeypatch.setattr(api, "model_buffer", buffer)
    monkeypatch.setattr(api, "ledger", api.PredictionStore(tmp_path / "ledger.sqlite"))
    monkeypatch.setattr(api.settings, "model_output_dir", str(tmp_path / "models"))
    rows = []
    for i, a in enumerate(amount):
        features = {"step": i + 201, "amount": float(a)}
        rows.append(
            (
                str(i),
                json.dumps(features),
                0.1,
                0,
                1,
                "original",
                "promotion-test:adaptive",
                i,
            )
        )
    api.ledger.record(rows)
    for i, a in enumerate(amount):
        api.ledger.feedback(str(i), int(a > 0))
    worker = api.BackgroundRetrainer()
    assert worker.execute_retrain()
    assert buffer.generation == 2
    assert buffer.snapshot()[0].metadata["model_revision"] != "original"
    assert len(list((tmp_path / "models").glob("generation-2-*/manifest.json"))) == 1
    assert (tmp_path / "models" / "active.json").exists()


def test_accepted_revision_is_restored_after_service_restart(
    serving, tmp_path, monkeypatch
):
    client, bundle, _ = serving
    initial = tmp_path / "initial"
    save_bundle(bundle, initial)
    accepted = copy.deepcopy(bundle)
    accepted.metadata["model_revision"] = "accepted-after-retrain"
    state = tmp_path / "state"
    save_bundle(accepted, state / "generation-2-accepted")
    (state / "active.json").write_text(
        json.dumps(
            {
                "family_id": bundle.metadata["run_id"]
                + ":"
                + bundle.metadata.get("model", "default"),
                "generation": 2,
                "package": "generation-2-accepted",
            }
        )
    )
    monkeypatch.setattr(api.settings, "model_bundle_path", str(initial))
    monkeypatch.setattr(api.settings, "model_output_dir", str(state))
    monkeypatch.setattr(api, "model_buffer", api.ModelDoubleBuffer())
    api.initialize_app_model()
    restored, generation = api.model_buffer.snapshot()
    assert (
        generation == 2
        and restored.metadata["model_revision"] == "accepted-after-retrain"
    )


def test_nested_feature_values_rejected_as_422(serving):
    client, _, row = serving
    row[next(iter(row))] = [1, 2]
    response = client.post("/v1/score", json={"trans_id": "nested", "features": row})
    assert response.status_code == 422
