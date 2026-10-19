"""
Remediation test suite for R07 (Rollback & Pointer Synchronization) and R09 (Timeout, Promotion & Feedback History).
Covers:
- Retrainer execution branches: Insufficient Data, Insufficient Classes, Training Timeout, Fit Failure, No Holdout Gain, Insufficient Margin, and Promotion.
- Persistent SQLite promotion history across service restarts.
- Dual-pointer synchronization (models/active.json and service/active_model.json).
- Authenticated rollback endpoint (previous generation, specific generation, base model gen 0).
- Corrupted/unready candidate rejection on rollback and service restart fallback.
- Append-only feedback audit trail with received_at, confirmed_at, label_source, replay, and conflict rejection.
"""

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import (
    app,
    ledger,
    model_buffer,
    retrainer,
    BackgroundRetrainer,
    PredictionStore,
    initialize_app_model,
)
from src.artifacts import ModelBundle, save_bundle, load_bundle
from src.config import API_KEY, RESULTS_DIR, settings
from src.provenance import create_verified_test_double_provenance


class DummyModel:
    def __init__(self, threshold=0.5, predict_val=0, prob_val=0.1):
        self.threshold = threshold
        self.predict_val = predict_val
        self.prob_val = prob_val
        self.trained = True
        self.feature_names = ["Amount", "V1"]

    def fit(self, X):
        return self

    def predict(self, X):
        n = len(X)
        if isinstance(self.predict_val, list):
            preds = (self.predict_val * (n // len(self.predict_val) + 1))[:n]
        else:
            preds = [self.predict_val] * n
        return preds, [self.prob_val] * n


class DummyPreprocessor:
    def __init__(self):
        self.required_input_columns = ["Amount", "V1"]
        self.feature_names = ["Amount", "V1"]

    def transform(self, frame):
        return frame


def create_test_bundle(ready=True, run_id="TEST_RUN_R07_R09", generation=0, model_revision="rev-0"):
    metadata = {
        "run_id": run_id,
        "model": "xgboost",
        "model_revision": model_revision,
        "parent_generation": generation,
        "data_kind": "real" if ready else "synthetic",
        "training_samples": 500,
        "created_at": "2026-10-19T12:00:00Z",
    }
    if ready:
        metadata["provenance"] = create_verified_test_double_provenance(f"test_{model_revision}")

    model = DummyModel(threshold=0.5, predict_val=0, prob_val=0.2)
    model.trained = ready
    return ModelBundle(
        model=model,
        preprocessor=DummyPreprocessor(),
        metadata=metadata,
    )


@pytest.fixture
def auth_headers():
    return {"X-API-Key": API_KEY}


# =========================================================================
# R09: Retrainer Execution Branches & Timeout Guard
# =========================================================================

def test_r09_retrainer_insufficient_data(tmp_path):
    db_file = tmp_path / "test_insufficient.sqlite"
    store = PredictionStore(str(db_file))
    r = BackgroundRetrainer(store=store, min_improvement_margin=0.01)

    bundle = create_test_bundle(ready=True)
    model_buffer.atomic_swap(bundle, generation=0)

    # With 0 rows
    success = r.execute_retrain()
    assert success is False
    assert r.last_retrain_status == "SKIPPED_INSUFFICIENT_DATA"
    assert len(r.promotion_log) == 1
    assert r.promotion_log[-1]["decision"] == "SKIPPED"
    assert "Insufficient labeled feedback" in r.promotion_log[-1]["reason"]


def test_r09_retrainer_insufficient_classes(tmp_path):
    db_file = tmp_path / "test_classes.sqlite"
    store = PredictionStore(str(db_file))
    r = BackgroundRetrainer(store=store, min_improvement_margin=0.01)

    bundle = create_test_bundle(ready=True)
    model_buffer.atomic_swap(bundle, generation=0)

    # Insert 120 records, but all label=0
    family = bundle.metadata["run_id"] + ":" + bundle.metadata.get("model", "default")
    records = [
        (f"tx_{i}", json.dumps({"Amount": 10.0, "V1": 1.0}), 0.1, 0, 0, "rev-0", family, float(i))
        for i in range(120)
    ]
    store.record(records)
    for i in range(120):
        store.feedback(f"tx_{i}", 0)

    success = r.execute_retrain()
    assert success is False
    assert r.last_retrain_status == "SKIPPED_INSUFFICIENT_CLASSES"
    assert r.promotion_log[-1]["decision"] == "SKIPPED"
    assert "Both classes" in r.promotion_log[-1]["reason"]


def test_r09_retrainer_timeout_branch(tmp_path):
    db_file = tmp_path / "test_timeout.sqlite"
    store = PredictionStore(str(db_file))
    # Set a tiny 0.05s timeout
    r = BackgroundRetrainer(store=store, min_improvement_margin=0.01, max_retrain_duration_sec=0.05)

    bundle = create_test_bundle(ready=True)
    model_buffer.atomic_swap(bundle, generation=0)

    # Insert 120 balanced records
    family = bundle.metadata["run_id"] + ":" + bundle.metadata.get("model", "default")
    records = [
        (f"tx_{i}", json.dumps({"Amount": float(i), "V1": float(i % 2)}), 0.1, 0, 0, "rev-0", family, float(i))
        for i in range(120)
    ]
    store.record(records)
    for i in range(120):
        store.feedback(f"tx_{i}", i % 2)

    # Patch candidate fit to sleep for 0.2s
    def slow_fit(df):
        time.sleep(0.2)
        return bundle.model

    with patch.object(DummyModel, "fit", side_effect=slow_fit):
        success = r.execute_retrain()
        assert success is False
        assert r.last_retrain_status == "REJECTED_TRAINING_TIMEOUT"
        assert r.promotion_log[-1]["decision"] == "REJECTED"
        assert "exceeded deadline" in r.promotion_log[-1]["reason"]


def test_r09_retrainer_fit_failure_branch(tmp_path):
    db_file = tmp_path / "test_fail.sqlite"
    store = PredictionStore(str(db_file))
    r = BackgroundRetrainer(store=store, min_improvement_margin=0.01)

    bundle = create_test_bundle(ready=True)
    model_buffer.atomic_swap(bundle, generation=0)

    family = bundle.metadata["run_id"] + ":" + bundle.metadata.get("model", "default")
    records = [
        (f"tx_{i}", json.dumps({"Amount": float(i), "V1": float(i % 2)}), 0.1, 0, 0, "rev-0", family, float(i))
        for i in range(120)
    ]
    store.record(records)
    for i in range(120):
        store.feedback(f"tx_{i}", i % 2)

    with patch.object(DummyModel, "fit", side_effect=RuntimeError("GPU OOM / LinAlg failure")):
        success = r.execute_retrain()
        assert success is False
        assert "FAILED" in r.last_retrain_status
        assert r.promotion_log[-1]["decision"] == "FAILED"
        assert "GPU OOM" in r.promotion_log[-1]["reason"]


def test_r09_retrainer_no_holdout_gain(tmp_path):
    db_file = tmp_path / "test_no_gain.sqlite"
    store = PredictionStore(str(db_file))
    r = BackgroundRetrainer(store=store, min_improvement_margin=0.01)

    bundle = create_test_bundle(ready=True)
    model_buffer.atomic_swap(bundle, generation=0)

    family = bundle.metadata["run_id"] + ":" + bundle.metadata.get("model", "default")
    records = [
        (f"tx_{i}", json.dumps({"Amount": float(i), "V1": float(i % 2)}), 0.1, 0, 0, "rev-0", family, float(i))
        for i in range(120)
    ]
    store.record(records)
    for i in range(120):
        store.feedback(f"tx_{i}", i % 2)

    # Both current and proposed output all 0s -> F1=0, gain=0 <= 0
    with patch.object(DummyModel, "predict", return_value=([0] * 24, [0.1] * 24)):
        success = r.execute_retrain()
        assert success is False
        assert r.last_retrain_status == "REJECTED_NO_HOLDOUT_GAIN"
        assert r.promotion_log[-1]["decision"] == "REJECTED"


def test_r09_retrainer_insufficient_margin(tmp_path):
    db_file = tmp_path / "test_insufficient_margin.sqlite"
    store = PredictionStore(str(db_file))
    # Require 0.05 margin
    r = BackgroundRetrainer(store=store, min_improvement_margin=0.05)

    bundle = create_test_bundle(ready=True)
    model_buffer.atomic_swap(bundle, generation=0)

    family = bundle.metadata["run_id"] + ":" + bundle.metadata.get("model", "default")
    records = [
        (f"tx_{i}", json.dumps({"Amount": float(i), "V1": float(i % 2)}), 0.1, 0, 0, "rev-0", family, float(i))
        for i in range(120)
    ]
    store.record(records)
    for i in range(120):
        store.feedback(f"tx_{i}", i % 2)

    # 24 holdout items, 12 are positive (isFraud=1).
    holdout_labels = [i % 2 for i in range(96, 120)]
    current_preds = list(holdout_labels)
    current_preds[1] = 0
    current_preds[3] = 0

    proposed_preds = list(holdout_labels)
    proposed_preds[1] = 0

    call_seq = [proposed_preds, current_preds]

    def mock_predict(df):
        preds = call_seq.pop(0) if call_seq else [0] * len(df)
        return preds, [0.8] * len(df)

    with patch.object(DummyModel, "predict", side_effect=mock_predict):
        success = r.execute_retrain()
        assert success is False
        assert r.last_retrain_status == "REJECTED_INSUFFICIENT_MARGIN"
        assert r.promotion_log[-1]["decision"] == "REJECTED"
        assert "below required margin" in r.promotion_log[-1]["reason"]


# =========================================================================
# R09: SQLite Promotion History Persistence Across Restarts
# =========================================================================

def test_r09_promotion_history_persistence_across_restarts(tmp_path):
    db_file = tmp_path / "persistent_promotions.sqlite"
    store1 = PredictionStore(str(db_file))
    r1 = BackgroundRetrainer(store=store1, min_improvement_margin=0.01)

    bundle = create_test_bundle(ready=True)
    model_buffer.atomic_swap(bundle, generation=0)

    # Trigger a skipped attempt
    r1.execute_retrain()
    assert len(r1.promotion_log) == 1

    # Simulate restart by instantiating new store and new retrainer on the same SQLite file
    store2 = PredictionStore(str(db_file))
    history = store2.get_promotion_history(10)
    assert len(history) == 1
    assert history[0]["status"] == "SKIPPED_INSUFFICIENT_DATA"
    assert history[0]["parent_generation"] == 0

    r2 = BackgroundRetrainer(store=store2, min_improvement_margin=0.01)
    assert len(r2.promotion_log) == 1
    assert r2.promotion_log[0]["attempt_id"] == history[0]["attempt_id"]


# =========================================================================
# R07: Dual Pointer Synchronization, Rollback Endpoint & Recovery
# =========================================================================

def test_r07_dual_pointer_and_rollback(tmp_path, auth_headers):
    client = TestClient(app)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    service_dir = tmp_path / "service"
    service_dir.mkdir(parents=True)

    base_bundle_path = tmp_path / "base_bundle"
    base_bundle = create_test_bundle(ready=True, generation=0, model_revision="rev-base")
    save_bundle(base_bundle, base_bundle_path)

    gen1_bundle_path = models_dir / "generation-1-rev1"
    gen1_bundle = create_test_bundle(ready=True, generation=1, model_revision="rev-1")
    save_bundle(gen1_bundle, gen1_bundle_path)

    gen2_bundle_path = models_dir / "generation-2-rev2"
    gen2_bundle = create_test_bundle(ready=True, generation=2, model_revision="rev-2")
    save_bundle(gen2_bundle, gen2_bundle_path)

    with patch.object(settings, "model_output_dir", str(models_dir)), \
         patch.object(settings, "model_bundle_path", str(base_bundle_path)), \
         patch("src.api.RESULTS_DIR", tmp_path):

        # 1. Activate gen 1
        resp = client.post(
            "/v1/models/activate",
            json={"package_name_or_path": "generation-1-rev1"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ACTIVATED"
        assert resp.json()["generation"] == 1

        # Check dual pointers
        active1 = json.loads((models_dir / "active.json").read_text())
        alt_active1 = json.loads((service_dir / "active_model.json").read_text())
        assert active1["package"] == "generation-1-rev1"
        assert alt_active1["package"] == "generation-1-rev1"

        # 2. Activate gen 2
        resp = client.post(
            "/v1/models/activate",
            json={"package_name_or_path": "generation-2-rev2"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["generation"] == 2

        active2 = json.loads((models_dir / "active.json").read_text())
        alt_active2 = json.loads((service_dir / "active_model.json").read_text())
        assert active2["package"] == "generation-2-rev2"
        assert alt_active2["package"] == "generation-2-rev2"

        # 3. Rollback to previous generation (should roll back to gen 1)
        resp_rb = client.post(
            "/v1/models/rollback",
            json={"reason": "Drift anomaly detected in gen 2"},
            headers=auth_headers,
        )
        assert resp_rb.status_code == 200
        rb_data = resp_rb.json()
        assert rb_data["status"] == "ROLLED_BACK"
        assert rb_data["generation"] == 1
        assert rb_data["previous_generation"] == 2
        assert rb_data["package"] == "generation-1-rev1"

        # Check dual pointers updated to gen 1
        active_rb = json.loads((models_dir / "active.json").read_text())
        alt_active_rb = json.loads((service_dir / "active_model.json").read_text())
        assert active_rb["package"] == "generation-1-rev1"
        assert active_rb["generation"] == 1
        assert alt_active_rb["package"] == "generation-1-rev1"

        # 4. Check promotions endpoint includes rollback record
        resp_prom = client.get("/v1/models/promotions", headers=auth_headers)
        assert resp_prom.status_code == 200
        prom_data = resp_prom.json()
        rollback_records = [p for p in prom_data["promotions"] if p["status"] == "ROLLED_BACK"]
        assert len(rollback_records) >= 1
        assert rollback_records[0]["candidate_generation"] == 1

        # 5. Rollback to generation 0 (base bundle)
        resp_rb0 = client.post(
            "/v1/models/rollback",
            json={"target_generation": 0, "reason": "Emergency reset to base"},
            headers=auth_headers,
        )
        assert resp_rb0.status_code == 200
        assert resp_rb0.json()["generation"] == 0


def test_r07_rollback_rejects_unready_or_corrupted(tmp_path, auth_headers):
    client = TestClient(app)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    base_bundle_path = tmp_path / "base_bundle"
    base_bundle = create_test_bundle(ready=True, generation=0)
    save_bundle(base_bundle, base_bundle_path)

    # Create unready bundle
    unready_path = models_dir / "generation-1-corrupted"
    unready_bundle = create_test_bundle(ready=False, generation=1)
    save_bundle(unready_bundle, unready_path)

    with patch.object(settings, "model_output_dir", str(models_dir)), \
         patch.object(settings, "model_bundle_path", str(base_bundle_path)), \
         patch("src.api.RESULTS_DIR", tmp_path):

        model_buffer.atomic_swap(base_bundle, generation=0)

        # Attempt to roll back / switch to corrupted package
        resp = client.post(
            "/v1/models/rollback",
            json={"package_name_or_path": "generation-1-corrupted"},
            headers=auth_headers,
        )
        assert resp.status_code == 422
        assert "provenance" in resp.json()["detail"].lower()


def test_r07_service_restart_with_corrupted_active_pointer(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    base_bundle_path = tmp_path / "base_bundle"
    base_bundle = create_test_bundle(ready=True, generation=0, model_revision="base-ok")
    save_bundle(base_bundle, base_bundle_path)

    # Corrupt pointer points to invalid bundle
    corrupt_pkg = models_dir / "generation-1-broken"
    corrupt_pkg.mkdir()
    (corrupt_pkg / "metadata.json").write_text("NOT_JSON")

    pointer = {
        "package": "generation-1-broken",
        "generation": 1,
        "family_id": "TEST_RUN_R07_R09:xgboost",
    }
    (models_dir / "active.json").write_text(json.dumps(pointer))

    with patch.object(settings, "model_output_dir", str(models_dir)), \
         patch.object(settings, "model_bundle_path", str(base_bundle_path)), \
         patch("src.api.RESULTS_DIR", tmp_path):

        initialize_app_model()
        cur_bundle, cur_gen = model_buffer.snapshot()
        # Must safely fall back to base bundle at generation 0
        assert cur_bundle.metadata["model_revision"] == "base-ok"
        assert cur_gen == 0


# =========================================================================
# R09: Feedback Audit Trail, Timestamps, Replay & Conflict
# =========================================================================

def test_r09_feedback_audit_trail_and_timestamps(tmp_path, auth_headers):
    client = TestClient(app)
    db_file = tmp_path / "test_feedback_audit.sqlite"
    test_store = PredictionStore(str(db_file))

    bundle = create_test_bundle(ready=True)
    model_buffer.atomic_swap(bundle, generation=0)

    with patch("src.api.ledger", test_store):
        # 1. Score a transaction
        tx = {
            "trans_id": "TX_AUDIT_101",
            "features": {"Amount": 150.0, "V1": 0.5},
            "timestamp": 1700000000.0,
        }
        resp = client.post("/v1/score", json=tx, headers=auth_headers)
        assert resp.status_code == 200

        # 2. Submit initial feedback with confirmation timestamp and source
        fb1 = {
            "trans_id": "TX_AUDIT_101",
            "is_fraud": 1,
            "label_source": "analyst_tier2_review",
            "confirmation_timestamp": 1700003600.0,
        }
        resp_fb1 = client.post("/v1/feedback", json=fb1, headers=auth_headers)
        assert resp_fb1.status_code == 200
        assert resp_fb1.json()["status"] == "RECORDED"

        # Check prediction record updated with timestamps
        with test_store.connect() as c:
            row = c.execute(
                "SELECT label, received_at, confirmed_at, label_source FROM predictions WHERE id=?",
                ("TX_AUDIT_101",)
            ).fetchone()
            assert row[0] == 1
            assert row[1] is not None  # received_at ISO string
            assert row[2] == 1700003600.0  # confirmed_at float
            assert row[3] == "analyst_tier2_review"

        # 3. Submit identical replay feedback
        resp_fb2 = client.post("/v1/feedback", json=fb1, headers=auth_headers)
        assert resp_fb2.status_code == 200
        assert resp_fb2.json()["status"] == "ALREADY_RECORDED"

        # 4. Submit conflicting feedback (is_fraud=0)
        fb_conflict = {
            "trans_id": "TX_AUDIT_101",
            "is_fraud": 0,
            "label_source": "dispute_reversal",
            "confirmation_timestamp": 1700007200.0,
        }
        resp_fb3 = client.post("/v1/feedback", json=fb_conflict, headers=auth_headers)
        assert resp_fb3.status_code == 409

        # 5. Query /v1/feedback/audit
        resp_audit = client.get("/v1/feedback/audit", headers=auth_headers)
        assert resp_audit.status_code == 200
        audit_data = resp_audit.json()
        assert audit_data["total"] == 3
        outcomes = [e["outcome"] for e in audit_data["events"]]
        assert "RECORDED_FRESH" in outcomes
        assert "IDEMPOTENT_REPLAY" in outcomes
        assert "CONFLICT_REJECTED" in outcomes
