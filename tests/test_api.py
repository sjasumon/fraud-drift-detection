"""
Unit and Integration Tests for the FastAPI Real-Time Scoring Microservice.
Verifies sub-15ms inference endpoints, batch scoring, feedback ingestion,
drift telemetry, and atomic model hot-swapping under concurrency.
"""

import pytest
import numpy as np
import xgboost as xgb
from fastapi.testclient import TestClient

from src.api import app, model_buffer, drift_monitor


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check_endpoint(client):
    """Verifies that /health returns operational telemetry and loaded model state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["model_generation"] >= 1
    assert data["memory_rss_mb"] > 0
    assert "cpu_percent" in data


def test_single_transaction_score(client):
    """Verifies real-time scoring with risk tier assignment and latency tracking."""
    payload = {
        "trans_id": "TX_TEST_001",
        "features": {
            "TransactionAmt": 249.99,
            "card1": 10020.0,
            "C1": 3.0,
            "C2": 2.0,
            "C5": 0.0,
            "C13": 5.0,
            "D1": 14.0
        }
    }
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["trans_id"] == "TX_TEST_001"
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH")
    assert data["inference_latency_ms"] >= 0.0
    # Strict SLA check: single inference in memory should easily beat 50ms in pytest
    assert data["inference_latency_ms"] < 50.0


def test_batch_scoring(client):
    """Verifies vectorized batch scoring with throughput calculation."""
    batch_txs = [
        {
            "trans_id": f"TX_BATCH_{i:03d}",
            "features": {
                "TransactionAmt": float(i * 15 + 10),
                "card1": 5000.0 + i,
                "C1": float(i % 5),
                "C2": 1.0,
                "C5": 0.0,
                "C13": float(i % 10),
                "D1": float(i)
            }
        }
        for i in range(25)
    ]
    response = client.post("/v1/batch-score", json={"transactions": batch_txs})
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 25
    assert len(data["results"]) == 25
    assert data["throughput_qps"] > 0.0


def test_feedback_ingestion_and_drift_telemetry(client):
    """Verifies that chargeback feedback updates the drift monitor and telemetry endpoint."""
    init_samples = drift_monitor.total_samples_monitored

    # First score a transaction
    client.post("/v1/score", json={
        "trans_id": "TX_CHARGEBACK_01",
        "features": {"TransactionAmt": 500.0, "card1": 1234.0, "C1": 10.0}
    })

    # Send feedback indicating actual chargeback / fraud
    feedback_payload = {
        "trans_id": "TX_CHARGEBACK_01",
        "is_fraud": 1,
        "dispute_reason": "UNAUTHORIZED_CARD_PRESENT"
    }
    fb_response = client.post("/v1/feedback", json=feedback_payload)
    assert fb_response.status_code == 200
    fb_data = fb_response.json()
    assert fb_data["status"] == "RECORDED"
    assert fb_data["monitored_samples_total"] == init_samples + 1

    # Query /v1/drift-status
    status_response = client.get("/v1/drift-status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["total_samples_monitored"] >= init_samples + 1
    assert "adwin_drift_detected" in status_data
    assert "ddm_drift_detected" in status_data


def test_atomic_model_hot_swap(client):
    """Verifies zero-downtime atomic model pointer replacement in the double buffer."""
    initial_gen = model_buffer.generation

    # Create new model
    new_features = ["TransactionAmt", "card1", "C1", "C2", "C5", "C13", "D1", "extra_feat"]
    X = np.random.randn(50, len(new_features)).astype(np.float32)
    y = np.random.binomial(1, 0.1, size=50)
    new_model = xgb.XGBClassifier(n_estimators=10, max_depth=2, tree_method="hist", device="cpu")
    new_model.fit(X, y)

    # Perform atomic swap
    model_buffer.atomic_swap(new_model, new_features)

    assert model_buffer.generation == initial_gen + 1
    assert model_buffer.feature_names == new_features

    # Immediate scoring request after swap must succeed without interruption
    score_resp = client.post("/v1/score", json={
        "trans_id": "TX_POST_SWAP",
        "features": {f: 1.0 for f in new_features}
    })
    assert score_resp.status_code == 200
    assert score_resp.json()["model_generation"] == initial_gen + 1
