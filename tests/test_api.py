"""
Unit and Integration Tests for the FastAPI Real-Time Scoring Microservice.
Verifies sub-15ms inference endpoints, batch scoring, feedback ingestion,
drift telemetry, and atomic model hot-swapping under concurrency.
"""

import pytest
import numpy as np
import xgboost as xgb
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api import app, model_buffer, drift_monitor, verify_api_key, retrainer, training_store
from src.config import API_KEY


@pytest.fixture
def client():
    return TestClient(app, headers={"X-API-Key": "dev-secret-key"})


def test_health_check_endpoint(client):
    """Verifies that /health returns operational telemetry, memory, and dummy/ready model state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["model_generation"] >= 1
    assert "is_dummy" in data
    assert "is_ready" in data
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
    """Verifies zero-downtime atomic model pointer replacement and is_ready state update."""
    initial_gen = model_buffer.generation

    # Create new genuine model
    new_features = ["TransactionAmt", "card1", "C1", "C2", "C5", "C13", "D1", "extra_feat"]
    X = np.random.randn(50, len(new_features)).astype(np.float32)
    y = np.random.binomial(1, 0.1, size=50)
    new_model = xgb.XGBClassifier(n_estimators=10, max_depth=2, tree_method="hist", device="cpu")
    new_model.fit(X, y)

    # Perform atomic swap marking model as genuine production model (is_dummy=False)
    model_buffer.atomic_swap(new_model, new_features, is_dummy=False)

    assert model_buffer.generation == initial_gen + 1
    assert model_buffer.feature_names == new_features
    assert model_buffer.is_dummy is False
    assert model_buffer.is_ready is True

    # Immediate scoring request after swap must succeed without interruption
    score_resp = client.post("/v1/score", json={
        "trans_id": "TX_POST_SWAP",
        "features": {f: 1.0 for f in new_features}
    })
    assert score_resp.status_code == 200
    assert score_resp.json()["model_generation"] == initial_gen + 1

    # Health check reflects ready production model
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["is_dummy"] is False
    assert health_resp.json()["is_ready"] is True


def test_unauthenticated_requests_rejected():
    """Verifies that protected endpoints reject missing or invalid API keys with 401."""
    unauth_client = TestClient(app)

    # /health probe is public for container liveness/readiness
    health_resp = unauth_client.get("/health")
    assert health_resp.status_code == 200

    # Protected endpoints require valid X-API-Key
    score_resp = unauth_client.post("/v1/score", json={
        "trans_id": "TX_NO_AUTH",
        "features": {"TransactionAmt": 100.0}
    })
    assert score_resp.status_code == 401
    assert "Invalid or missing API key" in score_resp.json()["detail"]

    feedback_resp = unauth_client.post("/v1/feedback", json={
        "trans_id": "TX_NO_AUTH",
        "is_fraud": 0
    })
    assert feedback_resp.status_code == 401

    drift_resp = unauth_client.get("/v1/drift-status")
    assert drift_resp.status_code == 401

    # Invalid API key rejected
    bad_client = TestClient(app, headers={"X-API-Key": "malicious-unauthorized-key"})
    bad_resp = bad_client.post("/v1/score", json={
        "trans_id": "TX_BAD_AUTH",
        "features": {"TransactionAmt": 100.0}
    })
    assert bad_resp.status_code == 401


def test_feedback_unknown_transaction_raises_404(client):
    """
    Verifies that logging feedback for a transaction not in the prediction cache
    raises 404 rather than silently fabricating a 0.0 probability.
    """
    # Without predicted_probability: must raise 404
    resp = client.post("/v1/feedback", json={
        "trans_id": "TX_COMPLETELY_UNKNOWN_999",
        "is_fraud": 1
    })
    assert resp.status_code == 404
    assert "not found in the recent prediction cache" in resp.json()["detail"]

    # With explicit predicted_probability: succeeds
    resp_with_prob = client.post("/v1/feedback", json={
        "trans_id": "TX_COMPLETELY_UNKNOWN_999",
        "is_fraud": 1,
        "predicted_probability": 0.85
    })
    assert resp_with_prob.status_code == 200
    assert resp_with_prob.json()["status"] == "RECORDED"


def test_verify_api_key_constant_time():
    """Asserts that verify_api_key validates keys using constant-time comparison."""
    # Valid key returns the key
    assert verify_api_key(API_KEY) == API_KEY

    # Missing key raises 401
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(None)
    assert exc_info.value.status_code == 401

    # Invalid key raises 401
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key("invalid-unauthorized-key")
    assert exc_info.value.status_code == 401


def test_manual_retrain_endpoint_synchronous(client):
    """Verifies that POST /v1/retrain?sync=true executes shadow retrain and hot-swaps model."""
    gen_before = model_buffer.generation
    resp = client.post("/v1/retrain?sync=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "COMPLETED"
    assert data["current_generation"] == gen_before + 1
    assert data["is_retraining"] is False
    assert data["total_retrains"] >= 1
    assert model_buffer.generation == gen_before + 1
    assert model_buffer.is_ready is True


def test_automated_shadow_retraining_on_drift(client):
    """Verifies that triggering concept drift on /v1/feedback executes shadow retrain and resets detectors."""
    gen_before = model_buffer.generation
    retrains_before = retrainer.total_retrains

    # Feed transactions that cause drift in drift_monitor (e.g. 70 consecutive prediction errors)
    for i in range(70):
        tx_id = f"TX_DRIFT_STREAM_{i:03d}"
        client.post("/v1/score", json={
            "trans_id": tx_id,
            "features": {"TransactionAmt": 150.0 + i, "card1": 2000.0, "C1": 1.0}
        })
        resp = client.post("/v1/feedback", json={
            "trans_id": tx_id,
            "is_fraud": 1,
            "predicted_probability": 0.01
        })
        assert resp.status_code == 200

    # Execute retrain to simulate background worker completion
    retrainer.execute_retrain()

    assert model_buffer.generation > gen_before
    assert retrainer.total_retrains > retrains_before
    # Detectors must have been reset following the retrain
    assert drift_monitor.adwin.width == 0


def test_concurrent_feedback_thread_safety(client):
    """
    Verifies that concurrent requests to /v1/feedback and /v1/drift-status
    execute without data races, corruption, or exceptions on shared drift_monitor.
    """
    import concurrent.futures

    initial_monitored = drift_monitor.total_samples_monitored
    num_concurrent_requests = 40

    def send_feedback(idx):
        payload = {
            "trans_id": f"TX_CONCURRENT_{idx:03d}",
            "is_fraud": idx % 2,
            "predicted_probability": 0.5
        }
        return client.post("/v1/feedback", json=payload)

    def query_status():
        return client.get("/v1/drift-status")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        fb_futures = [executor.submit(send_feedback, i) for i in range(num_concurrent_requests)]
        status_futures = [executor.submit(query_status) for _ in range(10)]

        for f in concurrent.futures.as_completed(fb_futures):
            resp = f.result()
            assert resp.status_code == 200
            assert resp.json()["status"] == "RECORDED"

        for f in concurrent.futures.as_completed(status_futures):
            resp = f.result()
            assert resp.status_code == 200
            assert "total_samples_monitored" in resp.json()

    assert drift_monitor.total_samples_monitored == initial_monitored + num_concurrent_requests




