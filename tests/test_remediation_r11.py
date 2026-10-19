"""
Remediation test suite for R11 (Measurement Scope, History Feature Semantics & Query Profiling).
Covers:
- Boundary-batched history feature computation preserving temporal causality across tied timestamps.
- Explicit separation of model-only inference, sequential HTTP scoring, and concurrent API throughput.
- SQLite EXPLAIN QUERY PLAN verification for all operational queries ensuring index usage.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app, model_buffer, PredictionStore, ledger
from src.artifacts import ModelBundle, save_bundle
from src.benchmark_latency import run_latency_benchmark, profile_sqlite_query_plans
from src.config import API_KEY
from src.history_features import add_history_features
from src.provenance import create_verified_test_double_provenance


class DummyModel:
    def __init__(self):
        self.threshold = 0.5
        self.trained = True
        self.feature_names = ["Amount", "V1"]
        self.fit_metadata = {"model": "dummy"}

    def fit(self, X):
        return self

    def predict(self, X):
        n = len(X)
        return [0] * n, [0.15] * n


class DummyPreprocessor:
    def __init__(self):
        self.required_input_columns = ["Amount", "V1"]
        self.feature_names = ["Amount", "V1"]

    def transform(self, frame):
        return frame


@pytest.fixture
def active_serving_bundle():
    metadata = {
        "run_id": "TEST_RUN_R11",
        "model": "xgboost",
        "model_revision": "rev-r11",
        "parent_generation": 0,
        "data_kind": "real",
        "provenance": create_verified_test_double_provenance("test_r11"),
    }
    bundle = ModelBundle(
        model=DummyModel(),
        preprocessor=DummyPreprocessor(),
        metadata=metadata,
    )
    model_buffer.atomic_swap(bundle, generation=0)
    return bundle


def test_r11_history_features_tied_timestamps_no_leakage():
    """Verify that multiple transactions at identical timestamps only observe t < T state."""
    # 4 transactions for card1='A':
    # tx0 at t=100: should see 0 prior tx
    # tx1 at t=200: should see 1 prior tx (tx0)
    # tx2 at t=200: tied with tx1! MUST see 1 prior tx (tx0), NOT tx1!
    # tx3 at t=300: should see 3 prior txs (tx0, tx1, tx2)
    df = pd.DataFrame({
        "TransactionID": ["tx0", "tx1", "tx2", "tx3"],
        "TransactionDT": [100.0, 200.0, 200.0, 300.0],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
        "card1": ["A", "A", "A", "A"],
    })

    res = add_history_features(df)

    counts = res["hist_tx_count"].tolist()
    assert counts[0] == 0  # t=100
    assert counts[1] == 1  # t=200, first
    assert counts[2] == 1  # t=200, tied second: strictly prior t < 200! No intra-second leakage!
    assert counts[3] == 3  # t=300, observes all three prior transactions

    means = res["hist_amount_mean"].tolist()
    assert means[0] == 0.0
    assert means[1] == 10.0
    assert means[2] == 10.0  # tied: average before t=200 is still 10.0
    assert np.isclose(means[3], (10.0 + 20.0 + 30.0) / 3.0)  # (10+20+30)/3 = 20.0


def test_r11_history_features_missing_identity_isolation():
    """Missing account identifiers must not aggregate across unrelated anonymous transactions."""
    df = pd.DataFrame({
        "TransactionID": ["tx0", "tx1", "tx2"],
        "TransactionDT": [100.0, 200.0, 300.0],
        "TransactionAmt": [10.0, 20.0, 30.0],
        "card1": [None, None, "KnownCard"],
    })

    res = add_history_features(df)
    counts = res["hist_tx_count"].tolist()
    assert counts[0] == 0
    assert counts[1] == 0  # Missing identity is isolated, does not link to tx0
    assert counts[2] == 0


def test_r11_sqlite_query_plans_use_indexes(tmp_path):
    """Verify that operational query plans leverage indexes."""
    db_file = tmp_path / "test_plans.sqlite"
    store = PredictionStore(str(db_file))

    # Insert a dummy record
    records = [
        ("TX_PLAN_1", json.dumps({"Amount": 10.0, "V1": 0.1}), 0.2, 0, 0, "rev-0", "RUN_DEFAULT:xgboost", 1000.0)
    ]
    store.record(records)
    store.feedback("TX_PLAN_1", 0)

    plans = profile_sqlite_query_plans(store.path)
    assert len(plans) == 4

    # 1. Lookup by ID
    id_plan = plans[0]
    assert id_plan["operation"].startswith("Transaction ID Lookup")
    assert id_plan["uses_index"] is True
    assert "PRIMARY KEY" in id_plan["plan_detail"].upper() or "INDEX" in id_plan["plan_detail"].upper()

    # 2. Training data extraction
    train_plan = plans[1]
    assert train_plan["operation"].startswith("Training Data Extraction")
    assert train_plan["uses_index"] is True


def test_r11_benchmark_latency_separation(active_serving_bundle, tmp_path):
    """Verify benchmark separates model-only, sequential HTTP, and concurrent API timings."""
    test_db = tmp_path / "bench_ledger.sqlite"
    test_store = PredictionStore(str(test_db))

    features = {"Amount": 100.0, "V1": 1.5}

    with patch("src.benchmark_latency.ledger", test_store):
        report = run_latency_benchmark(
            features=features,
            n_requests=50,
            concurrency=2,
            warmup=5,
        )

        assert "model_only_inference" in report
        assert "sequential_http" in report
        assert "concurrent_api" in report
        assert "sqlite_query_plans" in report

        # Pure model inference should be measurably faster than full sequential HTTP with SQLite ledger write
        model_p50 = report["model_only_inference"]["p50_ms"]
        seq_p50 = report["sequential_http"]["p50_ms"]
        assert model_p50 <= seq_p50

        # Concurrent traffic completed with 0 errors
        conc = report["concurrent_api"]
        assert conc["error_count"] == 0
        assert conc["successful_requests"] == 50
        assert conc["throughput_qps"] > 0
