"""
Unit tests for database management and persistence (src/db.py).
Tests schema creation, run logging, slice tracking, drift event persistence,
attribution logging, and SQL identifier safety using an in-memory SQLite engine.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine

from src.db import (
    DatabaseManager,
    ExperimentRun,
    TimeSliceRecord,
    DriftEventRecord,
    SliceMetricRecord,
    DriftAttributionRecord
)


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    return DatabaseManager(engine=engine)


def test_db_ensure_run_and_log_run(in_memory_db):
    """Verifies that ExperimentRun can be initialized and merged cleanly."""
    run_id = "RUN_TEST_001"
    in_memory_db.ensure_run_exists(run_id, dataset_name="TestSet", total_rows=500)

    with in_memory_db.get_session() as session:
        rec = session.query(ExperimentRun).filter_by(run_id=run_id).first()
        assert rec is not None
        assert rec.dataset_name == "TestSet"
        assert rec.total_rows == 500

    # Test updating run via log_run
    in_memory_db.log_run({
        "run_id": run_id,
        "dataset_name": "TestSet_Updated",
        "total_rows": 600,
        "time_slices_count": 4,
        "gpu_accelerated": "cpu",
        "static_f1_overall": 0.75,
        "adaptive_f1_overall": 0.88,
        "drift_alerts_count": 2,
        "notes": "Fast test execution"
    })

    with in_memory_db.get_session() as session:
        rec = session.query(ExperimentRun).filter_by(run_id=run_id).first()
        assert rec.dataset_name == "TestSet_Updated"
        assert rec.total_rows == 600
        assert rec.time_slices_count == 4
        assert rec.adaptive_f1_overall == 0.88


def test_db_log_slices(in_memory_db):
    """Verifies that time slice metadata records are persisted correctly."""
    run_id = "RUN_SLICES_001"
    slices = [
        {
            "slice_index": 0,
            "slice_name": "Week_1",
            "start_step": 1,
            "end_step": 168,
            "total_transactions": 1000,
            "fraud_transactions": 20,
            "fraud_ratio": 0.02
        },
        {
            "slice_index": 1,
            "slice_name": "Week_2",
            "start_step": 169,
            "end_step": 336,
            "total_transactions": 1100,
            "fraud_transactions": 25,
            "fraud_ratio": 0.0227
        }
    ]
    in_memory_db.log_slices(run_id, slices)

    with in_memory_db.get_session() as session:
        recs = session.query(TimeSliceRecord).filter_by(run_id=run_id).order_by(TimeSliceRecord.slice_index).all()
        assert len(recs) == 2
        assert recs[0].slice_name == "Week_1"
        assert recs[1].slice_name == "Week_2"
        assert recs[1].total_transactions == 1100


def test_db_log_drift_events(in_memory_db):
    """Verifies that drift detection alerts are persisted."""
    run_id = "RUN_DRIFT_001"
    events = [
        {
            "slice_index": 2,
            "step_number": 450,
            "detector_type": "ADWIN",
            "error_rate": 0.08,
            "confidence_score": 0.95,
            "status": "DRIFT",
            "notes": "Sudden mean error increase"
        }
    ]
    in_memory_db.log_drift_events(run_id, events)

    with in_memory_db.get_session() as session:
        recs = session.query(DriftEventRecord).filter_by(run_id=run_id).all()
        assert len(recs) == 1
        assert recs[0].detector_type == "ADWIN"
        assert recs[0].status == "DRIFT"
        assert recs[0].step_number == 450


def test_db_log_metrics(in_memory_db):
    """Verifies that evaluation slice metrics are persisted."""
    run_id = "RUN_METRICS_001"
    metrics = [
        {
            "slice_index": 0,
            "slice_name": "Week_1",
            "model_type": "static",
            "precision": 0.85,
            "recall": 0.70,
            "f1_score": 0.77,
            "auc_pr": 0.82
        },
        {
            "slice_index": 0,
            "slice_name": "Week_1",
            "model_type": "adaptive",
            "precision": 0.90,
            "recall": 0.78,
            "f1_score": 0.84,
            "auc_pr": 0.88
        }
    ]
    in_memory_db.log_metrics(run_id, metrics)

    with in_memory_db.get_session() as session:
        recs = session.query(SliceMetricRecord).filter_by(run_id=run_id).all()
        assert len(recs) == 2
        types = {r.model_type for r in recs}
        assert types == {"static", "adaptive"}


def test_db_log_attributions(in_memory_db):
    """Verifies that drift attribution records (SHAP + PSI) are persisted."""
    run_id = "RUN_ATTR_001"
    attributions = [
        {
            "slice_index": 3,
            "detector_type": "ADWIN",
            "feature_name": "TransactionAmt",
            "psi_score": 0.35,
            "shap_importance": 0.42,
            "attribution_score": 0.147,
            "rank_order": 1
        }
    ]
    in_memory_db.log_attributions(run_id, attributions)

    with in_memory_db.get_session() as session:
        recs = session.query(DriftAttributionRecord).filter_by(run_id=run_id).all()
        assert len(recs) == 1
        assert recs[0].feature_name == "TransactionAmt"
        assert recs[0].rank_order == 1
        assert pytest.approx(recs[0].attribution_score, rel=1e-3) == 0.147


def test_db_invalid_name_raises_value_error(monkeypatch):
    """Verifies that an invalid DB_NAME fails regex validation against SQL injection."""
    monkeypatch.setattr("src.db.DB_NAME", "fraud_db; DROP TABLE users;")
    with pytest.raises(ConnectionError, match="Insecure or invalid database name identifier"):
        DatabaseManager()


def test_db_unreachable_server_raises_connection_error(monkeypatch):
    """Verifies that unreachable MySQL host raises ConnectionError gracefully."""
    monkeypatch.setattr("src.db.DB_HOST", "127.0.0.1")
    monkeypatch.setattr("src.db.DB_PORT", 65534)
    with pytest.raises(ConnectionError, match="MySQL connection error"):
        DatabaseManager()
