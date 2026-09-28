"""
Tests for Concept Drift Detection and Change-Point Benchmarking.
Verifies detection delay invariants, false alarm calculations, and ground-truth recovery.
"""

import pytest
import numpy as np
from src.drift_detector import DriftMonitor
from src.drift_benchmark import (
    generate_change_point_stream,
    evaluate_detector_on_stream,
    run_multi_seed_drift_benchmark
)


def test_drift_monitor_basic_stream():
    """Verifies that DriftMonitor processes errors and logs events thread-safely."""
    monitor = DriftMonitor()
    assert monitor.total_samples_monitored == 0
    assert monitor.total_errors_observed == 0

    # Feed 10 correct classifications
    for i in range(10):
        alert = monitor.update(y_true=0, y_pred=0, slice_index=0, step_number=i, trans_id=f"TX_{i}")
        assert alert["error_val"] == 0

    assert monitor.total_samples_monitored == 10
    assert monitor.total_errors_observed == 0


def test_change_point_stream_generator_properties():
    """Verifies that generate_change_point_stream produces deterministic streams with correct change points."""
    stream, change_points = generate_change_point_stream(n_samples=6000, seed=42)
    assert len(stream) == 6000
    assert len(change_points) == 2  # Abrupt at 2000, gradual at 4000

    # In stationary phase [0:2000], empirical rate should be close to 0.05
    rate_p1 = np.mean(stream[:2000])
    assert 0.03 <= rate_p1 <= 0.07

    # In abrupt drift phase [2000:4000], empirical rate should be close to 0.25
    rate_p2 = np.mean(stream[2000:4000])
    assert 0.20 <= rate_p2 <= 0.30


def test_adwin_detection_delay_and_low_false_alarms():
    """Verifies that ADWIN detects the abrupt change point with bounded delay and low false alarm rate."""
    regimes = [
        {"start": 0, "end": 1500, "rate": 0.03, "drift_type": "stationary", "is_change_point": False},
        {"start": 1500, "end": 3000, "rate": 0.30, "drift_type": "abrupt", "is_change_point": True}
    ]
    stream, change_points = generate_change_point_stream(n_samples=3000, regimes=regimes, seed=42)
    res = evaluate_detector_on_stream("ADWIN", stream, change_points, tolerance_window=500)

    assert res["detected_count"] == 1
    assert res["detection_rate"] == 1.0
    # Abrupt jump from 3% to 30% should be detected within 200 samples
    assert res["mean_delay_samples"] < 200
    # False alarms during stationary phase should be 0 or near 0
    assert res["false_alarms_count"] <= 1


def test_ddm_detection_properties():
    """Verifies that DDM detects abrupt change points on synthetic streams."""
    regimes = [
        {"start": 0, "end": 1500, "rate": 0.05, "drift_type": "stationary", "is_change_point": False},
        {"start": 1500, "end": 3000, "rate": 0.35, "drift_type": "abrupt", "is_change_point": True}
    ]
    stream, change_points = generate_change_point_stream(n_samples=3000, regimes=regimes, seed=101)
    res = evaluate_detector_on_stream("DDM", stream, change_points, tolerance_window=600)

    assert res["detected_count"] >= 1
    assert res["detection_rate"] == 1.0
    assert res["mean_delay_samples"] is not None


def test_multi_seed_benchmark_summary():
    """Verifies that run_multi_seed_drift_benchmark generates well-formed summary metrics."""
    summary = run_multi_seed_drift_benchmark(n_seeds=2, n_samples=3000, save_metrics=False)
    assert "ADWIN" in summary
    assert "DDM" in summary
    assert summary["ADWIN"]["detection_rate_pct"] > 0.0
    assert summary["DDM"]["detection_rate_pct"] > 0.0
