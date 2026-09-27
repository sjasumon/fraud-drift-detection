"""
Unit tests for Statistical Rigor & Hypothesis Testing Suite.
Verifies sample statistics, student's t confidence intervals, bootstrap CIs,
and paired hypothesis significance testing.
"""

import numpy as np
import pytest
from pathlib import Path
import csv

from src.benchmark_statistical_rigor import (
    compute_sample_stats,
    compute_bootstrap_ci,
    compute_paired_tests,
    export_summary_csv
)


def test_compute_sample_stats_deterministic():
    vals = [10.0, 12.0, 14.0, 16.0, 18.0]
    res = compute_sample_stats(vals)
    assert res["mean"] == 14.0
    assert pytest.approx(res["std"], rel=1e-3) == 3.1623
    assert res["n_seeds"] == 5
    assert res["ci_95"][0] < res["mean"] < res["ci_95"][1]


def test_compute_bootstrap_ci_bounds():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])
    low, high = compute_bootstrap_ci(y_true, y_pred, n_bootstraps=200, alpha=0.05, metric_type="recall")
    assert 0.0 <= low <= high <= 100.0


def test_compute_paired_tests_identical():
    a = [0.95, 0.96, 0.97]
    b = [0.95, 0.96, 0.97]
    res = compute_paired_tests(a, b)
    assert res["mean_diff"] == 0.0
    assert res["p_value_t"] == 1.0
    assert not res["significant_05"]


def test_compute_paired_tests_significant():
    a = [80.0, 81.0, 80.5, 82.0, 81.5]
    b = [70.0, 71.0, 70.5, 72.0, 71.5]
    res = compute_paired_tests(a, b)
    assert res["mean_diff"] == 10.0
    assert res["significant_001"] is True
    assert res["p_value_t"] < 0.001


def test_export_summary_csv(tmp_path: Path):
    dummy_results = {
        "paysim_oos_f1": {
            "summary": {
                "models": {
                    "adaptive": {
                        "f1": {"mean": 0.9964, "std": 0.0009, "ci_95": [0.9953, 0.9975]},
                        "bootstrap_f1_ci": [0.9943, 0.9970]
                    }
                },
                "paired_tests": {
                    "adaptive_vs_static_f1": {"p_value_t": 0.2871}
                }
            }
        }
    }
    csv_out = tmp_path / "test_summary.csv"
    export_summary_csv(dummy_results, csv_out)
    assert csv_out.exists()
    with open(csv_out, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["benchmark"] == "PaySim Out-of-Sample Stream"
        assert rows[0]["architecture_or_cohort"] == "Adaptive"
