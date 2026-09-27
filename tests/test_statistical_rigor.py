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
    export_summary_csv,
    adjust_p_values_fdr_bh,
    adjust_p_values_bonferroni
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
        assert "cohens_d" in rows[0]
        assert "hedges_g" in rows[0]
        assert "p_value_fdr_bh" in rows[0]


def test_compute_paired_tests_cohens_d_and_hedges_g():
    """Verifies that compute_paired_tests returns Cohen's d, Hedges' g, and sample size limitations."""
    a = [0.85, 0.86, 0.84, 0.87, 0.88]
    b = [0.70, 0.72, 0.69, 0.71, 0.73]
    res = compute_paired_tests(a, b)

    assert res["mean_diff"] > 0.10
    assert res["cohens_d"] > 2.0  # huge effect size
    assert res["hedges_g"] < res["cohens_d"]  # Hedges' g is smaller than Cohen's d due to small-sample bias correction
    assert res["effect_size"] == "very large (huge)"
    assert res["n_seeds"] == 5
    assert "sample_size_note" in res
    assert "n=5" in res["sample_size_note"]


def test_adjust_p_values_fdr_and_bonferroni():
    """Verifies that FDR (Benjamini-Hochberg) and Bonferroni adjustments control family-wise error rates."""
    raw_p = [0.001, 0.02, 0.04]
    
    # Bonferroni: p * 3
    bonf = adjust_p_values_bonferroni(raw_p)
    assert bonf[0] == 0.003
    assert bonf[1] == 0.06
    assert bonf[2] == 0.12

    # FDR BH: monotonic non-decreasing, bounded by [0, 1]
    fdr = adjust_p_values_fdr_bh(raw_p)
    assert len(fdr) == 3
    assert fdr[0] <= fdr[1] <= fdr[2]
    assert fdr[0] < 0.05


def test_generate_statistical_errorbar_charts(tmp_path: Path, monkeypatch):
    """Verifies that statistical error-bar comparison plots render cleanly."""
    from src.benchmark_statistical_rigor import generate_statistical_errorbar_charts

    monkeypatch.setattr("src.benchmark_statistical_rigor.FIGURES_DIR", tmp_path)

    paysim_res = {
        "summary": {
            "models": {
                "static": {"f1": {"mean": 0.9955, "std": 0.0005}},
                "periodic": {"f1": {"mean": 0.9962, "std": 0.0004}},
                "sliding": {"f1": {"mean": 0.9960, "std": 0.0004}},
                "adaptive": {"f1": {"mean": 0.9978, "std": 0.0003}}
            },
            "paired_tests": {
                "adaptive_vs_static_f1": {"p_value_t": 0.00005}
            }
        }
    }
    ieee_uid_res = {
        "summary": {
            "cohort1_seen": {
                "static_recall": {"mean": 75.0, "std": 1.2},
                "adaptive_recall": {"mean": 82.0, "std": 1.0}
            },
            "cohort2_unseen": {
                "static_recall": {"mean": 65.0, "std": 1.5},
                "adaptive_recall": {"mean": 74.0, "std": 1.1}
            },
            "cohort3_zero_memory": {
                "static_recall": {"mean": 67.82, "std": 0.8},
                "adaptive_recall": {"mean": 77.09, "std": 0.6},
                "recall_gain": {"mean": 9.27, "std": 0.4},
                "paired_ttest_recall": {"p_value_t": 0.000007}
            }
        }
    }

    out = generate_statistical_errorbar_charts(paysim_res, ieee_uid_res, output_filename="test_errorbars.png")
    assert out.exists()


def test_export_summary_csv_with_ieee_uid(tmp_path: Path):
    """Verifies exporting summary CSV with both PaySim and IEEE-CIS UID Generalization results."""
    dummy_results = {
        "ieee_zero_memory_uid": {
            "summary": {
                "cohort1_seen": {
                    "static_recall": {"mean": 75.0, "std": 1.2, "ci_95": [73.5, 76.5]},
                    "adaptive_recall": {"mean": 82.0, "std": 1.0, "ci_95": [80.7, 83.3]}
                },
                "cohort2_unseen": {
                    "static_recall": {"mean": 65.0, "std": 1.5, "ci_95": [63.1, 66.9]},
                    "adaptive_recall": {"mean": 74.0, "std": 1.1, "ci_95": [72.6, 75.4]}
                },
                "cohort3_zero_memory": {
                    "static_recall": {"mean": 67.82, "std": 0.8, "ci_95": [66.8, 68.8]},
                    "adaptive_recall": {"mean": 77.09, "std": 0.6, "ci_95": [76.3, 77.8]},
                    "paired_ttest_recall": {
                        "p_value_t": 7.15e-6,
                        "cohens_d": 24.87,
                        "hedges_g": 20.19,
                        "p_value_t_fdr_bh": 1.43e-5
                    }
                }
            }
        }
    }
    csv_out = tmp_path / "test_ieee_summary.csv"
    export_summary_csv(dummy_results, csv_out)
    assert csv_out.exists()


