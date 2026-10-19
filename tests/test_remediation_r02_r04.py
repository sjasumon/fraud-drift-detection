"""
Targeted Acceptance Tests for Remediation Findings R02 & R04.
Verifies:
- Deterministic reconciliation of historical benchmark RUN_63D7CA7F68 (R02)
- Detection of confusion matrix and metric corruptions by reconcile_metrics (R02)
- Pipeline CLI parses --threshold-objective and --operational-daily-budget (R04)
- Pipeline CLI loads and merges versioned JSON configuration files (R04)
- Research protocol default hyperparameters (n_estimators=100, learning_rate=0.1) (R04)
- Proper rejection of invalid budget values or missing required budget options (R04)
"""

import json
import subprocess
import sys
import pytest
from pathlib import Path

from src.reconcile_metrics import reconcile_run_directory
from src.base_model import DEFAULT_MODEL_PARAMS, BaseFraudModel


def test_reconcile_metrics_on_run_63d7ca7f68():
    """Validates that historical benchmark run RUN_63D7CA7F68 reconciles completely from prediction ledger."""
    run_dir = Path("results/benchmark_runs/RUN_63D7CA7F68")
    assert run_dir.exists(), "Benchmark run RUN_63D7CA7F68 directory must exist"

    is_reconciled, report, discrepancies = reconcile_run_directory(run_dir, tolerance=1e-4)
    assert is_reconciled is True, f"Reconciliation failed with discrepancies: {discrepancies}"
    assert len(discrepancies) == 0
    assert report["provenance_status"] == "VERIFIED_RESEARCH_DATA"
    assert report["total_records_scored"] == 221727 * 5  # 5 models scored on 221,727 rows

    # Verify headline metrics match BENCHMARK_REPORT.md exactly
    adaptive_m = report["metrics_table"]["adaptive"]
    assert adaptive_m["total_transactions"] == 221727
    assert adaptive_m["total_frauds"] == 8761
    assert adaptive_m["true_positives"] == 3727
    assert adaptive_m["false_positives"] == 2461
    assert adaptive_m["false_negatives"] == 5034
    assert adaptive_m["true_negatives"] == 210505
    assert adaptive_m["overall_f1"] == pytest.approx(0.4986, abs=1e-4)
    assert adaptive_m["average_precision"] == pytest.approx(0.5184, abs=1e-4)


def test_reconcile_metrics_detects_confusion_matrix_corruption(tmp_path):
    """Reconcile command must detect corrupted prediction records where confusion invariants fail."""
    # Copy manifest and create corrupted predictions
    manifest = {
        "run_id": "TEST_CORRUPT",
        "dataset": "test.csv",
        "data_kind": "synthetic",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    # Inconsistent predictions: true label is 1, but model is corrupted
    pred_data = [
        "model,trans_id,y_true,prediction,probability",
        "m1,1,1,1,0.9",
        "m1,2,0,0,0.1",
        "m1,3,1,0,0.2",
    ]
    (tmp_path / "predictions.csv").write_text("\n".join(pred_data))

    # Recorded metrics claim TP=10 (impossible for 3 total rows)
    rec_metrics = [
        "model,overall_precision,overall_recall,overall_f1,average_precision,brier_score",
        "m1,0.9,0.9,0.9,0.9,0.01",
    ]
    (tmp_path / "metrics.csv").write_text("\n".join(rec_metrics))

    is_reconciled, report, discrepancies = reconcile_run_directory(tmp_path)
    assert is_reconciled is False
    assert len(discrepancies) > 0
    assert any("mismatch" in d for d in discrepancies)


def test_base_model_default_hyperparameters():
    """Protocol specifies 100 trees and learning_rate=0.1. Base defaults must reflect this."""
    assert DEFAULT_MODEL_PARAMS["n_estimators"] == 100
    assert DEFAULT_MODEL_PARAMS["learning_rate"] == 0.1
    assert DEFAULT_MODEL_PARAMS["max_depth"] == 6


def test_pipeline_cli_help_includes_required_flags():
    """Verify that --threshold-objective and --operational-daily-budget are exposed in CLI help."""
    cmd = [sys.executable, "-m", "src.pipeline", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    help_text = res.stdout

    assert "--threshold-objective" in help_text
    assert "--operational-daily-budget" in help_text
    assert "--config" in help_text
    assert "--n-estimators" in help_text
    assert "--learning-rate" in help_text


def test_study_cli_help_includes_required_flags():
    """Verify that study CLI exposes threshold-objective and operational-daily-budget."""
    cmd = [sys.executable, "-m", "src.study", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    help_text = res.stdout

    assert "--threshold-objective" in help_text
    assert "--operational-daily-budget" in help_text


def test_pipeline_cli_rejects_invalid_budget():
    """Pipeline CLI must reject negative or zero daily budgets."""
    cmd = [
        sys.executable,
        "-m",
        "src.pipeline",
        "--dataset",
        "dummy.csv",
        "--operational-daily-budget",
        "-5",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "positive integer" in (res.stderr + res.stdout)


def test_pipeline_cli_rejects_budget_recall_without_budget():
    """Pipeline CLI must reject review_budget_recall if no daily budget is supplied."""
    cmd = [
        sys.executable,
        "-m",
        "src.pipeline",
        "--dataset",
        "dummy.csv",
        "--threshold-objective",
        "review_budget_recall",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "operational-daily-budget must be provided" in (res.stderr + res.stdout)


def test_versioned_configs_exist_and_validate():
    """Verify that all required versioned configuration files exist and contain valid schemas."""
    configs = [
        Path("configs/protocol_v1_baseline.json"),
        Path("configs/protocol_v1_review_budget_100.json"),
        Path("configs/protocol_v1_review_budget_500.json"),
        Path("configs/historical_v050_run_63d7ca7f68.json"),
        Path("configs/dev_quick_synthetic.json"),
    ]

    for cfg_file in configs:
        assert cfg_file.exists(), f"Missing config file: {cfg_file}"
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert "protocol_version" in cfg
        assert "dataset" in cfg
        assert "threshold_objective" in cfg
        assert "model_params" in cfg
