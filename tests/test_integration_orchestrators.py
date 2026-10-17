"""
Integration tests for high-level research orchestrators:
1. src/pipeline.py (run_pipeline)
2. src/delayed_audit.py (main & run_latency_simulation_regime)
3. src/evaluate_uid_split.py (run_single_audit & CLI run)
Guarantees full-system reproducibility, zero data leakage, and artifact generation on synthetic streams.
"""

import pytest
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

from src.data_generator import generate_paysim_stream
from src.db import DatabaseManager, ExperimentRun
from src.pipeline import run_pipeline
from src.delayed_audit import main as delayed_audit_main
from src.evaluate_uid_split import run_single_audit, run as evaluate_uid_split_run


@pytest.fixture
def synthetic_paysim_csv(tmp_path: Path) -> Path:
    """Creates a fast, self-contained 4-week synthetic PaySim transaction stream."""
    csv_file = tmp_path / "paysim_stream_4w.csv"
    generate_paysim_stream(
        total_weeks=4, tx_per_week=40, fraud_rate=0.08, seed=42, output_path=csv_file
    )
    return csv_file


@pytest.fixture
def synthetic_ieee_csv(tmp_path: Path, synthetic_stream_df: pd.DataFrame) -> Path:
    """Writes the IEEE synthetic fixture to a CSV file for ingestion (spanning >14 days for FR-1.2)."""
    csv_file = tmp_path / "ieee_stream_3w.csv"
    df = synthetic_stream_df.copy()
    # Ensure time span satisfies FR-1.2 (>= 14 days / 336 hours)
    df["TransactionDT"] = df["step"] * 6 * 3600
    df.to_csv(csv_file, index=False)
    return csv_file


def test_pipeline_synthetic_execution(
    synthetic_paysim_csv: Path, tmp_path: Path, monkeypatch
):
    """
    Executes full pipeline on synthetic PaySim stream with an in-memory SQLite DatabaseManager.
    Validates end-to-end model training, drift detection, evaluation metrics, and DB logging.
    """
    # In-memory SQLite DB
    db_engine = create_engine("sqlite:///:memory:")
    db_mgr = DatabaseManager(engine=db_engine)

    result = run_pipeline(
        dataset_path=synthetic_paysim_csv,
        dataset_type="paysim",
        db=db_mgr,
        output_dir=tmp_path / "runs",
        data_kind="synthetic",
    )

    assert result is not None
    assert result["run_id"].startswith("RUN_")
    assert result["dataset_slug"] == "paysim"
    assert "overall_summary" in result

    summary = result["overall_summary"]
    assert "static_model" in summary
    assert "adaptive_model" in summary
    assert "out_of_sample_stream" in summary

    # Verify database persistence
    with db_mgr.get_session() as session:
        run_record = (
            session.query(ExperimentRun).filter_by(run_id=result["run_id"]).first()
        )
        assert run_record is not None
        assert run_record.dataset_name == synthetic_paysim_csv.name
        assert run_record.time_slices_count == 4


def test_delayed_audit_synthetic_execution(
    synthetic_paysim_csv: Path, tmp_path: Path, monkeypatch
):
    """
    Executes delayed audit workflow on synthetic data under controlled latency regimes.
    Verifies metric calculation, maturity queues, and 4-panel figure rendering.
    """
    monkeypatch.setattr("src.delayed_audit.METRICS_DIR", tmp_path)
    monkeypatch.setattr("src.delayed_audit.FIGURES_DIR", tmp_path)

    fast_regimes = [
        {"name": "Immediate Feedback (0w)", "delay_slices": 0, "mode": "fixed"},
        {"name": "Standard Delay (1w)", "delay_slices": 1, "mode": "fixed"},
    ]

    delayed_audit_main(data_file_path=str(synthetic_paysim_csv), regimes=fast_regimes)

    summary_file = tmp_path / "delayed_feedback_benchmark.json"
    plot_file = tmp_path / "delayed_feedback_impact.png"
    assert summary_file.exists()
    assert plot_file.exists()


def test_evaluate_uid_split_synthetic_execution(synthetic_stream_df: pd.DataFrame):
    """
    Executes UID entity-memorization vs concept drift generalization audit on synthetic stream.
    Tests Method A (clean cases) and Method B (collision-free) cohorts with both standard and feature-blind modes.
    """
    df = synthetic_stream_df.copy()
    # Exercise a genuine 3-week stream; the old 72-hour fixture had no OOS slice.
    df["step"] = df["step"] * 7
    df["TransactionDT"] = (df["step"] - 1) * 3600

    # Precompute UID annotations
    valid_mask = df["card2"].notna() & df["addr1"].notna()
    uid_clean = pd.Series(index=df.index, dtype=object)
    uid_clean[valid_mask] = (
        df.loc[valid_mask, "card1"].astype(int).astype(str)
        + "_"
        + df.loc[valid_mask, "card2"].astype(int).astype(str)
        + "_"
        + df.loc[valid_mask, "addr1"].astype(int).astype(str)
    )
    uid_cf = uid_clean.copy()
    missing_mask = ~valid_mask
    uid_cf[missing_mask] = "MISSING_" + df.loc[missing_mask, "TransactionID"].astype(
        str
    )

    df["uid_clean"] = uid_clean.values
    df["uid_cf"] = uid_cf.values
    df["is_clean"] = valid_mask.values

    # Test Standard Features (IDs included)
    res_std = run_single_audit(
        df_raw=df,
        exclude_card_identifiers=False,
        title="TEST STANDARD FEATURES",
        random_seed=42,
    )
    assert "global_oos" in res_std
    assert res_std["global_oos"]["total_tx"] > 0
    assert "clean" in res_std
    assert "collision_free" in res_std
    assert "seen" in res_std["collision_free"]
    assert "unseen" in res_std["collision_free"]
    assert "strict" in res_std["collision_free"]

    # Test Feature-Blind (IDs excluded)
    res_blind = run_single_audit(
        df_raw=df,
        exclude_card_identifiers=True,
        title="TEST FEATURE BLIND",
        random_seed=42,
    )
    assert "global_oos" in res_blind
    assert "clean" in res_blind


def test_evaluate_uid_split_cli_run(synthetic_ieee_csv: Path):
    """Executes evaluate_uid_split.run CLI entrypoint on synthetic dataset."""
    evaluate_uid_split_run(
        ["--dataset", str(synthetic_ieee_csv), "--mode", "feature-blind"]
    )
