"""
Unit tests for the Unified Multi-Dataset Generalization Suite.
Validates Feedzai Bank Account Fraud (NeurIPS 2022) and PaySim Mobile Money
schema adaptation, categorical handling, and temporal partitioning.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.ingestion import DatasetIngestion
from src.preprocessing import Preprocessor


def test_feedzai_baf_ingestion_schema(tmp_path: Path):
    """Verifies that Feedzai BAF dataset correctly adapts to the platform's unified schema."""
    feedzai_path = Path("data/feedzai/Base.csv")
    if feedzai_path.exists():
        ingestor = DatasetIngestion(feedzai_path)
        df, report = ingestor.load_and_validate()
        assert ingestor.dataset_type == "Feedzai-BAF"
        assert report["total_rows"] == 1_000_000
        assert report["fraud_count"] == 11_029
    else:
        # Fallback to self-contained synthetic fixture so tests never skip on clean clones
        sample_path = tmp_path / "synthetic_feedzai_base.csv"
        mock_df = pd.DataFrame({
            "fraud_bool": [0, 1] * 60,
            "month": [0] * 60 + [1] * 60,
            "proposed_credit_limit": [1200.0] * 120,
            "payment_type": ["AA", "AB", "AC", "AA"] * 30
        })
        mock_df.to_csv(sample_path, index=False)
        ingestor = DatasetIngestion(sample_path)
        df, report = ingestor.load_and_validate()
        assert ingestor.dataset_type == "Feedzai-BAF"
        assert report["total_rows"] == 120
        assert report["fraud_count"] == 60

    assert "isFraud" in df.columns
    assert "trans_id" in df.columns
    assert "step" in df.columns
    assert "month" in df.columns


def test_feedzai_baf_monthly_partitioning():
    """Verifies that Feedzai BAF is partitioned into exactly 8 non-overlapping monthly slices."""
    feedzai_path = Path("data/feedzai/Base.csv")
    if feedzai_path.exists():
        sample_df = pd.read_csv(feedzai_path, nrows=10_000)
    else:
        # Fallback to self-contained 8-month fixture so tests never skip on clean clones
        sample_df = pd.DataFrame({
            "fraud_bool": [0, 1] * 40,
            "month": [m for m in range(8) for _ in range(10)],
            "proposed_credit_limit": [1500.0] * 80
        })

    sample_df["isFraud"] = sample_df["fraud_bool"].astype(int)
    sample_df["trans_id"] = [f"BAF_{i}" for i in range(len(sample_df))]
    intra_rank = sample_df.groupby("month").cumcount()
    month_counts = sample_df.groupby("month")["trans_id"].transform("count")
    intra_step = (intra_rank / np.maximum(month_counts - 1, 1) * 719).astype(int)
    sample_df["step"] = (sample_df["month"] * 720 + 1 + intra_step).astype(int)

    preprocessor = Preprocessor()
    slices = preprocessor.partition_time_slices(sample_df)

    assert len(slices) >= 1
    assert slices[0]["slice_name"].startswith("Month_")
    total_slice_rows = sum(len(s["data"]) for s in slices)
    assert total_slice_rows == len(sample_df)
    if not feedzai_path.exists():
        assert len(slices) == 8


def test_paysim_ingestion_schema():
    """Verifies that PaySim dataset loads and computes accounting balance features."""
    sample_df = pd.DataFrame({
        "step": [1, 100, 250, 400],  # 400 hours = 16.6 days (>14 days check)
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
        "amount": [100.0, 500.0, 1000.0, 50.0],
        "nameOrig": ["C1", "C2", "C3", "C4"],
        "oldbalanceOrg": [100.0, 500.0, 1000.0, 50.0],
        "newbalanceOrig": [0.0, 0.0, 0.0, 0.0],
        "nameDest": ["M1", "C5", "C6", "M2"],
        "oldbalanceDest": [0.0, 0.0, 500.0, 0.0],
        "newbalanceDest": [0.0, 500.0, 1500.0, 0.0],
        "isFraud": [0, 1, 1, 0],
        "isFlaggedFraud": [0, 0, 0, 0]
    })
    sample_path = Path("data/paysim/sample_test.csv")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)

    try:
        ingestor = DatasetIngestion(sample_path)
        df, report = ingestor.load_and_validate()

        assert ingestor.dataset_type == "PaySim"
        assert "isFraud" in df.columns
        assert "trans_id" in df.columns
        assert "errorBalanceOrig" in df.columns
        assert "errorBalanceDest" in df.columns
    finally:
        if sample_path.exists():
            sample_path.unlink()


def test_preprocessor_multi_dataset_categoricals():
    """Verifies that candidate categoricals from Feedzai and PaySim are encoded without error."""
    mock_df = pd.DataFrame({
        "step": [1, 2, 3, 4],
        "trans_id": ["TX1", "TX2", "TX3", "TX4"],
        "isFraud": [0, 1, 0, 0],
        "payment_type": ["AA", "AB", "AC", "AA"],
        "employment_status": ["CA", "CB", "CA", "CC"],
        "housing_status": ["BA", "BB", "BA", "BC"],
        "device_os": ["windows", "linux", "mac", "windows"],
        "proposed_credit_limit": [500.0, 1000.0, 1500.0, 2000.0]
    })

    preprocessor = Preprocessor()
    preprocessor.fit(mock_df)
    transformed = preprocessor.transform(mock_df)

    assert "payment_type_AA" in preprocessor.feature_names or any("payment_type" in f for f in preprocessor.feature_names)
    assert "employment_status_CA" in preprocessor.feature_names or any("employment_status" in f for f in preprocessor.feature_names)
    assert "device_os_windows" in preprocessor.feature_names or any("device_os" in f for f in preprocessor.feature_names)
    assert len(transformed) == len(mock_df)


def test_feedzai_baf_non_degenerate_step_ranges():
    """Verifies that Feedzai BAF monthly partitions report non-degenerate start_step < end_step."""
    mock_baf = pd.DataFrame({
        "month": [0, 0, 0, 1, 1, 1],
        "fraud_bool": [0, 1, 0, 0, 0, 1],
        "proposed_credit_limit": [1000.0] * 6,
    })
    tmp_path = Path("data/feedzai/sample_baf.csv")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    mock_baf.to_csv(tmp_path, index=False)

    try:
        ingestor = DatasetIngestion(tmp_path)
        df, _ = ingestor.load_and_validate()
        assert ingestor.dataset_type == "Feedzai-BAF"

        preprocessor = Preprocessor()
        slices = preprocessor.partition_time_slices(df)
        assert len(slices) == 2
        for s in slices:
            assert s["start_step"] < s["end_step"]
        # Month 0 spans 1 to 720, Month 1 spans 721 to 1440
        assert slices[0]["start_step"] == 1
        assert slices[0]["end_step"] == 720
        assert slices[1]["start_step"] == 721
        assert slices[1]["end_step"] == 1440
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_feedzai_baf_degenerate_step_fallback():
    """Verifies that partition_time_slices guards against start_step == end_step when steps are uniform."""
    mock_df = pd.DataFrame({
        "month": [0, 0, 1, 1],
        "step": [1, 1, 721, 721],  # uniform steps per month
        "isFraud": [0, 1, 0, 0],
        "trans_id": ["T1", "T2", "T3", "T4"]
    })
    preprocessor = Preprocessor()
    slices = preprocessor.partition_time_slices(mock_df)
    assert len(slices) == 2
    assert slices[0]["start_step"] == 1
    assert slices[0]["end_step"] == 720
    assert slices[1]["start_step"] == 721
    assert slices[1]["end_step"] == 1440

