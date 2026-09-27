"""
Unit tests for synthetic stream generation matching PaySim schema (src/data_generator.py).
Tests schema adherence, class balance, temporal monotonic ordering, and multi-phase drift dynamics.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_generator import generate_paysim_stream


def test_generate_paysim_stream_basic(tmp_path: Path):
    """Verifies that generate_paysim_stream generates valid dataframe and CSV file."""
    out_csv = tmp_path / "synthetic_stream.csv"
    df = generate_paysim_stream(
        total_weeks=2,
        tx_per_week=50,
        fraud_rate=0.04,
        seed=42,
        output_path=out_csv
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert out_csv.exists()

    expected_cols = [
        "trans_id", "step", "type", "amount", "nameOrig",
        "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest",
        "newbalanceDest", "isFraud", "isFlaggedFraud", "drift_phase", "week"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify temporal ordering
    assert df["step"].is_monotonic_increasing
    # Verify binary fraud column
    assert set(df["isFraud"].unique()).issubset({0, 1})
    # Verify non-empty amounts
    assert (df["amount"] > 0).all()


def test_generate_paysim_stream_all_phases():
    """Verifies that all 4 concept drift phases are generated across a multi-phase horizon."""
    df = generate_paysim_stream(
        total_weeks=18,
        tx_per_week=15,
        fraud_rate=0.10,
        seed=101
    )

    phases = set(df["drift_phase"].unique())
    assert "Phase_0_Baseline" in phases
    assert "Phase_1_Sudden_MicroStructuring" in phases
    assert "Phase_2_Gradual_MerchantCamouflage" in phases
    assert "Phase_3_Mule_Network" in phases

    # Verify merchant prefixes for payment destinations
    merchant_tx = df[df["type"] == "PAYMENT"]
    if len(merchant_tx) > 0:
        assert merchant_tx["nameDest"].str.startswith("M").all()
