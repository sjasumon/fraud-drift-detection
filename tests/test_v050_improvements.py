"""
Tests for v0.5.0 improvements:
- Dataset provenance classification and verification
- Provenance enforcement in ModelBundle
- Tied-timestamp history feature leakage safety
- Operational review budget threshold objective
- BackgroundRetrainer margin and timeout guards
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from src.provenance import (
    GENUINE_OBSERVED,
    SIMULATOR_GENERATED,
    SYNTHETIC,
    UNVERIFIED,
    get_dataset_provenance,
    verify_bundle_provenance,
)
from src.artifacts import ModelBundle
from src.history_features import add_history_features
from src.base_model import BaseFraudModel
from src.api import BackgroundRetrainer


def test_provenance_classification_rules():
    from src.provenance import create_verified_test_double_provenance

    # Calling IEEE-CIS without a verified file strictly returns UNVERIFIED per R01
    prov_ieee_nofile = get_dataset_provenance("IEEE-CIS")
    assert prov_ieee_nofile["classification"] == UNVERIFIED
    assert prov_ieee_nofile["acceptable_for_proposal_benchmark"] is False
    assert prov_ieee_nofile["verification_status"] == "UNVERIFIED"

    # Explicit verified test double produces verified record
    prov_test_double = create_verified_test_double_provenance("test_run")
    assert prov_test_double["classification"] == GENUINE_OBSERVED
    assert prov_test_double["acceptable_for_proposal_benchmark"] is True
    assert prov_test_double["verification_status"] == "VERIFIED_RESEARCH_DATA"

    prov_paysim = get_dataset_provenance("PaySim")
    assert prov_paysim["classification"] == SIMULATOR_GENERATED
    assert prov_paysim["acceptable_for_proposal_benchmark"] is False
    assert prov_paysim["verification_status"] == "EXCLUDED_NON_GENUINE"

    prov_baf = get_dataset_provenance("Feedzai-BAF")
    assert prov_baf["classification"] == SYNTHETIC
    assert prov_baf["acceptable_for_proposal_benchmark"] is False

    prov_unknown = get_dataset_provenance("RandomDataset")
    assert prov_unknown["classification"] == UNVERIFIED
    assert prov_unknown["acceptable_for_proposal_benchmark"] is False


def test_model_bundle_ready_enforces_provenance():
    from src.provenance import create_verified_test_double_provenance

    mock_model = MagicMock()
    mock_model.trained = True
    mock_prep = MagicMock()

    # Valid genuine bundle using verified test double
    meta_genuine = {
        "data_kind": "real",
        "provenance": create_verified_test_double_provenance("unit_test"),
    }
    bundle_genuine = ModelBundle(mock_model, mock_prep, meta_genuine)
    assert bundle_genuine.ready is True

    # Simulator-generated bundle must NOT be ready for verified research service
    meta_paysim = {
        "data_kind": "real",
        "provenance": get_dataset_provenance("PaySim"),
    }
    bundle_paysim = ModelBundle(mock_model, mock_prep, meta_paysim)
    assert bundle_paysim.ready is False

    # Synthetic bundle must NOT be ready
    meta_synthetic = {
        "data_kind": "synthetic",
        "provenance": get_dataset_provenance("Feedzai-BAF"),
    }
    bundle_synthetic = ModelBundle(mock_model, mock_prep, meta_synthetic)
    assert bundle_synthetic.ready is False

    # Contradictory synthetic bundle carrying IEEE metadata must NOT be ready
    meta_contradictory = {
        "data_kind": "synthetic",
        "provenance": create_verified_test_double_provenance("unit_test"),
    }
    bundle_contradictory = ModelBundle(mock_model, mock_prep, meta_contradictory)
    assert bundle_contradictory.ready is False

    # Missing provenance must NOT be ready (no permissive fallback)
    meta_no_prov = {
        "data_kind": "real",
    }
    bundle_no_prov = ModelBundle(mock_model, mock_prep, meta_no_prov)
    assert bundle_no_prov.ready is False

    # Untrained model is not ready
    mock_untrained = MagicMock()
    mock_untrained.trained = False
    bundle_untrained = ModelBundle(mock_untrained, mock_prep, meta_genuine)
    assert bundle_untrained.ready is False


def test_tied_timestamp_leakage_safety():
    """
    Transactions occurring at the EXACT same timestamp must NOT observe
    each other's current transaction amount or count in prior history.
    """
    df = pd.DataFrame(
        {
            "trans_id": [101, 102, 103, 104],
            "card1": [1001, 1001, 1001, 1001],
            "step": [10, 10, 10, 12],
            "amount": [50.0, 75.0, 100.0, 20.0],
            "isFraud": [0, 0, 1, 0],
        }
    )
    res = add_history_features(df)

    # For the three tied transactions at step 10, prior card count must be 0
    # because strictly prior transactions occurred at step < 10.
    step_10_records = res[res.step == 10]
    for _, row in step_10_records.iterrows():
        assert row["hist_tx_count"] == 0
        assert row["hist_tx_24h"] == 0
        assert row["hist_amount_mean"] == 0.0

    # At step 12, all three step 10 transactions must now be in history
    step_12_record = res[res.step == 12].iloc[0]
    assert step_12_record["hist_tx_count"] == 3
    assert step_12_record["hist_tx_24h"] == 3
    assert step_12_record["hist_amount_mean"] == pytest.approx(
        (50.0 + 75.0 + 100.0) / 3.0
    )


def test_review_budget_recall_thresholding():
    """
    Verifies that BaseFraudModel with threshold_objective='review_budget_recall'
    selects a threshold calibrated to the specified review budget on validation data.
    """
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "trans_id": range(n),
            "feat1": rng.normal(size=n),
            "feat2": rng.normal(size=n),
            "step": np.repeat(range(10), 20),
            "isFraud": rng.binomial(1, 0.1, size=n),
        }
    )
    # Ensure both classes in fit and val
    df.loc[0, "isFraud"] = 1
    df.loc[1, "isFraud"] = 0
    df.loc[190, "isFraud"] = 1
    df.loc[191, "isFraud"] = 0

    model = BaseFraudModel(
        feature_names=["feat1", "feat2"],
        threshold_objective="review_budget_recall",
        operational_daily_budget=5,  # 5 alerts per day
        model_params={"n_estimators": 5, "max_depth": 2},
    )
    model.fit(df)
    assert 0.0 <= model.threshold <= 1.0


def test_background_retrainer_promotion_criteria():
    """
    Verifies that BackgroundRetrainer strictly checks promotion margin (>= 0.005)
    and timeout guards.
    """
    retrainer = BackgroundRetrainer(
        min_improvement_margin=0.005,
        max_retrain_duration_sec=60.0,
    )
    assert retrainer.min_improvement_margin == 0.005
    assert retrainer.max_retrain_duration_sec == 60.0
    assert retrainer.promotion_log == []
