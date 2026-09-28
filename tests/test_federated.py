"""
Unit and Invariant Tests for Federated Multi-Bank Fraud Detection Module (src/federated.py).

Validates:
- Non-IID distribution divergence across simulated banking institutions
- Strict zero raw-data leakage between decentralized clients and aggregator
- Cross-institution blindness of isolated single-bank models
- Collaborative federated recall advantage on cross-bank fraud attacks
- Mathematical validity of evidence-combination aggregation modes
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from src.federated import (
    generate_non_iid_bank_data,
    BankClient,
    FederatedEnsembleAggregator,
    evaluate_model_performance,
    run_federated_fraud_benchmark,
)


def test_non_iid_divergence():
    """Verify that simulated banks have statistically significant non-IID feature distributions."""
    data = generate_non_iid_bank_data(n_samples_per_bank=500, seed=42)
    assert "Bank_A" in data and "Bank_B" in data and "Bank_C" in data

    df_a = data["Bank_A"]
    df_b = data["Bank_B"]
    df_c = data["Bank_C"]

    # Verify sample counts
    assert len(df_a) == 500
    assert len(df_b) == 500
    assert len(df_c) == 500

    # KS test on TransactionAmt must confirm non-IID divergence (p < 1e-4)
    ks_stat_ab, ks_p_ab = stats.ks_2samp(df_a["TransactionAmt"], df_b["TransactionAmt"])
    assert ks_p_ab < 1e-4
    assert ks_stat_ab > 0.5  # Distinct ticket sizes

    # Velocity distribution divergence (Bank A micro-testing vs Bank B wire transfers)
    ks_stat_vel, ks_p_vel = stats.ks_2samp(df_a["velocity_count"], df_b["velocity_count"])
    assert ks_p_vel < 1e-4


def test_zero_raw_data_leakage():
    """Verify the strict privacy invariant: Aggregator never accesses raw DataFrames or records."""
    data = generate_non_iid_bank_data(n_samples_per_bank=300, seed=42)
    feature_cols = ["TransactionAmt", "velocity_count", "errorBalance", "hour_of_day", "foreign_dest"]

    clients = []
    for bank_name, df in data.items():
        train_df = df.iloc[:200]
        test_df = df.iloc[200:]
        client = BankClient(bank_name, train_df, test_df, feature_cols)
        client.train_local_model()
        clients.append(client)

    aggregator = FederatedEnsembleAggregator(clients, feature_cols)
    aggregator.collect_and_aggregate()

    # The aggregator must hold model objects and weights, NOT private dataframes
    assert len(aggregator.client_models) == 3
    assert hasattr(aggregator, "_private_train") is False
    assert hasattr(aggregator, "_private_test") is False

    for client in clients:
        # Private data remains encapsulated within client
        assert hasattr(client, "_private_train")
        assert len(client._private_train) == 200


def test_federated_aggregator_probability_bounds_and_modes():
    """Verify predict_proba bounds [0, 1] and prediction outputs across aggregation modes."""
    data = generate_non_iid_bank_data(n_samples_per_bank=300, seed=42)
    feature_cols = ["TransactionAmt", "velocity_count", "errorBalance", "hour_of_day", "foreign_dest"]

    clients = []
    for bank_name, df in data.items():
        client = BankClient(bank_name, df.iloc[:200], df.iloc[200:], feature_cols)
        client.train_local_model()
        clients.append(client)

    aggregator = FederatedEnsembleAggregator(clients, feature_cols)
    aggregator.collect_and_aggregate()

    test_X, _ = clients[0].get_test_data()

    # Test all 3 aggregation modes
    for mode in ["noisy_or", "max", "weighted_mean"]:
        probs = aggregator.predict_proba(test_X, mode=mode)
        preds = aggregator.predict(test_X, threshold=0.5, mode=mode)

        assert len(probs) == len(test_X)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert set(np.unique(preds)).issubset({0, 1})

    with pytest.raises(ValueError):
        aggregator.predict_proba(test_X, mode="invalid_mode")


def test_cross_bank_blindness_and_federated_advantage():
    """Verify that isolated local models underperform on foreign banks while federated model generalizes."""
    results = run_federated_fraud_benchmark(n_samples_per_bank=1000, seed=42)

    assert "metadata" in results
    assert results["metadata"]["threat_model"]["raw_data_transferred_bytes"] == 0
    assert results["metadata"]["threat_model"]["adversary"] == "Semi-honest (honest-but-curious)"

    global_eval = results["global_eval"]
    assert "federated_model" in global_eval
    assert "isolated_models_mean_recall" in global_eval

    # Federated model must outperform isolated models on the global pooled stream
    assert global_eval["federated_advantage_over_isolated"] >= 0.0
    assert global_eval["federated_model"]["recall"] >= global_eval["isolated_models_mean_recall"]
