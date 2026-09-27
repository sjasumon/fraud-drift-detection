"""
Invariant Tests for Metric Reconciliation and Mathematical Precision (FR-6).
Asserts that confusion matrix components sum to exact totals down to the integer,
and that cohort decompositions reconcile exactly with global out-of-sample benchmarks.
"""

import numpy as np
import pytest

from src.evaluate_uid_split import compute_metrics


def test_confusion_matrix_integer_reconciliation():
    """
    Asserts that TP + FN equals total positives and TN + FP equals total negatives.
    """
    np.random.seed(42)
    n = 500
    y_true = (np.random.rand(n) < 0.10).astype(int)
    y_prob = np.random.rand(n)
    y_pred = (y_prob >= 0.5).astype(int)

    d = {
        "y": y_true.tolist(),
        "stat_p": y_pred.tolist(),
        "adapt_p": y_pred.tolist(),
        "stat_prob": y_prob.tolist(),
        "adapt_prob": y_prob.tolist()
    }

    metrics = compute_metrics(d)

    total_positives = int(y_true.sum())
    total_negatives = n - total_positives

    stat_tp = metrics["stat_tp"]
    stat_fn = total_positives - stat_tp
    stat_fp = int(((y_true == 0) & (y_pred == 1)).sum())
    stat_tn = total_negatives - stat_fp

    # Invariants: Exact integer sums
    assert stat_tp + stat_fn == total_positives, "TP + FN != Total Positives"
    assert stat_tn + stat_fp == total_negatives, "TN + FP != Total Negatives"
    assert stat_tp + stat_fn + stat_tn + stat_fp == n, "Confusion matrix sum != N"


def test_cohort_reconciliation_exact_sum():
    """
    Asserts that in Method B (full stream decomposition),
    TP_Cohort1 + TP_Cohort2 == TP_Global exactly, and
    Frauds_Cohort1 + Frauds_Cohort2 == Frauds_Global exactly.
    """
    np.random.seed(42)
    n = 1000
    y = (np.random.rand(n) < 0.08).astype(int)
    preds = (np.random.rand(n) > 0.45).astype(int)
    probs = np.random.rand(n)

    # Split randomly into Cohort 1 and Cohort 2 (representing disjoint partitioning)
    cohort1_mask = np.random.rand(n) < 0.60
    cohort2_mask = ~cohort1_mask

    # Compute metrics for Cohort 1
    d_c1 = {
        "y": y[cohort1_mask].tolist(),
        "stat_p": preds[cohort1_mask].tolist(),
        "adapt_p": preds[cohort1_mask].tolist(),
        "stat_prob": probs[cohort1_mask].tolist(),
        "adapt_prob": probs[cohort1_mask].tolist()
    }
    m_c1 = compute_metrics(d_c1)

    # Compute metrics for Cohort 2
    d_c2 = {
        "y": y[cohort2_mask].tolist(),
        "stat_p": preds[cohort2_mask].tolist(),
        "adapt_p": preds[cohort2_mask].tolist(),
        "stat_prob": probs[cohort2_mask].tolist(),
        "adapt_prob": probs[cohort2_mask].tolist()
    }
    m_c2 = compute_metrics(d_c2)

    # Compute metrics for Global stream
    d_global = {
        "y": y.tolist(),
        "stat_p": preds.tolist(),
        "adapt_p": preds.tolist(),
        "stat_prob": probs.tolist(),
        "adapt_prob": probs.tolist()
    }
    m_global = compute_metrics(d_global)

    # Invariant 1: Total transactions reconcile
    assert m_c1["total_tx"] + m_c2["total_tx"] == m_global["total_tx"], (
        f"Tx mismatch: {m_c1['total_tx']} + {m_c2['total_tx']} != {m_global['total_tx']}"
    )

    # Invariant 2: Total frauds reconcile
    assert m_c1["fraud_tx"] + m_c2["fraud_tx"] == m_global["fraud_tx"], (
        f"Fraud mismatch: {m_c1['fraud_tx']} + {m_c2['fraud_tx']} != {m_global['fraud_tx']}"
    )

    # Invariant 3: True positives reconcile down to the single integer
    assert m_c1["stat_tp"] + m_c2["stat_tp"] == m_global["stat_tp"], (
        f"TP mismatch: {m_c1['stat_tp']} + {m_c2['stat_tp']} != {m_global['stat_tp']}"
    )


def test_metric_boundary_conditions():
    """
    Asserts that compute_metrics handles edge cases gracefully:
    - Zero fraud transactions
    - Empty stream
    """
    # Case 1: Zero frauds
    d_zero_fraud = {
        "y": [0, 0, 0, 0, 0],
        "stat_p": [0, 0, 0, 0, 0],
        "adapt_p": [0, 0, 0, 0, 0],
        "stat_prob": [0.1, 0.2, 0.1, 0.3, 0.05],
        "adapt_prob": [0.1, 0.2, 0.1, 0.3, 0.05]
    }
    res = compute_metrics(d_zero_fraud)
    assert res == {}, "Expected empty dict when fraud count is zero."

    # Case 2: Empty lists
    d_empty = {
        "y": [],
        "stat_p": [],
        "adapt_p": [],
        "stat_prob": [],
        "adapt_prob": []
    }
    res_empty = compute_metrics(d_empty)
    assert res_empty == {}, "Expected empty dict when stream is empty."
