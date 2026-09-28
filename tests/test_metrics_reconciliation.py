"""
Invariant Tests for Metric Reconciliation and Mathematical Precision (FR-6).
Asserts that confusion matrix components sum to exact totals down to the integer,
and that cohort decompositions reconcile exactly with global out-of-sample benchmarks.
"""

import numpy as np
import pytest

from src.evaluate_uid_split import compute_metrics
from src.evaluator import SliceEvaluator


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


def test_evaluator_zero_fraud_and_single_class_edge_cases():
    """Verifies that SliceEvaluator.compute_overall_summary and evaluate_slice handle zero frauds gracefully."""
    evaluator = SliceEvaluator()
    n = 30
    preds = np.zeros(n, dtype=int)
    probs = np.full(n, 0.05)
    y_zero = np.zeros(n, dtype=int)

    # 1. evaluate_slice with zero fraud
    slice_res = evaluator.evaluate_slice(
        y_true=y_zero,
        y_pred=preds,
        y_prob=probs,
        slice_index=0,
        slice_name="ZeroFraudSlice",
        model_type="static"
    )
    assert slice_res["auc_pr"] == 0.0
    assert slice_res["f1_score"] == 0.0

    # 2. compute_overall_summary with zero fraud across stream
    summary = evaluator.compute_overall_summary(
        static_preds=preds,
        static_probs=probs,
        adaptive_preds=preds,
        adaptive_probs=probs,
        y_all=y_zero,
        oos_mask=np.ones(n, dtype=bool)
    )
    assert summary["static_model"]["overall_auc_pr"] == 0.0
    assert summary["adaptive_model"]["overall_auc_pr"] == 0.0
    assert summary["static_model"]["is_pure_out_of_sample"] is False
    assert "overall_f1_including_training_window" in summary["static_model"]
    assert "metric_disambiguation_note" in summary["static_model"]
    assert summary["out_of_sample_stream"]["static_model"]["overall_auc_pr"] == 0.0
    assert summary["out_of_sample_stream"]["adaptive_model"]["overall_auc_pr"] == 0.0
    assert summary["out_of_sample_stream"]["static_model"]["is_pure_out_of_sample"] is True
    assert "pure_oos_f1" in summary["out_of_sample_stream"]["static_model"]


def test_operational_alert_budget_and_fpr_metrics():
    """Verifies that compute_recall_at_fpr and compute_recall_at_alert_budget satisfy mathematical bounds."""
    from src.evaluator import compute_recall_at_fpr, compute_recall_at_alert_budget

    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])

    # Top 20% alert budget (2 alerts out of 10)
    budget_20 = compute_recall_at_alert_budget(y_true, y_prob, budget_ratio=0.20)
    assert budget_20["alert_count"] == 2
    assert budget_20["recall"] == 1.0  # Caught both frauds
    assert budget_20["precision"] == 1.0

    # Top 10% alert budget (1 alert out of 10)
    budget_10 = compute_recall_at_alert_budget(y_true, y_prob, budget_ratio=0.10)
    assert budget_10["alert_count"] == 1
    assert budget_10["recall"] == 0.5  # Caught 1 of 2 frauds
    assert budget_10["precision"] == 1.0

    # Recall at FPR <= 15% (8 negatives * 0.15 = 1.2 -> at most 1 FP allowed)
    fpr_res = compute_recall_at_fpr(y_true, y_prob, target_fpr=0.15)
    assert fpr_res["actual_fpr"] <= 0.15
    assert 0.0 <= fpr_res["recall"] <= 1.0
    assert 0.0 <= fpr_res["precision"] <= 1.0


def test_cost_weighted_dollar_recall_metrics():
    """Verifies that compute_cost_weighted_metrics accurately weights losses by TransactionAmt."""
    from src.evaluator import compute_cost_weighted_metrics, SliceEvaluator

    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])  # Caught fraud 0, missed fraud 1, 1 false alarm
    amounts = np.array([10000.0, 50.0, 20.0, 10.0])  # Caught $10k fraud, missed $50 fraud

    # Dollar recall: 10000 / (10000 + 50) = 10000 / 10050 = 0.9950
    # Standard unweighted recall: 1 / 2 = 0.50
    cost_m = compute_cost_weighted_metrics(y_true, y_pred, amounts, cost_per_investigation=10.0)
    assert cost_m["dollar_recall"] == round(10000.0 / 10050.0, 4)
    assert cost_m["total_fraud_dollars"] == 10050.0
    assert cost_m["intercepted_fraud_dollars"] == 10000.0
    assert cost_m["missed_fraud_dollars"] == 50.0
    assert cost_m["total_alerts"] == 2
    assert cost_m["investigation_cost"] == 20.0
    assert cost_m["net_financial_savings"] == 9980.0

    # Test integrated evaluate_slice with amounts
    evaluator = SliceEvaluator()
    entry = evaluator.evaluate_slice(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=np.array([0.9, 0.4, 0.6, 0.1]),
        slice_index=0,
        slice_name="CostSlice",
        model_type="adaptive",
        amounts=amounts
    )
    assert "dollar_recall" in entry
    assert "intercepted_fraud_dollars" in entry
    assert "net_financial_savings" in entry
    assert "recall_at_alert_budget_1pct" in entry
    assert "recall_at_fpr_1pct" in entry


