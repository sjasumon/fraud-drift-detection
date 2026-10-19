"""
Unit & Behavioral Tests for Remediation Findings R05 & R06.
R05: Statistics, AP paired moving-block bootstrap, Recall@Budget paired bootstrap,
     and H1 one-sided non-inferiority testing.
R06: Capacity policy, deterministic tie-breaking, causal online quota allocator,
     persistence across restart, and 5-alert quota regression.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.capacity_policy import (
    allocate_batch_ranking,
    allocate_online_stream,
    OnlineDailyQuotaAllocator,
    POLICY_RETROSPECTIVE_BATCH,
    POLICY_CAUSAL_ONLINE,
)
from src.reporting import block_bootstrap_metric_delta, block_bootstrap_delta


# =========================================================================
# Finding R06: Capacity Policy & Quota Tests
# =========================================================================

def test_r06_5_alert_quota_regression_tied_probabilities():
    """
    R06 Regression: When 50 items share identical probability crossing the threshold,
    allocate_batch_ranking MUST return exactly 5 alerts (not 40 or 50).
    Ties must be broken deterministically by trans_id.
    """
    n = 50
    df = pd.DataFrame({
        "trans_id": [f"T_{i:04d}" for i in range(n)],
        "probability": [0.85] * n,  # all tied
        "y_true": [1 if i % 5 == 0 else 0 for i in range(n)],
        "prediction": [1] * n,
    })
    budget = 5
    result, _ = allocate_batch_ranking(df, daily_budget=budget)
    assert len(result) == 5, f"Expected exactly {budget} alerts, got {len(result)}"
    assert result["trans_id"].tolist() == [f"T_{i:04d}" for i in range(5)]
    assert result["is_alert"].sum() == 5


def test_r06_batch_ranking_validates_budget():
    """Budget must be a positive integer."""
    df = pd.DataFrame({
        "trans_id": ["T_1", "T_2"],
        "probability": [0.9, 0.1],
        "y_true": [1, 0],
        "prediction": [1, 0],
    })
    with pytest.raises(ValueError, match="positive integer"):
        allocate_batch_ranking(df, daily_budget=0)
    with pytest.raises(ValueError, match="positive integer"):
        allocate_batch_ranking(df, daily_budget=-5)


def test_r06_batch_ranking_deterministic_ordering():
    """Sorts primarily by probability descending, secondarily by trans_id ascending."""
    df = pd.DataFrame({
        "trans_id": ["B_02", "A_01", "C_03", "D_04"],
        "probability": [0.70, 0.70, 0.95, 0.50],
        "y_true": [0, 1, 1, 0],
        "prediction": [1, 1, 1, 0],
    })
    top2, _ = allocate_batch_ranking(df, daily_budget=2)
    assert len(top2) == 2
    # Highest prob is C_03 (0.95), second is tie between A_01 and B_02 (0.70), A_01 wins lexicographically
    assert top2.iloc[0]["trans_id"] == "C_03"
    assert top2.iloc[1]["trans_id"] == "A_01"


def test_r06_causal_online_allocator_enforces_budget_and_advances_day(tmp_path: Path):
    """
    Causal online quota tracker:
    - Only admits up to daily_budget alerts per day.
    - Day advance resets quota.
    - Idempotent on repeated trans_id.
    """
    allocator = OnlineDailyQuotaAllocator(daily_budget=2, threshold=0.5)

    # Day 1: 4 transactions above threshold
    res1 = allocator.process_transaction("TX_01", 0.9, day=1)
    res2 = allocator.process_transaction("TX_02", 0.8, day=1)
    res3 = allocator.process_transaction("TX_03", 0.7, day=1)  # Quota exhausted
    res4 = allocator.process_transaction("TX_04", 0.6, day=1)  # Quota exhausted

    assert res1.is_alert is True and res1.reason == "admitted"
    assert res2.is_alert is True and res2.reason == "admitted"
    assert res3.is_alert is False and res3.reason == "quota_exhausted"
    assert res4.is_alert is False and res4.reason == "quota_exhausted"

    # Replay of TX_01 is idempotent
    res1_replay = allocator.process_transaction("TX_01", 0.9, day=1)
    assert res1_replay.is_alert is True
    assert res1_replay.reason == "admitted_prior"

    # Advance to Day 2: Quota resets to 2
    res5 = allocator.process_transaction("TX_05", 0.85, day=2)
    assert res5.is_alert is True and res5.reason == "admitted"
    assert allocator.day_allocated == 1
    assert allocator.daily_remaining == 1


def test_r06_causal_online_allocator_persistence_across_restart(tmp_path: Path):
    """Allocator state persists to disk and recovers cleanly on restart."""
    state_file = tmp_path / "quota_state.json"
    allocator = OnlineDailyQuotaAllocator(daily_budget=3, threshold=0.4, state_path=state_file)

    allocator.process_transaction("TX_10", 0.7, day=5)
    allocator.process_transaction("TX_11", 0.8, day=5)
    assert allocator.daily_remaining == 1

    # Simulate restart by creating a new instance with the same state_path
    restarted = OnlineDailyQuotaAllocator(daily_budget=3, threshold=0.4, state_path=state_file)
    assert restarted.current_day == 5
    assert restarted.daily_remaining == 1
    assert restarted.day_allocated == 2

    # Process 3rd alert on Day 5
    res = restarted.process_transaction("TX_12", 0.9, day=5)
    assert res.is_alert is True
    assert restarted.daily_remaining == 0

    # 4th alert rejected
    res_overflow = restarted.process_transaction("TX_13", 0.95, day=5)
    assert res_overflow.is_alert is False
    assert res_overflow.reason == "quota_exhausted"


# =========================================================================
# Finding R05: Paired Bootstrap Statistics & Non-Inferiority Tests
# =========================================================================

def _create_synthetic_evaluation_df(n_per_slice=50, n_slices=6, seed=42):
    """Helper to create matched predictions across slices."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_slices):
        for i in range(n_per_slice):
            tid = f"TX_S{s}_{i:03d}"
            y = int(rng.random() < 0.20)  # 20% fraud
            prob_static = float(rng.uniform(0.1, 0.7))
            prob_adaptive = float(np.clip(prob_static + rng.normal(0.05, 0.02), 0.0, 1.0))

            rows.append({
                "seed": 42,
                "window": 0,
                "slice_index": s,
                "day": s,
                "trans_id": tid,
                "model": "static",
                "y_true": y,
                "probability": prob_static,
                "prediction": int(prob_static >= 0.5),
            })
            rows.append({
                "seed": 42,
                "window": 0,
                "slice_index": s,
                "day": s,
                "trans_id": tid,
                "model": "adaptive",
                "y_true": y,
                "probability": prob_adaptive,
                "prediction": int(prob_adaptive >= 0.5),
            })
            # Cloned weekly identical to adaptive
            rows.append({
                "seed": 42,
                "window": 0,
                "slice_index": s,
                "day": s,
                "trans_id": tid,
                "model": "weekly",
                "y_true": y,
                "probability": prob_adaptive,
                "prediction": int(prob_adaptive >= 0.5),
            })
    return pd.DataFrame(rows)


def test_r05_identical_models_yield_zero_paired_difference():
    """Identical models (or comparing model to itself) MUST yield exactly 0 delta and 0 CI."""
    df = _create_synthetic_evaluation_df()
    res = block_bootstrap_metric_delta(
        df,
        candidate_model="adaptive",
        baseline_model="weekly",  # weekly is identical to adaptive in df
        metric="average_precision",
        n_bootstrap=100,
    )
    assert res["point_estimate"] == 0.0
    assert res["mean_delta"] == 0.0
    assert res["ci_95"] == [0.0, 0.0]
    assert res["ci_99"] == [0.0, 0.0]
    assert res["empirical_p_value"] == 0.0
    assert res["is_non_inferior"] is True


def test_r05_rejects_mismatched_labels():
    """If ground-truth labels differ between candidate and baseline, must raise ValueError."""
    df = _create_synthetic_evaluation_df()
    # Corrupt one y_true in adaptive
    df.loc[(df.model == "adaptive") & (df.trans_id == "TX_S0_000"), "y_true"] = 1 - df.loc[(df.model == "adaptive") & (df.trans_id == "TX_S0_000"), "y_true"].values[0]

    with pytest.raises(ValueError, match="Paired ground-truth labels differ"):
        block_bootstrap_metric_delta(df, "adaptive", "static", metric="average_precision")


def test_r05_rejects_missing_or_unpaired_records():
    """If candidate and baseline have different counts or missing keys, must raise ValueError."""
    df = _create_synthetic_evaluation_df()
    # Drop one row from static
    df = df[~((df.model == "static") & (df.trans_id == "TX_S0_000"))]

    with pytest.raises(ValueError, match="Predictions must be paired exactly"):
        block_bootstrap_metric_delta(df, "adaptive", "static", metric="average_precision")


def test_r05_average_precision_matches_independent_calculation():
    """Point estimate delta must match independent scikit-learn average_precision_score difference."""
    df = _create_synthetic_evaluation_df()
    res = block_bootstrap_metric_delta(
        df,
        candidate_model="adaptive",
        baseline_model="static",
        metric="average_precision",
        n_bootstrap=50,
        seed=123,
    )

    y = df[df.model == "adaptive"]["y_true"].to_numpy()
    p_ad = df[df.model == "adaptive"]["probability"].to_numpy()
    p_st = df[df.model == "static"]["probability"].to_numpy()

    expected_delta = float(average_precision_score(y, p_ad) - average_precision_score(y, p_st))
    assert pytest.approx(res["point_estimate"], abs=1e-6) == expected_delta
    assert res["ci_95"][0] <= res["mean_bootstrap_delta"] <= res["ci_95"][1]
    assert res["metric"] == "average_precision"


def test_r05_recall_at_budget_paired_bootstrap():
    """Recall@Budget delta correctly aggregates top-K detections across days."""
    df = _create_synthetic_evaluation_df(n_per_slice=20, n_slices=4)
    budget = 5
    res = block_bootstrap_metric_delta(
        df,
        candidate_model="adaptive",
        baseline_model="static",
        metric="recall_at_budget",
        daily_budget=budget,
        capacity_policy="retrospective_batch",
        n_bootstrap=50,
        seed=42,
    )

    assert "point_estimate" in res
    assert res["daily_budget"] == budget
    assert res["capacity_policy"] == "retrospective_batch"
    assert res["ci_95"][0] <= res["ci_95"][1]


def test_r05_h1_one_sided_non_inferiority_test():
    """
    Hypothesis H1 test: candidate recall >= baseline recall - margin.
    When candidate is identical or superior, is_non_inferior is True and empirical p-val <= 0.05.
    When candidate is artificially degraded by 0.30, is_non_inferior is False.
    """
    df = _create_synthetic_evaluation_df()

    # Test 1: identical models (adaptive vs weekly) with 0.01 margin
    res_ident = block_bootstrap_metric_delta(
        df,
        candidate_model="adaptive",
        baseline_model="weekly",
        metric="average_precision",
        non_inferiority_margin=0.01,
        n_bootstrap=50,
    )
    assert res_ident["is_non_inferior"] is True
    assert res_ident["empirical_p_value"] == 0.0

    # Test 2: degraded candidate model (inverting signal to ensure severe underperformance)
    df_degraded = df.copy()
    mask = df_degraded.model == "adaptive"
    df_degraded.loc[mask, "probability"] = np.where(df_degraded.loc[mask, "y_true"] == 1, 0.05, 0.95)
    df_degraded.loc[mask, "prediction"] = (df_degraded.loc[mask, "probability"] >= 0.5).astype(int)
    res_degraded = block_bootstrap_metric_delta(
        df_degraded,
        candidate_model="adaptive",
        baseline_model="static",
        metric="average_precision",
        non_inferiority_margin=0.01,
        n_bootstrap=50,
    )
    # Degraded model should fail non-inferiority
    assert res_degraded["is_non_inferior"] is False
    assert res_degraded["empirical_p_value"] > 0.05


def test_r05_backward_compatibility_block_bootstrap_delta():
    """Existing callers of block_bootstrap_delta receive expected dictionary structure."""
    df = _create_synthetic_evaluation_df(n_slices=4, n_per_slice=20)
    res = block_bootstrap_delta(df, "adaptive", baseline="static", n_bootstrap=30)
    assert "mean_delta" in res
    assert "ci_95" in res
    assert "n_bootstrap" in res
    assert "temporal_blocks" in res
    assert len(res["ci_95"]) == 2
