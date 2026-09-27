"""
Unit tests for the Delayed Feedback & Verification Latency Simulation Module.
Validates exact temporal maturity intervals, absence of lookahead leakage in training buffers,
and queue state reconciliation.
"""

import pytest
import numpy as np
import pandas as pd

from src.delayed_feedback import DelayedFeedbackQueue


@pytest.fixture
def sample_slice_data():
    """Generates synthetic transactions for testing delayed feedback queues."""
    n_rows = 100
    df = pd.DataFrame({
        "trans_id": [f"TX_{i:04d}" for i in range(n_rows)],
        "step": np.arange(n_rows),
        "TransactionAmt": np.random.uniform(10, 500, size=n_rows),
        "isFraud": np.random.binomial(1, 0.05, size=n_rows)
    })
    return df


def test_fixed_delay_maturity_exact_slices(sample_slice_data):
    """Verifies that transactions mature exactly at slice t + delay under fixed latency."""
    delay = 4
    queue = DelayedFeedbackQueue(delay_slices=delay, latency_mode="fixed")

    # Enqueue at slice 0
    assigned = queue.enqueue(sample_slice_data, current_slice=0)
    assert len(assigned) == len(sample_slice_data)
    assert np.all(assigned == 4)
    assert queue.get_pending_count() == len(sample_slice_data)
    assert queue.get_matured_count() == 0

    # Slices 0, 1, 2, 3: Nothing should mature
    for s in range(4):
        matured = queue.dequeue_matured(current_slice=s)
        assert len(matured) == 0
        assert queue.get_matured_count() == 0

    # Slice 4: All slice 0 transactions must mature
    matured_at_4 = queue.dequeue_matured(current_slice=4)
    assert len(matured_at_4) == len(sample_slice_data)
    assert queue.get_matured_count() == len(sample_slice_data)
    assert queue.get_pending_count() == 0

    # Slice 5: Queue is empty, 0 new matures
    matured_at_5 = queue.dequeue_matured(current_slice=5)
    assert len(matured_at_5) == 0


def test_zero_delay_immediate_feedback(sample_slice_data):
    """Verifies that delay_slices=0 acts as an immediate verification baseline."""
    queue = DelayedFeedbackQueue(delay_slices=0, latency_mode="fixed")
    queue.enqueue(sample_slice_data, current_slice=5)
    
    matured = queue.dequeue_matured(current_slice=5)
    assert len(matured) == len(sample_slice_data)
    assert queue.get_pending_count() == 0
    assert queue.get_matured_count() == len(sample_slice_data)


def test_zero_temporal_leakage_in_training_data(sample_slice_data):
    """Verifies that get_all_matured_training_data contains NO unverified pending records."""
    delay = 4
    queue = DelayedFeedbackQueue(delay_slices=delay, latency_mode="fixed")
    seed_df = sample_slice_data.iloc[:20].copy()

    # Stream slices 0 through 6
    for s in range(7):
        slice_df = sample_slice_data.copy()
        slice_df["trans_id"] = [f"TX_S{s}_{i}" for i in range(len(slice_df))]
        queue.enqueue(slice_df, current_slice=s)
        queue.dequeue_matured(current_slice=s)

    # At slice 6, transactions from slice 0, 1, 2 have matured (0+4=4, 1+4=5, 2+4=6).
    # Slices 3, 4, 5, 6 must STILL be pending.
    matured_train_df = queue.get_all_matured_training_data(initial_train_df=seed_df)

    expected_count = len(seed_df) + (3 * len(sample_slice_data))
    assert len(matured_train_df) == expected_count

    # Verify no records from Slices 3, 4, 5, 6 exist in the training set
    for s in [3, 4, 5, 6]:
        forbidden_prefix = f"TX_S{s}_"
        assert not matured_train_df["trans_id"].str.startswith(forbidden_prefix).any()


def test_lognormal_delay_distribution(sample_slice_data):
    """Verifies that LogNormal delays produce valid positive future maturity slices."""
    queue = DelayedFeedbackQueue(latency_mode="lognormal", lognormal_mu=1.38, random_seed=42)
    assigned = queue.enqueue(sample_slice_data, current_slice=10)

    assert len(assigned) == len(sample_slice_data)
    # Every transaction must mature strictly after arrival slice 10
    assert np.all(assigned > 10)
    # Check reasonable empirical range (most within 1 to 15 slices)
    delays = assigned - 10
    assert np.all(delays >= 1)
    assert np.median(delays) >= 2


def test_invalid_parameters_raise_error():
    """Verifies that invalid delay parameters trigger immediate ValueError."""
    with pytest.raises(ValueError, match="delay_slices must be >= 0"):
        DelayedFeedbackQueue(delay_slices=-1)

    with pytest.raises(ValueError, match="latency_mode must be 'fixed' or 'lognormal'"):
        DelayedFeedbackQueue(latency_mode="exponential")
