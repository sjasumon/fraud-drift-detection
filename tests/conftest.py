"""
Pytest configuration and synthetic fixtures for concept-drift-aware fraud detection tests.
Provides synthetic streaming datasets with known temporal partitions, missing-value patterns,
and categorical variations to test mathematical invariants under controlled conditions.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_stream_df() -> pd.DataFrame:
    """
    Creates a synthetic 3-slice transaction stream (300 rows total, 100 per slice).
    - Slice 0 (step 1-24): Training baseline.
    - Slice 1 (step 25-48): Stream slice 1.
    - Slice 2 (step 49-72): Stream slice 2, introducing an unseen category 'Z'.
    Includes complete rows and rows with missing card2 / addr1 to test UID tokenization.
    """
    np.random.seed(42)
    n_rows = 300

    # 3 non-overlapping slices: steps 1-24, 25-48, 49-72
    steps = np.concatenate(
        [
            np.random.randint(1, 25, size=100),
            np.random.randint(25, 49, size=100),
            np.random.randint(49, 73, size=100),
        ]
    )

    tx_ids = np.arange(1000000, 1000000 + n_rows)
    tx_dt = (steps - 1) * 3600 + np.random.randint(0, 3600, size=n_rows)

    # Categories: Slice 0 & 1 have 'W', 'H', 'C'. Slice 2 introduces 'Z'
    cats = []
    for i in range(n_rows):
        if i < 200:
            cats.append(np.random.choice(["W", "H", "C"]))
        else:
            cats.append(np.random.choice(["W", "H", "Z"]))

    # Card and address identifiers
    card1 = np.random.choice([1001, 1002, 1003, 1004, 1005], size=n_rows).astype(float)
    card2 = np.random.choice(
        [201.0, 202.0, 203.0, np.nan], size=n_rows, p=[0.35, 0.35, 0.15, 0.15]
    )
    addr1 = np.random.choice([301.0, 302.0, np.nan], size=n_rows, p=[0.45, 0.45, 0.10])

    # Numerical features
    amt = np.random.exponential(scale=100.0, size=n_rows) + 5.0
    c1 = np.random.poisson(lam=3.0, size=n_rows).astype(float)
    d1 = np.random.uniform(0, 500, size=n_rows)

    # Labels with ~5% fraud prevalence
    is_fraud = (np.random.uniform(0, 1, size=n_rows) < 0.08).astype(int)
    # Guarantee at least some frauds in each slice
    is_fraud[10] = 1
    is_fraud[120] = 1
    is_fraud[240] = 1

    df = pd.DataFrame(
        {
            "TransactionID": tx_ids,
            "trans_id": [str(x) for x in tx_ids],
            "TransactionDT": tx_dt,
            "step": steps,
            "TransactionAmt": amt,
            "amount": amt,
            "ProductCD": cats,
            "card1": card1,
            "card2": card2,
            "addr1": addr1,
            "C1": c1,
            "D1": d1,
            "isFraud": is_fraud,
        }
    )

    # Sort chronologically by step then trans_id
    df = df.sort_values(by=["step", "trans_id"]).reset_index(drop=True)
    return df
