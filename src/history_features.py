"""Label-free entity history computed strictly before each event.
Events with tied timestamps are scored against strictly prior history before shared
state is updated, preventing intra-second ordering leakage.
This is an offline feature builder; online callers must supply these historical
values from their causal feature store.
"""

from collections import deque
import numpy as np
import pandas as pd


def add_history_features(df):
    df = df.copy()
    time_col = "TransactionDT" if "TransactionDT" in df else "step"
    id_col = "TransactionID" if "TransactionID" in df else "trans_id"
    amount_col = "TransactionAmt" if "TransactionAmt" in df else "amount"

    # Sort primarily by timestamp, preserving stable input order for ties
    df = df.sort_values([time_col, id_col], kind="stable").reset_index(drop=True)

    keys = [c for c in ["card1", "card2", "addr1"] if c in df]
    if not keys and "nameOrig" in df:
        keys = ["nameOrig"]
    if not keys:
        raise ValueError("Historical features require card or sender identifiers")

    n_rows = len(df)
    times = df[time_col].to_numpy(dtype=float)
    if time_col == "step":
        # Convert 1-based step hours to seconds
        times = (times - 1.0) * 3600.0
    times_hours = times / 3600.0
    amounts = df[amount_col].to_numpy(dtype=float)

    is_null = df[keys].isna().any(axis=1).to_numpy()
    if len(keys) == 1:
        raw_keys = df[keys[0]].astype(str).to_numpy()
    else:
        raw_keys = list(zip(*(df[c].astype(str) for c in keys)))

    trans_ids = df[id_col].astype(str).to_numpy()

    counts = np.zeros(n_rows, dtype=int)
    means = np.zeros(n_rows, dtype=float)
    deviations = np.zeros(n_rows, dtype=float)
    recent_24h = np.zeros(n_rows, dtype=int)

    states = {}  # key -> (n, total_amt, deque_of_hours)

    # Group by identical timestamp to prevent intra-second ordering leakage
    diffs = np.flatnonzero(times[:-1] != times[1:]) + 1
    boundaries = np.concatenate(([0], diffs, [n_rows]))

    for b_idx in range(len(boundaries) - 1):
        start = boundaries[b_idx]
        end = boundaries[b_idx + 1]

        # Step 1: Compute historical features using strictly prior observable state (t < T)
        batch_updates = []
        for i in range(start, end):
            # Missing identities cannot safely link transactions across accounts
            key = ("missing", trans_ids[i]) if is_null[i] else raw_keys[i]
            t_h = times_hours[i]
            a = amounts[i]

            n, total, queue = states.get(key, (0, 0.0, deque()))
            while queue and queue[0] <= t_h - 24.0:
                queue.popleft()

            m = total / n if n > 0 else 0.0
            counts[i] = n
            means[i] = m
            deviations[i] = a - m if n > 0 else 0.0
            recent_24h[i] = len(queue)

            batch_updates.append((key, a, t_h))

        # Step 2: Apply updates from this tied timestamp window to shared history state
        for key, a, t_h in batch_updates:
            n, total, queue = states.get(key, (0, 0.0, deque()))
            queue.append(t_h)
            states[key] = (n + 1, total + a, queue)

    df["hist_tx_count"] = counts
    df["hist_amount_mean"] = means
    df["hist_amount_deviation"] = deviations
    df["hist_tx_24h"] = recent_24h
    return df
