"""Label-free entity history computed strictly before each event.
Same-timestamp events use a deterministic transaction-ID order. This is an offline
feature builder; online callers must supply these historical values from their feature store.
"""

from collections import deque
import pandas as pd


def add_history_features(df):
    df = df.copy()
    order = (
        ["TransactionDT", "TransactionID"]
        if "TransactionDT" in df
        else ["step", "trans_id"]
    )
    df = df.sort_values(order, kind="stable").reset_index(drop=True)
    keys = [c for c in ["card1", "card2", "addr1"] if c in df]
    if not keys and "nameOrig" in df:
        keys = ["nameOrig"]
    if not keys:
        raise ValueError("Historical features require card or sender identifiers")
    amount = "TransactionAmt" if "TransactionAmt" in df else "amount"
    states, counts, means, deviations, recent = {}, [], [], [], []
    for row in df.to_dict("records"):
        # Missing identities cannot safely link transactions.
        key = (
            tuple(str(row[c]) for c in keys)
            if all(pd.notna(row[c]) for c in keys)
            else ("missing", str(row["trans_id"]))
        )
        t = float(row.get("TransactionDT", (row["step"] - 1) * 3600)) / 3600
        n, total, queue = states.get(key, (0, 0.0, deque()))
        while queue and queue[0] <= t - 24:
            queue.popleft()
        a = float(row[amount])
        mean = total / n if n else 0.0
        counts.append(n)
        means.append(mean)
        deviations.append(a - mean if n else 0.0)
        recent.append(len(queue))
        queue.append(t)
        states[key] = (n + 1, total + a, queue)
    df["hist_tx_count"] = counts
    df["hist_amount_mean"] = means
    df["hist_amount_deviation"] = deviations
    df["hist_tx_24h"] = recent
    return df
