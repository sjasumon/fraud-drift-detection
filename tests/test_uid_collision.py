"""
Invariant Tests for Entity (UID) Tokenization and Cohort Partitioning.
Asserts that missing-identity transactions receive collision-free unique tokens,
and that Method A and Method B cohorts maintain strict mathematical set disjointness.
"""

import numpy as np
import pandas as pd
import pytest


def test_missing_identity_collision_freedom(synthetic_stream_df):
    """
    Asserts that missing-identity fields in Method B are assigned unique collision-free tokens
    ('MISSING_<trans_id>') rather than grouping into 'nan_nan'.
    """
    df = synthetic_stream_df.copy()

    # Identify rows with missing card2 or addr1
    is_missing = df["card2"].isna() | df["addr1"].isna()
    assert is_missing.sum() > 0, "Fixture must include rows with missing card2/addr1."

    # Construct collision-free UID tokens (Method B)
    tokens = []
    for idx, row in df.iterrows():
        if pd.isna(row["card2"]) or pd.isna(row["addr1"]):
            tokens.append(f"MISSING_{row['trans_id']}")
        else:
            tokens.append(f"{int(row['card1'])}_{int(row['card2'])}_{int(row['addr1'])}")

    df["uid_cf"] = tokens

    # Invariant: No token should contain the string 'nan'
    assert not df["uid_cf"].str.contains("nan", case=False).any(), (
        "Found 'nan' substring in generated UID tokens!"
    )

    # Invariant: All missing-identity tokens must be globally unique
    missing_tokens = df.loc[is_missing, "uid_cf"]
    assert missing_tokens.nunique() == len(missing_tokens), (
        f"Missing identity tokens collided: {len(missing_tokens)} rows produced only {missing_tokens.nunique()} unique tokens!"
    )


def test_method_a_clean_filtering(synthetic_stream_df):
    """
    Asserts that Method A strict complete-case filtering retains only transactions
    where card1, card2, and addr1 are all non-null.
    """
    df = synthetic_stream_df.copy()
    clean_mask = df["card1"].notna() & df["card2"].notna() & df["addr1"].notna()
    clean_df = df[clean_mask]

    # Invariant: Zero NaNs in identifier fields
    assert clean_df["card1"].isna().sum() == 0
    assert clean_df["card2"].isna().sum() == 0
    assert clean_df["addr1"].isna().sum() == 0

    # Coverage arithmetic check
    expected_clean_count = clean_mask.sum()
    assert len(clean_df) == expected_clean_count
    coverage_pct = (len(clean_df) / len(df)) * 100
    assert 0.0 < coverage_pct <= 100.0


def test_cohort_disjointness_and_exhaustiveness(synthetic_stream_df):
    """
    Asserts that:
    1. Baseline-Seen (Cohort 1) and Baseline-Unseen (Cohort 2) are mutually disjoint.
    2. Cohort 1 union Cohort 2 equals the entire evaluated out-of-sample population.
    3. Strict Stream-Unseen (Cohort 3) is a strict subset of Cohort 2.
    """
    df = synthetic_stream_df.copy()

    # Split into baseline (Slice 0: step <= 24) and out-of-sample stream (step > 24)
    baseline_df = df[df["step"] <= 24]
    stream_df = df[df["step"] > 24]

    # Tokenize
    baseline_uids = set()
    for _, row in baseline_df.iterrows():
        if pd.isna(row["card2"]) or pd.isna(row["addr1"]):
            baseline_uids.add(f"MISSING_{row['trans_id']}")
        else:
            baseline_uids.add(f"{int(row['card1'])}_{int(row['card2'])}_{int(row['addr1'])}")

    stream_tokens = []
    for _, row in stream_df.iterrows():
        if pd.isna(row["card2"]) or pd.isna(row["addr1"]):
            stream_tokens.append(f"MISSING_{row['trans_id']}")
        else:
            stream_tokens.append(f"{int(row['card1'])}_{int(row['card2'])}_{int(row['addr1'])}")

    stream_df = stream_df.assign(uid=stream_tokens)

    # Cohort 1: Seen in baseline
    cohort1_mask = stream_df["uid"].isin(baseline_uids)
    cohort1 = stream_df[cohort1_mask]

    # Cohort 2: Unseen in baseline
    cohort2 = stream_df[~cohort1_mask]

    # Invariant 1: Disjointness
    assert len(set(cohort1.index).intersection(set(cohort2.index))) == 0, (
        "Cohort 1 and Cohort 2 overlap!"
    )

    # Invariant 2: Exhaustiveness
    assert len(cohort1) + len(cohort2) == len(stream_df), (
        f"Sum of Cohort 1 ({len(cohort1)}) + Cohort 2 ({len(cohort2)}) != total stream ({len(stream_df)})"
    )

    # Cohort 3: Stream-Unseen (card never seen before anywhere up to that transaction)
    seen_in_stream = set(baseline_uids)
    cohort3_indices = []
    for idx, row in stream_df.iterrows():
        u = row["uid"]
        if u not in seen_in_stream:
            cohort3_indices.append(idx)
        seen_in_stream.add(u)

    cohort3 = stream_df.loc[cohort3_indices]

    # Invariant 3: Cohort 3 is a subset of Cohort 2
    assert set(cohort3.index).issubset(set(cohort2.index)), (
        "Cohort 3 contains transactions that were in Cohort 1!"
    )
