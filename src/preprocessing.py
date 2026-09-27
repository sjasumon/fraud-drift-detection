"""
FR-2: Preprocessing and Time-Slice Partitioning Module.
Optimized for IEEE-CIS Fraud Detection and PaySim transaction streams.
Handles missing values, categorical encoding, class imbalance, and sequential weekly partitioning.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.config import DEFAULT_HOURS_PER_WEEK

logger = logging.getLogger("fraud_drift.preprocessing")


class Preprocessor:
    """Preprocesses raw transaction data and partitions into sequential temporal slices."""

    def __init__(self, hours_per_slice: int = DEFAULT_HOURS_PER_WEEK, exclude_card_identifiers: bool = False):
        self.hours_per_slice = hours_per_slice
        self.exclude_card_identifiers = exclude_card_identifiers
        self.encoder: Optional[OneHotEncoder] = None
        self.fitted = False
        self.feature_names: List[str] = []
        self.imbalance_weights: Dict[str, float] = {}
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.imputation_values: Dict[str, float] = {}

    def extract_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives temporal and accounting features."""
        df = df.copy()

        # Temporal features
        df["hour_of_day"] = (df["step"] - 1) % 24
        df["day_of_week"] = ((df["step"] - 1) // 24) % 7
        df["is_overnight"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)

        # Amount features
        amt_col = None
        for cand in ["TransactionAmt", "amount", "proposed_credit_limit"]:
            if cand in df.columns:
                amt_col = cand
                break

        if amt_col is not None:
            df["log_amount"] = np.log1p(np.maximum(df[amt_col], 0.0))
            df["amt_decimal"] = df[amt_col] - np.floor(df[amt_col])

        # PaySim specific features if present
        if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
            df["orig_balance_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
            if amt_col is not None:
                df["orig_reconcile_err"] = (df["newbalanceOrig"] + df[amt_col]) - df["oldbalanceOrg"]
                df["amount_to_orig_ratio"] = df[amt_col] / (df["oldbalanceOrg"] + 1.0)
            df["orig_emptied"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)

        if "oldbalanceDest" in df.columns and "newbalanceDest" in df.columns:
            df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
            if amt_col is not None:
                df["dest_reconcile_err"] = (df["oldbalanceDest"] + df[amt_col]) - df["newbalanceDest"]
            df["dest_was_zero"] = (df["oldbalanceDest"] == 0).astype(int)

        return df

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """
        FR-2: Fits imputation medians, one-hot category vocabularies, and class imbalance
        weights strictly from the training partition (e.g. Weeks 1-5).
        Ensures zero temporal leakage from future streaming slices.
        """
        df = self.extract_engineered_features(df)

        # Identify candidate categorical columns
        possible_cats = [
            "ProductCD", "card4", "card6", "type",
            "payment_type", "employment_status", "housing_status", "source", "device_os"
        ]
        self.categorical_cols = [c for c in possible_cats if c in df.columns]

        # Base exclusion set
        exclude_cols = {
            "TransactionID", "trans_id", "isFraud", "isFlaggedFraud", "TransactionDT",
            "step", "month", "fraud_bool", "nameOrig", "nameDest", "drift_phase", "week",
            "P_emaildomain", "R_emaildomain", "uid", "uid_clean", "uid_cf", "is_clean"
        }

        # Optionally exclude raw card and address identifier codes
        if self.exclude_card_identifiers:
            exclude_cols.update({"card1", "card2", "card3", "card5", "addr1", "addr2"})

        self.numeric_cols = [
            c for c in df.columns
            if c not in exclude_cols and c not in self.categorical_cols
        ]

        # Compute median imputation values strictly on the training partition
        logger.info(f"FR-2.1: Computing median imputation for {len(self.numeric_cols)} numerical features on training partition...")
        for col in self.numeric_cols:
            median_val = float(df[col].median(skipna=True))
            if np.isnan(median_val):
                median_val = 0.0
            self.imputation_values[col] = median_val

        # Fit OneHotEncoder strictly on training partition
        if self.categorical_cols:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            # Fit on non-null string representation
            train_cats = df[self.categorical_cols].fillna("UNKNOWN").astype(str)
            self.encoder.fit(train_cats)
            cat_feature_names = list(self.encoder.get_feature_names_out(self.categorical_cols))
        else:
            cat_feature_names = []

        self.feature_names = self.numeric_cols + cat_feature_names

        # Compute class imbalance weight strictly on training partition
        total_samples = len(df)
        pos_samples = int(df["isFraud"].sum())
        neg_samples = total_samples - pos_samples
        scale_pos_weight = float(neg_samples) / max(1.0, float(pos_samples))

        self.imbalance_weights = {
            "negative_samples": neg_samples,
            "positive_samples": pos_samples,
            "scale_pos_weight": round(scale_pos_weight, 2),
            "fraud_prevalence_pct": round((pos_samples / total_samples) * 100, 3)
        }
        self.fitted = True

        logger.info(
            f"FR-2: Preprocessor fitted strictly on training partition ({total_samples:,} samples). "
            f"Feature count: {len(self.feature_names)}. "
            f"Class balance: {self.imbalance_weights['fraud_prevalence_pct']:.2f}% positive "
            f"(scale_pos_weight={scale_pos_weight:.2f})."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a transaction slice using strictly the parameters fitted on the training baseline.
        Zero future statistics leak into this transformation.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform().")

        df = self.extract_engineered_features(df)

        # Impute numerical features using baseline-fitted medians
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.imputation_values[col])
            else:
                df[col] = self.imputation_values[col]

        # Encode categorical features using baseline-fitted OneHotEncoder
        if self.categorical_cols and self.encoder is not None:
            slice_cats = df[self.categorical_cols].fillna("UNKNOWN").astype(str)
            encoded_cats = self.encoder.transform(slice_cats)
            cat_feature_names = list(self.encoder.get_feature_names_out(self.categorical_cols))
            encoded_df = pd.DataFrame(encoded_cats, columns=cat_feature_names, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)

        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Convenience method: fits strictly on provided data, then transforms it.
        """
        self.fit(df)
        return self.transform(df), self.imbalance_weights

    def partition_time_slices(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        FR-2.4: Orders dataset by transaction time and partitions into sequential,
        non-overlapping time slices (weekly or monthly).
        """
        # If discrete month column exists (e.g. Feedzai BAF: 8 months)
        if "month" in df.columns and df["month"].nunique() <= 12:
            slices = []
            for m_idx, (m_val, m_group) in enumerate(df.groupby("month", sort=True)):
                sort_candidates = [c for c in ["step", "trans_id", "TransactionID"] if c in m_group.columns]
                if sort_candidates:
                    slice_df = m_group.sort_values(by=sort_candidates).reset_index(drop=True)
                else:
                    slice_df = m_group.copy().reset_index(drop=True)
                fraud_count = int(slice_df["isFraud"].sum())
                total_count = len(slice_df)
                min_step = int(slice_df["step"].min()) if "step" in slice_df.columns else int(m_val * 720 + 1)
                max_step = int(slice_df["step"].max()) if "step" in slice_df.columns else int((m_val + 1) * 720)
                if min_step == max_step:
                    min_step = int(m_val * 720 + 1)
                    max_step = int((m_val + 1) * 720)
                slices.append({
                    "slice_index": m_idx,
                    "slice_name": f"Month_{m_idx + 1:02d}",
                    "start_step": min_step,
                    "end_step": max_step,
                    "total_transactions": total_count,
                    "fraud_transactions": fraud_count,
                    "fraud_ratio": round(fraud_count / float(total_count), 4),
                    "data": slice_df
                })
            logger.info(f"FR-2.4: Dataset partitioned by month into {len(slices)} non-overlapping sequential monthly slices.")
            return slices

        df = df.sort_values(by=["step", "trans_id"]).reset_index(drop=True)

        min_step = int(df["step"].min())
        max_step = int(df["step"].max())

        slices = []
        slice_idx = 0
        current_start = min_step

        while current_start <= max_step:
            current_end = current_start + self.hours_per_slice - 1
            mask = (df["step"] >= current_start) & (df["step"] <= current_end)
            slice_df = df[mask].copy().reset_index(drop=True)

            if len(slice_df) > 0:
                fraud_count = int(slice_df["isFraud"].sum())
                total_count = len(slice_df)
                slices.append({
                    "slice_index": slice_idx,
                    "slice_name": f"Week_{slice_idx + 1:02d}",
                    "start_step": current_start,
                    "end_step": current_end,
                    "total_transactions": total_count,
                    "fraud_transactions": fraud_count,
                    "fraud_ratio": round(fraud_count / float(total_count), 4),
                    "data": slice_df
                })
                slice_idx += 1

            current_start = current_end + 1

        logger.info(f"FR-2.4: Dataset partitioned into {len(slices)} non-overlapping sequential time slices.")
        return slices
