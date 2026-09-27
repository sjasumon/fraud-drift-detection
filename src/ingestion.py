"""
FR-1: Data Ingestion and Validation Module.
Supports both IEEE-CIS Fraud Detection and PaySim datasets.
Validates schemas, temporal dimensions, row counts, and class balance.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import DEFAULT_HOURS_PER_WEEK

logger = logging.getLogger("fraud_drift.ingestion")


class IngestionError(Exception):
    """Raised when dataset fails FR-1 validation checks."""
    pass


class DatasetIngestion:
    """Ingests and validates transaction streams according to FR-1 requirements."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.validation_report: Dict[str, Any] = {}
        self.dataset_type = "unknown"

    def load_and_validate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads CSV and performs validation against FR-1.1, FR-1.2, FR-1.3.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")

        logger.info(f"Loading transaction dataset from {self.file_path}...")

        # Detect dataset schema by reading header
        sample_df = pd.read_csv(self.file_path, nrows=5)

        if "TransactionID" in sample_df.columns and "TransactionDT" in sample_df.columns:
            self.dataset_type = "IEEE-CIS"
            df = self._load_ieee_cis()
        elif "fraud_bool" in sample_df.columns and "month" in sample_df.columns:
            self.dataset_type = "Feedzai-BAF"
            df = self._load_feedzai_baf()
        elif "step" in sample_df.columns and "amount" in sample_df.columns:
            self.dataset_type = "PaySim"
            df = self._load_paysim()
        else:
            raise IngestionError(
                f"[FR-1.1 Violation] Unrecognized dataset schema in {self.file_path}. "
                f"Missing required columns (either IEEE-CIS: TransactionID, TransactionDT, TransactionAmt, isFraud; "
                f"Feedzai-BAF: fraud_bool, month; or PaySim: step, amount, isFraud)."
            )

        # Validate label column is binary (0 or 1)
        unique_labels = set(df["isFraud"].dropna().unique())
        if not unique_labels.issubset({0, 1}):
            raise IngestionError(
                f"[FR-1.1 Violation] Fraud label 'isFraud' must be binary (0 or 1). Found: {unique_labels}"
            )

        # FR-1.2 Time Dimension Check
        min_step = int(df["step"].min())
        max_step = int(df["step"].max())
        total_hours = max_step - min_step + 1
        total_days = total_hours / 24.0
        total_weeks = total_hours / float(DEFAULT_HOURS_PER_WEEK)

        # FR-1.2: Spanning only 1-2 days is NOT acceptable
        if total_days < 14.0:
            raise IngestionError(
                f"[FR-1.2 Violation] Dataset time span is insufficient: spans only {total_days:.1f} days "
                f"({total_hours} hours). FR-1.2 explicitly requires multiple weeks or months to demonstrate drift."
            )

        # FR-1.3 Validation Metrics Report
        total_rows = len(df)
        fraud_count = int(df["isFraud"].sum())
        legit_count = total_rows - fraud_count
        fraud_ratio = fraud_count / float(total_rows)

        self.validation_report = {
            "dataset_type": self.dataset_type,
            "file_name": self.file_path.name,
            "total_rows": total_rows,
            "columns_count": len(df.columns),
            "fraud_count": fraud_count,
            "legit_count": legit_count,
            "fraud_ratio_pct": round(fraud_ratio * 100, 3),
            "step_min": min_step,
            "step_max": max_step,
            "total_hours": total_hours,
            "total_days": round(total_days, 1),
            "total_weeks": round(total_weeks, 1),
            "validation_status": "PASSED"
        }

        logger.info(
            f"Validation PASSED for {self.dataset_type}: {total_rows:,} records, "
            f"{fraud_ratio*100:.2f}% fraud, spanning {total_weeks:.1f} weeks ({total_days:.1f} days)."
        )

        return df, self.validation_report

    def _load_ieee_cis(self) -> pd.DataFrame:
        """Loads and selects high-signal features from IEEE-CIS train_transaction.csv."""
        # Key high-signal columns from IEEE-CIS
        selected_cols = [
            "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
            "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "dist1", "P_emaildomain",
            "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
            "D1", "D2", "D3", "D4",
            "V12", "V13", "V29", "V30", "V53", "V54", "V75", "V76", "V90", "V91",
            "V107", "V127", "V130", "V283", "V307", "V310", "V312"
        ]

        # Read only required columns for high efficiency
        available_cols = set(pd.read_csv(self.file_path, nrows=1).columns)
        load_cols = [c for c in selected_cols if c in available_cols]

        logger.info(f"Loading {len(load_cols)} high-signal features from IEEE-CIS...")
        df = pd.read_csv(self.file_path, usecols=load_cols)

        # Standardize transaction ID
        df["trans_id"] = df["TransactionID"].astype(str)

        # Compute hourly time step from TransactionDT (seconds)
        min_dt = df["TransactionDT"].min()
        df["step"] = ((df["TransactionDT"] - min_dt) // 3600).astype(int) + 1
        df["amount"] = df["TransactionAmt"]

        # Sort strictly by time step
        df = df.sort_values(by=["step", "trans_id"]).reset_index(drop=True)
        return df

    def _load_feedzai_baf(self) -> pd.DataFrame:
        """Loads and prepares Feedzai Bank Account Fraud (NeurIPS 2022) dataset."""
        logger.info(f"Loading Feedzai BAF dataset from {self.file_path}...")
        df = pd.read_csv(self.file_path)

        # Standardize target label
        df["isFraud"] = df["fraud_bool"].astype(int)

        # Standardize unique transaction tracking identifier
        df["trans_id"] = "BAF_" + pd.RangeIndex(len(df)).astype(str)

        # Step: map month (0-7) to hourly simulation steps (each month = 720 hours = 30 days)
        df["step"] = (df["month"] * 720 + 1).astype(int)

        # Monetary amount proxy (proposed credit limit)
        if "proposed_credit_limit" in df.columns:
            df["amount"] = df["proposed_credit_limit"]
        else:
            df["amount"] = 1000.0

        # Sort strictly by month and trans_id
        df = df.sort_values(by=["month", "trans_id"]).reset_index(drop=True)
        return df

    def _load_paysim(self) -> pd.DataFrame:
        """Loads PaySim mobile money dataset."""
        logger.info(f"Loading PaySim dataset from {self.file_path}...")
        df = pd.read_csv(self.file_path)
        if "trans_id" not in df.columns:
            df["trans_id"] = "PS_" + pd.RangeIndex(len(df)).astype(str)

        # Engineer accounting balance delta features
        if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
            df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
        if "oldbalanceDest" in df.columns and "newbalanceDest" in df.columns:
            df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

        df = df.sort_values(by=["step", "trans_id"]).reset_index(drop=True)
        return df
