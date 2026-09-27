"""
Delayed Feedback & Verification Latency Simulation Module.

In real-world financial fraud streaming environments, ground-truth fraud labels
do not arrive instantaneously. Instead, they arrive with substantial latency
(typically 30 to 90 days) via cardholder disputes, chargebacks, and investigative resolution.

This module simulates realistic label latency dynamics:
1. Fixed-latency queues (e.g. 4-week / 8-week delay).
2. Stochastic LogNormal latency queues modeling empirical chargeback curves.
3. Dual-buffer management: unverified pending queue vs. matured ground-truth buffer.
4. Strict enforcement of zero-leakage training sets (retraining strictly on matured labels).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("fraud_drift.delayed_feedback")


class DelayedFeedbackQueue:
    """
    Manages an asynchronous label maturity queue for streaming transactions.
    
    Transactions enter the queue upon inference at slice `t`.
    Their ground-truth labels remain sealed/unverified until slice `t + delay`.
    Only matured transactions are released for model retraining or supervised drift detection.
    """

    def __init__(
        self,
        delay_slices: int = 4,
        latency_mode: str = "fixed",
        lognormal_mu: float = 1.38,  # ~4 weeks median if exp(1.38) ~= 3.97
        lognormal_sigma: float = 0.45,
        random_seed: int = 42
    ):
        """
        Args:
            delay_slices: Fixed number of slices to delay labels (if mode='fixed').
            latency_mode: 'fixed' or 'lognormal'.
            lognormal_mu: Location parameter for LogNormal distribution in slices.
            lognormal_sigma: Scale parameter for LogNormal distribution.
            random_seed: Random seed for stochastic latency sampling.
        """
        if delay_slices < 0:
            raise ValueError(f"delay_slices must be >= 0, got {delay_slices}")
        if latency_mode not in ("fixed", "lognormal"):
            raise ValueError(f"latency_mode must be 'fixed' or 'lognormal', got {latency_mode}")

        self.delay_slices = delay_slices
        self.latency_mode = latency_mode
        self.lognormal_mu = lognormal_mu
        self.lognormal_sigma = lognormal_sigma
        self.rng = np.random.default_rng(random_seed)

        # Internal storage: List of dicts or records containing transaction metadata and maturity slice
        self._queue: List[Dict[str, Any]] = []
        self._matured_history: List[pd.DataFrame] = []
        self._total_enqueued = 0
        self._total_matured = 0

    def enqueue(self, slice_df: pd.DataFrame, current_slice: int) -> np.ndarray:
        """
        Enqueues incoming transactions from current_slice, assigning each a maturity slice.
        
        Args:
            slice_df: DataFrame of transactions arriving in current_slice.
            current_slice: Integer index of the current time slice (e.g. week index).
            
        Returns:
            np.ndarray of assigned maturity slice indices.
        """
        n = len(slice_df)
        if n == 0:
            return np.array([], dtype=int)

        if self.latency_mode == "fixed":
            maturity_slices = np.full(n, current_slice + self.delay_slices, dtype=int)
        elif self.latency_mode == "lognormal":
            # Sample continuous delays and round to discrete slice offsets (min 1 slice if delay > 0)
            sampled_delays = self.rng.lognormal(mean=self.lognormal_mu, sigma=self.lognormal_sigma, size=n)
            discrete_delays = np.maximum(1, np.round(sampled_delays)).astype(int)
            maturity_slices = current_slice + discrete_delays
        else:
            maturity_slices = np.full(n, current_slice + self.delay_slices, dtype=int)

        tagged_df = slice_df.copy(deep=False)
        tagged_df["_maturity_slice"] = maturity_slices
        tagged_df["_arrival_slice"] = current_slice

        self._queue.append({
            "arrival_slice": current_slice,
            "data": tagged_df
        })
        self._total_enqueued += n
        return maturity_slices

    def dequeue_matured(self, current_slice: int) -> pd.DataFrame:
        """
        Extracts and returns all transactions whose labels have matured at or before current_slice.
        These transactions are moved from the pending queue into the matured buffer.
        
        Args:
            current_slice: Current time slice index.
            
        Returns:
            DataFrame of transactions whose ground-truth labels just matured.
        """
        matured_chunks = []
        remaining_queue = []

        for entry in self._queue:
            df = entry["data"]
            # Check if any rows in this chunk mature at or before current_slice
            matured_mask = df["_maturity_slice"] <= current_slice
            unmatured_mask = df["_maturity_slice"] > current_slice

            if matured_mask.any():
                matured_rows = df[matured_mask].copy()
                matured_chunks.append(matured_rows)

            if unmatured_mask.any():
                remaining_df = df[unmatured_mask].copy()
                remaining_queue.append({
                    "arrival_slice": entry["arrival_slice"],
                    "data": remaining_df
                })

        self._queue = remaining_queue

        if matured_chunks:
            result_df = pd.concat(matured_chunks, ignore_index=True)
            self._matured_history.append(result_df)
            self._total_matured += len(result_df)
            return result_df
        else:
            return pd.DataFrame()

    def get_all_matured_training_data(self, initial_train_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Returns the complete accumulated training dataset composed strictly of
        initial seed transactions plus all transactions whose labels have matured to date.
        
        Guarantees ZERO lookahead bias: transactions whose chargeback window is still open
        are strictly excluded from training.
        """
        chunks = []
        if initial_train_df is not None and len(initial_train_df) > 0:
            chunks.append(initial_train_df)

        if self._matured_history:
            chunks.extend(self._matured_history)

        if chunks:
            full_df = pd.concat(chunks, ignore_index=True)
            # Clean temporary tracking columns if present
            cols_to_drop = [c for c in ["_maturity_slice", "_arrival_slice"] if c in full_df.columns]
            if cols_to_drop:
                return full_df.drop(columns=cols_to_drop)
            return full_df
        return pd.DataFrame()

    def get_pending_count(self) -> int:
        """Returns the number of transactions currently pending maturity."""
        return sum(len(entry["data"]) for entry in self._queue)

    def get_matured_count(self) -> int:
        """Returns the total number of transactions that have matured so far."""
        return self._total_matured

    def get_status_summary(self, current_slice: int) -> Dict[str, Union[int, float]]:
        """Returns summary statistics of the queue status at current_slice."""
        pending = self.get_pending_count()
        return {
            "current_slice": current_slice,
            "delay_slices": self.delay_slices,
            "latency_mode": self.latency_mode,
            "total_enqueued": self._total_enqueued,
            "total_matured": self._total_matured,
            "total_pending": pending,
            "maturity_rate": (self._total_matured / self._total_enqueued) if self._total_enqueued > 0 else 0.0
        }
