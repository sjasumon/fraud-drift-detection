"""History management and retraining shared by periodic, sliding and adaptive models."""

import numpy as np
import pandas as pd
from src.base_model import BaseFraudModel


class StreamModel(BaseFraudModel):
    def __init__(
        self,
        *args,
        window_slices=None,
        retrain_interval_slices=4,
        half_life_slices=None,
        max_retrains=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.window_slices = window_slices
        self.retrain_interval_slices = retrain_interval_slices
        self.half_life_slices = half_life_slices
        self.max_retrains = max_retrains
        self.cumulative_history = []
        self.retrain_log = []
        self.retrain_count = 0
        self.out_of_sample_slices_seen = 0
        self.last_training_id_digest = None

    @property
    def adaptation_count(self):
        return self.retrain_count

    @property
    def window_history(self):
        return self.cumulative_history

    def fit_initial(self, df):
        self.fit(df)
        # Seed once; replayed training rows are ignored by update_with_slice.
        self.cumulative_history = [df.assign(_history_slice=-1).copy()]
        import hashlib

        if "trans_id" in df:
            self.last_training_id_digest = hashlib.sha256(
                "\n".join(sorted(df.trans_id.astype(str))).encode()
            ).hexdigest()

    def _append(self, df, slice_idx):
        old = (
            pd.concat(self.cumulative_history, ignore_index=True)
            if self.cumulative_history
            else pd.DataFrame()
        )
        tagged = df.assign(_history_slice=slice_idx)
        if len(old) and "trans_id" in df:
            tagged = tagged[~tagged.trans_id.isin(old.trans_id)]
        if len(tagged):
            self.cumulative_history.append(tagged.copy())
        if self.window_slices is not None:
            self.cumulative_history = [
                d
                for d in self.cumulative_history
                if int(d._history_slice.max()) > slice_idx - self.window_slices
            ]

    def _retrain(self, slice_idx, training_df_override=None):
        if self.max_retrains is not None and self.retrain_count >= self.max_retrains:
            return False
        df = (
            training_df_override
            if training_df_override is not None
            else pd.concat(self.cumulative_history, ignore_index=True)
        )
        if "trans_id" in df:
            df = df.drop_duplicates("trans_id")
        order = [
            c
            for c in ["_history_slice", "TransactionDT", "step", "trans_id"]
            if c in df
        ]
        if order:
            df = df.sort_values(order, kind="stable")
        anchor = (
            int(df._history_slice.max())
            if "_history_slice" in df and len(df)
            else slice_idx
        )
        if self.window_slices is not None and "_history_slice" in df:
            # Recent memory is relative to the latest VERIFIED arrival, not the wall-clock
            # slice; a window shorter than verification lag must not become permanently empty.
            df = df[df._history_slice > anchor - self.window_slices]
        if (
            len(df) < 10
            or df.iloc[: self.validation_start(len(df))].isFraud.nunique() < 2
        ):
            return False
        import hashlib

        ids = (
            df.trans_id.astype(str).to_numpy()
            if "trans_id" in df
            else df.index.astype(str).to_numpy()
        )
        digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
        if digest == self.last_training_id_digest:
            return False
        weights = None
        if self.half_life_slices is not None and "_history_slice" in df:
            weights = np.exp2(
                -(anchor - df._history_slice.to_numpy()) / self.half_life_slices
            )
        self.fit(df, sample_weight=weights)
        self.last_training_id_digest = digest
        self.retrain_count += 1
        self.retrain_log.append(
            {
                "adaptation_num": self.retrain_count,
                "slice_idx": slice_idx,
                "accumulated_samples": len(df),
                "fraud_count": int(df.isFraud.sum()),
                "duration_seconds": self.train_duration_sec,
                "device": self.device_used,
                "latest_training_arrival_slice": anchor,
                "verification_staleness_slices": slice_idx - anchor,
                "training_id_digest": digest,
                "threshold": self.threshold,
            }
        )
        return True
