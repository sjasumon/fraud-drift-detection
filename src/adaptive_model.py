"""Drift-triggered retraining, optionally using recent or age-weighted history."""

from src.stream_model import StreamModel


class DriftAdaptiveModel(StreamModel):
    def update_with_slice(
        self,
        slice_df,
        drift_triggered,
        slice_idx,
        training_df_override=None,
        is_out_of_sample=True,
    ):
        if training_df_override is None:
            self._append(slice_df, slice_idx)
        if is_out_of_sample and drift_triggered:
            return self._retrain(slice_idx, training_df_override)
        return False
