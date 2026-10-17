"""Scheduled retraining with the shared estimator and validation policy."""

from src.stream_model import StreamModel


class PeriodicRetrainModel(StreamModel):
    def update_with_slice(self, slice_df, slice_idx, is_out_of_sample=True):
        self._append(slice_df, slice_idx)
        if is_out_of_sample:
            self.out_of_sample_slices_seen += 1
            if self.out_of_sample_slices_seen % self.retrain_interval_slices == 0:
                return self._retrain(slice_idx)
        return False
