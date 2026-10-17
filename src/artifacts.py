"""Versioned, integrity-checked model packages. Load only packages you trust (joblib)."""

import copy
import hashlib
import json
import os
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    model: object
    preprocessor: object
    metadata: dict

    def transform(self, rows):
        frame = pd.DataFrame(rows)
        required = self.preprocessor.required_input_columns
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError("Missing required features: " + ", ".join(missing))
        for row in rows:
            if set(required) - set(row):
                raise ValueError(
                    "Each transaction must contain the full input schema; use null for observed missing values"
                )
        out = self.preprocessor.transform(frame)
        if not np.isfinite(out[self.model.feature_names].to_numpy(dtype=float)).all():
            raise ValueError("Features contain non-finite numeric values")
        return out

    def predict(self, rows):
        return self.model.predict(self.transform(rows))

    @property
    def ready(self):
        return bool(self.model.trained and self.metadata.get("data_kind") == "real")


def save_bundle(bundle, directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Model package already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-package-", dir=directory.parent))
    try:
        serving = copy.copy(bundle)
        serving.model = copy.copy(bundle.model)
        # Training buffers belong to experiments/verified storage, never serving artifacts.
        if hasattr(serving.model, "cumulative_history"):
            serving.model.cumulative_history = []
        serving.model.retrain_log = []
        joblib.dump(serving, staging / "bundle.joblib")
        digest = hashlib.sha256((staging / "bundle.joblib").read_bytes()).hexdigest()
        manifest = dict(
            serving.metadata,
            format_version=2,
            sha256=digest,
            input_features=serving.preprocessor.required_input_columns,
            transformed_features=serving.model.feature_names,
            threshold=serving.model.threshold,
            fit_metadata=serving.model.fit_metadata,
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(staging, directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return directory


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    data = directory / "bundle.joblib"
    if hashlib.sha256(data.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Model package checksum mismatch")
    bundle = joblib.load(data)
    if not isinstance(bundle, ModelBundle):
        raise ValueError("Unsupported model package")
    if list(bundle.model.feature_names) != list(bundle.preprocessor.feature_names):
        raise ValueError("Model and preprocessing schemas do not match")
    if not np.isfinite(bundle.model.threshold) or not 0 <= bundle.model.threshold <= 1:
        raise ValueError("Invalid model decision threshold")
    return bundle
