"""Shared estimator, chronological calibration and decision policy for every strategy."""

import logging
import time
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from src.config import USE_GPU, GPU_DEVICE, RANDOM_SEED

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    n_jobs=2,
)


def fit_xgb_with_fallback(X, y, xgb_params, model_name="Model", sample_weight=None):
    params = dict(xgb_params)
    requested = GPU_DEVICE if USE_GPU else "cpu"
    t0 = time.perf_counter()
    try:
        model = xgb.XGBClassifier(**params, tree_method="hist", device=requested)
        model.fit(X, y, sample_weight=sample_weight)
    except xgb.core.XGBoostError:
        if not USE_GPU:
            raise
        logger.warning("GPU training failed; retrying on CPU", exc_info=True)
        model = xgb.XGBClassifier(**params, tree_method="hist", device="cpu")
        model.fit(X, y, sample_weight=sample_weight)
    import json

    device = json.loads(model.get_booster().save_config())["learner"]["generic_param"][
        "device"
    ]
    return model, device, time.perf_counter() - t0


class BaseFraudModel:
    def __init__(
        self,
        feature_names,
        scale_pos_weight=1.0,
        random_seed=RANDOM_SEED,
        model_params=None,
        tune_threshold=True,
        calibrate=True,
        threshold_objective="validation_f1",
        operational_daily_budget=None,
    ):
        self.feature_names = list(feature_names)
        self.scale_pos_weight = scale_pos_weight
        self.random_seed = random_seed
        self.model_params = dict(DEFAULT_MODEL_PARAMS, **(model_params or {}))
        self.tune_threshold = tune_threshold
        self.calibrate = calibrate
        self.threshold_objective = threshold_objective
        self.operational_daily_budget = operational_daily_budget
        self.model = None
        self.calibrator = None
        self.threshold = 0.5
        self.trained = False
        self.device_used = "cpu"
        self.train_duration_sec = 0.0
        self.fit_metadata = {}

    def _fit_xgb(self, X, y, params, model_name=None):
        self.model, self.device_used, self.train_duration_sec = fit_xgb_with_fallback(
            X, y, params, model_name or type(self).__name__
        )
        self.trained = True
        return self.model

    @staticmethod
    def validation_start(n):
        return max(1, int(n * 0.8))

    def fit(self, df, sample_weight=None):
        if len(df) < 10 or df.isFraud.nunique() < 2:
            raise ValueError(
                "Training requires at least 10 labelled rows and both classes"
            )
        cut = self.validation_start(len(df))
        fit_df, validation = df.iloc[:cut], df.iloc[cut:]
        if fit_df.isFraud.nunique() < 2:
            raise ValueError(
                "Chronological fitting partition must contain both classes"
            )
        start = time.perf_counter()
        y = fit_df.isFraud.to_numpy()
        params = dict(
            self.model_params,
            random_state=self.random_seed,
            scale_pos_weight=float((y == 0).sum()) / max(1, int(y.sum())),
        )
        weights = None if sample_weight is None else np.asarray(sample_weight)[:cut]
        self.model, self.device_used, _ = fit_xgb_with_fallback(
            fit_df[self.feature_names].to_numpy(), y, params, sample_weight=weights
        )
        self.calibrator = None
        self.threshold = 0.5
        valid_y = validation.isFraud.to_numpy()
        probs = self.model.predict_proba(validation[self.feature_names].to_numpy())[
            :, 1
        ]
        calibration_status = "disabled_or_insufficient_validation_classes"
        if self.calibrate and min(int(valid_y.sum()), int((valid_y == 0).sum())) >= 5:
            self.calibrator = LogisticRegression(C=1.0, random_state=self.random_seed)
            self.calibrator.fit(self._logit(probs), valid_y)
            probs = self.calibrator.predict_proba(self._logit(probs))[:, 1]
            calibration_status = "sigmoid_on_chronological_validation"
        if self.tune_threshold and len(np.unique(valid_y)) == 2:
            if self.threshold_objective == "review_budget_recall" and self.operational_daily_budget:
                valid_hours = len(validation)
                if "step" in validation:
                    valid_hours = max(1.0, float(validation["step"].max() - validation["step"].min() + 1))
                days = max(1.0, valid_hours / 24.0)
                max_alerts = int(self.operational_daily_budget * days)
                if max_alerts < len(probs):
                    sorted_probs = np.sort(probs)
                    cand = float(sorted_probs[-max_alerts])
                    # Guard against score ties causing alert inflation above target budget
                    if np.sum(probs >= cand) > max_alerts:
                        higher_distinct = sorted_probs[sorted_probs > cand]
                        if len(higher_distinct) > 0:
                            self.threshold = float(higher_distinct[0])
                        else:
                            self.threshold = cand
                    else:
                        self.threshold = cand
                else:
                    self.threshold = float(np.min(probs))
            else:
                precision, recall, thresholds = precision_recall_curve(valid_y, probs)
                f1 = (
                    2
                    * precision[:-1]
                    * recall[:-1]
                    / np.maximum(precision[:-1] + recall[:-1], 1e-15)
                )
                # Prefer the highest threshold when validation F1 ties.
                best = np.flatnonzero(np.isclose(f1, f1.max()))[-1]
                self.threshold = float(thresholds[best])
        self.train_duration_sec = time.perf_counter() - start
        self.trained = True
        self.fit_metadata = {
            "fit_rows": len(fit_df),
            "validation_rows": len(validation),
            "calibration": calibration_status,
            "threshold": self.threshold,
            "threshold_objective": self.threshold_objective
            if self.tune_threshold
            else "fixed_0.5",
            "operational_daily_budget": self.operational_daily_budget,
        }
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, df):
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        probs = self.model.predict_proba(df[self.feature_names].to_numpy())[:, 1]
        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(self._logit(probs))[:, 1]
        return (probs >= self.threshold).astype(int), probs
