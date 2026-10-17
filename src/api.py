"""Serve an explicitly loaded model package with durable prediction/label pairing.
Run a single serving worker; external coordination is required for multiple workers.
"""

import copy
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict, field_validator
from sklearn.metrics import f1_score
from src.artifacts import load_bundle, save_bundle
from src.config import API_KEY, RESULTS_DIR, settings
from src.drift_detector import DriftMonitor

app = FastAPI(title="Fraud scoring", version="0.5.0")
header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key=Security(header)):
    if not api_key or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            401, "Invalid or missing API key. Provide a valid 'X-API-Key' header."
        )
    return api_key


class TransactionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("features")
    @classmethod
    def flat_finite_features(cls, values):
        for key, value in values.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"Feature {key} must be a scalar or null")
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"Feature {key} must be finite or null")
        return values

    trans_id: str = Field(min_length=1, max_length=200)
    features: dict[str, Any]
    timestamp: float | None = None


class BatchTransactionPayload(BaseModel):
    transactions: list[TransactionPayload] = Field(max_length=10000)


class FeedbackPayload(BaseModel):
    trans_id: str
    is_fraud: int = Field(ge=0, le=1)
    dispute_reason: str | None = None


class ModelDoubleBuffer:
    def __init__(self):
        self._snapshot = (None, 0)
        self._lock = threading.Lock()

    def snapshot(self):
        return self._snapshot

    def atomic_swap(self, bundle, generation=None):
        with self._lock:
            self._snapshot = (
                bundle,
                self._snapshot[1] + 1 if generation is None else generation,
            )

    @property
    def is_ready(self):
        bundle, _ = self._snapshot
        return bool(bundle and bundle.ready)

    @property
    def generation(self):
        return self._snapshot[1]


class PredictionStore:
    """SQLite ledger persists original decisions and idempotent feedback across restarts."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, features TEXT NOT NULL, probability REAL NOT NULL, prediction INTEGER NOT NULL, generation INTEGER NOT NULL, model_id TEXT NOT NULL, family_id TEXT NOT NULL, event_time REAL NOT NULL, label INTEGER)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_family_label_time ON predictions (family_id, label, event_time DESC)"
            )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, records):
        with self.connect() as c:
            c.executemany(
                "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,NULL)", records
            )

    def feedback(self, tx_id, label):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT prediction,label,generation,model_id FROM predictions WHERE id=?",
                (tx_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(
                    404, "Transaction not found in the prediction ledger"
                )
            if row[1] is not None:
                if row[1] != label:
                    raise HTTPException(
                        409, "Conflicting label for an already verified transaction"
                    )
                return row, False
            c.execute("UPDATE predictions SET label=? WHERE id=?", (label, tx_id))
            return row, True

    def training_data(self, family_id, limit=10000):
        with self.connect() as c:
            rows = c.execute(
                "SELECT features,label,event_time FROM predictions WHERE label IS NOT NULL AND family_id=? ORDER BY event_time DESC,rowid DESC LIMIT ?",
                (family_id, limit),
            ).fetchall()
        rows.reverse()
        return rows


model_buffer = ModelDoubleBuffer()
drift_monitor = DriftMonitor()
ledger = PredictionStore(
    settings.prediction_db or str(RESULTS_DIR / "service" / "predictions.sqlite")
)
feedback_lock = threading.Lock()


def initialize_app_model():
    path = settings.model_bundle_path
    if not path:
        return
    bundle = load_bundle(path)
    active = (
        Path(settings.model_output_dir or str(RESULTS_DIR / "service" / "models"))
        / "active.json"
    )
    if active.exists():
        pointer = json.loads(active.read_text())
        family = (
            bundle.metadata["run_id"] + ":" + bundle.metadata.get("model", "default")
        )
        if pointer.get("family_id") == family:
            candidate = (active.parent / pointer["package"]).resolve()
            if not candidate.is_relative_to(active.parent.resolve()):
                raise ValueError("Invalid active model path")
            restored = load_bundle(candidate)
            if (
                restored.metadata["run_id"]
                + ":"
                + restored.metadata.get("model", "default")
                != family
            ):
                raise ValueError(
                    "Persisted model belongs to a different training family"
                )
            model_buffer.atomic_swap(restored, generation=int(pointer["generation"]))
            return
    model_buffer.atomic_swap(bundle)


initialize_app_model()


class BackgroundRetrainer:
    def __init__(self, min_improvement_margin=0.005, max_retrain_duration_sec=60.0):
        self._lock = threading.Lock()
        self.total_retrains = 0
        self.last_retrain_status = "NOT_RUN"
        self.min_improvement_margin = min_improvement_margin
        self.max_retrain_duration_sec = max_retrain_duration_sec
        self.promotion_log = []

    @property
    def is_retraining(self):
        return self._lock.locked()

    def execute_retrain(self):
        if not self._lock.acquire(blocking=False):
            return False
        try:
            bundle, generation = model_buffer.snapshot()
            if bundle is None or not bundle.ready:
                self.last_retrain_status = "SKIPPED_NOT_READY"
                return False
            rows = ledger.training_data(
                bundle.metadata["run_id"]
                + ":"
                + bundle.metadata.get("model", "default")
            )
            if len(rows) < 100:
                self.last_retrain_status = "SKIPPED_INSUFFICIENT_DATA"
                return False
            frame = bundle.transform([json.loads(r[0]) for r in rows])
            frame["isFraud"] = [r[1] for r in rows]
            # Separate promotion holdout, untouched by fitting/calibration/threshold selection.
            cut = int(len(frame) * 0.8)
            training, holdout = frame.iloc[:cut], frame.iloc[cut:]
            if (
                len(holdout) < 20
                or training.isFraud.nunique() < 2
                or holdout.isFraud.nunique() < 2
            ):
                self.last_retrain_status = "SKIPPED_INSUFFICIENT_CLASSES"
                return False

            fit_start = time.perf_counter()
            candidate = copy.deepcopy(bundle)
            candidate.model.fit(training)
            fit_duration = time.perf_counter() - fit_start

            if fit_duration > self.max_retrain_duration_sec:
                self.last_retrain_status = "REJECTED_TRAINING_TIMEOUT"
                self.promotion_log.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "generation": generation,
                        "status": "REJECTED_TRAINING_TIMEOUT",
                        "duration_sec": fit_duration,
                    }
                )
                return False

            proposed, _ = candidate.model.predict(holdout)
            current, _ = bundle.model.predict(holdout)
            curr_f1 = f1_score(holdout.isFraud, current, zero_division=0)
            prop_f1 = f1_score(holdout.isFraud, proposed, zero_division=0)
            gain = prop_f1 - curr_f1

            if gain <= 0.0:
                self.last_retrain_status = "REJECTED_NO_HOLDOUT_GAIN"
                self.promotion_log.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "generation": generation,
                        "status": "REJECTED_NO_HOLDOUT_GAIN",
                        "current_f1": curr_f1,
                        "proposed_f1": prop_f1,
                        "gain": gain,
                        "duration_sec": fit_duration,
                    }
                )
                return False
            if gain < self.min_improvement_margin:
                self.last_retrain_status = "REJECTED_INSUFFICIENT_MARGIN"
                self.promotion_log.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "generation": generation,
                        "status": "REJECTED_INSUFFICIENT_MARGIN",
                        "current_f1": curr_f1,
                        "proposed_f1": prop_f1,
                        "gain": gain,
                        "duration_sec": fit_duration,
                    }
                )
                return False

            candidate.metadata = dict(
                candidate.metadata,
                parent_generation=generation,
                promotion="chronological holdout F1 improvement with verified margin",
                model_revision=uuid.uuid4().hex,
                retrained_at=datetime.now(timezone.utc).isoformat(),
                f1_gain=gain,
                retrain_fit_duration_sec=fit_duration,
            )
            root = Path(
                settings.model_output_dir or str(RESULTS_DIR / "service" / "models")
            )
            path = (
                root
                / f"generation-{generation + 1}-{candidate.metadata['model_revision']}"
            )
            save_bundle(candidate, path)
            with feedback_lock:
                pointer = {
                    "package": path.name,
                    "generation": generation + 1,
                    "family_id": candidate.metadata["run_id"]
                    + ":"
                    + candidate.metadata.get("model", "default"),
                }
                temporary = root / ("active-" + uuid.uuid4().hex + ".tmp")
                temporary.write_text(json.dumps(pointer), encoding="utf-8")
                os.replace(temporary, root / "active.json")
                model_buffer.atomic_swap(candidate)
                drift_monitor.reset_detectors()
            self.total_retrains += 1
            self.last_retrain_status = "SUCCESS"
            self.promotion_log.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "generation": generation + 1,
                    "status": "PROMOTED",
                    "current_f1": curr_f1,
                    "proposed_f1": prop_f1,
                    "gain": gain,
                    "duration_sec": fit_duration,
                }
            )
            return True
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
            self.last_retrain_status = "FAILED: " + str(exc)
            return False
        finally:
            self._lock.release()


retrainer = BackgroundRetrainer()


@app.get("/health")
def health_check():
    bundle, generation = model_buffer.snapshot()
    return dict(
        status="READY" if bundle and bundle.ready else "NOT_READY",
        model_loaded=bundle is not None,
        is_ready=bool(bundle and bundle.ready),
        model_generation=generation,
        data_kind=bundle.metadata.get("data_kind") if bundle else None,
        is_retraining=retrainer.is_retraining,
        last_retrain_status=retrainer.last_retrain_status,
    )


@app.get("/v1/schema", dependencies=[Depends(verify_api_key)])
def schema():
    bundle, _ = model_buffer.snapshot()
    if bundle is None:
        raise HTTPException(503, "No model package loaded")
    return {
        "required_features": bundle.preprocessor.required_input_columns,
        "missing_values": "Use null for observed missing values; do not omit required keys",
        "historical_features": "If required, supply causal historical features from your feature store",
        "threshold": bundle.model.threshold,
        "model": bundle.metadata,
    }


def score_many(transactions):
    bundle, generation = model_buffer.snapshot()
    if bundle is None or not bundle.ready:
        raise HTTPException(503, "No validated real-data model package is ready")
    start = time.perf_counter()
    try:
        pred, prob = bundle.predict([t.features for t in transactions])
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    inference_ms = (time.perf_counter() - start) * 1000
    rows = []
    results = []
    for i, tx in enumerate(transactions):
        now = tx.timestamp if tx.timestamp is not None else time.time()
        rows.append(
            (
                tx.trans_id,
                json.dumps(tx.features, allow_nan=False),
                float(prob[i]),
                int(pred[i]),
                generation,
                bundle.metadata.get("model_revision", bundle.metadata["run_id"]),
                bundle.metadata["run_id"]
                + ":"
                + bundle.metadata.get("model", "default"),
                now,
            )
        )
        results.append(
            dict(
                trans_id=tx.trans_id,
                fraud_probability=float(prob[i]),
                is_fraud=bool(pred[i]),
                decision_threshold=bundle.model.threshold,
                risk_tier="REVIEW" if pred[i] else "LOW",
                model_version="0.5.0",
                model_generation=generation,
                inference_latency_ms=inference_ms / len(transactions),
            )
        )
    try:
        ledger.record(rows)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            409, "Transaction ID already scored; original decision is preserved"
        ) from exc
    return results, (time.perf_counter() - start) * 1000


@app.post("/v1/score", dependencies=[Depends(verify_api_key)])
def score_transaction(payload: TransactionPayload):
    return score_many([payload])[0][0]


@app.post("/v1/batch-score", dependencies=[Depends(verify_api_key)])
def score_batch(payload: BatchTransactionPayload):
    if not payload.transactions:
        return dict(results=[], total_processed=0, total_latency_ms=0, throughput_qps=0)
    results, ms = score_many(payload.transactions)
    return dict(
        results=results,
        total_processed=len(results),
        total_latency_ms=ms,
        throughput_qps=len(results) * 1000 / max(ms, 1e-9),
    )


@app.post("/v1/feedback", dependencies=[Depends(verify_api_key)])
def feedback(payload: FeedbackPayload, background_tasks: BackgroundTasks):
    with feedback_lock:
        original, fresh = ledger.feedback(payload.trans_id, payload.is_fraud)
        bundle, generation = model_buffer.snapshot()
        # Old-model decisions stay in the ledger but must not reset the new model's detector baseline.
        monitored = (
            fresh
            and bundle is not None
            and original[2] == generation
            and original[3]
            == bundle.metadata.get("model_revision", bundle.metadata["run_id"])
        )
        alert = {"any_drift": False}
        if monitored:
            alert = drift_monitor.update(
                payload.is_fraud,
                original[0],
                0,
                drift_monitor.total_samples_monitored,
                payload.trans_id,
            )
    if alert["any_drift"]:
        background_tasks.add_task(retrainer.execute_retrain)
    return dict(
        status="RECORDED" if fresh else "ALREADY_RECORDED",
        drift_alert_triggered=alert["any_drift"],
        monitored=monitored,
        model_generation=generation,
    )


@app.get("/v1/drift-status", dependencies=[Depends(verify_api_key)])
def drift_status():
    return dict(
        drift_monitor.get_telemetry_snapshot(), model_generation=model_buffer.generation
    )


@app.post("/v1/retrain", dependencies=[Depends(verify_api_key)])
def trigger_model_retrain(background_tasks: BackgroundTasks, sync: bool = False):
    if not model_buffer.is_ready:
        raise HTTPException(503, "Load a real-data model package first")
    if sync:
        success = retrainer.execute_retrain()
        return dict(status="COMPLETED" if success else retrainer.last_retrain_status)
    background_tasks.add_task(retrainer.execute_retrain)
    return dict(status="DISPATCHED")
