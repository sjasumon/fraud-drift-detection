"""Serve an explicitly loaded model package with durable prediction/label pairing.
Run a single serving worker; external coordination is required for multiple workers.
"""

import concurrent.futures
import copy
import json
import os
import secrets
import logging
import sqlite3
import threading
import time
import uuid
import math

logger = logging.getLogger("fraud_drift.api")
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict, field_validator
from sklearn.metrics import f1_score
from src.artifacts import load_bundle, save_bundle
from src.config import API_KEY, RESULTS_DIR, settings
from src.drift_detector import DriftMonitor

app = FastAPI(title="Fraud scoring", version="0.6.0")
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
    confirmation_timestamp: float | None = None
    label_source: str | None = "analyst_review"


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
    """SQLite ledger persists original decisions, audit trails, and idempotent feedback across restarts."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                "id TEXT PRIMARY KEY, "
                "features TEXT NOT NULL, "
                "probability REAL NOT NULL, "
                "prediction INTEGER NOT NULL, "
                "generation INTEGER NOT NULL, "
                "model_id TEXT NOT NULL, "
                "family_id TEXT NOT NULL, "
                "event_time REAL NOT NULL, "
                "label INTEGER, "
                "received_at TEXT, "
                "confirmed_at REAL, "
                "label_source TEXT"
                ")"
            )
            # Schema migration for existing databases
            for col, col_type in [("received_at", "TEXT"), ("confirmed_at", "REAL"), ("label_source", "TEXT")]:
                try:
                    c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass

            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_family_label_time ON predictions (family_id, label, event_time DESC)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_family_event_time ON predictions (family_id, event_time DESC)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS feedback_events ("
                "event_id TEXT PRIMARY KEY, "
                "trans_id TEXT NOT NULL, "
                "label INTEGER NOT NULL, "
                "label_source TEXT, "
                "event_time REAL, "
                "received_at TEXT NOT NULL, "
                "confirmed_at REAL, "
                "original_generation INTEGER, "
                "original_model_id TEXT, "
                "outcome TEXT NOT NULL"
                ")"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_events_trans_id ON feedback_events (trans_id)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_events_time ON feedback_events (received_at DESC)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS promotion_history ("
                "attempt_id TEXT PRIMARY KEY, "
                "timestamp TEXT NOT NULL, "
                "parent_generation INTEGER NOT NULL, "
                "candidate_generation INTEGER, "
                "parent_revision TEXT, "
                "candidate_revision TEXT, "
                "evidence_count INTEGER NOT NULL, "
                "current_f1 REAL, "
                "proposed_f1 REAL, "
                "gain REAL, "
                "margin_required REAL NOT NULL, "
                "duration_sec REAL NOT NULL, "
                "status TEXT NOT NULL, "
                "decision TEXT NOT NULL, "
                "reason TEXT NOT NULL"
                ")"
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
                "INSERT INTO predictions (id, features, probability, prediction, generation, model_id, family_id, event_time, label) "
                "VALUES (?,?,?,?,?,?,?,?,NULL)",
                records,
            )

    def feedback(self, tx_id, label, label_source="analyst_review", confirmation_timestamp=None):
        receipt_time = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            c = conn.cursor()
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT prediction, label, generation, model_id, event_time FROM predictions WHERE id=?",
                (tx_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise HTTPException(
                    404, "Transaction not found in the prediction ledger"
                )

            current_label = row[1]
            original_generation = row[2]
            original_model_id = row[3]
            event_time = row[4]

            if current_label is not None:
                if current_label != label:
                    # Conflict: persist audit event and commit before raising 409
                    c.execute(
                        "INSERT INTO feedback_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            "FBEV_" + uuid.uuid4().hex[:12].upper(),
                            tx_id,
                            label,
                            label_source,
                            event_time,
                            receipt_time,
                            confirmation_timestamp,
                            original_generation,
                            original_model_id,
                            "CONFLICT_REJECTED",
                        ),
                    )
                    conn.commit()
                    raise HTTPException(
                        409, "Conflicting label for an already verified transaction"
                    )
                # Idempotent replay: record audit event and commit
                c.execute(
                    "INSERT INTO feedback_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        "FBEV_" + uuid.uuid4().hex[:12].upper(),
                        tx_id,
                        label,
                        label_source,
                        event_time,
                        receipt_time,
                        confirmation_timestamp,
                        original_generation,
                        original_model_id,
                        "IDEMPOTENT_REPLAY",
                    ),
                )
                conn.commit()
                return row[:4], False

            # Fresh label: update predictions and insert feedback event, then commit
            c.execute(
                "UPDATE predictions SET label=?, received_at=?, confirmed_at=?, label_source=? WHERE id=?",
                (label, receipt_time, confirmation_timestamp, label_source, tx_id),
            )
            c.execute(
                "INSERT INTO feedback_events VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    "FBEV_" + uuid.uuid4().hex[:12].upper(),
                    tx_id,
                    label,
                    label_source,
                    event_time,
                    receipt_time,
                    confirmation_timestamp,
                    original_generation,
                    original_model_id,
                    "RECORDED_FRESH",
                ),
            )
            conn.commit()
            return row[:4], True
        except HTTPException:
            raise
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def training_data(self, family_id, limit=10000):
        with self.connect() as c:
            rows = c.execute(
                "SELECT features,label,event_time FROM predictions WHERE label IS NOT NULL AND family_id=? ORDER BY event_time DESC,rowid DESC LIMIT ?",
                (family_id, limit),
            ).fetchall()
        rows.reverse()
        return rows

    def record_promotion(self, record):
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO promotion_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record["attempt_id"],
                    record["timestamp"],
                    record["parent_generation"],
                    record.get("candidate_generation"),
                    record.get("parent_revision"),
                    record.get("candidate_revision"),
                    record["evidence_count"],
                    record.get("current_f1"),
                    record.get("proposed_f1"),
                    record.get("gain"),
                    record["margin_required"],
                    record["duration_sec"],
                    record["status"],
                    record["decision"],
                    record["reason"],
                ),
            )

    def get_promotion_history(self, limit=100):
        with self.connect() as c:
            cursor = c.execute(
                "SELECT attempt_id, timestamp, parent_generation, candidate_generation, "
                "parent_revision, candidate_revision, evidence_count, current_f1, proposed_f1, "
                "gain, margin_required, duration_sec, status, decision, reason "
                "FROM promotion_history ORDER BY rowid DESC LIMIT ?",
                (limit,),
            )
            cols = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(cols, r)) for r in rows]

    def get_feedback_events(self, limit=100):
        with self.connect() as c:
            cursor = c.execute(
                "SELECT event_id, trans_id, label, label_source, event_time, received_at, confirmed_at, original_generation, original_model_id, outcome "
                "FROM feedback_events ORDER BY rowid DESC LIMIT ?",
                (limit,),
            )
            cols = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(cols, r)) for r in rows]


model_buffer = ModelDoubleBuffer()
drift_monitor = DriftMonitor()
ledger = PredictionStore(
    settings.prediction_db or str(RESULTS_DIR / "service" / "predictions.sqlite")
)
feedback_lock = threading.Lock()


def initialize_app_model():
    path = settings.model_bundle_path
    if not path:
        logger.info(
            "No MODEL_BUNDLE_PATH configured. API starting in UNREADY state; model scoring unavailable."
        )
        return
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(
            f"Configured MODEL_BUNDLE_PATH does not exist: {path}. "
            "API starting in UNREADY state; model scoring unavailable."
        )
        return
    try:
        bundle = load_bundle(path_obj)
    except Exception as exc:
        logger.error(
            f"Failed to load model bundle at {path}: {exc}. "
            "API starting in UNREADY state; model scoring unavailable."
        )
        return

    models_dir = Path(settings.model_output_dir or str(RESULTS_DIR / "service" / "models"))
    active = models_dir / "active.json"
    if not active.exists():
        alt_active = Path(RESULTS_DIR / "service" / "active_model.json")
        if alt_active.exists():
            active = alt_active

    if active.exists():
        try:
            pointer = json.loads(active.read_text(encoding="utf-8"))
            family = (
                bundle.metadata["run_id"] + ":" + bundle.metadata.get("model", "default")
            )
            if pointer.get("family_id") == family:
                pkg = Path(pointer["package"])
                if pkg.is_absolute():
                    candidate = pkg
                elif (models_dir / pkg).exists():
                    candidate = models_dir / pkg
                else:
                    candidate = active.parent / pkg
                candidate = candidate.resolve()
                restored = load_bundle(candidate)
                if not restored.ready:
                    raise ValueError(
                        "Persisted model bundle fails genuine observed provenance check"
                    )
                if (
                    restored.metadata["run_id"]
                    + ":"
                    + restored.metadata.get("model", "default")
                    != family
                ):
                    raise ValueError(
                        "Persisted model belongs to a different training family"
                    )
                model_buffer.atomic_swap(restored, generation=int(pointer.get("generation", 0)))
                logger.info(
                    f"Successfully loaded active model {pointer.get('package')} at generation {pointer.get('generation')}"
                )
                return
        except Exception as exc:
            logger.warning(
                f"Failed to load active model pointer from {active}: {exc}. Using base bundle."
            )
    model_buffer.atomic_swap(bundle, generation=0)
    logger.info(f"Loaded base model bundle from {path} (ready={bundle.ready})")


initialize_app_model()


class BackgroundRetrainer:
    def __init__(self, store=None, min_improvement_margin=0.005, max_retrain_duration_sec=60.0):
        self._lock = threading.Lock()
        self.store = store
        self.total_retrains = 0
        self.last_retrain_status = "NOT_RUN"
        self.min_improvement_margin = min_improvement_margin
        self.max_retrain_duration_sec = max_retrain_duration_sec
        self.promotion_log: List[Dict[str, Any]] = (
            self.store.get_promotion_history(100) if self.store else []
        )

    @property
    def is_retraining(self):
        return self._lock.locked()

    def _record_attempt(
        self,
        parent_generation: int,
        status: str,
        decision: str,
        reason: str,
        duration_sec: float = 0.0,
        evidence_count: int = 0,
        current_f1: Optional[float] = None,
        proposed_f1: Optional[float] = None,
        gain: Optional[float] = None,
        candidate_generation: Optional[int] = None,
        parent_revision: Optional[str] = None,
        candidate_revision: Optional[str] = None,
    ):
        record = {
            "attempt_id": "ATTEMPT_" + uuid.uuid4().hex[:12].upper(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parent_generation": parent_generation,
            "candidate_generation": candidate_generation,
            "parent_revision": parent_revision,
            "candidate_revision": candidate_revision,
            "evidence_count": evidence_count,
            "current_f1": current_f1,
            "proposed_f1": proposed_f1,
            "gain": gain,
            "margin_required": self.min_improvement_margin,
            "duration_sec": duration_sec,
            "status": status,
            "decision": decision,
            "reason": reason,
        }
        self.promotion_log.append(record)
        if self.store:
            try:
                self.store.record_promotion(record)
            except Exception as e:
                logger.warning(f"Failed to persist promotion attempt: {e}")

    def execute_retrain(self):
        if not self._lock.acquire(blocking=False):
            return False
        bundle = None
        generation = 0
        try:
            bundle, generation = model_buffer.snapshot()
            if bundle is None or not bundle.ready:
                self.last_retrain_status = "SKIPPED_NOT_READY"
                self._record_attempt(
                    parent_generation=generation,
                    status="SKIPPED_NOT_READY",
                    decision="SKIPPED",
                    reason="Model bundle not ready or unverified",
                )
                return False

            family = (
                bundle.metadata["run_id"]
                + ":"
                + bundle.metadata.get("model", "default")
            )
            target_store = self.store or ledger
            rows = target_store.training_data(family)
            if len(rows) < 100:
                self.last_retrain_status = "SKIPPED_INSUFFICIENT_DATA"
                self._record_attempt(
                    parent_generation=generation,
                    evidence_count=len(rows),
                    status="SKIPPED_INSUFFICIENT_DATA",
                    decision="SKIPPED",
                    reason=f"Insufficient labeled feedback samples: {len(rows)} < 100",
                    parent_revision=bundle.metadata.get("model_revision", bundle.metadata.get("run_id")),
                )
                return False

            frame = bundle.transform([json.loads(r[0]) for r in rows])
            frame["isFraud"] = [r[1] for r in rows]
            cut = int(len(frame) * 0.8)
            training, holdout = frame.iloc[:cut], frame.iloc[cut:]
            if (
                len(holdout) < 20
                or training.isFraud.nunique() < 2
                or holdout.isFraud.nunique() < 2
            ):
                self.last_retrain_status = "SKIPPED_INSUFFICIENT_CLASSES"
                self._record_attempt(
                    parent_generation=generation,
                    evidence_count=len(rows),
                    status="SKIPPED_INSUFFICIENT_CLASSES",
                    decision="SKIPPED",
                    reason="Both classes (fraud and non-fraud) must be present in training and holdout",
                    parent_revision=bundle.metadata.get("model_revision", bundle.metadata.get("run_id")),
                )
                return False

            fit_start = time.perf_counter()
            candidate = copy.deepcopy(bundle)

            # Fit model with bounded worker timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(candidate.model.fit, training)
                try:
                    future.result(timeout=self.max_retrain_duration_sec)
                    fit_duration = time.perf_counter() - fit_start
                except concurrent.futures.TimeoutError:
                    fit_duration = time.perf_counter() - fit_start
                    self.last_retrain_status = "REJECTED_TRAINING_TIMEOUT"
                    self._record_attempt(
                        parent_generation=generation,
                        evidence_count=len(rows),
                        duration_sec=fit_duration,
                        status="REJECTED_TRAINING_TIMEOUT",
                        decision="REJECTED",
                        reason=f"Fitting exceeded deadline of {self.max_retrain_duration_sec}s",
                        parent_revision=bundle.metadata.get("model_revision", bundle.metadata.get("run_id")),
                    )
                    return False
                except Exception as exc:
                    fit_duration = time.perf_counter() - fit_start
                    self.last_retrain_status = f"FAILED: {exc}"
                    self._record_attempt(
                        parent_generation=generation,
                        evidence_count=len(rows),
                        duration_sec=fit_duration,
                        status=f"FAILED: {exc}",
                        decision="FAILED",
                        reason=str(exc),
                        parent_revision=bundle.metadata.get("model_revision", bundle.metadata.get("run_id")),
                    )
                    return False

            proposed, _ = candidate.model.predict(holdout)
            current, _ = bundle.model.predict(holdout)
            curr_f1 = float(f1_score(holdout.isFraud, current, zero_division=0))
            prop_f1 = float(f1_score(holdout.isFraud, proposed, zero_division=0))
            gain = float(prop_f1 - curr_f1)

            if gain <= 0.0:
                self.last_retrain_status = "REJECTED_NO_HOLDOUT_GAIN"
                self._record_attempt(
                    parent_generation=generation,
                    evidence_count=len(rows),
                    duration_sec=fit_duration,
                    current_f1=curr_f1,
                    proposed_f1=prop_f1,
                    gain=gain,
                    status="REJECTED_NO_HOLDOUT_GAIN",
                    decision="REJECTED",
                    reason=f"Proposed model F1 ({prop_f1:.4f}) did not exceed current F1 ({curr_f1:.4f})",
                    parent_revision=bundle.metadata.get("model_revision", bundle.metadata.get("run_id")),
                )
                return False
            if gain < self.min_improvement_margin:
                self.last_retrain_status = "REJECTED_INSUFFICIENT_MARGIN"
                self._record_attempt(
                    parent_generation=generation,
                    evidence_count=len(rows),
                    duration_sec=fit_duration,
                    current_f1=curr_f1,
                    proposed_f1=prop_f1,
                    gain=gain,
                    status="REJECTED_INSUFFICIENT_MARGIN",
                    decision="REJECTED",
                    reason=f"Holdout gain {gain:.4f} below required margin {self.min_improvement_margin:.4f}",
                    parent_revision=bundle.metadata.get("model_revision", bundle.metadata.get("run_id")),
                )
                return False

            candidate_revision = uuid.uuid4().hex
            candidate.metadata = dict(
                candidate.metadata,
                parent_generation=generation,
                promotion="chronological holdout F1 improvement with verified margin",
                model_revision=candidate_revision,
                retrained_at=datetime.now(timezone.utc).isoformat(),
                f1_gain=gain,
                retrain_fit_duration_sec=fit_duration,
            )
            root = Path(
                settings.model_output_dir or str(RESULTS_DIR / "service" / "models")
            )
            path = (
                root
                / f"generation-{generation + 1}-{candidate_revision}"
            )
            save_bundle(candidate, path)
            with feedback_lock:
                pointer = {
                    "package": path.name,
                    "generation": generation + 1,
                    "family_id": family,
                    "model_revision": candidate_revision,
                    "activated_at": datetime.now(timezone.utc).isoformat(),
                }
                root.mkdir(parents=True, exist_ok=True)
                temporary = root / ("active-" + uuid.uuid4().hex + ".tmp")
                temporary.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
                os.replace(temporary, root / "active.json")
                try:
                    alt_dir = Path(RESULTS_DIR / "service")
                    alt_dir.mkdir(parents=True, exist_ok=True)
                    alt_tmp = alt_dir / f"active_model-{uuid.uuid4().hex}.tmp"
                    alt_tmp.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
                    os.replace(alt_tmp, alt_dir / "active_model.json")
                except Exception:
                    pass
                model_buffer.atomic_swap(candidate, generation=generation + 1)
                drift_monitor.reset_detectors()
            self.total_retrains += 1
            self.last_retrain_status = "SUCCESS"
            self._record_attempt(
                parent_generation=generation,
                candidate_generation=generation + 1,
                evidence_count=len(rows),
                duration_sec=fit_duration,
                current_f1=curr_f1,
                proposed_f1=prop_f1,
                gain=gain,
                status="PROMOTED",
                decision="ACCEPTED",
                reason=f"Holdout gain {gain:.4f} >= {self.min_improvement_margin:.4f}",
                parent_revision=bundle.metadata.get("model_revision", bundle.metadata.get("run_id")),
                candidate_revision=candidate_revision,
            )
            return True
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
            self.last_retrain_status = "FAILED: " + str(exc)
            self._record_attempt(
                parent_generation=generation if bundle else 0,
                status=f"FAILED: {exc}",
                decision="FAILED",
                reason=str(exc),
            )
            return False
        finally:
            self._lock.release()


retrainer = BackgroundRetrainer(store=ledger)


@app.get("/health")
def health_check():
    bundle, generation = model_buffer.snapshot()
    return dict(
        status="READY" if bundle and bundle.ready else "NOT_READY",
        model_loaded=bundle is not None,
        is_ready=bool(bundle and bundle.ready),
        model_generation=generation,
        data_kind=bundle.metadata.get("data_kind") if bundle else None,
        model_version="0.6.0",
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
        reason = "No model package loaded" if bundle is None else "Loaded model package is unverified or not ready"
        raise HTTPException(
            503,
            f"Service Unavailable: {reason}. Operational scoring requires a ready model package verified on genuine observed data.",
        )
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
                model_version="0.6.0",
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
        original, fresh = ledger.feedback(
            payload.trans_id,
            payload.is_fraud,
            label_source=payload.label_source or "analyst_review",
            confirmation_timestamp=payload.confirmation_timestamp,
        )
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


class ModelActivationPayload(BaseModel):
    package_name_or_path: str = Field(
        min_length=1,
        description="Model package directory name within model_output_dir or absolute path",
    )


@app.post("/v1/models/activate", dependencies=[Depends(verify_api_key)])
def activate_model(payload: ModelActivationPayload):
    root = Path(settings.model_output_dir or str(RESULTS_DIR / "service" / "models"))
    candidate_path = Path(payload.package_name_or_path)
    if not candidate_path.is_absolute():
        candidate_path = root / payload.package_name_or_path

    if not candidate_path.exists():
        raise HTTPException(404, f"Model package directory not found: {candidate_path}")

    try:
        new_bundle = load_bundle(candidate_path)
    except Exception as exc:
        raise HTTPException(422, f"Failed to load model package: {exc}")

    if not new_bundle.ready:
        raise HTTPException(
            422,
            "Cannot activate model package: package fails verified genuine observed provenance requirements",
        )

    with feedback_lock:
        _, current_gen = model_buffer.snapshot()
        new_gen = current_gen + 1
        family_id = (
            new_bundle.metadata["run_id"]
            + ":"
            + new_bundle.metadata.get("model", "default")
        )
        pointer = {
            "package": candidate_path.name if candidate_path.parent == root else str(candidate_path),
            "generation": new_gen,
            "family_id": family_id,
            "activated_at": datetime.now(timezone.utc).isoformat(),
        }
        root.mkdir(parents=True, exist_ok=True)
        temp_pointer = root / f"active-{uuid.uuid4().hex}.tmp"
        temp_pointer.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        os.replace(temp_pointer, root / "active.json")
        try:
            alt_pointer_dir = Path(RESULTS_DIR / "service")
            alt_pointer_dir.mkdir(parents=True, exist_ok=True)
            alt_tmp = alt_pointer_dir / f"active_model-{uuid.uuid4().hex}.tmp"
            alt_tmp.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
            os.replace(alt_tmp, alt_pointer_dir / "active_model.json")
        except Exception:
            pass

        model_buffer.atomic_swap(new_bundle, generation=new_gen)
        drift_monitor.reset_detectors()

    return {
        "status": "ACTIVATED",
        "generation": new_gen,
        "model_revision": new_bundle.metadata.get("model_revision", new_bundle.metadata.get("run_id")),
        "package": pointer["package"],
    }


class ModelRollbackPayload(BaseModel):
    target_generation: Optional[int] = Field(
        None, description="Target generation to roll back to. If None, rolls back to previous generation."
    )
    package_name_or_path: Optional[str] = Field(
        None, description="Explicit package directory name or path to activate."
    )
    reason: Optional[str] = Field("Manual operational rollback", description="Reason for rollback")


@app.post("/v1/models/rollback", dependencies=[Depends(verify_api_key)])
def rollback_model(payload: ModelRollbackPayload):
    root = Path(settings.model_output_dir or str(RESULTS_DIR / "service" / "models"))
    base_bundle_path = Path(settings.model_bundle_path) if settings.model_bundle_path else None

    current_bundle, current_gen = model_buffer.snapshot()
    if current_bundle is None:
        raise HTTPException(503, "No model package currently loaded")

    family_id = (
        current_bundle.metadata["run_id"]
        + ":"
        + current_bundle.metadata.get("model", "default")
    )

    candidate_path = None
    target_gen = payload.target_generation

    if payload.package_name_or_path:
        cand = Path(payload.package_name_or_path)
        if not cand.is_absolute():
            cand = root / payload.package_name_or_path
        if not cand.exists():
            raise HTTPException(404, f"Target model package not found: {cand}")
        candidate_path = cand
        if target_gen is None:
            parts = cand.name.split("-")
            if len(parts) >= 2 and parts[0] == "generation" and parts[1].isdigit():
                target_gen = int(parts[1])
            else:
                target_gen = max(0, current_gen - 1)
    elif target_gen is not None:
        if target_gen == 0:
            if not base_bundle_path or not base_bundle_path.exists():
                raise HTTPException(404, "Base model bundle not found for generation 0")
            candidate_path = base_bundle_path
        else:
            matches = list(root.glob(f"generation-{target_gen}-*"))
            if not matches:
                raise HTTPException(404, f"No model package found for generation {target_gen} in {root}")
            candidate_path = sorted(matches)[-1]
    else:
        if current_gen <= 0:
            raise HTTPException(400, "Cannot roll back: already at generation 0 (base bundle)")
        target_gen = current_gen - 1
        if target_gen == 0:
            if not base_bundle_path or not base_bundle_path.exists():
                raise HTTPException(404, "Base model bundle not found for generation 0")
            candidate_path = base_bundle_path
        else:
            matches = list(root.glob(f"generation-{target_gen}-*"))
            if not matches:
                raise HTTPException(404, f"No model package found for generation {target_gen} in {root}")
            candidate_path = sorted(matches)[-1]

    try:
        rollback_bundle = load_bundle(candidate_path)
    except Exception as exc:
        raise HTTPException(422, f"Failed to load rollback candidate package: {exc}")

    if not rollback_bundle.ready:
        raise HTTPException(
            422,
            "Cannot roll back: candidate model package fails verified genuine observed provenance requirements",
        )

    cand_family = (
        rollback_bundle.metadata["run_id"]
        + ":"
        + rollback_bundle.metadata.get("model", "default")
    )
    if cand_family != family_id:
        raise HTTPException(
            422,
            f"Cannot roll back: candidate belongs to family {cand_family}, current family is {family_id}",
        )

    with feedback_lock:
        pointer = {
            "package": candidate_path.name if candidate_path.parent == root else str(candidate_path),
            "generation": target_gen,
            "family_id": family_id,
            "model_revision": rollback_bundle.metadata.get(
                "model_revision", rollback_bundle.metadata.get("run_id")
            ),
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "rollback_from_generation": current_gen,
            "rollback_reason": payload.reason,
        }
        root.mkdir(parents=True, exist_ok=True)
        temp_pointer = root / f"active-{uuid.uuid4().hex}.tmp"
        temp_pointer.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        os.replace(temp_pointer, root / "active.json")
        try:
            alt_pointer_dir = Path(RESULTS_DIR / "service")
            alt_pointer_dir.mkdir(parents=True, exist_ok=True)
            alt_tmp = alt_pointer_dir / f"active_model-{uuid.uuid4().hex}.tmp"
            alt_tmp.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
            os.replace(alt_tmp, alt_pointer_dir / "active_model.json")
        except Exception:
            pass

        model_buffer.atomic_swap(rollback_bundle, generation=target_gen)
        drift_monitor.reset_detectors()

    retrainer._record_attempt(
        parent_generation=current_gen,
        candidate_generation=target_gen,
        parent_revision=current_bundle.metadata.get(
            "model_revision", current_bundle.metadata.get("run_id")
        ),
        candidate_revision=rollback_bundle.metadata.get(
            "model_revision", rollback_bundle.metadata.get("run_id")
        ),
        status="ROLLED_BACK",
        decision="ROLLED_BACK",
        reason=payload.reason or f"Operational rollback from generation {current_gen} to {target_gen}",
    )

    return {
        "status": "ROLLED_BACK",
        "generation": target_gen,
        "previous_generation": current_gen,
        "model_revision": rollback_bundle.metadata.get(
            "model_revision", rollback_bundle.metadata.get("run_id")
        ),
        "package": pointer["package"],
        "reason": payload.reason,
    }


@app.get("/v1/models/promotions", dependencies=[Depends(verify_api_key)])
def get_promotions(limit: int = 50):
    records = (
        ledger.get_promotion_history(limit=limit)
        if ledger
        else retrainer.promotion_log[-limit:]
    )
    return {
        "total": len(records),
        "promotions": records,
    }


@app.get("/v1/feedback/audit", dependencies=[Depends(verify_api_key)])
def get_feedback_audit(limit: int = 100):
    records = ledger.get_feedback_events(limit=limit) if ledger else []
    return {
        "total": len(records),
        "events": records,
    }

