"""
Real-Time Concept-Drift-Aware Fraud Detection Microservice.
Built with FastAPI and Uvicorn.

Key Production Capabilities:
1. Sub-15ms Real-Time Inference (/v1/score and /v1/batch-score) via active double-buffer pointer.
2. Asynchronous Chargeback / Ground-Truth Feedback Stream (/v1/feedback) updating ADWIN & DDM detectors.
3. Automated Background Shadow Retraining: When concept drift is flagged by ADWIN/DDM on /v1/feedback,
   FastAPI dispatches an asynchronous background retrain worker that builds a shadow XGBoost model.
4. Atomic Double-Buffered Pointer Swap: Background retrain worker swaps active pointer with zero downtime
   or scoring interruptions, then automatically resets drift detectors (re-baselining error windows).
5. Dedicated Model Retraining API (/v1/retrain) supporting synchronous or asynchronous manual/orchestrated retrains.
6. Live Drift & Adaptation Telemetry (/v1/drift-status) and Health Probes (/health).
"""

import os
import secrets
import sys
import time
try:
    import psutil
except ImportError:
    psutil = None
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Security, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.config import USE_GPU, GPU_DEVICE, RANDOM_SEED, ADWIN_DELTA, API_KEY, APP_ENV
from src.drift_detector import DriftMonitor

logger = logging.getLogger("fraud_drift.api")

# --- Pydantic Data Contracts ---

class TransactionPayload(BaseModel):
    trans_id: str = Field(..., description="Unique transaction reference ID")
    features: Dict[str, float] = Field(..., description="Key-value mapping of engineered transaction features")
    timestamp: Optional[float] = Field(default=None, description="Unix epoch timestamp")


class BatchTransactionPayload(BaseModel):
    transactions: List[TransactionPayload] = Field(..., description="Array of transactions to score")


class ScoreResponse(BaseModel):
    trans_id: str
    fraud_probability: float
    is_fraud: bool
    risk_tier: str  # LOW, MEDIUM, HIGH
    inference_latency_ms: float
    model_version: str
    model_generation: int


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    total_processed: int
    total_latency_ms: float
    throughput_qps: float


class FeedbackPayload(BaseModel):
    trans_id: str = Field(..., description="Transaction ID corresponding to previous score")
    is_fraud: int = Field(..., ge=0, le=1, description="Ground truth outcome: 1 if fraud/chargeback, 0 if legitimate")
    dispute_reason: Optional[str] = Field(default=None, description="Reason code for chargeback")
    predicted_probability: Optional[float] = Field(default=None, description="Logged prediction probability")


class FeedbackResponse(BaseModel):
    status: str
    trans_id: str
    drift_alert_triggered: bool
    current_error_rate: float
    monitored_samples_total: int
    shadow_retrain_triggered: bool = False
    model_generation: int = 1


class DriftStatusResponse(BaseModel):
    total_samples_monitored: int
    total_errors_observed: int
    current_error_rate: float
    adwin_drift_detected: bool
    ddm_drift_detected: bool
    ddm_warning_detected: bool
    model_generation: int
    last_retrain_timestamp: Optional[str]
    is_retraining: bool = False
    total_retrains: int = 0
    last_retrain_status: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    model_generation: int
    is_dummy: bool = Field(..., description="True if serving a synthetic warmup placeholder model")
    is_ready: bool = Field(..., description="True if serving a genuine production-trained model")
    gpu_enabled: bool
    gpu_device: str
    memory_rss_mb: float
    cpu_percent: float
    is_retraining: bool = False
    total_retrains: int = 0


class RetrainResponse(BaseModel):
    status: str
    message: str
    current_generation: int
    is_retraining: bool
    total_retrains: int
    last_retrain_duration_sec: float
    last_retrain_status: Optional[str] = None


# --- Atomic Double-Buffer Model Container ---

class ModelDoubleBuffer:
    """
    Thread-safe double-buffered model wrapper for zero-downtime serving.
    Allows concurrent read requests to access active_model without locking,
    while background retrain threads build shadow_model and perform an atomic pointer swap.
    """

    def __init__(self, initial_model=None, feature_names: Optional[List[str]] = None, is_dummy: bool = True):
        self._active_model = initial_model
        self._feature_names = feature_names or []
        self._generation = 1
        self._is_dummy = is_dummy
        self._last_retrain_time: Optional[str] = datetime.utcnow().isoformat()
        self._swap_lock = threading.Lock()

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def last_retrain_time(self) -> Optional[str]:
        return self._last_retrain_time

    @property
    def is_loaded(self) -> bool:
        return self._active_model is not None

    @property
    def is_dummy(self) -> bool:
        return self._is_dummy

    @property
    def is_ready(self) -> bool:
        return self.is_loaded and not self._is_dummy

    def predict_single(self, features_dict: Dict[str, float]) -> Tuple[float, bool]:
        """Performs non-blocking inference on the active model pointer."""
        model = self._active_model
        if model is None:
            raise RuntimeError("Scoring model is not initialized.")

        # Vectorize features in canonical order
        vector = np.array(
            [features_dict.get(f, 0.0) for f in self._feature_names],
            dtype=np.float32
        ).reshape(1, -1)

        probs = model.predict_proba(vector)[:, 1]
        p = float(probs[0])
        return p, bool(p >= 0.5)

    def predict_batch(self, features_list: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Performs vectorized batch inference on the active model pointer."""
        model = self._active_model
        if model is None:
            raise RuntimeError("Scoring model is not initialized.")

        matrix = np.array([
            [f_dict.get(f, 0.0) for f in self._feature_names]
            for f_dict in features_list
        ], dtype=np.float32)

        probs = model.predict_proba(matrix)[:, 1]
        preds = (probs >= 0.5).astype(bool)
        return probs, preds

    def atomic_swap(self, new_model, new_feature_names: Optional[List[str]] = None, is_dummy: bool = False):
        """Atomically hot-swaps the active model pointer with the newly trained shadow model."""
        with self._swap_lock:
            self._active_model = new_model
            if new_feature_names is not None:
                self._feature_names = new_feature_names
            self._is_dummy = is_dummy
            self._generation += 1
            self._last_retrain_time = datetime.utcnow().isoformat()
            logger.info(
                f"ModelDoubleBuffer: Atomic hot-swap complete. "
                f"Active model generation is now #{self._generation} (is_dummy={self._is_dummy}, is_ready={self.is_ready})."
            )


# --- Feedback Training Store & Background Shadow Retrainer ---

class FeedbackTrainingStore:
    """
    Thread-safe store for accumulating labeled feedback data and initial training seed
    to train shadow models when concept drift is flagged.
    """
    def __init__(self, max_samples: int = 10000):
        self._lock = threading.Lock()
        self._max_samples = max_samples
        self._feature_names: List[str] = []
        self._X: List[List[float]] = []
        self._y: List[int] = []

    def set_seed_data(self, feature_names: List[str], X: np.ndarray, y: np.ndarray):
        with self._lock:
            self._feature_names = list(feature_names)
            self._X = X.tolist()
            self._y = y.tolist()

    def add_sample(self, features: Dict[str, float], is_fraud: int):
        with self._lock:
            if not self._feature_names:
                self._feature_names = list(features.keys())
            vector = [features.get(f, 0.0) for f in self._feature_names]
            self._X.append(vector)
            self._y.append(int(is_fraud))
            if len(self._X) > self._max_samples:
                self._X.pop(0)
                self._y.pop(0)

    def get_dataset(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        with self._lock:
            return (
                list(self._feature_names),
                np.array(self._X, dtype=np.float32),
                np.array(self._y, dtype=np.int32)
            )

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._y)


class BackgroundRetrainer:
    """
    Manages background shadow-model training and atomic hot-swapping
    when concept drift is flagged by ADWIN / DDM.
    Guarantees non-blocking scoring during retrain and coalesces concurrent triggers.
    """
    def __init__(self, model_buffer: ModelDoubleBuffer, drift_monitor: DriftMonitor, training_store: FeedbackTrainingStore):
        self.model_buffer = model_buffer
        self.drift_monitor = drift_monitor
        self.training_store = training_store
        self._retrain_lock = threading.Lock()
        self._is_retraining = False
        self._total_retrains = 0
        self._last_retrain_duration: float = 0.0
        self._last_retrain_status: Optional[str] = None

    @property
    def is_retraining(self) -> bool:
        return self._is_retraining

    @property
    def total_retrains(self) -> int:
        return self._total_retrains

    @property
    def last_retrain_duration(self) -> float:
        return self._last_retrain_duration

    @property
    def last_retrain_status(self) -> Optional[str]:
        return self._last_retrain_status

    def trigger_retrain_async(self, background_tasks: Optional[BackgroundTasks] = None) -> bool:
        """Dispatches shadow model retraining into a background thread or task."""
        if self._is_retraining:
            logger.info("BackgroundRetrainer: Retrain already in progress, coalescing trigger.")
            return False

        if background_tasks is not None:
            background_tasks.add_task(self.execute_retrain)
        else:
            thread = threading.Thread(target=self.execute_retrain, daemon=True, name="ShadowRetrainWorker")
            thread.start()
        return True

    def execute_retrain(self) -> bool:
        """
        Builds shadow model on accumulated data, performs atomic pointer swap,
        and resets drift detectors.
        """
        if not self._retrain_lock.acquire(blocking=False):
            logger.info("BackgroundRetrainer: Another worker acquired retrain lock; skipping.")
            return False

        self._is_retraining = True
        t0 = time.time()
        logger.info("BackgroundRetrainer: Starting background shadow model training...")
        try:
            import xgboost as xgb
            feature_names, X, y = self.training_store.get_dataset()

            if len(y) < 10 or len(np.unique(y)) < 2:
                logger.warning(
                    f"BackgroundRetrainer: Insufficient training data ({len(y)} samples, classes={np.unique(y)}). "
                    "Skipping shadow retrain."
                )
                self._last_retrain_status = "SKIPPED_INSUFFICIENT_DATA"
                return False

            pos = int(np.sum(y))
            neg = len(y) - pos
            scale_pos = float(neg) / max(1.0, float(pos))

            shadow_model = xgb.XGBClassifier(
                n_estimators=30,
                max_depth=4,
                learning_rate=0.08,
                scale_pos_weight=scale_pos,
                tree_method="hist",
                device="cpu",
                random_state=RANDOM_SEED
            )
            shadow_model.fit(X, y)

            # Atomic hot-swap pointer to activate new model without dropping in-flight scoring
            self.model_buffer.atomic_swap(shadow_model, feature_names, is_dummy=False)

            # Re-baseline detectors: flush pre-retrain error statistics
            self.drift_monitor.reset_detectors()

            dur = time.time() - t0
            self._last_retrain_duration = round(dur, 3)
            self._total_retrains += 1
            self._last_retrain_status = f"SUCCESS (gen={self.model_buffer.generation})"
            logger.info(
                f"BackgroundRetrainer: Shadow model generation #{self.model_buffer.generation} "
                f"trained and swapped successfully in {dur:.3f}s. Drift detectors reset."
            )
            return True
        except Exception as e:
            self._last_retrain_status = f"FAILED: {str(e)}"
            logger.error(f"BackgroundRetrainer: Failed during shadow retrain: {e}", exc_info=True)
            return False
        finally:
            self._is_retraining = False
            self._retrain_lock.release()


# --- FastAPI Application & State Initialization ---

app = FastAPI(
    title="Concept-Drift-Aware Fraud Detection Microservice",
    description="Sub-15ms real-time scoring microservice with atomic model hot-swapping and live ADWIN/DDM drift telemetry.",
    version="0.2.0"
)

# Global State Singletons
start_time = time.time()
model_buffer = ModelDoubleBuffer()
drift_monitor = DriftMonitor()
training_store = FeedbackTrainingStore()
retrainer = BackgroundRetrainer(model_buffer, drift_monitor, training_store)
recent_predictions: Dict[str, float] = {}  # In-memory LRU cache of recent transaction probabilities
recent_features: Dict[str, Dict[str, float]] = {}  # Cached features for ground-truth feedback pairing
predictions_lock = threading.Lock()
MAX_CACHE_SIZE = 100000


# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Enforces constant-time API Key authentication for sensitive inference, feedback, and telemetry endpoints.
    Uses secrets.compare_digest to prevent timing-attack side-channels and mitigates
    adversarial label-poisoning attacks on the feedback stream.
    """
    if not api_key or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide a valid 'X-API-Key' header."
        )
    return api_key


@app.on_event("startup")
def startup_security_check():
    """Validates runtime configuration hygiene upon microservice initialization."""
    if APP_ENV.lower() in ("production", "prod", "staging"):
        if secrets.compare_digest(API_KEY, "dev-secret-key") or len(API_KEY) < 16:
            raise RuntimeError(
                f"FATAL: Production mode ('{APP_ENV}') detected with insecure API_KEY. "
                "Set a secure, high-entropy API_KEY in production."
            )
        logger.info("Microservice initialized securely in '%s' mode.", APP_ENV)
    else:
        if secrets.compare_digest(API_KEY, "dev-secret-key"):
            logger.warning(
                "Microservice initialized in '%s' mode with default API_KEY ('dev-secret-key'). "
                "Ensure production deployments provide a non-default secret.", APP_ENV
            )


def assign_risk_tier(prob: float) -> str:
    """Categorizes risk probability into actionable banking tiers."""
    if prob >= 0.75:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Liveness & readiness probe reporting system telemetry, memory, and model generation."""
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_rss = mem_info.rss / (1024 * 1024)
            cpu_pct = process.cpu_percent(interval=None)
        except Exception:
            mem_rss, cpu_pct = 0.0, 0.0
    else:
        mem_rss, cpu_pct = 0.0, 0.0

    return HealthResponse(
        status="HEALTHY",
        uptime_seconds=time.time() - start_time,
        model_loaded=model_buffer.is_loaded,
        model_generation=model_buffer.generation,
        is_dummy=model_buffer.is_dummy,
        is_ready=model_buffer.is_ready,
        gpu_enabled=USE_GPU,
        gpu_device=GPU_DEVICE if USE_GPU else "cpu",
        memory_rss_mb=round(mem_rss, 2),
        cpu_percent=round(cpu_pct, 2),
        is_retraining=retrainer.is_retraining,
        total_retrains=retrainer.total_retrains
    )


@app.post("/v1/score", response_model=ScoreResponse, dependencies=[Depends(verify_api_key)], tags=["Inference"])
def score_transaction(payload: TransactionPayload):
    """
    Sub-15ms real-time scoring endpoint for payment gateways.
    Reads active model via atomic lock-free pointer.
    """
    t0 = time.perf_counter()
    try:
        prob, is_fraud = model_buffer.predict_single(payload.features)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Cache prediction & features for downstream chargeback feedback correlation
    with predictions_lock:
        recent_predictions[payload.trans_id] = prob
        recent_features[payload.trans_id] = payload.features
        if len(recent_predictions) > MAX_CACHE_SIZE:
            first_keys = list(recent_predictions.keys())[:10_000]
            for k in first_keys:
                recent_predictions.pop(k, None)
                recent_features.pop(k, None)

    return ScoreResponse(
        trans_id=payload.trans_id,
        fraud_probability=round(prob, 4),
        is_fraud=is_fraud,
        risk_tier=assign_risk_tier(prob),
        inference_latency_ms=round(latency_ms, 2),
        model_version="0.2.0",
        model_generation=model_buffer.generation
    )


@app.post("/v1/batch-score", response_model=BatchScoreResponse, dependencies=[Depends(verify_api_key)], tags=["Inference"])
def score_batch(payload: BatchTransactionPayload):
    """Vectorized high-throughput batch scoring endpoint."""
    t0 = time.perf_counter()
    if not payload.transactions:
        return BatchScoreResponse(results=[], total_processed=0, total_latency_ms=0.0, throughput_qps=0.0)

    try:
        features_list = [tx.features for tx in payload.transactions]
        probs, preds = model_buffer.predict_batch(features_list)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    throughput = (len(payload.transactions) / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

    results = []
    with predictions_lock:
        for i, tx in enumerate(payload.transactions):
            p = float(probs[i])
            is_f = bool(preds[i])
            recent_predictions[tx.trans_id] = p
            recent_features[tx.trans_id] = tx.features
            results.append(ScoreResponse(
                trans_id=tx.trans_id,
                fraud_probability=round(p, 4),
                is_fraud=is_f,
                risk_tier=assign_risk_tier(p),
                inference_latency_ms=round(total_latency_ms / len(payload.transactions), 3),
                model_version="0.2.0",
                model_generation=model_buffer.generation
            ))

    return BatchScoreResponse(
        results=results,
        total_processed=len(payload.transactions),
        total_latency_ms=round(total_latency_ms, 2),
        throughput_qps=round(throughput, 1)
    )


@app.post("/v1/feedback", response_model=FeedbackResponse, dependencies=[Depends(verify_api_key)], tags=["Stream Feedback"])
def log_chargeback_feedback(payload: FeedbackPayload, background_tasks: BackgroundTasks):
    """
    Ingests asynchronous ground-truth dispute/chargeback labels into the drift monitor stream.
    Updates ADWIN and DDM statistical test streams. If concept drift is flagged,
    automatically dispatches an asynchronous background retrain task that trains a shadow
    XGBoost model, atomically swaps the serving pointer, and resets drift detectors.
    """
    predicted_prob = payload.predicted_probability
    with predictions_lock:
        tx_features = recent_features.get(payload.trans_id)

    if predicted_prob is None:
        with predictions_lock:
            if payload.trans_id not in recent_predictions:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f"Transaction '{payload.trans_id}' was not found in the recent prediction cache. "
                        "Provide 'predicted_probability' explicitly to log ground-truth feedback."
                    )
                )
            predicted_prob = recent_predictions[payload.trans_id]

    # Accumulate labeled sample for background shadow retraining if features exist
    if tx_features:
        training_store.add_sample(tx_features, payload.is_fraud)

    predicted_binary = int(predicted_prob >= 0.5)

    alert = drift_monitor.update(
        y_true=payload.is_fraud,
        y_pred=predicted_binary,
        slice_index=0,
        step_number=drift_monitor.total_samples_monitored,
        trans_id=payload.trans_id
    )

    # Automated background retrain trigger on concept drift
    retrain_dispatched = False
    if alert["any_drift"]:
        retrain_dispatched = retrainer.trigger_retrain_async(background_tasks=background_tasks)

    err_rate = (drift_monitor.total_errors_observed / drift_monitor.total_samples_monitored) if drift_monitor.total_samples_monitored > 0 else 0.0

    return FeedbackResponse(
        status="RECORDED",
        trans_id=payload.trans_id,
        drift_alert_triggered=alert["any_drift"],
        current_error_rate=round(err_rate, 4),
        monitored_samples_total=drift_monitor.total_samples_monitored,
        shadow_retrain_triggered=retrain_dispatched,
        model_generation=model_buffer.generation
    )


@app.get("/v1/drift-status", response_model=DriftStatusResponse, dependencies=[Depends(verify_api_key)], tags=["Monitoring"])
def get_drift_status():
    """Returns real-time drift telemetry from the error stream."""
    err_rate = (drift_monitor.total_errors_observed / drift_monitor.total_samples_monitored) if drift_monitor.total_samples_monitored > 0 else 0.0
    return DriftStatusResponse(
        total_samples_monitored=drift_monitor.total_samples_monitored,
        total_errors_observed=drift_monitor.total_errors_observed,
        current_error_rate=round(err_rate, 4),
        adwin_drift_detected=drift_monitor.adwin.drift_detected,
        ddm_drift_detected=drift_monitor.ddm.drift_detected,
        ddm_warning_detected=drift_monitor.ddm.warning_detected,
        model_generation=model_buffer.generation,
        last_retrain_timestamp=model_buffer.last_retrain_time,
        is_retraining=retrainer.is_retraining,
        total_retrains=retrainer.total_retrains,
        last_retrain_status=retrainer.last_retrain_status
    )


@app.post("/v1/retrain", response_model=RetrainResponse, dependencies=[Depends(verify_api_key)], tags=["Model Lifecycle"])
def trigger_model_retrain(background_tasks: BackgroundTasks, sync: bool = False):
    """
    Triggers shadow model retraining on accumulated training and feedback data.
    If sync=True, trains synchronously and blocks until atomic pointer swap completes.
    If sync=False (default), dispatches retrain to background worker and returns immediately.
    """
    if retrainer.is_retraining:
        return RetrainResponse(
            status="IN_PROGRESS",
            message="Shadow retraining is already in progress.",
            current_generation=model_buffer.generation,
            is_retraining=True,
            total_retrains=retrainer.total_retrains,
            last_retrain_duration_sec=retrainer.last_retrain_duration,
            last_retrain_status=retrainer.last_retrain_status
        )

    if sync:
        success = retrainer.execute_retrain()
        return RetrainResponse(
            status="COMPLETED" if success else "SKIPPED_OR_FAILED",
            message=f"Synchronous retrain finished with status: {retrainer.last_retrain_status}",
            current_generation=model_buffer.generation,
            is_retraining=False,
            total_retrains=retrainer.total_retrains,
            last_retrain_duration_sec=retrainer.last_retrain_duration,
            last_retrain_status=retrainer.last_retrain_status
        )
    else:
        dispatched = retrainer.trigger_retrain_async(background_tasks=background_tasks)
        return RetrainResponse(
            status="DISPATCHED" if dispatched else "IN_PROGRESS",
            message="Background shadow retrain dispatched.",
            current_generation=model_buffer.generation,
            is_retraining=True,
            total_retrains=retrainer.total_retrains,
            last_retrain_duration_sec=retrainer.last_retrain_duration,
            last_retrain_status=retrainer.last_retrain_status
        )


def initialize_app_model():
    """Initializes a lightweight dummy/warmup XGBoost model into the buffer on startup."""
    import xgboost as xgb
    dummy_features = ["TransactionAmt", "card1", "C1", "C2", "C5", "C13", "D1"]
    np.random.seed(RANDOM_SEED)
    X = np.random.randn(200, len(dummy_features)).astype(np.float32)
    y = np.random.binomial(1, 0.1, size=200)

    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        tree_method="hist",
        device="cpu",
        random_state=RANDOM_SEED
    )
    model.fit(X, y)
    model_buffer.atomic_swap(model, dummy_features, is_dummy=True)
    training_store.set_seed_data(dummy_features, X, y)
    logger.info("initialize_app_model: Model buffer pre-seeded with synthetic warmup placeholder (is_dummy=True, is_ready=False).")


# Auto-seed on module import
initialize_app_model()
