"""
Real-Time Concept-Drift-Aware Fraud Detection Microservice.
Built with FastAPI and Uvicorn.

Key Production Capabilities:
1. Sub-15ms Real-Time Inference (/v1/score and /v1/batch-score)
2. Asynchronous Chargeback / Ground-Truth Feedback Stream (/v1/feedback)
3. Live Concept Drift Telemetry & Statistical Error Tracking (/v1/drift-status)
4. Atomic Double-Buffered Model Pointer: Lock-free scoring requests during background model retraining.
5. Readiness and Liveness Health Probes (/health)
"""

import os
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

from src.config import USE_GPU, GPU_DEVICE, RANDOM_SEED, ADWIN_DELTA, API_KEY
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


class DriftStatusResponse(BaseModel):
    total_samples_monitored: int
    total_errors_observed: int
    current_error_rate: float
    adwin_drift_detected: bool
    ddm_drift_detected: bool
    ddm_warning_detected: bool
    model_generation: int
    last_retrain_timestamp: Optional[str]


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
recent_predictions: Dict[str, float] = {}  # In-memory LRU cache of recent transaction probabilities
predictions_lock = threading.Lock()


# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Enforces API Key authentication for sensitive inference, feedback, and telemetry endpoints.
    Mitigates adversarial label-poisoning attacks on the feedback stream.
    """
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide a valid 'X-API-Key' header."
        )
    return api_key


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
        cpu_percent=round(cpu_pct, 2)
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

    # Cache prediction for downstream chargeback feedback correlation
    with predictions_lock:
        recent_predictions[payload.trans_id] = prob
        if len(recent_predictions) > 100_000:
            # Simple eviction of oldest entries
            first_keys = list(recent_predictions.keys())[:10_000]
            for k in first_keys:
                del recent_predictions[k]

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
def log_chargeback_feedback(payload: FeedbackPayload):
    """
    Ingests asynchronous ground-truth dispute/chargeback labels into the drift monitor stream.
    Updates ADWIN and DDM statistical test streams.
    """
    predicted_prob = payload.predicted_probability
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

    predicted_binary = int(predicted_prob >= 0.5)

    alert = drift_monitor.update(
        y_true=payload.is_fraud,
        y_pred=predicted_binary,
        slice_index=0,
        step_number=drift_monitor.total_samples_monitored,
        trans_id=payload.trans_id
    )

    err_rate = (drift_monitor.total_errors_observed / drift_monitor.total_samples_monitored) if drift_monitor.total_samples_monitored > 0 else 0.0

    return FeedbackResponse(
        status="RECORDED",
        trans_id=payload.trans_id,
        drift_alert_triggered=alert["any_drift"],
        current_error_rate=round(err_rate, 4),
        monitored_samples_total=drift_monitor.total_samples_monitored
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
        last_retrain_timestamp=model_buffer.last_retrain_time
    )


def initialize_app_model():
    """Initializes a lightweight dummy/warmup XGBoost model into the buffer on startup."""
    import xgboost as xgb
    dummy_features = ["TransactionAmt", "card1", "C1", "C2", "C5", "C13", "D1"]
    X = np.random.randn(200, len(dummy_features)).astype(np.float32)
    y = np.random.binomial(1, 0.1, size=200)

    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        tree_method="hist",
        device="cpu"
    )
    model.fit(X, y)
    model_buffer.atomic_swap(model, dummy_features, is_dummy=True)
    logger.info("initialize_app_model: Model buffer pre-seeded with synthetic warmup placeholder (is_dummy=True, is_ready=False).")


# Auto-seed on module import
initialize_app_model()
