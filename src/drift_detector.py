"""
FR-4: Drift Detection Instrumentation Module.
Implements ADWIN (Adaptive Windowing) and DDM (Drift Detection Method)
applied to model prediction error streams, logging all statistical alerts
and correlating them with adaptation triggers.
"""

import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from river import drift

from src.config import (
    ADWIN_DELTA, DDM_MIN_INSTANCES, DDM_WARNING_LEVEL, DDM_DRIFT_LEVEL
)

logger = logging.getLogger("fraud_drift.detector")


class DriftMonitor:
    """
    FR-4: Instruments real-time concept drift detection over prediction error streams.
    Applies ADWIN and DDM concurrently to capture sudden and gradual distribution shifts.
    Equipped with internal reentrant thread synchronization for safe concurrent microservice updates.
    """

    def __init__(
        self,
        adwin_delta: float = ADWIN_DELTA,
        ddm_min_instances: int = DDM_MIN_INSTANCES,
        ddm_warning_level: float = DDM_WARNING_LEVEL,
        ddm_drift_level: float = DDM_DRIFT_LEVEL
    ):
        self.adwin_delta = adwin_delta
        self.ddm_min_instances = ddm_min_instances
        self.ddm_warning_level = ddm_warning_level
        self.ddm_drift_level = ddm_drift_level
        self._lock = threading.Lock()
        self.adwin = drift.ADWIN(delta=adwin_delta)
        self.ddm = drift.binary.DDM(
            warm_start=ddm_min_instances,
            warning_threshold=ddm_warning_level,
            drift_threshold=ddm_drift_level
        )
        self.drift_events: List[Dict[str, Any]] = []
        self.total_samples_monitored = 0
        self.total_errors_observed = 0

    def update(
        self,
        y_true: int,
        y_pred: int,
        slice_index: int,
        step_number: int,
        trans_id: str
    ) -> Dict[str, Any]:
        """
        Thread-safely updates detectors with single transaction prediction error.
        Error = 1 if misclassified, 0 if correctly classified.
        """
        with self._lock:
            error_val = 1 if (y_pred != y_true) else 0
            self.total_samples_monitored += 1
            self.total_errors_observed += error_val

            # Update ADWIN
            self.adwin.update(error_val)
            adwin_drift = self.adwin.drift_detected

            # Update DDM
            self.ddm.update(error_val)
            ddm_drift = self.ddm.drift_detected
            ddm_warning = self.ddm.warning_detected

            alert = {
                "adwin_drift": adwin_drift,
                "ddm_drift": ddm_drift,
                "ddm_warning": ddm_warning,
                "any_drift": adwin_drift or ddm_drift,
                "error_val": error_val
            }

            # Log drift events (FR-4.2)
            if adwin_drift:
                event = {
                    "slice_index": slice_index,
                    "step_number": step_number,
                    "trans_id": trans_id,
                    "detector_type": "ADWIN",
                    "error_rate": round(self.adwin.estimation, 4) if hasattr(self.adwin, "estimation") else None,
                    "confidence_score": round(self.adwin.width, 1) if hasattr(self.adwin, "width") else 0.0,
                    "status": "DRIFT",
                    "notes": f"ADWIN flagged distribution shift at step {step_number} in slice {slice_index + 1}"
                }
                self.drift_events.append(event)
                logger.info(f"[FR-4 DRIFT ALERT] ADWIN: Slice {slice_index + 1}, Step {step_number}")

            if ddm_drift:
                event = {
                    "slice_index": slice_index,
                    "step_number": step_number,
                    "trans_id": trans_id,
                    "detector_type": "DDM",
                    "error_rate": round(self.total_errors_observed / max(1, self.total_samples_monitored), 4),
                    "confidence_score": None,
                    "status": "DRIFT",
                    "notes": f"DDM flagged drift level reached at step {step_number} in slice {slice_index + 1}"
                }
                self.drift_events.append(event)
                logger.info(f"[FR-4 DRIFT ALERT] DDM: Slice {slice_index + 1}, Step {step_number}")

            return alert

    def reset_detectors(self):
        """Thread-safely resets detection windows after model adaptation event."""
        with self._lock:
            self.adwin = drift.ADWIN(delta=self.adwin_delta)
            self.ddm = drift.binary.DDM(
                warm_start=self.ddm_min_instances,
                warning_threshold=self.ddm_warning_level,
                drift_threshold=self.ddm_drift_level
            )
            logger.info("Drift detectors reset following model adaptation.")

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Returns a thread-safe snapshot of current drift monitoring telemetry."""
        with self._lock:
            total_samples = self.total_samples_monitored
            total_errors = self.total_errors_observed
            err_rate = (total_errors / total_samples) if total_samples > 0 else 0.0
            return {
                "total_samples_monitored": total_samples,
                "total_errors_observed": total_errors,
                "current_error_rate": round(err_rate, 4),
                "adwin_drift_detected": getattr(self.adwin, "drift_detected", False),
                "ddm_drift_detected": getattr(self.ddm, "drift_detected", False),
                "ddm_warning_detected": getattr(self.ddm, "warning_detected", False),
            }
