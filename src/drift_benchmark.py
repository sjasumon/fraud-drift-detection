"""
Concept Drift Detection Benchmark Module.
Evaluates detection delay (Mean Time to Detect) and false alarm rates on synthetic
streams with known ground-truth change points, comparing ADWIN and DDM across
abrupt, subtle, and gradual drift regimes.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from river import drift

from src.config import METRICS_DIR, RANDOM_SEED, ADWIN_DELTA, DDM_MIN_INSTANCES, DDM_WARNING_LEVEL, DDM_DRIFT_LEVEL

logger = logging.getLogger("fraud_drift.drift_benchmark")


def generate_change_point_stream(
    n_samples: int = 6000,
    regimes: Optional[List[Dict[str, Any]]] = None,
    seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Generates a 1D synthetic binary error stream with known change points.
    
    Default configuration:
    - Phase 1 [0:2000]: Stationary baseline with error rate p = 0.05
    - Phase 2 [2000:4000]: Abrupt concept drift to error rate p = 0.25 (True Change Point at t=2000)
    - Phase 3 [4000:6000]: Gradual recovery back to p = 0.05 over 500 samples (True Change Point at t=4000)
    """
    rng = np.random.default_rng(seed)
    if regimes is None:
        regimes = [
            {"start": 0, "end": 2000, "rate": 0.05, "drift_type": "stationary", "is_change_point": False},
            {"start": 2000, "end": 4000, "rate": 0.25, "drift_type": "abrupt", "is_change_point": True},
            {"start": 4000, "end": 6000, "rate": 0.05, "drift_type": "gradual", "is_change_point": True, "ramp_width": 500}
        ]

    stream = np.zeros(n_samples, dtype=int)
    change_points = []

    for reg in regimes:
        start = reg["start"]
        end = min(n_samples, reg["end"])
        length = end - start
        if length <= 0:
            continue

        if reg.get("is_change_point", False):
            change_points.append({
                "step": start,
                "drift_type": reg.get("drift_type", "abrupt"),
                "target_rate": reg["rate"],
                "ramp_width": reg.get("ramp_width", 0)
            })

        if reg.get("ramp_width", 0) > 0 and reg.get("drift_type") == "gradual":
            # Linear transition from previous rate
            prev_rate = stream[start - 1] if start > 0 else 0.05
            # Find the actual previous rate from regimes
            target_rate = reg["rate"]
            ramp = min(reg["ramp_width"], length)
            probs = np.linspace(prev_rate, target_rate, ramp)
            if length > ramp:
                probs = np.concatenate([probs, np.full(length - ramp, target_rate)])
            stream[start:end] = rng.binomial(1, probs)
        else:
            p = reg["rate"]
            stream[start:end] = rng.binomial(1, p, size=length)

    return stream, change_points


def evaluate_detector_on_stream(
    detector_name: str,
    stream: np.ndarray,
    change_points: List[Dict[str, Any]],
    tolerance_window: int = 800,
    detector_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates detection delay (samples to detection after change point) and false alarms.
    
    Returns:
    - delays: List of detection delays (in samples) for detected change points
    - detections: Number of true change points detected within tolerance window
    - false_alarms: Number of alerts triggered outside change-point tolerance windows
    - false_alarm_rate_per_1k: False alarms normalized per 1,000 stationary samples
    """
    detector_params = detector_params or {}
    if detector_name.upper() == "ADWIN":
        delta = detector_params.get("delta", ADWIN_DELTA)
        detector = drift.ADWIN(delta=delta)
    elif detector_name.upper() == "DDM":
        min_inst = detector_params.get("min_instances", DDM_MIN_INSTANCES)
        w_lvl = detector_params.get("warning_level", DDM_WARNING_LEVEL)
        d_lvl = detector_params.get("drift_level", DDM_DRIFT_LEVEL)
        detector = drift.binary.DDM(warm_start=min_inst, warning_threshold=w_lvl, drift_threshold=d_lvl)
    else:
        raise ValueError(f"Unsupported detector: {detector_name}")

    alerts = []
    for step, val in enumerate(stream):
        detector.update(val)
        if detector.drift_detected:
            alerts.append(step)

    # Correlate alerts with ground-truth change points
    detected_change_points = []
    delays = []
    matched_alerts = set()

    for cp in change_points:
        cp_step = cp["step"]
        # Find first alert in [cp_step, cp_step + tolerance_window]
        valid_alerts = [a for a in alerts if cp_step <= a <= cp_step + tolerance_window]
        if valid_alerts:
            first_alert = valid_alerts[0]
            delay = first_alert - cp_step
            delays.append(delay)
            matched_alerts.add(first_alert)
            detected_change_points.append({
                "change_point_step": cp_step,
                "drift_type": cp["drift_type"],
                "detected_at_step": first_alert,
                "delay_samples": delay,
                "status": "DETECTED"
            })
        else:
            detected_change_points.append({
                "change_point_step": cp_step,
                "drift_type": cp["drift_type"],
                "detected_at_step": None,
                "delay_samples": None,
                "status": "MISSED"
            })

    # False alarms: alerts outside all [cp_step, cp_step + tolerance_window] intervals
    false_alarms = [a for a in alerts if not any(cp["step"] <= a <= cp["step"] + tolerance_window for cp in change_points)]
    
    total_samples = len(stream)
    tolerance_coverage = sum(min(tolerance_window, total_samples - cp["step"]) for cp in change_points)
    stationary_samples = max(1, total_samples - tolerance_coverage)
    fa_per_1k = (len(false_alarms) / stationary_samples) * 1000.0

    return {
        "detector_name": detector_name.upper(),
        "total_stream_samples": total_samples,
        "true_change_points_count": len(change_points),
        "detected_count": len(delays),
        "detection_rate": round(len(delays) / len(change_points), 4) if change_points else 0.0,
        "delays_samples": delays,
        "mean_delay_samples": round(float(np.mean(delays)), 2) if delays else None,
        "median_delay_samples": round(float(np.median(delays)), 2) if delays else None,
        "false_alarms_count": len(false_alarms),
        "false_alarm_rate_per_1k": round(float(fa_per_1k), 3),
        "change_point_audit": detected_change_points
    }


def run_multi_seed_drift_benchmark(
    n_seeds: int = 5,
    n_samples: int = 6000,
    output_file: Optional[Path] = None,
    save_metrics: bool = True
) -> Dict[str, Any]:
    """
    Executes a multi-seed benchmark evaluating ADWIN and DDM detection latency and false alarms.
    """
    seeds = [42 + i * 101 for i in range(n_seeds)]
    adwin_delays_abrupt = []
    adwin_delays_gradual = []
    adwin_false_alarms = []

    ddm_delays_abrupt = []
    ddm_delays_gradual = []
    ddm_false_alarms = []

    for seed in seeds:
        stream, change_points = generate_change_point_stream(n_samples=n_samples, seed=seed)

        # 1. ADWIN
        adwin_res = evaluate_detector_on_stream("ADWIN", stream, change_points)
        adwin_false_alarms.append(adwin_res["false_alarms_count"])
        for audit in adwin_res["change_point_audit"]:
            if audit["status"] == "DETECTED":
                if audit["drift_type"] == "abrupt":
                    adwin_delays_abrupt.append(audit["delay_samples"])
                elif audit["drift_type"] == "gradual":
                    adwin_delays_gradual.append(audit["delay_samples"])

        # 2. DDM
        ddm_res = evaluate_detector_on_stream("DDM", stream, change_points)
        ddm_false_alarms.append(ddm_res["false_alarms_count"])
        for audit in ddm_res["change_point_audit"]:
            if audit["status"] == "DETECTED":
                if audit["drift_type"] == "abrupt":
                    ddm_delays_abrupt.append(audit["delay_samples"])
                elif audit["drift_type"] == "gradual":
                    ddm_delays_gradual.append(audit["delay_samples"])

    summary = {
        "benchmark_description": "Concept drift detection delay and false alarm benchmark on synthetic stream",
        "n_seeds": n_seeds,
        "n_samples_per_seed": n_samples,
        "ADWIN": {
            "abrupt_drift_mean_delay_samples": round(float(np.mean(adwin_delays_abrupt)), 2) if adwin_delays_abrupt else None,
            "abrupt_drift_std_delay_samples": round(float(np.std(adwin_delays_abrupt)), 2) if adwin_delays_abrupt else None,
            "gradual_drift_mean_delay_samples": round(float(np.mean(adwin_delays_gradual)), 2) if adwin_delays_gradual else None,
            "gradual_drift_std_delay_samples": round(float(np.std(adwin_delays_gradual)), 2) if adwin_delays_gradual else None,
            "mean_false_alarms_per_stream": round(float(np.mean(adwin_false_alarms)), 2),
            "detection_rate_pct": 100.0 if len(adwin_delays_abrupt) == n_seeds else round(len(adwin_delays_abrupt) / n_seeds * 100.0, 1)
        },
        "DDM": {
            "abrupt_drift_mean_delay_samples": round(float(np.mean(ddm_delays_abrupt)), 2) if ddm_delays_abrupt else None,
            "abrupt_drift_std_delay_samples": round(float(np.std(ddm_delays_abrupt)), 2) if ddm_delays_abrupt else None,
            "gradual_drift_mean_delay_samples": round(float(np.mean(ddm_delays_gradual)), 2) if ddm_delays_gradual else None,
            "gradual_drift_std_delay_samples": round(float(np.std(ddm_delays_gradual)), 2) if ddm_delays_gradual else None,
            "mean_false_alarms_per_stream": round(float(np.mean(ddm_false_alarms)), 2),
            "detection_rate_pct": 100.0 if len(ddm_delays_abrupt) == n_seeds else round(len(ddm_delays_abrupt) / n_seeds * 100.0, 1)
        }
    }

    if save_metrics:
        out_file = output_file or (METRICS_DIR / "drift_detection_benchmark.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        logger.info("Drift detection benchmark metrics saved to %s", out_file)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Concept Drift Detection Benchmark")
    parser.add_argument("--seeds", type=int, default=5, help="Number of evaluation random seeds")
    parser.add_argument("--samples", type=int, default=6000, help="Stream length per seed")
    args = parser.parse_args()

    summary = run_multi_seed_drift_benchmark(n_seeds=args.seeds, n_samples=args.samples)
    print("\n=== Concept Drift Detection Benchmark Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
