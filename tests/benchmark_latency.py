"""
Microservice Inference Latency SLA Benchmark.

Measures real-time scoring latency distributions (p50, p95, p99) over 1,000 requests
to mathematically document sub-15ms SLAs for production deployment.
"""

import time
import json
import logging
from typing import List, Dict, Any
import numpy as np
from fastapi.testclient import TestClient

from src.config import METRICS_DIR
from src.api import app

logger = logging.getLogger("fraud_drift.benchmark")


def run_latency_benchmark(n_requests: int = 1000) -> Dict[str, Any]:
    client = TestClient(app)

    # Warmup
    for _ in range(50):
        client.post("/v1/score", json={
            "trans_id": "WARMUP",
            "features": {"TransactionAmt": 100.0, "card1": 5000.0, "C1": 1.0, "C2": 1.0, "C5": 0.0, "C13": 2.0, "D1": 5.0}
        })

    latencies_ms: List[float] = []

    start_bench = time.perf_counter()
    for i in range(n_requests):
        payload = {
            "trans_id": f"BENCH_{i:04d}",
            "features": {
                "TransactionAmt": float(50 + (i % 200)),
                "card1": float(1000 + (i % 5000)),
                "C1": float(i % 5),
                "C2": 1.0,
                "C5": 0.0,
                "C13": float(i % 10),
                "D1": float(i % 30)
            }
        }
        t0 = time.perf_counter()
        resp = client.post("/v1/score", json=payload)
        t_dur = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(t_dur)

    total_bench_duration = time.perf_counter() - start_bench
    latencies_arr = np.array(latencies_ms)

    p50 = float(np.percentile(latencies_arr, 50))
    p90 = float(np.percentile(latencies_arr, 90))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    mean_lat = float(np.mean(latencies_arr))
    throughput_qps = float(n_requests / total_bench_duration)

    summary = {
        "total_requests": n_requests,
        "total_duration_sec": round(total_bench_duration, 3),
        "throughput_qps": round(throughput_qps, 1),
        "mean_latency_ms": round(mean_lat, 3),
        "p50_latency_ms": round(p50, 3),
        "p90_latency_ms": round(p90, 3),
        "p95_latency_ms": round(p95, 3),
        "p99_latency_ms": round(p99, 3),
        "min_latency_ms": round(float(np.min(latencies_arr)), 3),
        "max_latency_ms": round(float(np.max(latencies_arr)), 3),
        "sla_met_sub_15ms": p95 < 15.0
    }

    out_file = METRICS_DIR / "latency_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("FASTAPI REAL-TIME INFERENCE LATENCY SLA BENCHMARK")
    print("=" * 60)
    print(f"Requests Evaluated : {n_requests:,}")
    print(f"Throughput         : {throughput_qps:,.1f} QPS")
    print(f"Median (p50)       : {p50:.3f} ms")
    print(f"95th Percentile    : {p95:.3f} ms")
    print(f"99th Percentile    : {p99:.3f} ms")
    print(f"Sub-15ms SLA Met   : {'PASS (EXCEEDED)' if summary['sla_met_sub_15ms'] else 'FAIL'}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_latency_benchmark(1000)
