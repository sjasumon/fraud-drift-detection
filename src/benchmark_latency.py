"""Benchmark authenticated successful inference using a loaded real model package.
Usage: MODEL_BUNDLE_PATH=... python -m src.benchmark_latency --features example.json
This measures in-process HTTP handling, not production network/concurrent load.
"""

import argparse
import json
import time
import uuid
import numpy as np
from fastapi.testclient import TestClient
from src.api import app
from src.config import API_KEY


def run_latency_benchmark(features, n_requests=1000):
    if n_requests < 1:
        raise ValueError("n_requests must be positive")
    client = TestClient(app, headers={"X-API-Key": API_KEY})
    if not client.get("/health").json()["is_ready"]:
        raise RuntimeError("Load a real-data model package before benchmarking")
    values = []
    start = time.perf_counter()
    for i in range(n_requests + 20):
        t = time.perf_counter()
        response = client.post(
            "/v1/score",
            json={"trans_id": "LATENCY_" + uuid.uuid4().hex, "features": features},
        )
        response.raise_for_status()
        assert "fraud_probability" in response.json()
        if i >= 20:
            values.append((time.perf_counter() - t) * 1000)
    return {
        "requests": n_requests,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "scope": "authenticated in-process requests, including durable ledger writes",
        "elapsed_seconds_including_warmup": time.perf_counter() - start,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--requests", type=int, default=1000)
    a = p.parse_args()
    print(
        json.dumps(
            run_latency_benchmark(json.load(open(a.features)), a.requests), indent=2
        )
    )


if __name__ == "__main__":
    main()
