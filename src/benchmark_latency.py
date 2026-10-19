"""Benchmark authenticated inference latency, concurrent API throughput, and SQLite query plans.
Supports:
1. Pure model-only inference latency (in-memory preprocessing + estimator predict).
2. Sequential HTTP request latency (including JSON deserialization, auth, validation, model scoring, and durable SQLite ledger write).
3. Concurrent HTTP API load (ThreadPoolExecutor client measuring QPS, p50/p95/p99 latency, and error rate).
4. SQLite EXPLAIN QUERY PLAN profiling on real operational queries.

Usage:
    python -m src.benchmark_latency --features example.json --requests 500 --concurrency 4
"""

import argparse
import concurrent.futures
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi.testclient import TestClient

from src.api import app, model_buffer, ledger
from src.config import API_KEY, RESULTS_DIR


def profile_sqlite_query_plans(db_path: Path) -> List[Dict[str, Any]]:
    """Runs EXPLAIN QUERY PLAN on standard operational queries in the prediction ledger."""
    queries = [
        (
            "Transaction ID Lookup (Score / Feedback check)",
            "EXPLAIN QUERY PLAN SELECT prediction, label, generation, model_id, event_time FROM predictions WHERE id = ?",
            ("TX_BENCHMARK_LOOKUP",),
        ),
        (
            "Training Data Extraction (Holdout / Retraining query)",
            "EXPLAIN QUERY PLAN SELECT features, label, event_time FROM predictions WHERE label IS NOT NULL AND family_id = ? ORDER BY event_time DESC, rowid DESC LIMIT ?",
            ("RUN_DEFAULT:xgboost", 1000),
        ),
        (
            "Promotion History Query (GET /v1/models/promotions)",
            "EXPLAIN QUERY PLAN SELECT attempt_id, timestamp, parent_generation, candidate_generation, evidence_count, gain, status, decision FROM promotion_history ORDER BY timestamp DESC LIMIT ?",
            (50,),
        ),
        (
            "Feedback Audit Query (GET /v1/feedback/audit)",
            "EXPLAIN QUERY PLAN SELECT event_id, trans_id, label, label_source, received_at, outcome FROM feedback_events ORDER BY rowid DESC LIMIT ?",
            (100,),
        ),
    ]

    results = []
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        for name, sql, params in queries:
            cursor.execute(sql, params)
            plan_rows = cursor.fetchall()
            # SQLite EXPLAIN QUERY PLAN returns: (id, parent, notused, detail)
            details = [row[3] for row in plan_rows]
            results.append({
                "operation": name,
                "sql": sql.replace("EXPLAIN QUERY PLAN ", ""),
                "plan_detail": " | ".join(details),
                "uses_index": any("INDEX" in d.upper() or "PRIMARY KEY" in d.upper() for d in details),
            })
    finally:
        conn.close()
    return results


def run_latency_benchmark(
    features: Dict[str, Any],
    n_requests: int = 500,
    concurrency: int = 4,
    warmup: int = 20,
) -> Dict[str, Any]:
    if n_requests < 1:
        raise ValueError("n_requests must be positive")

    client = TestClient(app, headers={"X-API-Key": API_KEY})
    health = client.get("/health").json()
    if not health.get("is_ready"):
        raise RuntimeError("Load a real-data model package before benchmarking")

    bundle, generation = model_buffer.snapshot()

    # -------------------------------------------------------------
    # 1. Pure Model-Only Inference (No HTTP, No SQLite ledger)
    # -------------------------------------------------------------
    model_latencies_ms = []
    for _ in range(warmup):
        bundle.predict([features])

    t0_model = time.perf_counter()
    for _ in range(n_requests):
        t_start = time.perf_counter()
        bundle.predict([features])
        model_latencies_ms.append((time.perf_counter() - t_start) * 1000.0)
    model_elapsed = time.perf_counter() - t0_model

    # -------------------------------------------------------------
    # 2. Sequential HTTP Scoring (In-process Client, Full Pipeline + SQLite)
    # -------------------------------------------------------------
    seq_latencies_ms = []
    for i in range(warmup):
        client.post(
            "/v1/score",
            json={"trans_id": f"WARMUP_SEQ_{i}", "features": features},
        )

    t0_seq = time.perf_counter()
    for i in range(n_requests):
        tx_id = f"LATENCY_SEQ_{uuid.uuid4().hex[:12]}"
        t_start = time.perf_counter()
        resp = client.post(
            "/v1/score",
            json={"trans_id": tx_id, "features": features},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP scoring failed: {resp.status_code} - {resp.text}")
        seq_latencies_ms.append((time.perf_counter() - t_start) * 1000.0)
    seq_elapsed = time.perf_counter() - t0_seq

    # -------------------------------------------------------------
    # 3. Concurrent HTTP API Traffic (Multi-threaded Client)
    # -------------------------------------------------------------
    concurrent_latencies_ms = []
    errors = 0

    def send_one(req_id: int):
        thread_client = TestClient(app, headers={"X-API-Key": API_KEY})
        tx_id = f"LATENCY_CONC_{uuid.uuid4().hex[:12]}_{req_id}"
        t_start = time.perf_counter()
        resp = thread_client.post(
            "/v1/score",
            json={"trans_id": tx_id, "features": features},
        )
        lat = (time.perf_counter() - t_start) * 1000.0
        return resp.status_code, lat

    t0_conc = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_one, i) for i in range(n_requests)]
        for f in concurrent.futures.as_completed(futures):
            code, lat = f.result()
            if code == 200:
                concurrent_latencies_ms.append(lat)
            else:
                errors += 1
    conc_elapsed = time.perf_counter() - t0_conc

    # -------------------------------------------------------------
    # 4. Profile SQLite Query Plans
    # -------------------------------------------------------------
    query_plans = profile_sqlite_query_plans(ledger.path)

    return {
        "scope": "Comprehensive latency, throughput, and query plan profile",
        "benchmark_parameters": {
            "requests": n_requests,
            "concurrency": concurrency,
            "warmup_requests": warmup,
            "model_version": health.get("model_version", "0.6.0"),
            "model_generation": generation,
        },
        "model_only_inference": {
            "description": "In-memory feature preprocessing and model.predict without HTTP or disk I/O",
            "p50_ms": float(np.percentile(model_latencies_ms, 50)),
            "p90_ms": float(np.percentile(model_latencies_ms, 90)),
            "p95_ms": float(np.percentile(model_latencies_ms, 95)),
            "p99_ms": float(np.percentile(model_latencies_ms, 99)),
            "mean_ms": float(np.mean(model_latencies_ms)),
            "total_elapsed_sec": float(model_elapsed),
            "effective_qps": float(n_requests / max(model_elapsed, 1e-6)),
        },
        "sequential_http": {
            "description": "Sequential HTTP calls including authentication, schema validation, and durable SQLite ledger writes",
            "p50_ms": float(np.percentile(seq_latencies_ms, 50)),
            "p90_ms": float(np.percentile(seq_latencies_ms, 90)),
            "p95_ms": float(np.percentile(seq_latencies_ms, 95)),
            "p99_ms": float(np.percentile(seq_latencies_ms, 99)),
            "mean_ms": float(np.mean(seq_latencies_ms)),
            "total_elapsed_sec": float(seq_elapsed),
            "effective_qps": float(n_requests / max(seq_elapsed, 1e-6)),
        },
        "concurrent_api": {
            "description": f"Concurrent HTTP calls with concurrency={concurrency} including authentication and SQLite writes",
            "concurrency_level": concurrency,
            "successful_requests": len(concurrent_latencies_ms),
            "error_count": errors,
            "p50_ms": float(np.percentile(concurrent_latencies_ms, 50)) if concurrent_latencies_ms else 0.0,
            "p90_ms": float(np.percentile(concurrent_latencies_ms, 90)) if concurrent_latencies_ms else 0.0,
            "p95_ms": float(np.percentile(concurrent_latencies_ms, 95)) if concurrent_latencies_ms else 0.0,
            "p99_ms": float(np.percentile(concurrent_latencies_ms, 99)) if concurrent_latencies_ms else 0.0,
            "mean_ms": float(np.mean(concurrent_latencies_ms)) if concurrent_latencies_ms else 0.0,
            "total_elapsed_sec": float(conc_elapsed),
            "throughput_qps": float(len(concurrent_latencies_ms) / max(conc_elapsed, 1e-6)),
        },
        "sqlite_query_plans": query_plans,
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark model scoring and query plans")
    p.add_argument("--features", required=True, help="Path to JSON file with transaction features")
    p.add_argument("--requests", type=int, default=500, help="Number of benchmark iterations")
    p.add_argument("--concurrency", type=int, default=4, help="Concurrency level for load benchmark")
    p.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    args = p.parse_args()

    with open(args.features, "r", encoding="utf-8") as f:
        features = json.load(f)

    result = run_latency_benchmark(
        features=features,
        n_requests=args.requests,
        concurrency=args.concurrency,
    )

    out_str = json.dumps(result, indent=2)
    print(out_str)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_str, encoding="utf-8")


if __name__ == "__main__":
    main()
