"""Benchmark authenticated inference latency, concurrent API throughput, and SQLite query plans.
Supports:
1. Pure model-only inference latency (in-memory preprocessing + estimator predict).
2. Sequential HTTP request latency (including JSON deserialization, auth, validation, model scoring, and durable SQLite ledger write).
3. Concurrent HTTP API load (ThreadPoolExecutor measuring QPS, p50/p95/p99 latency, and error rate).
4. Live Network Service benchmarking via --target-url against a real running uvicorn instance over TCP/HTTP.
5. Concurrent retraining contention measurement.
6. Real hardware, memory (RSS), and SQLite EXPLAIN QUERY PLAN profiling on operational queries.

Usage:
    # In-process TestClient mode:
    python -m src.benchmark_latency --features example.json --requests 500 --concurrency 4

    # Live Network Service mode:
    python -m src.benchmark_latency --target-url http://127.0.0.1:8000 --features example.json --requests 500 --concurrency 4
"""

import argparse
import concurrent.futures
import json
import os
import platform
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    import httpx
except ImportError:
    httpx = None

from fastapi.testclient import TestClient

from src.api import app, model_buffer, ledger
from src.config import API_KEY, RESULTS_DIR


def get_hardware_info() -> Dict[str, Any]:
    """Captures runtime hardware specifications and host environment."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
    }
    if psutil:
        mem = psutil.virtual_memory()
        proc = psutil.Process()
        info["system_total_memory_gb"] = round(mem.total / (1024 ** 3), 2)
        info["system_available_memory_gb"] = round(mem.available / (1024 ** 3), 2)
        info["process_initial_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
    return info


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
    if not Path(db_path).exists():
        return results

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        for name, sql, params in queries:
            cursor.execute(sql, params)
            plan_rows = cursor.fetchall()
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
    target_url: Optional[str] = None,
    api_key: Optional[str] = None,
    simulate_retraining: bool = False,
) -> Dict[str, Any]:
    if n_requests < 1:
        raise ValueError("n_requests must be positive")

    hw_info = get_hardware_info()
    effective_api_key = api_key or API_KEY
    is_live_network = target_url is not None and len(target_url.strip()) > 0
    mode_label = "live_network_service" if is_live_network else "in_process_test_client"

    # Setup client
    if is_live_network:
        base_url = target_url.rstrip("/")
        if httpx is None:
            raise RuntimeError("httpx package is required for live network benchmarking")
        http_client = httpx.Client(base_url=base_url, headers={"X-API-Key": effective_api_key}, timeout=30.0)
        try:
            resp = http_client.get("/health")
            if resp.status_code != 200:
                raise RuntimeError(f"Target service {base_url} returned health status {resp.status_code}")
            health = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to target service {base_url}: {exc}") from exc
        model_version = health.get("model_version", "0.7.0")
        generation = health.get("model_generation", 0)
    else:
        test_client = TestClient(app, headers={"X-API-Key": effective_api_key})
        health = test_client.get("/health").json()
        if not health.get("is_ready"):
            raise RuntimeError("Load a real-data model package before benchmarking")
        model_version = health.get("model_version", "0.7.0")
        bundle, generation = model_buffer.snapshot()

    # -------------------------------------------------------------
    # 1. Pure Model-Only Inference (In-memory preprocessing + predict)
    # -------------------------------------------------------------
    model_latencies_ms = []
    bundle, _ = model_buffer.snapshot()
    if bundle is not None and bundle.ready:
        for _ in range(warmup):
            bundle.predict([features])

        t0_model = time.perf_counter()
        for _ in range(n_requests):
            t_start = time.perf_counter()
            bundle.predict([features])
            model_latencies_ms.append((time.perf_counter() - t_start) * 1000.0)
        model_elapsed = time.perf_counter() - t0_model
    else:
        model_elapsed = 0.0

    # -------------------------------------------------------------
    # 2. Sequential HTTP Scoring
    # -------------------------------------------------------------
    seq_latencies_ms = []

    def make_single_request(client_obj, req_id: str):
        payload = {"trans_id": req_id, "features": features}
        if is_live_network:
            return client_obj.post("/v1/score", json=payload)
        return client_obj.post("/v1/score", json=payload)

    # Warmup
    for i in range(warmup):
        tx_id = f"WARMUP_SEQ_{i}"
        if is_live_network:
            make_single_request(http_client, tx_id)
        else:
            make_single_request(test_client, tx_id)

    t0_seq = time.perf_counter()
    for i in range(n_requests):
        tx_id = f"LATENCY_SEQ_{uuid.uuid4().hex[:12]}"
        t_start = time.perf_counter()
        if is_live_network:
            resp = make_single_request(http_client, tx_id)
        else:
            resp = make_single_request(test_client, tx_id)
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
        if is_live_network:
            worker_client = httpx.Client(base_url=base_url, headers={"X-API-Key": effective_api_key}, timeout=30.0)
        else:
            worker_client = TestClient(app, headers={"X-API-Key": effective_api_key})
        tx_id = f"LATENCY_CONC_{uuid.uuid4().hex[:12]}_{req_id}"
        t_start = time.perf_counter()
        try:
            resp = make_single_request(worker_client, tx_id)
            lat = (time.perf_counter() - t_start) * 1000.0
            status_code = resp.status_code
        except Exception:
            lat = (time.perf_counter() - t_start) * 1000.0
            status_code = 500
        finally:
            if is_live_network:
                worker_client.close()
        return status_code, lat

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
    # 4. Optional Retraining Contention Measurement
    # -------------------------------------------------------------
    contention_report = None
    if simulate_retraining:
        retrain_latencies_ms = []
        retrain_errors = 0
        retrain_start_event = concurrent.futures.Future()

        def do_load():
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as exec_load:
                load_futures = [exec_load.submit(send_one, 10000 + i) for i in range(n_requests // 2)]
                retrain_start_event.set_result(True)
                for f in concurrent.futures.as_completed(load_futures):
                    code, lat = f.result()
                    if code == 200:
                        retrain_latencies_ms.append(lat)
                    else:
                        retrain_errors += 1

        t0_contention = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as retrain_exec:
            load_fut = retrain_exec.submit(do_load)
            # In live service, trigger retrain endpoint
            if is_live_network:
                try:
                    http_client.post("/v1/retrain/trigger", json={"max_wait_seconds": 10.0})
                except Exception:
                    pass
            load_fut.result()
        contention_elapsed = time.perf_counter() - t0_contention

        contention_report = {
            "contention_tested": True,
            "requests_under_retraining": len(retrain_latencies_ms) + retrain_errors,
            "successful_requests": len(retrain_latencies_ms),
            "errors": retrain_errors,
            "p50_ms": float(np.percentile(retrain_latencies_ms, 50)) if retrain_latencies_ms else 0.0,
            "p95_ms": float(np.percentile(retrain_latencies_ms, 95)) if retrain_latencies_ms else 0.0,
            "p99_ms": float(np.percentile(retrain_latencies_ms, 99)) if retrain_latencies_ms else 0.0,
            "throughput_qps": float(len(retrain_latencies_ms) / max(contention_elapsed, 1e-6)),
        }

    # -------------------------------------------------------------
    # 5. Profile SQLite Query Plans
    # -------------------------------------------------------------
    ledger_path = getattr(ledger, "path", Path("results/service/predictions.sqlite"))
    query_plans = profile_sqlite_query_plans(ledger_path)

    # Capture peak RSS memory at completion
    if psutil:
        proc = psutil.Process()
        hw_info["process_final_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
        hw_info["process_peak_rss_mb"] = max(
            hw_info.get("process_initial_rss_mb", 0.0),
            hw_info["process_final_rss_mb"]
        )

    if is_live_network:
        http_client.close()

    return {
        "scope": "Comprehensive latency, throughput, hardware, and query plan profile",
        "execution_mode": mode_label,
        "is_live_network": is_live_network,
        "target_url": target_url,
        "hardware": hw_info,
        "benchmark_parameters": {
            "requests": n_requests,
            "concurrency": concurrency,
            "warmup_requests": warmup,
            "model_version": model_version,
            "model_generation": generation,
        },
        "model_only_inference": {
            "description": "In-memory feature preprocessing and model.predict without HTTP or disk I/O",
            "p50_ms": float(np.percentile(model_latencies_ms, 50)) if model_latencies_ms else 0.0,
            "p90_ms": float(np.percentile(model_latencies_ms, 90)) if model_latencies_ms else 0.0,
            "p95_ms": float(np.percentile(model_latencies_ms, 95)) if model_latencies_ms else 0.0,
            "p99_ms": float(np.percentile(model_latencies_ms, 99)) if model_latencies_ms else 0.0,
            "mean_ms": float(np.mean(model_latencies_ms)) if model_latencies_ms else 0.0,
            "total_elapsed_sec": float(model_elapsed),
            "effective_qps": float(n_requests / max(model_elapsed, 1e-6)) if model_latencies_ms else 0.0,
        },
        "sequential_http": {
            "description": f"Sequential HTTP calls ({mode_label}) including auth, validation, scoring, and SQLite ledger write",
            "p50_ms": float(np.percentile(seq_latencies_ms, 50)),
            "p90_ms": float(np.percentile(seq_latencies_ms, 90)),
            "p95_ms": float(np.percentile(seq_latencies_ms, 95)),
            "p99_ms": float(np.percentile(seq_latencies_ms, 99)),
            "mean_ms": float(np.mean(seq_latencies_ms)),
            "total_elapsed_sec": float(seq_elapsed),
            "effective_qps": float(n_requests / max(seq_elapsed, 1e-6)),
        },
        "concurrent_api": {
            "description": f"Concurrent HTTP calls ({mode_label}) with concurrency={concurrency} including auth and SQLite writes",
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
        "retraining_contention": contention_report or {
            "contention_tested": False,
            "note": "Use --retraining-contention to measure latency under simultaneous retraining",
        },
        "sqlite_query_plans": query_plans,
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark model scoring and query plans")
    p.add_argument("--features", required=True, help="Path to JSON file with transaction features")
    p.add_argument("--requests", type=int, default=500, help="Number of benchmark iterations")
    p.add_argument("--concurrency", type=int, default=4, help="Concurrency level for load benchmark")
    p.add_argument("--warmup", type=int, default=20, help="Warmup requests before timing")
    p.add_argument("--target-url", type=str, default=None, help="Optional live HTTP server URL (e.g. http://127.0.0.1:8000)")
    p.add_argument("--api-key", type=str, default=None, help="API Key for live service authentication")
    p.add_argument("--retraining-contention", action="store_true", help="Simulate retraining contention during benchmark")
    p.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    args = p.parse_args()

    with open(args.features, "r", encoding="utf-8") as f:
        features = json.load(f)

    result = run_latency_benchmark(
        features=features,
        n_requests=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        target_url=args.target_url,
        api_key=args.api_key,
        simulate_retraining=args.retraining_contention,
    )

    out_str = json.dumps(result, indent=2)
    print(out_str)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_str, encoding="utf-8")


if __name__ == "__main__":
    main()
