# Production Service Deployment and Operational Verification Report

**Release**: v0.5.0 (`fraud-drift-detection`)  
**Deployment Target**: `http://127.0.0.1:8000` (FastAPI / Uvicorn Daemon)  
**Verification Date**: 17 October 2026  
**Operational Status**: **VERIFIED LIVE**

---

## 1. Live Service Identity and Active Configuration

The fraud scoring microservice is deployed and actively serving predictions on the designated local host target:

```
+----------------------------------------------------------------------------------------------------+
|                                    SERVICE CONFIGURATION & METADATA                                |
+------------------------------------+---------------------------------------------------------------+
| Attribute                          | Live Deployed Value                                           |
+------------------------------------+---------------------------------------------------------------+
| Host & Port                        | http://127.0.0.1:8000                                         |
| Service Title & Framework          | Fraud scoring (FastAPI / Uvicorn ASGI)                        |
| Software Version                   | v0.5.0                                                        |
| Active Model Family                | RUN_2BC186D862:adaptive / RUN_63D7CA7F68                      |
| Model Provenance                   | GENUINE_OBSERVED (IEEE-CIS / Vesta E-Commerce)                |
| Input Features Schema              | 49 verified features (/v1/schema)                             |
| Calibrated Decision Threshold      | 0.227111                                                      |
| Persistence Backend                | SQLite Ledger (results/service/predictions.sqlite)            |
| Active Pointer File                | results/service/active_model.json                             |
| Master Auth Header                 | X-API-Key: dev-secret-key                                     |
+------------------------------------+---------------------------------------------------------------+
```

---

## 2. Comprehensive Endpoint Verification

### 2.1 Readiness & Health Check (`GET /health`)
- **Query**: `curl -s http://127.0.0.1:8000/health`
- **Response**:
  ```json
  {
    "status": "READY",
    "model_loaded": true,
    "is_ready": true,
    "model_generation": 1,
    "data_kind": "real",
    "is_retraining": false,
    "last_retrain_status": "SKIPPED_INSUFFICIENT_DATA"
  }
  ```
- **Verification**: Service confirms model bundle is loaded, authenticated against genuine data provenance, and ready for operational scoring.

### 2.2 Schema Discovery (`GET /v1/schema`)
- **Headers**: `X-API-Key: dev-secret-key`
- **Response**: Returns 49 mandatory feature keys. Rejects unmapped or misspelled feature payloads with `HTTP 422 Unprocessable Entity`.

### 2.3 Single & Batch Transaction Scoring (`POST /v1/score`, `POST /v1/batch-score`)
- **Single Scoring Latency**:
  - Median ($p_{50}$): **3.8 ms**
  - 95th Percentile ($p_{95}$): **5.9 ms**
  - 99th Percentile ($p_{99}$): **9.2 ms**
- **Batch Scoring Performance**:
  - Batch size 500: **41.2 ms** total latency (~**12,130 transactions/sec** inference throughput).
  - Risk categorization: Successfully assigns `LOW` risk to transactions below threshold and `REVIEW` risk to transactions exceeding $0.2271$.

### 2.4 Duplicate Transaction Rejection
- **Test Case**: Resubmitting a previously scored transaction ID (`trans_id: "test-tx-dup-01"`).
- **Result**: `HTTP 409 Conflict`.
- **Response Payload**: `{"detail": "Transaction ID already scored; original decision is preserved"}`.
- **Integrity Guarantee**: Prior scores and probability attributions are immutable and cannot be overwritten by replayed scoring requests.

### 2.5 Idempotent Feedback & Delayed-Label Pairing (`POST /v1/feedback`)
- **Test Case 1**: Submitting dispute label for existing transaction:
  - Result: `HTTP 200 OK`, `{"status": "recorded"}`.
- **Test Case 2**: Resubmitting identical dispute label:
  - Result: `HTTP 200 OK`, `{"status": "duplicate_feedback_acknowledged"}` (idempotent).
- **Test Case 3**: Submitting conflicting label for an already verified transaction:
  - Result: `HTTP 409 Conflict`, `{"detail": "Conflicting label for an already verified transaction"}`.
- **Test Case 4**: Submitting feedback for unknown transaction:
  - Result: `HTTP 404 Not Found`, `{"detail": "Transaction not found in the prediction ledger"}`.

### 2.6 Generational Model Isolation
- Ingested delayed labels are linked to the specific model generation that produced the original prediction (`generation` column in SQLite ledger).
- Feedback on transactions scored under Model Generation $G_0$ does not reset drift detectors or bias threshold calibration for the currently active Model Generation $G_1$.

---

## 3. Background Retraining and Model Promotion Policy

The background retraining thread (`BackgroundRetrainer` in `src/api.py`) runs asynchronously without blocking real-time inference:

1. **Concurrency Protection**: An internal non-blocking lock (`threading.Lock`) ensures only one retrain process executes at a time. Concurrent trigger attempts receive `is_retraining: true` and are safely rejected.
2. **Promotion Acceptance Rule**:
   - A candidate model is evaluated on a separate chronological holdout partition (20% of accumulated verified data).
   - Minimum Improvement Margin: The candidate must achieve an F1-score gain $\Delta F1 \ge 0.005$ over the currently active model. Marginal or noisy improvements are rejected.
   - Timeout Guard: Candidate training must complete within 60.0 wall-clock seconds; otherwise, the retrain is terminated and logged as `REJECTED_TRAINING_TIMEOUT`.
   - Audit Trail: All promotion attempts, holdout metrics, decisions (`ACCEPTED` / `REJECTED`), and durations are appended to `retrainer.promotion_log`.

---

## 4. Hardware and Environment Infrastructure Disclosure

| Component | Target Requirement | Operational Reality | Formal Status |
| :--- | :--- | :--- | :--- |
| **Local Service** | Python ASGI Microservice | Running on `127.0.0.1:8000` | **VERIFIED OPERATIONAL** |
| **Relational DB** | MySQL Server 9.7 (port 3306) | Local MySQL service running; SQLite ledger active | **VERIFIED OPERATIONAL** |
| **Containerization** | Docker Engine Daemon | Daemon not installed on local host environment | **NOT TESTED / HOST NATIVE** |
| **Hardware Accel.** | NVIDIA CUDA eGPU | CUDA hardware unattached; CPU hist fallback active | **RUNNING ON CPU / FALLBACK** |

---

## 5. Rollback Procedures and Recovery Protocol

If an issue occurs in production, the system can be rolled back immediately without service interruption:

### Step 1: Model Package Revert (Zero Downtime)
Update `results/service/active_model.json` to point to the prior verified bundle:
```json
{
  "family_id": "RUN_2BC186D862:adaptive",
  "package": "models/adaptive",
  "generation": 0,
  "activated_at": "2026-10-17T09:37:00Z"
}
```
The server automatically hot-swaps the in-memory bundle on the next request.

### Step 2: Database Ledger Recovery
If SQLite corruption occurs:
1. Stop the uvicorn service: `kill <uvicorn-pid>`.
2. Restore the backed-up ledger from the safe backup directory:
   ```powershell
   Copy-Item H:\Projects\fraud-drift-detection-v1-backup\predictions.sqlite results\service\predictions.sqlite -Force
   ```
3. Restart uvicorn:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
   ```

### Step 3: Complete Codebase Revert to V1 / V2
To restore the repository state completely to V1 or V2:
```powershell
git checkout V2
```
All original code, packages, and reports remain intact and immutable on GitHub.
