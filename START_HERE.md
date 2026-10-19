# Start Here — Release v0.6.0 (Remediation Tag V4)

1. Read [VALIDATION.md](VALIDATION.md) for the complete test verification report (177 tests passed, 0 failed).
2. Read [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) and [PROPOSAL_RESULTS.md](PROPOSAL_RESULTS.md) for the reconciled benchmark on `RUN_63D7CA7F68` and hypothesis evaluations ($H_1$ Supported, $H_2$ Not Supported / Rejected in Protocol v1, $H_3$ Supported).
3. Read [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) for the literature review (Dal Pozzolo et al., Carminati et al. BankSealer), feedback queue latency limitations, and experimental protocol.
4. Run metric reconciliation on the verified benchmark:
   ```bash
   python -m src.reconcile_metrics --run-dir results/benchmark_runs/RUN_63D7CA7F68 --bootstrap-intervals
   ```
5. Run the complete automated test suite:
   ```bash
   pytest tests/ -v
   ```
6. Start the operational FastAPI scoring microservice:
   ```bash
   uvicorn src.api:app --host 127.0.0.1 --port 8000
   ```
7. Microservice endpoints available:
   - `GET /health`: Health, model readiness, and generation status.
   - `GET /v1/schema`: Required input schema and threshold metadata.
   - `POST /v1/score`: Single transaction scoring with durable SQLite ledger write.
   - `POST /v1/batch-score`: Batch scoring with throughput metrics.
   - `POST /v1/feedback`: Feedback ingestion recording `received_at`, `confirmed_at`, `label_source`, and conflict rejections.
   - `GET /v1/feedback/audit`: Append-only feedback audit log.
   - `GET /v1/models/promotions`: Persistent promotion and rollback history.
   - `POST /v1/models/activate`: Model activation with cryptographic provenance checks.
   - `POST /v1/models/rollback`: Authenticated generation rollback and state revert.
   - `POST /v1/retrain`: Chronological holdout retraining with bounded worker timeout.

Installable wheel included in `dist/fraud_drift_detection-0.6.0-py3-none-any.whl`.
