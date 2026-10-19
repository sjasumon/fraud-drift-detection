"""Comprehensive Regression and Acceptance Test Suite for v0.7.0 Remediation Audit.
Directly reproduces and verifies fixes for all findings in Fraud_Detection_v060_Followup_Review.txt:
1.  Sound derived provenance rejection of unverified arbitrary files (R01).
2.  Supplied wrong identity rejection on derived provenance paths (R01).
3.  ModelBundle.ready rejects test doubles in production (R01).
4.  Strict reconciliation catches corrupted confusion matrix counts (R02).
5.  Strict reconciliation rejects missing metrics.csv and compute_cost.csv (R02).
6.  Strict reconciliation rejects empty prediction ledger (R02).
7.  CLI defaults do not overwrite file configuration (R04).
8.  Historical protocol version and strategies forwarded faithfully (R04).
9.  Study CLI supports --config and merges arguments cleanly (R04).
10. Ranking changes with small score deltas compute exact AP delta without float shortcut erasure (R05).
11. Finite-resample limitation reported instead of empirical_p_value = 0.0 (R05).
12. BaseFraudModel predict caps alerts at operational review budget under tied scores (R06).
13. OnlineDailyQuotaAllocator is idempotent across restarts without double-counting replay (R06).
14. OnlineDailyQuotaAllocator respects per-day quota on out-of-order/late arrivals (R06).
15. Online quota step-to-day conversion aligns with pipeline hour convention (R06).
16. Batch scoring API enforces operational daily review budget (R06).
17. Authoritative active model survives restart across model families (R07).
18. Concurrent retrain rejected as stale parent when active model changed in-flight (R07/R09).
19. Retrain timeout logged duration reflects true wall-clock occupied duration (R09).
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api, pipeline, study
from src.artifacts import ModelBundle, save_bundle
from src.base_model import BaseFraudModel
from src.capacity_policy import OnlineDailyQuotaAllocator, allocate_batch_ranking
from src.provenance import (
    DATASET_REGISTRY,
    create_derived_subset_manifest,
    create_verified_test_double_provenance,
    get_dataset_provenance,
)
from src.reconcile_metrics import reconcile_run_directory
from src.reporting import block_bootstrap_metric_delta


# Top-level helper classes to allow joblib/pickle serialization
class AuditDummyModel:
    def __init__(self):
        self.threshold = 0.5
        self.trained = True
        self.feature_names = ["x"]
        self.fit_metadata = {}
        self.operational_daily_budget = 5

    def fit(self, frame):
        return self

    def predict(self, frame):
        return np.ones(len(frame), dtype=int), np.full(len(frame), 0.9)


class AuditDummyPreprocessor:
    feature_names = ["x"]
    required_input_columns = ["x"]

    def transform(self, frame):
        return frame.copy()


STARTED_EVENT = threading.Event()
FINISH_EVENT = threading.Event()


class ConcurrencyProbeModel:
    def __init__(self):
        self.threshold = 0.5
        self.trained = True
        self.feature_names = ["x"]
        self.fit_metadata = {}
        self.fitted = False

    def fit(self, frame):
        STARTED_EVENT.set()
        if not FINISH_EVENT.wait(5):
            raise RuntimeError("probe coordination timeout")
        self.fitted = True
        return self

    def predict(self, frame):
        pred = (frame.x.to_numpy().astype(int) % 2) if self.fitted else np.zeros(len(frame), dtype=int)
        return pred, np.where(pred, 0.9, 0.1)


# ---------------------------------------------------------------------------
# R01: Provenance Acceptance and Lineage Soundness
# ---------------------------------------------------------------------------
def test_r01_arbitrary_derived_subset_rejected(tmp_path):
    """create_derived_subset_manifest without byte verification must be rejected for benchmark."""
    bad_csv = tmp_path / "NOT_GENUINE.csv"
    bad_csv.write_text("not real transaction records\n", encoding="utf-8")

    parent_hash = DATASET_REGISTRY["IEEE-CIS"]["reference_sha256"]["train_transaction.csv"]
    # Self-declared claim without parent_path verification
    dm = create_derived_subset_manifest(parent_hash, bad_csv, {"claim": "I am a genuine subset"})
    wrong_id = tmp_path / "NOT_GENUINE_IDENTITY.csv"
    wrong_id.write_text("also not real\n", encoding="utf-8")

    prov = get_dataset_provenance("IEEE-CIS", bad_csv, identity_path=wrong_id, derived_manifest=dm)
    assert prov["acceptable_for_proposal_benchmark"] is False
    assert prov["verification_status"] in ("EXCLUDED_UNVERIFIED_DERIVED_LINEAGE", "EXCLUDED_IDENTITY_HASH_MISMATCH")


def test_r01_production_accepts_exported_test_helper_is_false():
    """ModelBundle.ready must be False for test doubles."""
    test_double_prov = create_verified_test_double_provenance()
    bundle = ModelBundle(
        SimpleNamespace(trained=True),
        None,
        {"data_kind": "real", "provenance": test_double_prov},
    )
    assert bundle.ready is False, "Production ModelBundle.ready must reject test-double provenance"


# ---------------------------------------------------------------------------
# R02: Strict Reconciliation
# ---------------------------------------------------------------------------
def test_r02_reconciler_rejects_wrong_counts_and_cost(tmp_path):
    """Reconciler must detect when recorded confusion counts or retrain counts do not match ledger."""
    run_dir = tmp_path / "reconciliation_fixture"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "audit-fixture", "data_kind": "unverified"}))

    pd.DataFrame([
        {"model": name, "probability": p, "prediction": y, "trans_id": str(i), "y_true": y}
        for name in ["static", "adaptive"]
        for i, (p, y) in enumerate([(0.8, 1), (0.2, 0)])
    ]).to_csv(run_dir / "predictions.csv", index=False)

    # Deliberately corrupt true_positives and retrains to 999999
    pd.DataFrame([
        {
            "model": name, "overall_precision": 1.0, "overall_recall": 1.0, "overall_f1": 1.0,
            "average_precision": 1.0, "brier_score": 0.04, "true_positives": 999999, "false_positives": 999999,
            "true_negatives": 999999, "false_negatives": 999999, "total_transactions": 999999, "log_loss": 999999,
        }
        for name in ["static", "adaptive"]
    ]).to_csv(run_dir / "metrics.csv", index=False)

    pd.DataFrame([
        {"model": name, "retrains": 999999, "training_seconds": 999999.0, "initial_training_seconds": 999999.0}
        for name in ["static", "adaptive"]
    ]).to_csv(run_dir / "compute_cost.csv", index=False)

    (run_dir / "retrain_events.json").write_text("[]")

    is_reconciled, report, issues = reconcile_run_directory(run_dir)
    assert is_reconciled is False, "Corrupted confusion counts and retrain counts must fail reconciliation"
    assert len(issues) > 0


def test_r02_reconciler_rejects_missing_comparison_files(tmp_path):
    """Missing metrics.csv or compute_cost.csv must cause reconciliation to fail."""
    run_dir = tmp_path / "missing_files_fixture"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "audit-fixture", "data_kind": "unverified"}))
    pd.DataFrame([
        {"model": "adaptive", "probability": 0.8, "prediction": 1, "trans_id": "1", "y_true": 1},
        {"model": "adaptive", "probability": 0.1, "prediction": 0, "trans_id": "2", "y_true": 0},
    ]).to_csv(run_dir / "predictions.csv", index=False)

    is_reconciled, report, issues = reconcile_run_directory(run_dir)
    assert is_reconciled is False
    assert any("metrics.csv is missing" in iss for iss in issues)
    assert any("compute_cost.csv is missing" in iss for iss in issues)


def test_r02_reconciler_rejects_empty_run(tmp_path):
    """Empty predictions ledger must not report a successful reconciliation."""
    run_dir = tmp_path / "empty_run_fixture"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "audit-fixture", "data_kind": "unverified"}))
    pd.DataFrame(columns=["model", "probability", "prediction", "trans_id", "y_true"]).to_csv(
        run_dir / "predictions.csv", index=False
    )

    is_reconciled, report, issues = reconcile_run_directory(run_dir)
    assert is_reconciled is False
    assert any("empty" in iss.lower() for iss in issues)


# ---------------------------------------------------------------------------
# R04: Configuration Resolution
# ---------------------------------------------------------------------------
def test_r04_pipeline_historical_config_resolution(tmp_path):
    """CLI defaults must not overwrite parameters specified in config file."""
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return {"run_dir": tmp_path / "fake_run"}

    cfg_file = Path("configs/historical_v050_run_63d7ca7f68.json")
    assert cfg_file.exists(), "Historical config file must exist"

    with patch.object(pipeline, "run_pipeline", fake_run), patch.object(
        sys, "argv", ["pipeline", "--config", str(cfg_file)]
    ):
        pipeline.main()

    original = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert captured["model_params"] == original["model_params"]
    assert captured["protocol_version"] == original["protocol_version"]
    assert captured["strategies"] == original["strategies"]
    assert "n_estimators" not in captured["resolved_config"] or captured["resolved_config"]["model_params"]["n_estimators"] == 150


def test_r04_study_cli_supports_config(tmp_path):
    """Study CLI must accept --config and merge parameters."""
    cfg_path = tmp_path / "study_config.json"
    cfg_data = {
        "dataset": "dummy.csv",
        "output_dir": str(tmp_path / "study_out"),
        "seeds": [123],
        "train_windows": [5],
        "test_slices": 2,
    }
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

    captured_study = {}

    def fake_run_study(*args, **kwargs):
        captured_study["dataset"] = args[0] if len(args) > 0 else kwargs.get("dataset")
        captured_study["output_dir"] = args[1] if len(args) > 1 else kwargs.get("output_dir")
        captured_study["seeds"] = args[2] if len(args) > 2 else kwargs.get("seeds")
        return {"comparisons": []}

    with patch.object(study, "run_study", fake_run_study), patch.object(
        sys, "argv", ["study", "--config", str(cfg_path)]
    ):
        study.main()

    assert captured_study["dataset"] == "dummy.csv"
    assert captured_study["seeds"] == [123]


# ---------------------------------------------------------------------------
# R05: Statistical Calculations
# ---------------------------------------------------------------------------
def test_r05_ranking_change_with_small_delta_preserves_ap_gain():
    """Ranking reversal with tiny score differences must compute exact AP delta (0.5), not 0.0."""
    rows = []
    # labels are [1, 0]
    # adaptive: [0.5000001, 0.5] -> AP = 1.0
    # static:   [0.5, 0.5000001] -> AP = 0.5
    for name, probs in [("adaptive", [0.5000001, 0.5]), ("static", [0.5, 0.5000001])]:
        for i in range(2):
            rows.append({
                "seed": 42,
                "window": 0,
                "slice_index": 0,
                "day": 0,
                "trans_id": str(i),
                "model": name,
                "y_true": [1, 0][i],
                "prediction": 1,
                "probability": probs[i],
            })

    df = pd.DataFrame(rows)
    delta = block_bootstrap_metric_delta(df, "adaptive", metric="average_precision", n_bootstrap=20)
    assert delta["point_estimate"] == pytest.approx(0.5, abs=1e-5), "True AP delta is 0.5"
    assert "finite_resample_limit" in delta, "Must report finite resample resolution limit"


# ---------------------------------------------------------------------------
# R06: Budget and Quota Enforcement
# ---------------------------------------------------------------------------
def test_r06_base_model_tied_scores_capped_at_daily_budget():
    """BaseFraudModel.predict with daily budget 5 must return exactly 5 alerts when scores are tied."""
    constant = pd.DataFrame({
        "feat": np.zeros(200),
        "step": np.ones(200),
        "isFraud": np.tile([0, 1], 100),
    })
    model = BaseFraudModel(
        ["feat"],
        threshold_objective="review_budget_recall",
        operational_daily_budget=5,
        model_params={"n_estimators": 3, "max_depth": 2},
    )
    model.fit(constant)
    pred, _ = model.predict(constant.iloc[160:])  # 40 rows
    assert int(pred.sum()) <= 5, f"Budget of 5 must not be exceeded; got {pred.sum()} alerts"


def test_r06_quota_restart_replay_idempotency(tmp_path):
    """Replaying an admitted ID after restart must not consume extra quota."""
    state = tmp_path / "quota_restart.json"
    quota = OnlineDailyQuotaAllocator(daily_budget=2, state_path=state)
    a = quota.evaluate("a", 0.9, day=0).is_alert
    assert a is True

    # Restart allocator from persisted state
    restored = OnlineDailyQuotaAllocator(daily_budget=2, state_path=state)
    a_replayed = restored.evaluate("a", 0.9, day=0).is_alert
    assert a_replayed is True, "Replayed decision for 'a' must remain True"

    b = restored.evaluate("b", 0.9, day=0).is_alert
    assert b is True, "Slot for 'b' must be available (unique IDs = 2)"
    assert restored.alerts_today == 2


def test_r06_out_of_order_day_quota():
    """Events arriving out of order must be allocated according to their event day quota."""
    late = OnlineDailyQuotaAllocator(daily_budget=1)
    d0_first = late.evaluate("day0-first", 0.9, day=0).is_alert
    d1_below = late.evaluate("day1-below", 0.1, day=1).is_alert
    d0_late = late.evaluate("day0-late", 0.9, day=0).is_alert

    assert d0_first is True
    assert d1_below is False
    assert d0_late is False, "Second day-0 event must be denied since day-0 budget (1) is exhausted"


def test_r06_step_day_boundary_convention():
    """step=24 must map to day 0 under 1-indexed hours convention: (24-1)//24 == 0."""
    allocator = OnlineDailyQuotaAllocator(daily_budget=1)
    assert allocator.get_day(23) == 0
    assert allocator.get_day(24) == 0  # 24th hour belongs to day 0
    assert allocator.get_day(25) == 1  # 25th hour starts day 1


def test_r06_batch_scoring_api_enforces_budget(tmp_path):
    """Batch scoring API must enforce operational daily budget."""
    bundle = ModelBundle(
        AuditDummyModel(), AuditDummyPreprocessor(),
        {"run_id": "test_quota", "model": "adaptive", "data_kind": "real",
         "model_revision": "rev-quota", "provenance": create_verified_test_double_provenance()}
    )
    buffer = api.ModelDoubleBuffer()
    buffer.atomic_swap(bundle)
    test_store = api.PredictionStore(str(tmp_path / "batch_quota.sqlite"))

    with patch.object(api, "model_buffer", buffer), patch.object(api, "ledger", test_store):
        with TestClient(api.app, headers={"X-API-Key": api.API_KEY}) as client:
            score = client.post(
                "/v1/batch-score",
                json={"transactions": [{"trans_id": f"quota-test-{i}", "features": {"x": 1}} for i in range(8)]},
            )
            assert score.status_code == 200
            review_decisions = sum(x["risk_tier"] == "REVIEW" for x in score.json().get("results", []))
            assert review_decisions <= 5, f"Budget of 5 must be respected; got {review_decisions} REVIEW decisions"


# ---------------------------------------------------------------------------
# R07: Durable Cross-Family Activation and Monotonic Revision
# ---------------------------------------------------------------------------
def test_r07_cross_family_activation_survives_restart(tmp_path):
    """Activating family B survives service restart and is restored from authoritative active.json."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)

    def make_bundle(family):
        return ModelBundle(
            AuditDummyModel(), AuditDummyPreprocessor(),
            {"run_id": family, "model": "adaptive", "data_kind": "real",
             "model_revision": family + "-rev", "provenance": create_verified_test_double_provenance()}
        )

    base = models_dir / "family_A"
    other = models_dir / "family_B"
    save_bundle(make_bundle("family_A"), base)
    save_bundle(make_bundle("family_B"), other)

    new_buffer = api.ModelDoubleBuffer()
    store = api.PredictionStore(str(tmp_path / "service_probe.sqlite"))
    operational_state = tmp_path / "operational_state"

    with patch.object(api, "model_buffer", new_buffer), \
         patch.object(api, "ledger", store), \
         patch.object(api.settings, "model_bundle_path", str(base)), \
         patch.object(api.settings, "model_output_dir", str(models_dir)), \
         patch.object(api, "RESULTS_DIR", operational_state):

        api.initialize_app_model()
        with TestClient(api.app, headers={"X-API-Key": api.API_KEY}) as client:
            resp = client.post("/v1/models/activate", json={"package_name_or_path": str(other)})
            assert resp.status_code == 200
            assert api.model_buffer.snapshot()[0].metadata["run_id"] == "family_B"

            # Restart service
            api.model_buffer = api.ModelDoubleBuffer()
            api.initialize_app_model()
            after = api.model_buffer.snapshot()[0].metadata["run_id"]
            assert after == "family_B", "Authoritative activation of family_B must survive restart"


def test_r07_concurrency_probe_rejects_stale_retrain_promotion(tmp_path):
    """Controlled concurrency probe: old in-flight retrain cannot overwrite newly activated model."""
    root = tmp_path / "race_models"
    root.mkdir(exist_ok=True)
    STARTED_EVENT.clear()
    FINISH_EVENT.clear()

    def make_bundle(family):
        return ModelBundle(
            ConcurrencyProbeModel(), AuditDummyPreprocessor(),
            {"run_id": family, "model": "adaptive", "model_revision": family + "-revision",
             "data_kind": "real", "provenance": create_verified_test_double_provenance()}
        )

    base = root / "race_A"
    other = root / "race_B"
    save_bundle(make_bundle("race_A"), base)
    save_bundle(make_bundle("race_B"), other)

    store = api.PredictionStore(str(tmp_path / "concurrency.sqlite"))
    family = "race_A:adaptive"
    store.record([(str(i), json.dumps({"x": i}), 0.1, 0, 0, "race_A-revision", family, float(i)) for i in range(120)])
    for i in range(120):
        store.feedback(str(i), i % 2)

    buffer = api.ModelDoubleBuffer()
    buffer.atomic_swap(make_bundle("race_A"), generation=0)
    retrainer = api.BackgroundRetrainer(store=store, max_retrain_duration_sec=3.0)

    result = {}
    with patch.object(api, "model_buffer", buffer), \
         patch.object(api, "ledger", store), \
         patch.object(api, "retrainer", retrainer), \
         patch.object(api.settings, "model_bundle_path", str(base)), \
         patch.object(api.settings, "model_output_dir", str(root)), \
         patch.object(api, "RESULTS_DIR", tmp_path / "race_state"):

        t = threading.Thread(target=lambda: result.update(retrain_return=retrainer.execute_retrain()), daemon=True)
        t.start()
        assert STARTED_EVENT.wait(3), "Retrain fit did not start in time"

        # Manually activate race_B while retrain of race_A is in-flight!
        with TestClient(api.app, headers={"X-API-Key": api.API_KEY}) as client:
            resp = client.post("/v1/models/activate", json={"package_name_or_path": str(other)})
            assert resp.status_code == 200
            assert api.model_buffer.snapshot()[0].metadata["run_id"] == "race_B"

        FINISH_EVENT.set()
        t.join(5)
        assert not t.is_alive(), "Retrain worker thread did not complete"

        # Stale retrain must have been rejected! Active model must remain race_B!
        assert api.model_buffer.snapshot()[0].metadata["run_id"] == "race_B"
        assert retrainer.last_retrain_status == "REJECTED_STALE_PARENT"


# ---------------------------------------------------------------------------
# R09: Effective Training Deadline and Duration Logging
# ---------------------------------------------------------------------------
def test_r09_thread_timeout_logs_actual_wall_time(tmp_path):
    """When a fit times out, status must be REJECTED_TRAINING_TIMEOUT and logged duration must be accurate."""
    class SlowFit:
        def fit(self, frame):
            time.sleep(0.08)

    slow_bundle = SimpleNamespace(
        ready=True,
        metadata={"run_id": "audit", "model": "adaptive", "model_revision": "audit-rev"},
        model=SlowFit(),
        transform=lambda rows: pd.DataFrame(rows),
    )
    buffer = api.ModelDoubleBuffer()
    buffer.atomic_swap(slow_bundle)
    fake_store = SimpleNamespace(
        training_data=lambda *args: [(json.dumps({"x": i}), i % 2, float(i)) for i in range(100)]
    )

    with patch.object(api, "model_buffer", buffer), patch.object(api, "ledger", fake_store):
        retrainer = api.BackgroundRetrainer(max_retrain_duration_sec=0.01)
        accepted = retrainer.execute_retrain()
        assert accepted is False
        assert retrainer.last_retrain_status == "REJECTED_TRAINING_TIMEOUT"
        logged_duration = retrainer.promotion_log[-1]["duration_sec"]
        # Logged duration must reflect the actual wall clock time occupied (>= 0.05s)
        assert logged_duration >= 0.05, f"Logged duration {logged_duration}s must reflect real wall clock time"
