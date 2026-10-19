"""Chronological test-before-learn benchmark; all outputs belong to one immutable run."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import time
import uuid
from pathlib import Path
import pandas as pd
from src.config import RESULTS_DIR, RANDOM_SEED, TRAINING_WINDOW_RATIO
from src.ingestion import DatasetIngestion
from src.preprocessing import Preprocessor
from src.base_model import BaseFraudModel
from src.static_model import StaticBaselineModel
from src.adaptive_model import DriftAdaptiveModel
from src.periodic_model import PeriodicRetrainModel
from src.sliding_model import SlidingWindowModel
from src.drift_detector import DriftMonitor
from src.delayed_feedback import DelayedFeedbackQueue
from src.history_features import add_history_features
from src.artifacts import ModelBundle, save_bundle
from src.reporting import generate_report


def run_pipeline(
    dataset_path=None,
    exclude_card_identifiers=False,
    dataset_type=None,
    db=None,
    output_dir=None,
    seed=RANDOM_SEED,
    n_train_slices=None,
    evaluation_slices=None,
    adaptive_window=None,
    half_life_slices=None,
    max_retrains=None,
    delay_slices=0,
    latency_mode="fixed",
    identity_path=None,
    all_features=False,
    history_features=False,
    pre_transaction_only=False,
    data_kind="real",
    model_params=None,
    explain=False,
    lognormal_mu=1.38,
    lognormal_sigma=0.45,
    raw_dataframe=None,
    raw_slices=None,
    preprocessor=None,
    strategies=("static", "periodic", "weekly", "sliding", "adaptive"),
    threshold_objective="validation_f1",
    operational_daily_budget=None,
    protocol_version="1.0.0",
    resolved_config=None,
):
    started = time.perf_counter()
    if adaptive_window is not None and adaptive_window < 1:
        raise ValueError("adaptive_window must be positive")
    if half_life_slices is not None and half_life_slices <= 0:
        raise ValueError("half_life_slices must be positive")
    if max_retrains is not None and max_retrains < 0:
        raise ValueError("max_retrains must be nonnegative")
    if evaluation_slices is not None and evaluation_slices < 1:
        raise ValueError("evaluation_slices must be positive")
    if n_train_slices is not None and n_train_slices < 1:
        raise ValueError("n_train_slices must be positive")
    if data_kind not in {"real", "synthetic", "unverified"}:
        raise ValueError("data_kind must be real or synthetic")
    if threshold_objective not in {"validation_f1", "review_budget_recall"}:
        raise ValueError(
            f"Unsupported threshold_objective: '{threshold_objective}'. "
            "Supported objectives: 'validation_f1', 'review_budget_recall'"
        )
    if operational_daily_budget is not None and operational_daily_budget <= 0:
        raise ValueError("operational_daily_budget must be a positive integer")
    if threshold_objective == "review_budget_recall" and not operational_daily_budget:
        raise ValueError(
            "operational_daily_budget must be specified when threshold_objective is 'review_budget_recall'"
        )
    dataset_path = Path(dataset_path or "data/ieee/train_transaction.csv")
    ingestor = DatasetIngestion(
        dataset_path, identity_path=identity_path, all_features=all_features
    )
    if raw_dataframe is None:
        raw, validation = ingestor.load_and_validate()
    else:
        raw = raw_dataframe.copy()
        required = {"trans_id", "step", "isFraud"}
        if not required.issubset(raw):
            raise ValueError("Provided frame requires trans_id, step and isFraud")
        if raw.trans_id.isna().any() or raw.trans_id.duplicated().any():
            raise ValueError("Transaction IDs must be non-null and unique")
        if raw.isFraud.isna().any() or not set(raw.isFraud.unique()).issubset({0, 1}):
            raise ValueError("Labels must be binary")
        ingestor.dataset_type = (
            "IEEE-CIS"
            if "TransactionDT" in raw
            else ("Feedzai-BAF" if "month" in raw else "PaySim")
        )
        validation = {"source": "provided_dataframe", "total_rows": len(raw)}
    if history_features:
        raw = add_history_features(raw)
    slug = {"IEEE-CIS": "ieee_cis", "Feedzai-BAF": "feedzai_baf", "PaySim": "paysim"}[
        ingestor.dataset_type
    ]
    prep = preprocessor or Preprocessor(
        exclude_card_identifiers=exclude_card_identifiers,
        pre_transaction_only=pre_transaction_only,
    )
    slices = raw_slices if raw_slices is not None else prep.partition_time_slices(raw)
    if not {"static", "adaptive"}.issubset(strategies):
        raise ValueError("Strategies must include static and adaptive")
    n_train_slices = n_train_slices or (
        2 if len(slices) <= 8 else max(1, int(len(slices) * TRAINING_WINDOW_RATIO))
    )
    if not 1 <= n_train_slices < len(slices):
        raise ValueError("Need both initial training and out-of-sample slices")
    if evaluation_slices is not None and n_train_slices + evaluation_slices > len(
        slices
    ):
        raise ValueError("Requested evaluation period exceeds available slices")
    seed_raw = pd.concat(
        [s["data"] for s in slices[:n_train_slices]], ignore_index=True
    )
    # Unsupervised fitted statistics also exclude the threshold/calibration holdout.
    prep.fit(seed_raw.iloc[: BaseFraudModel.validation_start(len(seed_raw))])
    train = prep.transform(seed_raw)
    args = dict(
        feature_names=prep.feature_names,
        random_seed=seed,
        model_params=model_params,
        threshold_objective=threshold_objective,
        operational_daily_budget=operational_daily_budget,
    )
    initial = StaticBaselineModel(**args).fit(train)
    models = {"static": initial}
    interval = 2 if len(slices) <= 8 else 4
    for name, cls, extra in [
        ("periodic", PeriodicRetrainModel, dict(retrain_interval_slices=interval)),
        ("weekly", PeriodicRetrainModel, dict(retrain_interval_slices=1)),
        (
            "sliding",
            SlidingWindowModel,
            dict(window_slices=interval, retrain_interval_slices=interval),
        ),
        (
            "adaptive",
            DriftAdaptiveModel,
            dict(window_slices=adaptive_window, half_life_slices=half_life_slices),
        ),
    ]:
        if name not in strategies:
            continue
        model = cls(**args, max_retrains=max_retrains, **extra)
        for field in [
            "model",
            "calibrator",
            "threshold",
            "fit_metadata",
            "trained",
            "device_used",
            "train_duration_sec",
        ]:
            setattr(model, field, copy.deepcopy(getattr(initial, field)))
        model.last_training_id_digest = hashlib.sha256(
            "\n".join(sorted(train.trans_id.astype(str))).encode()
        ).hexdigest()
        model.cumulative_history = [
            prep.transform(s["data"]).assign(_history_slice=s["slice_index"])
            for s in slices[:n_train_slices]
        ]
        models[name] = model
    monitor = DriftMonitor()
    queue = DelayedFeedbackQueue(
        delay_slices=delay_slices,
        latency_mode=latency_mode,
        random_seed=seed,
        lognormal_mu=lognormal_mu,
        lognormal_sigma=lognormal_sigma,
    )
    records = []
    attributions = []
    events = []
    feedback_events = []
    selected = slices[
        n_train_slices : n_train_slices + evaluation_slices
        if evaluation_slices
        else None
    ]
    initial_train = pd.concat(models["adaptive"].cumulative_history, ignore_index=True)
    for s in selected:
        idx = s["slice_index"]
        frame = prep.transform(s["data"])
        tag = frame.assign(_history_slice=idx)
        for name, model in models.items():
            pred, prob = model.predict(frame)
            tag["_pred_" + name] = pred
            tag["_generation_" + name] = getattr(model, "retrain_count", 0)
            records.append(
                pd.DataFrame(
                    dict(
                        seed=seed,
                        window=n_train_slices,
                        slice_index=idx,
                        day=((frame.step.to_numpy() - 1) // 24).astype(int),
                        trans_id=frame.trans_id.astype(str).to_numpy(),
                        model=name,
                        y_true=frame.isFraud.to_numpy(dtype=int),
                        prediction=pred,
                        probability=prob,
                        threshold=model.threshold,
                        amount=frame.get(
                            "amount",
                            frame.get(
                                "TransactionAmt", pd.Series(0, index=frame.index)
                            ),
                        ).to_numpy(),
                        model_generation=getattr(model, "retrain_count", 0),
                    )
                )
            )
        queue.enqueue(tag, idx)
        matured = queue.dequeue_matured(idx)
        drift = False
        for tx in matured.to_dict("records"):
            current_generation = models["adaptive"].retrain_count
            monitored = int(tx["_generation_adaptive"]) == current_generation
            feedback_events.append(
                dict(
                    trans_id=str(tx["trans_id"]),
                    label_slice=idx,
                    prediction_generation=int(tx["_generation_adaptive"]),
                    active_generation=current_generation,
                    monitored=monitored,
                    original_prediction=int(tx["_pred_adaptive"]),
                )
            )
            if not monitored:
                continue
            alert = monitor.update(
                int(tx["isFraud"]),
                int(tx["_pred_adaptive"]),
                idx,
                int(tx["step"]),
                str(tx["trans_id"]),
            )
            drift = drift or alert["any_drift"]
        # Attribute the triggering model before changing it; preserve real detector names.
        if (
            drift
            and explain
            and models["adaptive"].max_retrains != models["adaptive"].retrain_count
        ):
            from src.drift_explainer import DriftAttributionEngine

            engine = DriftAttributionEngine(prep.feature_names)
            attributions.extend(
                engine.explain_drift_slice(
                    models["adaptive"],
                    train,
                    frame,
                    idx,
                    s["slice_name"],
                    detector_type=",".join(
                        sorted(
                            {
                                e["detector_type"]
                                for e in monitor.drift_events
                                if e["slice_index"] == idx
                            }
                        )
                    )
                    or "TEST_MONITOR",
                )
            )
        known = queue.get_all_matured_training_data(initial_train)
        known = known.sort_values(["_history_slice", "step", "trans_id"], kind="stable")
        for name, model in models.items():
            if name == "static":
                continue
            if name == "adaptive":
                trigger = drift
            else:
                model.out_of_sample_slices_seen += 1
                trigger = (
                    model.out_of_sample_slices_seen % model.retrain_interval_slices == 0
                )
            if trigger:
                if model._retrain(idx, training_df_override=known):
                    event = dict(model.retrain_log[-1], model=name)
                    events.append(event)
                    if name == "adaptive":
                        monitor.reset_detectors()
    run_id = "RUN_" + uuid.uuid4().hex[:10].upper()
    run_dir = Path(output_dir or RESULTS_DIR / "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "river"]:
        packages[name] = importlib.metadata.version(name)
    source_digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.rglob("*.py")):
        source_digest.update(str(source.relative_to(Path(__file__).parent)).encode())
        source_digest.update(source.read_bytes())
    prov = validation.get("provenance") if isinstance(validation, dict) else None
    if prov is None and hasattr(ingestor, "dataset_type"):
        from src.provenance import get_dataset_provenance
        prov = get_dataset_provenance(ingestor.dataset_type, dataset_path, identity_path)

    try:
        import subprocess

        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_rev = None

    manifest = dict(
        run_id=run_id,
        version="0.6.0",
        protocol_version=protocol_version or "1.0.0",
        git_revision=git_rev,
        resolved_configuration=resolved_config,
        source_sha256=source_digest.hexdigest(),
        data_kind=data_kind,
        dataset=dataset_path.name,
        provenance=prov,
        dataset_sha256=hashlib.sha256(
            dataset_path.read_bytes()
            if raw_dataframe is None
            else raw.to_csv(index=False).encode()
        ).hexdigest(),
        identity_sha256=hashlib.sha256(Path(identity_path).read_bytes()).hexdigest()
        if identity_path
        else None,
        seed=seed,
        n_train_slices=n_train_slices,
        evaluation_slice_indices=[s["slice_index"] for s in selected],
        adaptive_window=adaptive_window,
        half_life_slices=half_life_slices,
        max_retrains=max_retrains,
        delay_slices=delay_slices,
        latency_mode=latency_mode,
        lognormal_mu=lognormal_mu,
        lognormal_sigma=lognormal_sigma,
        threshold_objective=threshold_objective,
        operational_daily_budget=operational_daily_budget,
        model_params=initial.model_params,
        strategies=list(models),
        recent_window_basis="latest verified transaction arrival slice",
        initial_training_seconds=initial.train_duration_sec,
        calibration_policy="chronological final 20%, sigmoid if >=5 examples per class",
        initial_fit=initial.fit_metadata,
        device=initial.device_used,
        python=platform.python_version(),
        packages=packages,
        history_features=history_features,
        pre_transaction_only=pre_transaction_only,
        all_features=all_features,
        label_queue=queue.get_status_summary(selected[-1]["slice_index"]),
        time_basis="simulated intramonth hours"
        if slug == "feedzai_baf"
        else "dataset-relative hours",
        exclude_card_identifiers=exclude_card_identifiers,
        elapsed_seconds=time.perf_counter() - started,
        validation=validation,
    )
    try:
        import psutil

        manifest["process_rss_mb"] = psutil.Process().memory_info().rss / 1024**2
    except Exception:
        manifest["process_rss_mb"] = None
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.concat(records, ignore_index=True).to_csv(
        run_dir / "predictions.csv", index=False
    )
    (run_dir / "retrain_events.json").write_text(json.dumps(events, indent=2))
    (run_dir / "drift_events.json").write_text(
        json.dumps(monitor.drift_events, indent=2)
    )
    (run_dir / "attributions.json").write_text(json.dumps(attributions, indent=2))
    pd.DataFrame(
        feedback_events,
        columns=[
            "trans_id",
            "label_slice",
            "prediction_generation",
            "active_generation",
            "monitored",
            "original_prediction",
        ],
    ).to_csv(run_dir / "feedback_events.csv", index=False)
    summary = generate_report(run_dir)
    for name, model in models.items():
        save_bundle(
            ModelBundle(
                model,
                prep,
                dict(
                    run_id=run_id,
                    model=name,
                    data_kind=data_kind,
                    provenance=prov,
                    version="0.6.0",
                    model_revision=uuid.uuid4().hex,
                ),
            ),
            run_dir / "models" / name,
        )
    if db is not None:
        weekly = pd.read_csv(run_dir / "weekly_metrics.csv")
        db.log_metrics(
            run_id,
            [
                dict(
                    slice_index=int(r.slice_index),
                    model_type=r.model,
                    precision=r.overall_precision,
                    recall=r.overall_recall,
                    f1_score=r.overall_f1,
                    auc_pr=r.overall_auc_pr,
                )
                for r in weekly.itertuples()
            ],
        )
        db.log_drift_events(run_id, monitor.drift_events)
        if attributions:
            db.log_attributions(run_id, attributions)
        db.log_slices(run_id, slices)
        db.log_run(
            dict(
                run_id=run_id,
                dataset_name=dataset_path.name,
                total_rows=len(raw),
                time_slices_count=len(slices),
                gpu_accelerated=initial.device_used,
                static_f1_overall=summary["static_model"]["overall_f1"],
                adaptive_f1_overall=summary["adaptive_model"]["overall_f1"],
                static_f1_oos=summary["static_model"]["overall_f1"],
                adaptive_f1_oos=summary["adaptive_model"]["overall_f1"],
                drift_alerts_count=len(monitor.drift_events),
                notes="OOS-only v0.4.0",
            )
        )
    return dict(
        run_id=run_id,
        dataset_slug=slug,
        run_dir=run_dir,
        overall_summary={"out_of_sample_stream": summary, **summary},
        chart_path=run_dir / "overview.png",
        metrics_csv_path=run_dir / "weekly_metrics.csv",
        summary_json_path=run_dir / "summary.json",
        run_summary_json=run_dir / "summary.json",
        drift_events_count=len(monitor.drift_events),
        retrains_count=models["adaptive"].retrain_count,
        db_backend=db.backend_name if db else "filesystem",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="Path to versioned JSON experiment configuration file")
    p.add_argument("--dataset", help="Path to transaction CSV dataset")
    p.add_argument("--identity", help="Path to identity CSV dataset")
    p.add_argument("--output-dir", help="Directory for experiment run output artifacts")
    p.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    p.add_argument("--train-slices", type=int, help="Number of baseline training slices")
    p.add_argument("--evaluation-slices", type=int, help="Number of evaluation slices")
    p.add_argument("--adaptive-window", type=int, help="Sliding window size for adaptive model")
    p.add_argument("--half-life-slices", type=float, help="Exponential decay half-life in slices")
    p.add_argument("--max-retrains", type=int, help="Maximum permitted retraining budget")
    p.add_argument("--delay-slices", type=int, default=0, help="Dispute label arrival delay in slices")
    p.add_argument("--latency-mode", choices=["fixed", "lognormal"], default="fixed", help="Label feedback delay mode")
    p.add_argument("--lognormal-mu", type=float, default=1.38)
    p.add_argument("--lognormal-sigma", type=float, default=0.45)
    p.add_argument(
        "--threshold-objective",
        choices=["validation_f1", "review_budget_recall"],
        default="validation_f1",
        help="Decision threshold optimization policy ('validation_f1' or 'review_budget_recall')",
    )
    p.add_argument(
        "--operational-daily-budget",
        type=int,
        default=None,
        help="Target daily alert capacity (e.g. 5, 100, 500) for review budget thresholding",
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of gradient boosted trees (protocol default: 100)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="XGBoost learning rate (protocol default: 0.1)",
    )
    p.add_argument("--all-features", action="store_true", help="Include all raw dataset features")
    p.add_argument("--history-features", action="store_true", help="Include behavioral history features")
    p.add_argument("--pre-transaction-only", action="store_true", help="Exclude post-transaction balance leakage")
    p.add_argument("--exclude-identifiers", action="store_true", help="Exclude high-cardinality card/entity IDs")
    p.add_argument("--synthetic", action="store_true", help="Mark dataset as synthetic software test fixture")
    p.add_argument("--explain", action="store_true", help="Generate SHAP/attribution explanations")
    a = p.parse_args()

    # Load versioned configuration file if supplied
    cfg = {}
    if a.config:
        cfg_path = Path(a.config)
        if not cfg_path.exists():
            p.error(f"Configuration file not found: {a.config}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    dataset = a.dataset or cfg.get("dataset")
    if not dataset:
        p.error("Either --dataset or a --config defining 'dataset' must be provided")

    # Merge model hyperparameter settings
    model_params = dict(cfg.get("model_params", {}))
    if a.n_estimators is not None:
        model_params["n_estimators"] = a.n_estimators
    if a.learning_rate is not None:
        model_params["learning_rate"] = a.learning_rate

    # Validate operational review budget
    budget = a.operational_daily_budget if a.operational_daily_budget is not None else cfg.get("operational_daily_budget")
    if budget is not None and budget <= 0:
        p.error("--operational-daily-budget must be a positive integer")

    thresh_obj = a.threshold_objective if a.threshold_objective != "validation_f1" else cfg.get("threshold_objective", a.threshold_objective)
    if thresh_obj == "review_budget_recall" and not budget:
        p.error("--operational-daily-budget must be provided when threshold_objective is 'review_budget_recall'")

    r = run_pipeline(
        dataset,
        exclude_card_identifiers=a.exclude_identifiers or cfg.get("exclude_card_identifiers", False),
        output_dir=a.output_dir or cfg.get("output_dir"),
        seed=a.seed if a.seed != 42 else cfg.get("seed", 42),
        n_train_slices=a.train_slices or cfg.get("train_slices"),
        evaluation_slices=a.evaluation_slices or cfg.get("evaluation_slices"),
        adaptive_window=a.adaptive_window if a.adaptive_window is not None else cfg.get("adaptive_window"),
        half_life_slices=a.half_life_slices if a.half_life_slices is not None else cfg.get("half_life_slices"),
        max_retrains=a.max_retrains if a.max_retrains is not None else cfg.get("max_retrains"),
        delay_slices=a.delay_slices if a.delay_slices != 0 else cfg.get("delay_slices", 0),
        latency_mode=a.latency_mode if a.latency_mode != "fixed" else cfg.get("latency_mode", "fixed"),
        identity_path=a.identity or cfg.get("identity"),
        all_features=a.all_features or cfg.get("all_features", False),
        history_features=a.history_features or cfg.get("history_features", False),
        pre_transaction_only=a.pre_transaction_only or cfg.get("pre_transaction_only", False),
        data_kind="synthetic" if (a.synthetic or cfg.get("data_kind") == "synthetic") else "real",
        model_params=model_params or None,
        explain=a.explain or cfg.get("explain", False),
        lognormal_mu=a.lognormal_mu if a.lognormal_mu != 1.38 else cfg.get("lognormal_mu", 1.38),
        lognormal_sigma=a.lognormal_sigma if a.lognormal_sigma != 0.45 else cfg.get("lognormal_sigma", 0.45),
        threshold_objective=thresh_obj,
        operational_daily_budget=budget,
        resolved_config=dict(cfg, **{k: v for k, v in vars(a).items() if v is not None}),
    )
    print(r["run_dir"] / "report.html")


if __name__ == "__main__":
    main()
