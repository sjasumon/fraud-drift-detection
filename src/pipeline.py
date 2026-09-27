"""
End-to-End Orchestrator Pipeline for Concept-Drift-Aware Fraud Detection.
Configured for:
- IEEE-CIS Fraud Detection dataset (590,540 transactions spanning 26 weeks)
- Strict MySQL Database persistence (fraud_drift_db)
- GPU Hardware Acceleration (CUDA)
- Full Retrain Strategy upon ADWIN/DDM Concept Drift alerts
- Publication-quality metrics (Precision, Recall, F1, AUC-PR) and charts
"""

import sys
import uuid
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    RESULTS_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR,
    RANDOM_SEED, USE_GPU, GPU_DEVICE, TRAINING_WINDOW_RATIO
)
from src.db import DatabaseManager
from src.ingestion import DatasetIngestion
from src.preprocessing import Preprocessor
from src.static_model import StaticBaselineModel
from src.periodic_model import PeriodicRetrainModel
from src.sliding_model import SlidingWindowModel
from src.drift_detector import DriftMonitor
from src.adaptive_model import DriftAdaptiveModel
from src.evaluator import SliceEvaluator
from src.drift_explainer import DriftAttributionEngine

# Configure logging
log_file = LOGS_DIR / f"pipeline_run_{int(time.time())}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger("fraud_drift.pipeline")


def run_pipeline(
    dataset_path: Path = None,
    exclude_card_identifiers: bool = False,
    dataset_type: Optional[str] = None,
    db: Optional[DatabaseManager] = None
) -> Dict[str, Any]:
    """
    Executes the end-to-end experiment pipeline according to FRD requirements.
    Artifacts are namespaced by dataset slug and run_id to avoid cross-benchmark collisions.
    """
    run_id = f"RUN_{uuid.uuid4().hex[:10].upper()}"
    start_total_time = time.time()

    # 1. Initialize Database (MySQL Server 9.7 with graceful fallback to file-only persistence)
    if db is None:
        try:
            db = DatabaseManager()
            db_backend_name = db.backend_name
        except Exception as e:
            logger.warning(f"Database connection unavailable ({e}). Continuing with filesystem metrics logging.")
            db = None
            db_backend_name = "Filesystem Only (MySQL unavailable)"
    else:
        db_backend_name = db.backend_name

    logger.info("=" * 80)
    logger.info(f"STARTING CONCEPT-DRIFT-AWARE FRAUD DETECTION PIPELINE [{run_id}]")
    logger.info(f"Hardware: GPU Acceleration = {USE_GPU} ({GPU_DEVICE})")
    logger.info(f"Database Backend: {db_backend_name}")
    logger.info(f"Adaptation Strategy: FULL RETRAINING ON GPU")
    logger.info(f"Reproducibility Seed: {RANDOM_SEED}")
    logger.info("=" * 80)

    # 2. FR-1: Dataset Ingestion & Validation
    if dataset_path is None:
        candidate_paths = [
            Path("data/ieee/train_transaction.csv"),
            Path("ieee/train_transaction.csv"),
        ]
        for p in candidate_paths:
            if p.exists():
                dataset_path = p
                break
        if dataset_path is None:
            raise FileNotFoundError(
                f"IEEE-CIS dataset not found at {candidate_paths[0]} or {candidate_paths[1]}"
            )

    ingestor = DatasetIngestion(dataset_path)
    df_raw, validation_report = ingestor.load_and_validate()

    # Determine dataset slug for artifact namespacing
    ds_str = str(dataset_path).lower() if dataset_path else ""
    if dataset_type:
        dataset_slug = "ieee_cis" if dataset_type.lower() == "ieee" else dataset_type.lower()
    elif "ieee" in ds_str or "train_transaction" in ds_str:
        dataset_slug = "ieee_cis"
    elif "feedzai" in ds_str or "base.csv" in ds_str:
        dataset_slug = "feedzai_baf"
    elif "paysim" in ds_str:
        dataset_slug = "paysim"
    else:
        dataset_slug = dataset_path.stem.lower().replace("-", "_").replace(" ", "_") if dataset_path else "dataset"

    # 3. FR-2: Time-Slice Partitioning on RAW data (Zero Lookahead Leakage)
    preprocessor = Preprocessor(exclude_card_identifiers=exclude_card_identifiers)
    raw_slices = preprocessor.partition_time_slices(df_raw)

    # Support dynamic baseline window and retrain intervals across multi-dataset benchmarks
    if len(raw_slices) <= 8:
        n_train_slices = 2
        baseline_retrain_interval = 2
    else:
        n_train_slices = max(1, int(len(raw_slices) * TRAINING_WINDOW_RATIO))
        baseline_retrain_interval = 4

    train_raw_df = pd.concat([s["data"] for s in raw_slices[:n_train_slices]], ignore_index=True)

    # Fit preprocessor strictly on baseline training partition
    preprocessor.fit(train_raw_df)
    imbalance_info = preprocessor.imbalance_weights

    # Transform slices chronologically using baseline-fitted feature extractor
    time_slices = []
    for s in raw_slices:
        s_transformed = s.copy()
        s_transformed["data"] = preprocessor.transform(s["data"])
        time_slices.append(s_transformed)

    # Persist slice metadata strictly to MySQL if available
    if db is not None:
        db.log_slices(run_id, time_slices)

    # 4. FR-3: Static Baseline Model Setup
    train_slices = time_slices[:n_train_slices]
    train_df = pd.concat([s["data"] for s in train_slices], ignore_index=True)

    logger.info(
        f"FR-3.1: Assembling baseline training window from first {n_train_slices} slices "
        f"({len(train_df):,} transactions, {int(train_df['isFraud'].sum())} fraud instances, "
        f"scale_pos_weight={imbalance_info['scale_pos_weight']:.2f})."
    )

    static_model = StaticBaselineModel(
        feature_names=preprocessor.feature_names,
        scale_pos_weight=imbalance_info["scale_pos_weight"]
    )
    static_model.fit(train_df)

    # 4b. Periodic Retraining Baseline (Academic Baseline 2)
    periodic_model = PeriodicRetrainModel(
        feature_names=preprocessor.feature_names,
        retrain_interval_slices=baseline_retrain_interval,
        scale_pos_weight=imbalance_info["scale_pos_weight"]
    )
    periodic_model.fit_initial(train_df)

    # 4c. Sliding Window Retraining Baseline (Academic Baseline 3)
    sliding_model = SlidingWindowModel(
        feature_names=preprocessor.feature_names,
        window_slices=baseline_retrain_interval,
        retrain_interval_slices=baseline_retrain_interval,
        scale_pos_weight=imbalance_info["scale_pos_weight"]
    )
    sliding_model.fit_initial(train_df)

    # 5. FR-5: Drift-Adaptive Model Setup with Full Retraining on GPU
    adaptive_model = DriftAdaptiveModel(
        feature_names=preprocessor.feature_names,
        scale_pos_weight=imbalance_info["scale_pos_weight"]
    )
    adaptive_model.fit_initial(train_df)

    # 6. FR-4: Drift Detector Setup (ADWIN and DDM)
    drift_monitor = DriftMonitor()

    # 6b. P4: SHAP Drift Attribution Engine Setup
    drift_explainer = DriftAttributionEngine(feature_names=preprocessor.feature_names)
    all_attributions: List[Dict[str, Any]] = []

    # 7. FR-6: Sequential Time-Series Streaming Evaluation Loop
    evaluator = SliceEvaluator()

    all_static_preds, all_static_probs = [], []
    all_periodic_preds, all_periodic_probs = [], []
    all_sliding_preds, all_sliding_probs = [], []
    all_adaptive_preds, all_adaptive_probs = [], []
    all_y_true = []

    logger.info("\n" + "=" * 80)
    logger.info(f"BEGINNING SEQUENTIAL STREAMING EVALUATION ACROSS {len(time_slices)} WEEKS (4 MODELS)")
    logger.info("=" * 80)

    for s in time_slices:
        slice_idx = s["slice_index"]
        slice_name = s["slice_name"]
        slice_data = s["data"]
        y_true = slice_data["isFraud"].values
        is_oos = (slice_idx >= n_train_slices)

        # --- A. Static Model Inference (FR-3.3: Frozen model) ---
        static_preds, static_probs = static_model.predict(slice_data)
        evaluator.evaluate_slice(
            y_true=y_true,
            y_pred=static_preds,
            y_prob=static_probs,
            slice_index=slice_idx,
            slice_name=slice_name,
            model_type="static"
        )

        # --- B. Periodic Model Inference (Baseline 2: Every 4w) ---
        periodic_preds, periodic_probs = periodic_model.predict(slice_data)
        evaluator.evaluate_slice(
            y_true=y_true,
            y_pred=periodic_preds,
            y_prob=periodic_probs,
            slice_index=slice_idx,
            slice_name=slice_name,
            model_type="periodic"
        )

        # --- C. Sliding Window Model Inference (Baseline 3: Last 4w) ---
        sliding_preds, sliding_probs = sliding_model.predict(slice_data)
        evaluator.evaluate_slice(
            y_true=y_true,
            y_pred=sliding_preds,
            y_prob=sliding_probs,
            slice_index=slice_idx,
            slice_name=slice_name,
            model_type="sliding"
        )

        # --- D. Adaptive Model Inference (FR-5.3: Test-Before-Learn) ---
        adaptive_preds, adaptive_probs = adaptive_model.predict(slice_data)
        evaluator.evaluate_slice(
            y_true=y_true,
            y_pred=adaptive_preds,
            y_prob=adaptive_probs,
            slice_index=slice_idx,
            slice_name=slice_name,
            model_type="adaptive"
        )

        # Accumulate whole-dataset evaluation arrays
        all_static_preds.extend(static_preds)
        all_static_probs.extend(static_probs)
        all_periodic_preds.extend(periodic_preds)
        all_periodic_probs.extend(periodic_probs)
        all_sliding_preds.extend(sliding_preds)
        all_sliding_probs.extend(sliding_probs)
        all_adaptive_preds.extend(adaptive_preds)
        all_adaptive_probs.extend(adaptive_probs)
        all_y_true.extend(y_true)

        # --- E. Drift Monitoring over Prediction Error Stream (FR-4) ---
        slice_drift_triggered = False
        for i in range(len(slice_data)):
            trans_id = slice_data["trans_id"].iloc[i]
            step_num = int(slice_data["step"].iloc[i])
            yt = int(y_true[i])
            yp = int(adaptive_preds[i])

            alert = drift_monitor.update(
                y_true=yt,
                y_pred=yp,
                slice_index=slice_idx,
                step_number=step_num,
                trans_id=trans_id
            )
            if alert["any_drift"]:
                slice_drift_triggered = True

        # --- F. Update Baseline and Adaptive Models ---
        periodic_model.update_with_slice(slice_data, slice_idx=slice_idx, is_out_of_sample=is_oos)
        sliding_model.update_with_slice(slice_data, slice_idx=slice_idx, is_out_of_sample=is_oos)
        adaptive_model.update_with_slice(
            slice_df=slice_data,
            drift_triggered=slice_drift_triggered,
            slice_idx=slice_idx
        )
        if slice_drift_triggered:
            drift_monitor.reset_detectors()

        # --- G. P4: Mechanistic SHAP & PSI Attribution on Drift ---
        if slice_drift_triggered:
            slice_attrs = drift_explainer.explain_drift_slice(
                model=adaptive_model,
                baseline_df=train_df,
                current_df=slice_data,
                slice_index=slice_idx,
                slice_name=slice_name,
                detector_type="ADWIN",
                top_k=5
            )
            all_attributions.extend(slice_attrs)

            # Generate visualization for key transitional slices (e.g. W06, W12, W21)
            if slice_idx in (5, 11, 20):
                drift_explainer.render_attribution_plot(
                    attributions=slice_attrs,
                    slice_name=slice_name,
                    output_filename=f"drift_attribution_{slice_name.lower()}.png"
                )

        # Per-slice progress logging
        stat_f1 = [m["f1_score"] for m in evaluator.metrics_log if m["slice_index"] == slice_idx and m["model_type"] == "static"][0]
        period_f1 = [m["f1_score"] for m in evaluator.metrics_log if m["slice_index"] == slice_idx and m["model_type"] == "periodic"][0]
        slide_f1 = [m["f1_score"] for m in evaluator.metrics_log if m["slice_index"] == slice_idx and m["model_type"] == "sliding"][0]
        adapt_f1 = [m["f1_score"] for m in evaluator.metrics_log if m["slice_index"] == slice_idx and m["model_type"] == "adaptive"][0]
        drift_tag = " [DRIFT -> FULL GPU RETRAIN!]" if slice_drift_triggered else ""

        logger.info(
            f"Slice {slice_name} ({slice_idx+1:02d}/{len(time_slices)}): "
            f"Static F1={stat_f1:.4f} | Period F1={period_f1:.4f} | Slide F1={slide_f1:.4f} | Adapt F1={adapt_f1:.4f}{drift_tag}"
        )

    # 8. FR-6.3: Whole-Dataset Summary
    all_y_true = np.array(all_y_true)
    all_static_preds = np.array(all_static_preds)
    all_static_probs = np.array(all_static_probs)
    all_periodic_preds = np.array(all_periodic_preds)
    all_periodic_probs = np.array(all_periodic_probs)
    all_sliding_preds = np.array(all_sliding_preds)
    all_sliding_probs = np.array(all_sliding_probs)
    all_adaptive_preds = np.array(all_adaptive_preds)
    all_adaptive_probs = np.array(all_adaptive_probs)

    oos_mask = np.array([s["slice_index"] >= n_train_slices for s in time_slices for _ in range(len(s["data"]))])

    overall_summary = evaluator.compute_overall_summary(
        static_preds=all_static_preds,
        static_probs=all_static_probs,
        adaptive_preds=all_adaptive_preds,
        adaptive_probs=all_adaptive_probs,
        y_all=all_y_true,
        periodic_preds=all_periodic_preds,
        periodic_probs=all_periodic_probs,
        sliding_preds=all_sliding_preds,
        sliding_probs=all_sliding_probs,
        oos_mask=oos_mask
    )

    # Save metrics to CSV and JSON with namespacing
    metrics_df = pd.DataFrame(evaluator.metrics_log)

    # 1. Namespaced by run_id
    run_metrics_csv = METRICS_DIR / f"{dataset_slug}_{run_id}_slice_comparison_metrics.csv"
    metrics_df.to_csv(run_metrics_csv, index=False)

    run_summary_json = METRICS_DIR / f"{dataset_slug}_{run_id}_summary.json"
    with open(run_summary_json, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=4)

    # 2. Namespaced by dataset slug
    dataset_metrics_csv = METRICS_DIR / f"{dataset_slug}_slice_comparison_metrics.csv"
    metrics_df.to_csv(dataset_metrics_csv, index=False)

    dataset_summary_json = METRICS_DIR / f"{dataset_slug}_summary_metrics.json"
    with open(dataset_summary_json, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=4)

    # 3. Maintain root headline artifacts for IEEE-CIS (primary benchmark)
    metrics_csv_path = METRICS_DIR / "slice_comparison_metrics.csv"
    summary_json_path = METRICS_DIR / "summary_metrics.json"
    chart_output_filename = "model_comparison_f1_aucpr.png" if dataset_slug == "ieee_cis" else f"{dataset_slug}_model_comparison_f1_aucpr.png"

    if dataset_slug == "ieee_cis":
        metrics_df.to_csv(metrics_csv_path, index=False)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(overall_summary, f, indent=4)

    # 9. FR-6.2: Generate High-Resolution Visual Comparison Charts
    chart_path = evaluator.generate_comparison_plots(
        metrics_df=metrics_df,
        drift_events=drift_monitor.drift_events,
        output_filename=chart_output_filename,
        n_train_slices=n_train_slices
    )
    if dataset_slug == "ieee_cis":
        evaluator.generate_comparison_plots(
            metrics_df=metrics_df,
            drift_events=drift_monitor.drift_events,
            output_filename="ieee_cis_model_comparison_f1_aucpr.png",
            n_train_slices=n_train_slices
        )

    # 10. Persist Everything to MySQL Database if available
    if db is not None:
        db.log_metrics(run_id, evaluator.metrics_log)
        db.log_drift_events(run_id, drift_monitor.drift_events)
        if all_attributions:
            db.log_attributions(run_id, all_attributions)
            logger.info(f"P4: Persisted {len(all_attributions)} drift attribution records to MySQL `drift_attributions`.")

        oos_stream = overall_summary.get("out_of_sample_stream", {})
        static_oos_f1 = oos_stream.get("static_model", {}).get("overall_f1")
        adaptive_oos_f1 = oos_stream.get("adaptive_model", {}).get("overall_f1")

        run_record = {
            "run_id": run_id,
            "dataset_name": dataset_path.name,
            "total_rows": len(df_raw),
            "time_slices_count": len(time_slices),
            "gpu_accelerated": f"{static_model.device_used}",
            "static_f1_overall": overall_summary["static_model"]["overall_f1"],
            "adaptive_f1_overall": overall_summary["adaptive_model"]["overall_f1"],
            "static_f1_oos": static_oos_f1,
            "adaptive_f1_oos": adaptive_oos_f1,
            "drift_alerts_count": len(drift_monitor.drift_events),
            "notes": (
                f"Periodic retrains: {periodic_model.retrain_count}. "
                f"Sliding retrains: {sliding_model.retrain_count}. "
                f"Adaptive retrains: {adaptive_model.adaptation_count}. "
                f"Total execution time: {time.time() - start_total_time:.1f}s"
            )
        }
        db.log_run(run_record)

    total_pipeline_time = time.time() - start_total_time

    # 11. Final Console Summary Report
    logger.info("\n" + "=" * 96)
    logger.info("MULTI-MODEL STREAMING BENCHMARK COMPLETE — SUMMARY OF RESULTS")
    logger.info("=" * 96)
    logger.info(f"Dataset: {dataset_path.name} ({len(df_raw):,} transactions across {len(time_slices)} weeks)")
    logger.info(f"Dataset Slug: {dataset_slug}")
    logger.info(f"Hardware Acceleration: {static_model.device_used} (GPU)")
    logger.info(f"Database Logged: {db_backend_name}")
    logger.info(f"Total Drift Alerts: {len(drift_monitor.drift_events)}")
    logger.info("-" * 108)
    logger.info(f"{'Model Architecture':<28} | {'Retrains':<8} | {'In+OOS F1*':<11} | {'OOS F1':<10} | {'OOS Recall':<11} | {'OOS AUC-PR':<11}")
    logger.info("-" * 108)

    m_names = [
        ("Static Baseline (Frozen)", "static_model", 0),
        ("Periodic Retraining (4w)", "periodic_model", periodic_model.retrain_count),
        ("Sliding Window (4w)", "sliding_model", sliding_model.retrain_count),
        ("Drift-Adaptive (ADWIN/DDM)", "adaptive_model", adaptive_model.adaptation_count),
    ]

    oos = overall_summary.get("out_of_sample_stream", {})
    for label, key, n_retrains in m_names:
        w_f1 = overall_summary[key]["overall_f1"]
        oos_f1 = oos.get(key, {}).get("overall_f1", 0.0)
        oos_rec = oos.get(key, {}).get("overall_recall", 0.0)
        oos_auc = oos.get(key, {}).get("overall_auc_pr", 0.0)
        logger.info(f"{label:<28} | {n_retrains:<8} | {w_f1:<11.4f} | {oos_f1:<10.4f} | {oos_rec*100:<10.2f}% | {oos_auc:<11.4f}")

    logger.info("-" * 108)
    logger.info("* Note: 'In+OOS F1*' includes initial training warmup slices (Slices 1-5). Pure out-of-sample generalization is reported under 'OOS F1', 'OOS Recall', and 'OOS AUC-PR'.")
    logger.info(f"Outputs generated:")
    logger.info(f"  - Dataset Metrics: {dataset_metrics_csv}")
    logger.info(f"  - Dataset Summary: {dataset_summary_json}")
    logger.info(f"  - Run Summary:     {run_summary_json}")
    if dataset_slug == "ieee_cis":
        logger.info(f"  - Root Metrics:    {metrics_csv_path}")
        logger.info(f"  - Root Summary:    {summary_json_path}")
    logger.info(f"  - Chart:           {chart_path}")
    logger.info(f"  - Log File:        {log_file}")
    logger.info(f"Pipeline executed in {total_pipeline_time:.2f}s.")
    logger.info("=" * 80)

    return {
        "run_id": run_id,
        "dataset_slug": dataset_slug,
        "overall_summary": overall_summary,
        "chart_path": chart_path,
        "metrics_csv_path": metrics_csv_path if dataset_slug == "ieee_cis" else dataset_metrics_csv,
        "summary_json_path": summary_json_path if dataset_slug == "ieee_cis" else dataset_summary_json,
        "run_summary_json": run_summary_json,
        "drift_events_count": len(drift_monitor.drift_events),
        "db_backend": db_backend_name,
        "retrains_count": adaptive_model.adaptation_count
    }


def main():
    parser = argparse.ArgumentParser(description="Concept-Drift Fraud Detection Pipeline")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="ieee",
        choices=["ieee", "feedzai", "paysim"],
        help="Dataset archetype: ieee, feedzai (NeurIPS 2022), or paysim (mobile money)"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Explicit path to raw CSV dataset")
    parser.add_argument("--exclude-identifiers", action="store_true", help="Exclude raw card/address identifiers")
    args = parser.parse_args()

    if args.dataset:
        data_file = Path(args.dataset)
    else:
        if args.dataset_type == "feedzai":
            data_file = Path("data/feedzai/Base.csv")
        elif args.dataset_type == "paysim":
            data_file = Path("data/paysim/paysim_full.csv")
        else:
            default_path = "data/ieee/train_transaction.csv" if Path("data/ieee/train_transaction.csv").exists() else "ieee/train_transaction.csv"
            data_file = Path(default_path)

    run_pipeline(
        dataset_path=data_file,
        exclude_card_identifiers=args.exclude_identifiers,
        dataset_type=args.dataset_type if not args.dataset else None
    )


if __name__ == "__main__":
    main()
