"""
Deterministic Metrics & Evidence Reconciliation CLI (Finding R02).
Recomputes headline tables directly from raw prediction ledgers, validates
confusion matrix integer invariants, verifies Average Precision, retrospective
review-budget allocations, and training cost against recorded experiment artifacts.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss


def reconcile_run_directory(
    run_dir: Path,
    tolerance: float = 1e-4,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Deterministically reconciles all metrics for a single run from its predictions ledger.
    Returns (is_reconciled, report_dict, discrepancies).
    """
    run_dir = Path(run_dir)
    discrepancies: List[str] = []

    manifest_path = run_dir / "manifest.json"
    predictions_path = run_dir / "predictions.csv"
    metrics_path = run_dir / "metrics.csv"
    alert_budgets_path = run_dir / "alert_budgets.csv"
    compute_cost_path = run_dir / "compute_cost.csv"

    if not manifest_path.exists():
        discrepancies.append(f"Missing manifest.json at {manifest_path}")
        return False, {"is_reconciled": False, "discrepancies": discrepancies, "discrepancies_count": len(discrepancies), "run_id": run_dir.name, "data_kind": "unknown", "provenance_status": "MISSING"}, discrepancies
    if not predictions_path.exists():
        discrepancies.append(f"Missing predictions.csv at {predictions_path}")
        return False, {"is_reconciled": False, "discrepancies": discrepancies, "discrepancies_count": len(discrepancies), "run_id": run_dir.name, "data_kind": "unknown", "provenance_status": "MISSING"}, discrepancies

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id", run_dir.name)

    # 1. Load predictions ledger
    pred_df = pd.read_csv(predictions_path)
    total_rows = len(pred_df)
    models_present = sorted(pred_df["model"].unique()) if "model" in pred_df.columns else []

    if total_rows == 0 or len(models_present) == 0:
        discrepancies.append("predictions.csv is empty; cannot reconcile empty prediction ledger")
        return False, {
            "is_reconciled": False,
            "discrepancies": discrepancies,
            "discrepancies_count": len(discrepancies),
            "run_id": run_id,
            "total_records_scored": 0,
            "models_evaluated": [],
            "data_kind": manifest.get("data_kind", "unknown"),
            "provenance_status": manifest.get("provenance", {}).get("verification_status", "UNVERIFIED") if manifest.get("provenance") else "UNVERIFIED",
        }, discrepancies

    # Check required comparison files
    if not metrics_path.exists():
        discrepancies.append(f"Missing required comparison file: {metrics_path.name}")
    if not compute_cost_path.exists():
        discrepancies.append(f"Missing required comparison file: {compute_cost_path.name}")

    # Detect ground truth column: 'y_true', 'is_fraud', or 'isFraud'
    label_col = None
    for c in ["y_true", "is_fraud", "isFraud"]:
        if c in pred_df.columns:
            label_col = c
            break
    if label_col is None:
        discrepancies.append("predictions.csv missing ground truth column ('y_true', 'is_fraud', or 'isFraud')")
        return False, {
            "is_reconciled": False,
            "discrepancies": discrepancies,
            "discrepancies_count": len(discrepancies),
            "run_id": run_id,
            "data_kind": manifest.get("data_kind", "unknown"),
            "provenance_status": manifest.get("provenance", {}).get("verification_status", "UNVERIFIED") if manifest.get("provenance") else "UNVERIFIED",
        }, discrepancies

    # Invariants on predictions schema
    required_cols = {"model", "probability", "prediction", "trans_id"}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        discrepancies.append(f"predictions.csv missing required columns: {missing_cols}")
        return False, {
            "is_reconciled": False,
            "discrepancies": discrepancies,
            "discrepancies_count": len(discrepancies),
            "run_id": run_id,
            "data_kind": manifest.get("data_kind", "unknown"),
            "provenance_status": manifest.get("provenance", {}).get("verification_status", "UNVERIFIED") if manifest.get("provenance") else "UNVERIFIED",
        }, discrepancies

    # Filter to pure out-of-sample evaluation rows if flagged
    eval_df = pred_df
    if "is_evaluation" in pred_df.columns:
        eval_df = pred_df[pred_df["is_evaluation"] == True]

    recomputed_metrics: Dict[str, Dict[str, Any]] = {}
    recorded_metrics: Dict[str, Dict[str, Any]] = {}

    if not metrics_path.exists():
        discrepancies.append(f"metrics.csv is missing from run directory: {run_dir}")
    else:
        rec_metrics_df = pd.read_csv(metrics_path)
        for _, r in rec_metrics_df.iterrows():
            m_name = r["model"]
            clean_name = m_name.replace("_model", "")
            recorded_metrics[clean_name] = r.to_dict()

    # 2. Recompute metrics per model
    for m in models_present:
        clean_m = m.replace("_model", "")
        m_rows = eval_df[eval_df["model"] == m]
        y_true = m_rows[label_col].to_numpy(dtype=int)
        y_prob = m_rows["probability"].to_numpy(dtype=float)
        y_pred = m_rows["prediction"].to_numpy(dtype=int)

        n = len(y_true)
        n_fraud = int(y_true.sum())
        n_legit = n - n_fraud

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        # Exact integer confusion matrix invariants
        if tp + fn != n_fraud:
            discrepancies.append(f"Model {m}: TP({tp}) + FN({fn}) = {tp+fn} != Total Frauds({n_fraud})")
        if tn + fp != n_legit:
            discrepancies.append(f"Model {m}: TN({tn}) + FP({fp}) = {tn+fp} != Total Legit({n_legit})")
        if tp + fp + tn + fn != n:
            discrepancies.append(f"Model {m}: TP+FP+TN+FN({tp+fp+tn+fn}) != N({n})")

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / n_fraud) if n_fraud > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        ap = float(average_precision_score(y_true, y_prob)) if n_fraud > 0 else 0.0
        brier = float(brier_score_loss(y_true, y_prob))
        ll = float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15)))

        recomputed_metrics[clean_m] = {
            "total_transactions": n,
            "total_frauds": n_fraud,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "overall_precision": precision,
            "overall_recall": recall,
            "overall_f1": f1,
            "average_precision": ap,
            "brier_score": brier,
            "log_loss": ll,
        }

        # Compare against recorded metrics if available
        if clean_m in recorded_metrics:
            rec = recorded_metrics[clean_m]
            # Strictly compare recorded confusion counts
            for count_key, calc_val in [
                ("true_positives", tp),
                ("false_positives", fp),
                ("false_negatives", fn),
                ("true_negatives", tn),
                ("total_transactions", n),
                ("total_frauds", n_fraud),
            ]:
                if count_key in rec and not pd.isna(rec[count_key]):
                    rec_count = int(rec[count_key])
                    if rec_count != calc_val:
                        discrepancies.append(
                            f"Model {clean_m} {count_key} mismatch: recorded {rec_count} vs computed {calc_val}"
                        )

            for metric_key in ["overall_precision", "overall_recall", "overall_f1", "average_precision", "brier_score"]:
                if metric_key in rec and not pd.isna(rec[metric_key]):
                    rec_val = float(rec[metric_key])
                    calc_val = recomputed_metrics[clean_m][metric_key]
                    if not math.isclose(rec_val, calc_val, abs_tol=tolerance):
                        discrepancies.append(
                            f"Model {clean_m} {metric_key} mismatch: recorded {rec_val:.6f} vs computed {calc_val:.6f}"
                        )

    # 3. Reconcile Review Budgets
    recomputed_budgets: List[Dict[str, Any]] = []
    if alert_budgets_path.exists():
        recorded_budgets_df = pd.read_csv(alert_budgets_path)
        budgets_tested = sorted(recorded_budgets_df["daily_budget"].unique())
        
        # Verify day column exists or compute it from step
        budget_df = eval_df.copy()
        if "day" not in budget_df.columns:
            if "step" in budget_df.columns:
                budget_df["day"] = (budget_df["step"] - 1).clip(lower=0) // 24
            else:
                budget_df["day"] = 0

        for m in models_present:
            clean_m = m.replace("_model", "")
            m_sub = budget_df[budget_df["model"] == m]
            total_m_fraud = int(m_sub[label_col].sum())

            for k in budgets_tested:
                tp_k = alerts_k = 0
                for _, day_group in m_sub.groupby("day"):
                    top = day_group.sort_values(
                        ["probability", "trans_id"], ascending=[False, True]
                    ).head(k)
                    tp_k += int(top[label_col].sum())
                    alerts_k += len(top)

                prec_k = float(tp_k / alerts_k) if alerts_k > 0 else 0.0
                rec_k = float(tp_k / total_m_fraud) if total_m_fraud > 0 else 0.0

                b_entry = {
                    "model": clean_m,
                    "daily_budget": int(k),
                    "alerts": alerts_k,
                    "detected_fraud": tp_k,
                    "precision": prec_k,
                    "recall": rec_k,
                }
                recomputed_budgets.append(b_entry)

                # Match against recorded budget entry
                rec_match = recorded_budgets_df[
                    (recorded_budgets_df["model"].str.replace("_model", "") == clean_m)
                    & (recorded_budgets_df["daily_budget"] == k)
                ]
                if not rec_match.empty:
                    rec_b = rec_match.iloc[0]
                    if int(rec_b["alerts"]) != alerts_k:
                        discrepancies.append(
                            f"Budget {k}/day alerts mismatch for {clean_m}: recorded {rec_b['alerts']} vs recomputed {alerts_k}"
                        )
                    if int(rec_b["detected_fraud"]) != tp_k:
                        discrepancies.append(
                            f"Budget {k}/day detected fraud mismatch for {clean_m}: recorded {rec_b['detected_fraud']} vs recomputed {tp_k}"
                        )

    # 4. Reconcile Compute Cost & Retrains against retrain_events.json
    recomputed_cost: List[Dict[str, Any]] = []
    retrain_events_path = run_dir / "retrain_events.json"
    actual_events_by_model: Dict[str, int] = {}
    if retrain_events_path.exists():
        try:
            events_data = json.loads(retrain_events_path.read_text(encoding="utf-8"))
            if isinstance(events_data, list):
                for ev in events_data:
                    ev_model = str(ev.get("model", "")).replace("_model", "")
                    actual_events_by_model[ev_model] = actual_events_by_model.get(ev_model, 0) + 1
        except Exception:
            pass

    if not compute_cost_path.exists():
        discrepancies.append(f"compute_cost.csv is missing from run directory: {run_dir}")
    else:
        rec_cost_df = pd.read_csv(compute_cost_path)
        for _, r in rec_cost_df.iterrows():
            m_clean = str(r["model"]).replace("_model", "")
            rec_retrains = int(r["retrains"])
            actual_retrains = actual_events_by_model.get(m_clean, 0)
            if rec_retrains != actual_retrains:
                discrepancies.append(
                    f"Model {m_clean} retrains mismatch: recorded {rec_retrains} in compute_cost.csv vs {actual_retrains} events in retrain_events.json"
                )
            recomputed_cost.append({
                "model": m_clean,
                "recorded_retrains": rec_retrains,
                "actual_retrains": actual_retrains,
                "recorded_training_seconds": float(r["training_seconds"]),
                "recorded_initial_seconds": float(r.get("initial_training_seconds", 0.0)),
            })


    is_reconciled = len(discrepancies) == 0

    report = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dataset_name": manifest.get("dataset"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "data_kind": manifest.get("data_kind"),
        "provenance_status": manifest.get("provenance", {}).get("verification_status") if manifest.get("provenance") else "UNVERIFIED",
        "total_records_scored": total_rows,
        "models_evaluated": list(recomputed_metrics.keys()),
        "is_reconciled": is_reconciled,
        "discrepancies_count": len(discrepancies),
        "discrepancies": discrepancies,
        "metrics_table": recomputed_metrics,
        "alert_budgets_table": recomputed_budgets,
        "compute_cost_table": recomputed_cost,
    }

    return is_reconciled, report, discrepancies


def format_markdown_report(report: Dict[str, Any]) -> str:
    """Formats reconciliation findings into a clean Markdown table."""
    status_badge = "**STATUS: RECONCILED (PASS)**" if report["is_reconciled"] else f"**STATUS: MISMATCH ({report['discrepancies_count']} discrepancies)**"
    lines = [
        f"# Deterministic Metrics Reconciliation Report",
        f"",
        f"- **Run ID**: `{report['run_id']}`",
        f"- **Provenance**: `{report['provenance_status']}` ({report['data_kind']})",
        f"- **Dataset SHA-256**: `{report.get('dataset_sha256')}`",
        f"- **Reconciliation Outcome**: {status_badge}",
        f"",
        f"## 1. Headline Metric Reconciliation",
        f"",
        f"| Model | Total Tx | Frauds | TP | FP | FN | TN | Precision | Recall | F1-Score | Average Precision (AP) | Brier Score |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model, m in report.get("metrics_table", {}).items():
        lines.append(
            f"| **{model}** | {m['total_transactions']:,} | {m['total_frauds']:,} | "
            f"{m['true_positives']:,} | {m['false_positives']:,} | {m['false_negatives']:,} | {m['true_negatives']:,} | "
            f"{m['overall_precision']:.4f} | {m['overall_recall']:.4f} | {m['overall_f1']:.4f} | "
            f"**{m['average_precision']:.4f}** | {m['brier_score']:.4f} |"
        )

    lines.extend([
        f"",
        f"## 2. Review Capacity Budget Reconciliation",
        f"",
        f"| Model | Budget (alerts/day) | Alerts Examined | Detected Fraud | Precision | Recall |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for b in report.get("alert_budgets_table", []):
        lines.append(
            f"| **{b['model']}** | {b['daily_budget']} | {b['alerts']:,} | {b['detected_fraud']:,} | "
            f"{b['precision']*100:.2f}% | {b['recall']*100:.2f}% |"
        )

    if report.get("compute_cost_table"):
        lines.extend([
            f"",
            f"## 3. Compute Cost Reconciliation",
            f"",
            f"| Model | Retrains | Retrain Time (s) | Initial Fit Time (s) |",
            f"| :--- | :---: | :---: | :---: |",
        ])
        for c in report["compute_cost_table"]:
            lines.append(
                f"| **{c['model']}** | {c['recorded_retrains']} | {c['recorded_training_seconds']:.2f}s | {c['recorded_initial_seconds']:.2f}s |"
            )

    if report["discrepancies"]:
        lines.extend([
            f"",
            f"## Discrepancies Found ({len(report['discrepancies'])})",
            f"",
        ])
        for d in report["discrepancies"]:
            lines.append(f"- ❌ {d}")

    return "\n".join(lines)


def recompute_confidence_intervals(
    run_dir: Path,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Recomputes paired moving-block bootstrap intervals directly from prediction ledger."""
    from src.reporting import block_bootstrap_metric_delta

    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions.csv at {pred_path}")
    df = pd.read_csv(pred_path, dtype={"trans_id": str})
    results = {}
    models = sorted(df["model"].unique())
    for m in models:
        if m != "static":
            results[f"{m}_vs_static_f1"] = block_bootstrap_metric_delta(
                df, m, baseline="static", metric="f1", n_bootstrap=n_bootstrap, seed=seed
            )
            results[f"{m}_vs_static_ap"] = block_bootstrap_metric_delta(
                df, m, baseline="static", metric="average_precision", n_bootstrap=n_bootstrap, seed=seed
            )
            results[f"{m}_vs_static_recall_100"] = block_bootstrap_metric_delta(
                df, m, baseline="static", metric="recall_at_budget", daily_budget=100, n_bootstrap=n_bootstrap, seed=seed
            )
    if "adaptive" in models and "weekly" in models:
        results["h1_adaptive_vs_weekly_recall_100"] = block_bootstrap_metric_delta(
            df, "adaptive", baseline="weekly", metric="recall_at_budget", daily_budget=100, non_inferiority_margin=0.01, n_bootstrap=n_bootstrap, seed=seed
        )
        results["h1_adaptive_vs_weekly_ap"] = block_bootstrap_metric_delta(
            df, "adaptive", baseline="weekly", metric="average_precision", non_inferiority_margin=0.01, n_bootstrap=n_bootstrap, seed=seed
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Deterministic metrics reconciliation command (R02 & R05)")
    parser.add_argument(
        "--run-dir",
        default="results/benchmark_runs/RUN_63D7CA7F68",
        help="Path to experiment run output directory containing predictions.csv and manifest.json",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Numerical tolerance for metric float comparison (default: 1e-4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON verification report",
    )
    parser.add_argument(
        "--bootstrap-intervals",
        action="store_true",
        help="Recompute and display paired moving-block bootstrap confidence intervals (R05)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap iterations when computing intervals (default: 200)",
    )
    parser.add_argument(
        "--output-file",
        help="Optional path to write the formatted markdown reconciliation report",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    if not run_path.exists():
        print(f"Error: Run directory not found: {run_path}", file=sys.stderr)
        sys.exit(1)

    is_reconciled, report, discrepancies = reconcile_run_directory(run_path, tolerance=args.tolerance)

    if args.bootstrap_intervals:
        print(f"Recomputing paired moving-block bootstrap confidence intervals ({args.n_bootstrap} resamples)...")
        ci_results = recompute_confidence_intervals(run_path, n_bootstrap=args.n_bootstrap)
        report["confidence_intervals"] = ci_results
        if not args.json:
            print("\n## Paired Moving-Block Bootstrap Confidence Intervals (R05)")
            for key, val in ci_results.items():
                print(f"- **{key}** ({val['metric']}):")
                print(f"  - Point delta: {val['point_estimate']:.5f}")
                print(f"  - Mean bootstrap delta: {val['mean_bootstrap_delta']:.5f}")
                print(f"  - 95% CI: [{val['ci_95'][0]:.5f}, {val['ci_95'][1]:.5f}]")
                if "non_inferiority_margin" in val:
                    print(f"  - Non-inferiority margin: {val['non_inferiority_margin']}")
                    print(f"  - Non-inferior: {val['is_non_inferior']} (empirical p-val: {val['empirical_p_value']:.4f})")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        md = format_markdown_report(report)
        print(md)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(format_markdown_report(report), encoding="utf-8")
        print(f"\nSaved reconciliation report to: {out_path}")

    sys.exit(0 if is_reconciled else 1)


if __name__ == "__main__":
    main()
