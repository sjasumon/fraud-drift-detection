"""Metrics and plots derived exclusively from a run's saved prediction ledger."""

import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def metrics(df):
    y, p, pred = (
        df.y_true.to_numpy(),
        df.probability.to_numpy(),
        df.prediction.to_numpy(),
    )
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    ap = float(average_precision_score(y, p)) if y.sum() else None
    return dict(
        overall_precision=float(precision_score(y, pred, zero_division=0)),
        overall_recall=float(recall_score(y, pred, zero_division=0)),
        overall_f1=float(f1_score(y, pred, zero_division=0)),
        average_precision=ap,
        overall_auc_pr=ap,
        brier_score=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        total_frauds=tp + fn,
        total_transactions=len(df),
        evaluation_scope="Pure Out-of-Sample",
        is_pure_out_of_sample=True,
    )


def budget_metrics(df, budgets=(100, 500)):
    output = []
    for name, group in df.groupby("model"):
        for k in budgets:
            tp = alerts = 0
            for _, day in group.groupby(
                [c for c in ["seed", "window", "day"] if c in group]
            ):
                top = day.sort_values(
                    ["probability", "trans_id"], ascending=[False, True]
                ).head(k)
                tp += int(top.y_true.sum())
                alerts += len(top)
            total = int(group.y_true.sum())
            output.append(
                dict(
                    model=name,
                    daily_budget=k,
                    alerts=alerts,
                    detected_fraud=tp,
                    precision=tp / alerts if alerts else 0,
                    recall=tp / total if total else 0,
                    interpretation="retrospective daily ranking; requires batch review",
                )
            )
    return pd.DataFrame(output)


def block_bootstrap_metric_delta(
    predictions,
    candidate_model,
    baseline_model="static",
    baseline=None,
    metric="average_precision",
    daily_budget=100,
    capacity_policy="retrospective_batch",
    n_bootstrap=500,
    seed=42,
    block_length=2,
    non_inferiority_margin=0.01,
):
    """Paired moving-block bootstrap with the SAME temporal draws across seeds.

    Supports 'average_precision', 'recall_at_budget', and 'f1' metrics.
    Preserves matched original probabilities, decisions, and temporal pairing.
    Computes percentile confidence intervals and one-sided non-inferiority test for H1.
    """
    if baseline is not None:
        baseline_model = baseline
    if n_bootstrap < 1 or block_length < 1:
        raise ValueError("n_bootstrap and block_length must be positive")

    pred = predictions.copy()
    if "seed" not in pred.columns:
        pred["seed"] = 0
    if "window" not in pred.columns:
        pred["window"] = 0
    if "day" not in pred.columns:
        pred["day"] = pred["slice_index"]

    key = [c for c in ["seed", "window", "slice_index", "trans_id"] if c in pred.columns]
    if "trans_id" not in key or "slice_index" not in key:
        raise ValueError("predictions must contain at least 'slice_index' and 'trans_id'")

    a = pred[pred.model == candidate_model]
    b = pred[pred.model == baseline_model]
    if a.empty:
        raise ValueError(
            f"Candidate model '{candidate_model}' not found in predictions; cannot create paired comparison"
        )
    if b.empty:
        raise ValueError(
            f"Baseline model '{baseline_model}' not found in predictions; cannot create paired comparison"
        )

    paired = a.merge(b, on=key, suffixes=("_a", "_b"), validate="one_to_one")
    if len(paired) != len(a) or len(paired) != len(b):
        raise ValueError("Predictions must be paired exactly")
    if paired.empty:
        raise ValueError("Paired predictions must not be empty")
    if not paired.y_true_a.equals(paired.y_true_b):
        raise ValueError("Paired ground-truth labels differ")

    seeds = sorted(paired.seed.unique())
    windows = sorted(paired.window.unique())
    timeline = {
        w: sorted(paired.loc[paired.window == w, "slice_index"].unique())
        for w in windows
    }
    n_periods = sum(len(ts) for ts in timeline.values())

    has_prob = "probability_a" in paired.columns and "probability_b" in paired.columns
    is_identical = (candidate_model == baseline_model) or (
        (np.allclose(paired.probability_a.to_numpy(), paired.probability_b.to_numpy(), atol=1e-12) if has_prob else True)
        and np.array_equal(paired.prediction_a.to_numpy(), paired.prediction_b.to_numpy())
    )

    if is_identical:
        return {
            "candidate_model": candidate_model,
            "baseline_model": baseline_model,
            "metric": metric,
            "point_estimate": 0.0,
            "mean_delta": 0.0,
            "mean_bootstrap_delta": 0.0,
            "ci_95": [0.0, 0.0],
            "ci_99": [0.0, 0.0],
            "non_inferiority_margin": non_inferiority_margin,
            "empirical_p_value": 0.0,
            "is_non_inferior": True,
            "n_bootstrap": n_bootstrap,
            "n_seeds": len(seeds),
            "temporal_blocks": n_periods,
            "block_length_slices": block_length,
            "daily_budget": daily_budget if "budget" in metric else None,
            "capacity_policy": capacity_policy if "budget" in metric else None,
            "method": f"paired circular moving-block bootstrap; metric={metric}; shared time draws across seeds",
            "status": "exploratory_short_stream"
            if n_periods < 6
            else "conditional_on_observed_stream",
            "limitation": "Conditional on supplied periods; seeds do not add independent transaction populations.",
        }

    clean_metric = metric.lower().replace("-", "_")

    if clean_metric in ("average_precision", "ap", "auc_pr"):
        ap_a = float(average_precision_score(paired.y_true_a, paired.probability_a))
        ap_b = float(average_precision_score(paired.y_true_b, paired.probability_b))
        point_delta = ap_a - ap_b

        slice_lookup = {int(s): {} for s in seeds}
        for (s, w, t), g in paired.groupby(["seed", "window", "slice_index"]):
            slice_lookup[int(s)][(w, t)] = (
                g.y_true_a.to_numpy(),
                g.probability_a.to_numpy(),
                g.probability_b.to_numpy(),
            )

        temporal_rng = np.random.default_rng(seed)
        seed_rng = np.random.default_rng(seed + 1)
        deltas = []
        for _ in range(n_bootstrap):
            selected = []
            for w, ts in timeline.items():
                length = min(block_length, len(ts))
                draw = []
                while len(draw) < len(ts):
                    start = int(temporal_rng.integers(len(ts)))
                    draw.extend(ts[(start + j) % len(ts)] for j in range(length))
                selected.extend((w, t) for t in draw[: len(ts)])
            sampled_seeds = seed_rng.choice(seeds, size=len(seeds), replace=True)

            draw_deltas = []
            for s in sampled_seeds:
                s_dict = slice_lookup[int(s)]
                y_arr = np.concatenate([s_dict[k][0] for k in selected])
                pa_arr = np.concatenate([s_dict[k][1] for k in selected])
                pb_arr = np.concatenate([s_dict[k][2] for k in selected])
                if y_arr.sum() > 0:
                    d_ap = float(
                        average_precision_score(y_arr, pa_arr)
                        - average_precision_score(y_arr, pb_arr)
                    )
                else:
                    d_ap = 0.0
                draw_deltas.append(d_ap)
            deltas.append(float(np.mean(draw_deltas)))

    elif "budget" in clean_metric or ("recall" in clean_metric and "f1" not in clean_metric):
        from src.capacity_policy import allocate_batch_ranking, allocate_online_stream

        day_counts = []
        day_keys = [c for c in ["seed", "window", "slice_index", "day"] if c in paired.columns]
        for key_vals, g in paired.groupby(day_keys):
            s = key_vals[0] if "seed" in day_keys else 0
            w = key_vals[1] if "window" in day_keys else 0
            sl = key_vals[2] if "slice_index" in day_keys else key_vals[0]
            total_fraud = int(g.y_true_a.sum())

            df_a = pd.DataFrame({
                "trans_id": g.trans_id,
                "probability": g.probability_a,
                "y_true": g.y_true_a,
                "prediction": g.prediction_a,
            })
            df_b = pd.DataFrame({
                "trans_id": g.trans_id,
                "probability": g.probability_b,
                "y_true": g.y_true_b,
                "prediction": g.prediction_b,
            })

            if capacity_policy == "retrospective_batch":
                top_a, _ = allocate_batch_ranking(df_a, daily_budget=daily_budget)
                top_b, _ = allocate_batch_ranking(df_b, daily_budget=daily_budget)
            else:
                top_a, _ = allocate_online_stream(df_a, daily_budget=daily_budget)
                top_b, _ = allocate_online_stream(df_b, daily_budget=daily_budget)

            tp_a = int(top_a.y_true.sum()) if not top_a.empty else 0
            tp_b = int(top_b.y_true.sum()) if not top_b.empty else 0
            day_counts.append((s, w, sl, tp_a, tp_b, total_fraud))

        counts_df = pd.DataFrame(
            day_counts,
            columns=["seed", "window", "slice_index", "tp_a", "tp_b", "total_fraud"],
        )
        slice_agg = (
            counts_df.groupby(["seed", "window", "slice_index"])[["tp_a", "tp_b", "total_fraud"]]
            .sum()
            .reset_index()
        )

        slice_lookup = {int(s): {} for s in seeds}
        for row in slice_agg.itertuples(index=False):
            slice_lookup[int(row.seed)][(row.window, row.slice_index)] = np.array(
                [row.tp_a, row.tp_b, row.total_fraud], dtype=float
            )

        total_frauds_all = counts_df.total_fraud.sum()
        point_delta = float(
            (counts_df.tp_a.sum() - counts_df.tp_b.sum()) / max(1, total_frauds_all)
        )

        temporal_rng = np.random.default_rng(seed)
        seed_rng = np.random.default_rng(seed + 1)
        deltas = []
        for _ in range(n_bootstrap):
            selected = []
            for w, ts in timeline.items():
                length = min(block_length, len(ts))
                draw = []
                while len(draw) < len(ts):
                    start = int(temporal_rng.integers(len(ts)))
                    draw.extend(ts[(start + j) % len(ts)] for j in range(length))
                selected.extend((w, t) for t in draw[: len(ts)])
            sampled_seeds = seed_rng.choice(seeds, size=len(seeds), replace=True)

            draw_deltas = []
            for s in sampled_seeds:
                s_dict = slice_lookup[int(s)]
                sum_counts = sum((s_dict[k] for k in selected), np.zeros(3))
                tp_a_sum, tp_b_sum, fraud_sum = sum_counts
                draw_deltas.append(float((tp_a_sum - tp_b_sum) / max(1.0, fraud_sum)))
            deltas.append(float(np.mean(draw_deltas)))

    else:
        # F1 metric
        rows = []
        for (s, w, t), g in paired.groupby(["seed", "window", "slice_index"]):
            y = g.y_true_a.to_numpy()
            values = []
            for c in ["prediction_a", "prediction_b"]:
                p_c = g[c].to_numpy()
                values.extend([
                    int(((y == 1) & (p_c == 1)).sum()),
                    int(((y == 0) & (p_c == 1)).sum()),
                    int(((y == 1) & (p_c == 0)).sum()),
                ])
            rows.append([s, w, t, *values])

        blocks = pd.DataFrame(
            rows,
            columns=["seed", "window", "slice_index", "ta", "fa", "na", "tb", "fb", "nb"],
        )
        slice_lookup = {int(s): {} for s in seeds}
        for row in blocks.itertuples(index=False):
            slice_lookup[int(row.seed)][(row.window, row.slice_index)] = np.array(
                row[3:], dtype=float
            )

        def delta_f1(counts):
            ta, fa, na, tb, fb, nb = counts
            return 2 * ta / max(1, 2 * ta + fa + na) - 2 * tb / max(1, 2 * tb + fb + nb)

        point_delta = float(
            np.mean([delta_f1(sum(v.values(), np.zeros(6))) for v in slice_lookup.values()])
        )

        temporal_rng = np.random.default_rng(seed)
        seed_rng = np.random.default_rng(seed + 1)
        deltas = []
        for _ in range(n_bootstrap):
            selected = []
            for w, ts in timeline.items():
                length = min(block_length, len(ts))
                draw = []
                while len(draw) < len(ts):
                    start = int(temporal_rng.integers(len(ts)))
                    draw.extend(ts[(start + j) % len(ts)] for j in range(length))
                selected.extend((w, t) for t in draw[: len(ts)])
            sampled = seed_rng.choice(seeds, size=len(seeds), replace=True)
            deltas.append(
                float(
                    np.mean(
                        [
                            delta_f1(sum((slice_lookup[int(s)][k] for k in selected), np.zeros(6)))
                            for s in sampled
                        ]
                    )
                )
            )

    ci_95 = np.quantile(deltas, [0.025, 0.975]).tolist()
    ci_99 = np.quantile(deltas, [0.005, 0.995]).tolist()
    emp_p = float(np.mean(np.array(deltas) <= -non_inferiority_margin))
    is_ni = bool(ci_95[0] >= -non_inferiority_margin)

    return {
        "candidate_model": candidate_model,
        "baseline_model": baseline_model,
        "metric": metric,
        "point_estimate": point_delta,
        "mean_delta": float(np.mean(deltas)),
        "mean_bootstrap_delta": float(np.mean(deltas)),
        "ci_95": ci_95,
        "ci_99": ci_99,
        "non_inferiority_margin": non_inferiority_margin,
        "empirical_p_value": emp_p,
        "is_non_inferior": is_ni,
        "n_bootstrap": n_bootstrap,
        "n_seeds": len(seeds),
        "temporal_blocks": n_periods,
        "block_length_slices": block_length,
        "daily_budget": daily_budget if "budget" in metric else None,
        "capacity_policy": capacity_policy if "budget" in metric else None,
        "method": f"paired circular moving-block bootstrap; metric={metric}; shared time draws across seeds",
        "status": "exploratory_short_stream"
        if n_periods < 6
        else "conditional_on_observed_stream",
        "limitation": "Conditional on supplied periods; seeds do not add independent transaction populations.",
    }


def block_bootstrap_delta(
    predictions, model, baseline="static", n_bootstrap=500, seed=42, block_length=2
):
    """Paired moving-block bootstrap for backwards compatibility (evaluates F1 delta)."""
    return block_bootstrap_metric_delta(
        predictions=predictions,
        candidate_model=model,
        baseline_model=baseline,
        metric="f1",
        n_bootstrap=n_bootstrap,
        seed=seed,
        block_length=block_length,
    )


def generate_report(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    df = pd.read_csv(run_dir / "predictions.csv", dtype={"trans_id": str})
    events = json.loads((run_dir / "retrain_events.json").read_text())
    summary = {name + "_model": metrics(g) for name, g in df.groupby("model")}
    (run_dir / "summary.json").write_text(
        json.dumps({"out_of_sample_stream": summary}, indent=2, allow_nan=False)
    )
    table = pd.DataFrame(summary).T
    table.to_csv(run_dir / "metrics.csv", index_label="model")
    budgets = budget_metrics(df)
    budgets.to_csv(run_dir / "alert_budgets.csv", index=False)
    weekly = []
    for (name, t), g in df.groupby(["model", "slice_index"]):
        weekly.append(dict(model=name, slice_index=t, **metrics(g)))
    weekly = pd.DataFrame(weekly)
    weekly.to_csv(run_dir / "weekly_metrics.csv", index=False)
    ci = {}
    for m in df.model.unique():
        if m != "static":
            ci[m] = block_bootstrap_delta(df, m)
            ci[f"{m}_vs_static_ap"] = block_bootstrap_metric_delta(
                df, m, baseline="static", metric="average_precision", n_bootstrap=200
            )
            ci[f"{m}_vs_static_recall_100"] = block_bootstrap_metric_delta(
                df, m, baseline="static", metric="recall_at_budget", daily_budget=100, n_bootstrap=200
            )
    if "adaptive" in df.model.unique() and "weekly" in df.model.unique():
        ci["h1_adaptive_vs_weekly_recall_100"] = block_bootstrap_metric_delta(
            df, "adaptive", baseline="weekly", metric="recall_at_budget", daily_budget=100, non_inferiority_margin=0.01, n_bootstrap=200
        )
        ci["h1_adaptive_vs_weekly_ap"] = block_bootstrap_metric_delta(
            df, "adaptive", baseline="weekly", metric="average_precision", non_inferiority_margin=0.01, n_bootstrap=200
        )
    (run_dir / "confidence_intervals.json").write_text(json.dumps(ci, indent=2))
    errors = df[
        ((df.y_true == 0) & (df.prediction == 1))
        | ((df.y_true == 1) & (df.prediction == 0))
    ].copy()
    errors["error_type"] = np.where(errors.y_true == 1, "missed_fraud", "false_alarm")
    errors.sort_values("probability", ascending=False).groupby(
        ["model", "error_type"], group_keys=False
    ).head(20).to_csv(run_dir / "error_examples.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for name, g in df.groupby("model"):
        precision, recall, _ = precision_recall_curve(g.y_true, g.probability)
        axes[0, 0].plot(recall, precision, label=name)
        week = weekly[weekly.model == name]
        axes[0, 1].plot(week.slice_index + 1, week.overall_f1, label=name)
        frac, mean = calibration_curve(
            g.y_true, g.probability, n_bins=8, strategy="quantile"
        )
        axes[1, 0].plot(mean, frac, marker=".", label=name)
        cost = sum(e["duration_seconds"] for e in events if e["model"] == name)
        axes[1, 1].scatter(cost, metrics(g)["overall_recall"], label=name)
    axes[0, 0].set(
        xlabel="Recall", ylabel="Precision", title="Out-of-sample precision–recall"
    )
    axes[0, 1].set(xlabel="Slice", ylabel="F1", title="Chronological performance")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1, 0].set(
        xlabel="Mean score", ylabel="Observed fraud fraction", title="Calibration"
    )
    axes[1, 1].set(
        xlabel="Retraining seconds", ylabel="Recall", title="Measured retraining cost"
    )
    for ax in axes.flat:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{manifest['run_id']} — {manifest['data_kind']} data")
    fig.tight_layout()
    fig.savefig(run_dir / "overview.png", dpi=150)
    plt.close(fig)
    costs = pd.DataFrame(
        [
            {
                "model": m,
                "retrains": sum(e["model"] == m for e in events),
                "training_seconds": sum(
                    e["duration_seconds"] for e in events if e["model"] == m
                ),
                "initial_training_seconds": manifest.get("initial_training_seconds", 0),
            }
            for m in df.model.unique()
        ]
    )
    costs.to_csv(run_dir / "compute_cost.csv", index=False)
    # These curves diagnose an operating point; future labels never select a deployed threshold.
    threshold_rows = []
    for name, g in df.groupby("model"):
        for threshold in np.linspace(0, 1, 21):
            decision = (g.probability >= threshold).astype(int)
            threshold_rows.append(
                dict(
                    model=name,
                    threshold=float(threshold),
                    precision=float(
                        precision_score(g.y_true, decision, zero_division=0)
                    ),
                    recall=float(recall_score(g.y_true, decision, zero_division=0)),
                    f1=float(f1_score(g.y_true, decision, zero_division=0)),
                    scope="OOS diagnostic only; not used for model selection",
                )
            )
    pd.DataFrame(threshold_rows).to_csv(
        run_dir / "threshold_diagnostics.csv", index=False
    )
    columns = [
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "average_precision",
        "brier_score",
        "false_positives",
        "false_negatives",
    ]
    page = f"""<!doctype html><html><meta charset="utf-8"><title>Fraud experiment report</title>
    <style>body{{font:16px system-ui;margin:40px;max-width:1200px;color:#182237}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}img{{width:100%}}pre{{white-space:pre-wrap}}</style>
    <h1>Fraud detection experiment</h1><p>{html.escape(manifest["run_id"])} · {html.escape(manifest["data_kind"])} data · out-of-sample only</p>
    <p>Thresholds and calibration use prior chronological validation data. Daily alert budgets are retrospective review rankings. Average precision is used consistently. Timestamp basis: {html.escape(manifest.get("time_basis", "dataset-relative"))}.</p>
    {table[columns].round(4).to_html()}<img src="overview.png" alt="PR, weekly F1, calibration and training cost plots">
    <h2>Daily review budgets</h2>{budgets.to_html(index=False)}<h2>Compute</h2>{costs.to_html(index=False)}
    <h2>Paired uncertainty</h2><pre>{html.escape(json.dumps(ci, indent=2))}</pre>
    <h2>Reproducibility</h2><pre>{html.escape(json.dumps(manifest, indent=2))}</pre>
    <p>Error examples: error_examples.csv. Predictions and labels: predictions.csv. Threshold diagnostics are descriptive only. No publication claims are inferred from synthetic runs.</p></html>"""
    (run_dir / "report.html").write_text(page)
    return summary
