"""Run predefined strategy ablations over seeds and non-overlapping evaluation windows.
These are exploratory comparisons. Select settings on development periods and confirm
once on a separately reserved future period before making generalization claims.
"""

import argparse
import itertools
import json
from pathlib import Path
import pandas as pd
from src.pipeline import run_pipeline
from src.reporting import block_bootstrap_delta, metrics, budget_metrics


def run_study(
    dataset,
    output_dir,
    seeds=(42, 101, 202, 303, 404),
    train_windows=(5, 10, 15),
    test_slices=5,
    windows=(None, 2, 4, 8),
    half_lives=(None,),
    budgets=(None,),
    delays=(0,),
    pre_transaction_modes=(False,),
    data_kind="real",
    identity_path=None,
    all_features=False,
    history_features=False,
    exclude_card_identifiers=False,
    latency_mode="fixed",
    lognormal_mu=1.38,
    lognormal_sigma=0.45,
    threshold_objective="validation_f1",
    operational_daily_budget=None,
):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if not train_windows or len(set(train_windows)) != len(train_windows):
        raise ValueError("Training windows must be nonempty and unique")
    train_windows = sorted(train_windows)
    if len(train_windows) > 1 and any(
        b < a + test_slices for a, b in zip(train_windows, train_windows[1:])
    ):
        raise ValueError("Evaluation periods must not overlap")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    runs = []
    for window, half_life, budget, delay, pre_only in itertools.product(
        windows, half_lives, budgets, delays, pre_transaction_modes
    ):
        frames = []
        events = []
        for seed, start in itertools.product(seeds, train_windows):
            r = run_pipeline(
                dataset,
                output_dir=output_dir / "runs",
                seed=seed,
                n_train_slices=start,
                evaluation_slices=test_slices,
                adaptive_window=window,
                half_life_slices=half_life,
                max_retrains=budget,
                delay_slices=delay,
                pre_transaction_only=pre_only,
                data_kind=data_kind,
                identity_path=identity_path,
                all_features=all_features,
                history_features=history_features,
                exclude_card_identifiers=exclude_card_identifiers,
                latency_mode=latency_mode,
                threshold_objective=threshold_objective,
                operational_daily_budget=operational_daily_budget,
                lognormal_mu=lognormal_mu,
                lognormal_sigma=lognormal_sigma,
            )
            frames.append(
                pd.read_csv(r["run_dir"] / "predictions.csv", dtype={"trans_id": str})
            )
            runs.append(str(r["run_dir"].relative_to(output_dir)))
            events.extend(
                dict(e, seed=seed, window=start)
                for e in json.loads((r["run_dir"] / "retrain_events.json").read_text())
            )
        predictions = pd.concat(frames, ignore_index=True)
        budgets_summary = budget_metrics(predictions)
        config = {
            "window_slices": window,
            "half_life_slices": half_life,
            "max_retrains": budget,
            "delay_slices": delay,
            "pre_transaction_only": pre_only,
            "all_features": all_features,
            "history_features": history_features,
            "exclude_identifiers": exclude_card_identifiers,
            "latency_mode": latency_mode,
        }
        for model in predictions.model.unique():
            row = dict(config, model=model)
            per_seed = [
                metrics(g)
                for _, g in predictions[predictions.model == model].groupby("seed")
            ]
            for metric in [
                "overall_precision",
                "overall_recall",
                "overall_f1",
                "overall_auc_pr",
                "brier_score",
                "false_positives",
                "false_negatives",
            ]:
                values = [x[metric] for x in per_seed if x[metric] is not None]
                row["mean_seed_" + metric.replace("overall_", "")] = (
                    sum(values) / len(values) if values else None
                )
            model_events = [e for e in events if e["model"] == model]
            row["mean_retrains_per_run"] = len(model_events) / (
                len(seeds) * len(train_windows)
            )
            row["mean_training_seconds_per_run"] = sum(
                e["duration_seconds"] for e in model_events
            ) / (len(seeds) * len(train_windows))
            row["daily_review_budgets"] = budgets_summary[
                budgets_summary.model == model
            ].to_dict("records")
            row["run_ids"] = [
                Path(r).name for r in runs[-len(seeds) * len(train_windows) :]
            ]
            if model != "static":
                row["paired_f1_delta"] = block_bootstrap_delta(predictions, model)
            summary.append(row)
    result = {
        "protocol": "predefined exploratory seed/time-window comparisons; no automatic winner selection",
        "seeds": list(seeds),
        "train_windows": list(train_windows),
        "test_slices": test_slices,
        "runs": runs,
        "comparisons": summary,
        "data_kind": data_kind,
        "version": "0.4.1",
    }
    (output_dir / "study.json").write_text(json.dumps(result, indent=2))
    pd.json_normalize(summary).to_csv(output_dir / "study.csv", index=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    p.add_argument("--train-windows", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--test-slices", type=int, default=5)
    p.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8],
        help="0 means all history",
    )
    p.add_argument(
        "--half-lives",
        type=float,
        nargs="+",
        default=[0],
        help="0 means no age weighting",
    )
    p.add_argument(
        "--budgets", type=int, nargs="+", default=[-1], help="-1 means no retrain cap"
    )
    p.add_argument("--delays", type=int, nargs="+", default=[0])
    p.add_argument("--paysim-ablation", action="store_true")
    p.add_argument("--identity")
    p.add_argument("--all-features", action="store_true")
    p.add_argument("--history-features", action="store_true")
    p.add_argument("--exclude-identifiers", action="store_true")
    p.add_argument("--latency-mode", choices=["fixed", "lognormal"], default="fixed")
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
        help="Target daily alert capacity for review budget thresholding",
    )
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    run_study(
        a.dataset,
        a.output_dir,
        a.seeds,
        a.train_windows,
        a.test_slices,
        [w or None for w in a.windows],
        [h or None for h in a.half_lives],
        [b if b >= 0 else None for b in a.budgets],
        a.delays,
        [False, True] if a.paysim_ablation else [False],
        data_kind="synthetic" if a.synthetic else "real",
        identity_path=a.identity,
        all_features=a.all_features,
        history_features=a.history_features,
        exclude_card_identifiers=a.exclude_identifiers,
        latency_mode=a.latency_mode,
        lognormal_mu=a.lognormal_mu,
        lognormal_sigma=a.lognormal_sigma,
        threshold_objective=a.threshold_objective,
        operational_daily_budget=a.operational_daily_budget,
    )


if __name__ == "__main__":
    main()
