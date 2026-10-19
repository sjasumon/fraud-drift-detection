"""Run predefined strategy ablations over seeds and non-overlapping evaluation windows.
These are exploratory comparisons. Select settings on development periods and confirm
once on a separately reserved future period before making generalization claims.
"""

import argparse
import itertools
import json
from pathlib import Path
import sys
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
        "version": "0.7.0",
    }
    (output_dir / "study.json").write_text(json.dumps(result, indent=2))
    pd.json_normalize(summary).to_csv(output_dir / "study.csv", index=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="Path to study JSON configuration file")
    p.add_argument("--dataset", required=False)
    p.add_argument("--output-dir", required=False)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--train-windows", type=int, nargs="+", default=None)
    p.add_argument("--test-slices", type=int, default=None)
    p.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=None,
        help="0 means all history",
    )
    p.add_argument(
        "--half-lives",
        type=float,
        nargs="+",
        default=None,
        help="0 means no age weighting",
    )
    p.add_argument(
        "--budgets", type=int, nargs="+", default=None, help="-1 means no retrain cap"
    )
    p.add_argument("--delays", type=int, nargs="+", default=None)
    p.add_argument("--paysim-ablation", action="store_true", default=None)
    p.add_argument("--identity", default=None)
    p.add_argument("--all-features", action="store_true", default=None)
    p.add_argument("--history-features", action="store_true", default=None)
    p.add_argument("--exclude-identifiers", action="store_true", default=None)
    p.add_argument("--latency-mode", choices=["fixed", "lognormal"], default=None)
    p.add_argument("--lognormal-mu", type=float, default=None)
    p.add_argument("--lognormal-sigma", type=float, default=None)
    p.add_argument(
        "--threshold-objective",
        choices=["validation_f1", "review_budget_recall"],
        default=None,
        help="Decision threshold optimization policy ('validation_f1' or 'review_budget_recall')",
    )
    p.add_argument(
        "--operational-daily-budget",
        type=int,
        default=None,
        help="Target daily alert capacity for review budget thresholding",
    )
    p.add_argument("--synthetic", action="store_true", default=None)

    explicit_flags = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            flag = arg.lstrip("-").split("=")[0].replace("-", "_")
            explicit_flags.add(flag)

    a = p.parse_args()

    cfg = {}
    if a.config:
        cfg_path = Path(a.config)
        if not cfg_path.exists():
            p.error(f"Configuration file not found: {a.config}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    dataset = a.dataset or cfg.get("dataset")
    if not dataset:
        p.error("Either --dataset or a --config defining 'dataset' must be provided")

    output_dir = a.output_dir or cfg.get("output_dir")
    if not output_dir:
        p.error("Either --output-dir or a --config defining 'output_dir' must be provided")

    seeds = a.seeds if "seeds" in explicit_flags and a.seeds is not None else cfg.get("seeds", [42, 101, 202, 303, 404])
    train_windows = a.train_windows if "train_windows" in explicit_flags and a.train_windows is not None else cfg.get("train_windows", [5, 10, 15])
    test_slices = a.test_slices if "test_slices" in explicit_flags and a.test_slices is not None else cfg.get("test_slices", 5)
    windows = a.windows if "windows" in explicit_flags and a.windows is not None else cfg.get("windows", [0, 2, 4, 8])
    half_lives = a.half_lives if "half_lives" in explicit_flags and a.half_lives is not None else cfg.get("half_lives", [0])
    budgets = a.budgets if "budgets" in explicit_flags and a.budgets is not None else cfg.get("budgets", [-1])
    delays = a.delays if "delays" in explicit_flags and a.delays is not None else cfg.get("delays", [0])
    paysim_ablation = a.paysim_ablation if "paysim_ablation" in explicit_flags and a.paysim_ablation is not None else cfg.get("paysim_ablation", False)
    identity = a.identity if "identity" in explicit_flags and a.identity is not None else cfg.get("identity", None)
    all_features = a.all_features if "all_features" in explicit_flags and a.all_features is not None else cfg.get("all_features", False)
    history_features = a.history_features if "history_features" in explicit_flags and a.history_features is not None else cfg.get("history_features", False)
    exclude_identifiers = a.exclude_identifiers if "exclude_identifiers" in explicit_flags and a.exclude_identifiers is not None else cfg.get("exclude_identifiers", False)
    latency_mode = a.latency_mode if "latency_mode" in explicit_flags and a.latency_mode is not None else cfg.get("latency_mode", "fixed")
    lognormal_mu = a.lognormal_mu if "lognormal_mu" in explicit_flags and a.lognormal_mu is not None else cfg.get("lognormal_mu", 1.38)
    lognormal_sigma = a.lognormal_sigma if "lognormal_sigma" in explicit_flags and a.lognormal_sigma is not None else cfg.get("lognormal_sigma", 0.45)
    threshold_objective = a.threshold_objective if "threshold_objective" in explicit_flags and a.threshold_objective is not None else cfg.get("threshold_objective", "validation_f1")
    operational_daily_budget = a.operational_daily_budget if "operational_daily_budget" in explicit_flags and a.operational_daily_budget is not None else cfg.get("operational_daily_budget", None)
    synthetic = a.synthetic if "synthetic" in explicit_flags and a.synthetic is not None else (cfg.get("data_kind") == "synthetic" or cfg.get("synthetic", False))

    run_study(
        dataset,
        output_dir,
        seeds,
        train_windows,
        test_slices,
        [w or None for w in windows],
        [h or None for h in half_lives],
        [b if b >= 0 else None for b in budgets],
        delays,
        [False, True] if paysim_ablation else [False],
        data_kind="synthetic" if synthetic else "real",
        identity_path=identity,
        all_features=all_features,
        history_features=history_features,
        exclude_card_identifiers=exclude_identifiers,
        latency_mode=latency_mode,
        lognormal_mu=lognormal_mu,
        lognormal_sigma=lognormal_sigma,
        threshold_objective=threshold_objective,
        operational_daily_budget=operational_daily_budget,
    )


if __name__ == "__main__":
    main()

