"""Compare label-delay regimes through the shared experiment engine."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from src.config import METRICS_DIR, FIGURES_DIR


def generate_latency_audit_plots(
    all_results, output_filename="delayed_feedback_impact.png"
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in all_results:
        axes[0].scatter(r["delay_slices"], r["oos_recall"], label=r["regime_name"])
        axes[1].scatter(r["training_seconds"], r["oos_recall"], label=r["regime_name"])
    axes[0].set(
        xlabel="Delay in slices (nominal for lognormal)",
        ylabel="Recall",
        title="Observed delayed-label regimes",
    )
    axes[1].set(
        xlabel="Measured retraining seconds", ylabel="Recall", title="Compute trade-off"
    )
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    path = Path(output_filename)
    if not path.is_absolute():
        path = FIGURES_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(data_file_path=None, regimes=None, output_dir=None, data_kind="unverified"):
    from src.pipeline import run_pipeline

    if data_file_path is None:
        raise ValueError(
            "Specify a dataset path; use fraud-latency-audit --dataset PATH"
        )
    regimes = regimes or [
        dict(name=f"Fixed {d} slices", delay_slices=d, mode="fixed") for d in [0, 4, 8]
    ] + [dict(name="Stochastic lognormal", delay_slices=4, mode="lognormal")]
    root = Path(output_dir or METRICS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for r in regimes:
        run = run_pipeline(
            data_file_path,
            output_dir=root / "delayed_runs",
            delay_slices=r["delay_slices"],
            latency_mode=r["mode"],
            lognormal_mu=r.get("lognormal_mu", 1.38),
            lognormal_sigma=r.get("lognormal_sigma", 0.45),
            data_kind=data_kind,
        )
        m = run["overall_summary"]["out_of_sample_stream"]["adaptive_model"]
        events = json.loads((run["run_dir"] / "retrain_events.json").read_text())
        results.append(
            dict(
                regime_name=r["name"],
                delay_slices=r["delay_slices"],
                latency_mode=r["mode"],
                lognormal_mu=r.get("lognormal_mu", 1.38),
                lognormal_sigma=r.get("lognormal_sigma", 0.45),
                run_id=run["run_id"],
                run_dir=str(run["run_dir"]),
                retrains_count=run["retrains_count"],
                oos_tp=m["true_positives"],
                oos_fp=m["false_positives"],
                oos_fn=m["false_negatives"],
                oos_tn=m["true_negatives"],
                oos_precision=m["overall_precision"],
                oos_recall=m["overall_recall"],
                oos_f1=m["overall_f1"],
                oos_aucpr=m["overall_auc_pr"],
                training_seconds=sum(
                    e["duration_seconds"] for e in events if e["model"] == "adaptive"
                ),
            )
        )
    (root / "delayed_feedback_benchmark.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    generate_latency_audit_plots(
        results,
        str(Path(output_dir) / "delayed_feedback_impact.png")
        if output_dir
        else "delayed_feedback_impact.png",
    )
    return results


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--delays", nargs="+", type=int, default=[0, 4, 8])
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    results = main(
        a.dataset,
        [
            dict(name=f"Fixed {d} slices", delay_slices=d, mode="fixed")
            for d in a.delays
        ],
        a.output_dir,
        "synthetic" if a.synthetic else "real",
    )
    print(Path(a.output_dir) / "delayed_feedback_benchmark.json")


if __name__ == "__main__":
    cli()
