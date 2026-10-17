"""Generate a clearly labelled small synthetic dataset and reproducible report."""

import argparse
from pathlib import Path
from src.data_generator import generate_paysim_stream
from src.pipeline import run_pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="results/demo")
    p.add_argument("--weeks", type=int, default=8)
    p.add_argument("--transactions-per-week", type=int, default=300)
    a = p.parse_args()
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "synthetic_paysim.csv"
    generate_paysim_stream(
        total_weeks=a.weeks,
        tx_per_week=a.transactions_per_week,
        fraud_rate=0.08,
        seed=42,
        output_path=dataset,
    )
    result = run_pipeline(
        dataset,
        output_dir=root / "runs",
        data_kind="synthetic",
        adaptive_window=4,
        pre_transaction_only=True,
        model_params={"n_estimators": 40},
    )
    print(result["run_dir"] / "report.html")


if __name__ == "__main__":
    main()
