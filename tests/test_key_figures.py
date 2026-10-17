"""Reports must consume explicit saved run artifacts, never historical logs."""

import json
import pandas as pd
import pytest
from src.reporting import generate_report, block_bootstrap_delta


def test_report_requires_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_report(tmp_path)


def test_generate_report_from_prediction_ledger(tmp_path):
    rows = []
    for model in ["static", "adaptive"]:
        for t in range(3):
            for i in range(20):
                y = int(i % 5 == 0)
                rows.append(
                    dict(
                        seed=42,
                        window=2,
                        slice_index=t + 2,
                        day=t,
                        trans_id=f"{t}-{i}",
                        model=model,
                        y_true=y,
                        probability=0.8 if y else 0.1,
                        prediction=y,
                    )
                )
    pd.DataFrame(rows).to_csv(tmp_path / "predictions.csv", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_id": "TEST", "data_kind": "synthetic"})
    )
    (tmp_path / "retrain_events.json").write_text("[]")
    summary = generate_report(tmp_path)
    assert summary["adaptive_model"]["overall_f1"] == 1
    assert (tmp_path / "report.html").exists() and (tmp_path / "overview.png").exists()
    assert "synthetic" in (tmp_path / "report.html").read_text()
    ci = json.loads((tmp_path / "confidence_intervals.json").read_text())
    assert ci["adaptive"]["ci_95"] == [0, 0]


def test_bootstrap_requires_matching_predictions():
    df = pd.DataFrame(
        [
            dict(
                seed=1,
                window=1,
                slice_index=1,
                trans_id="a",
                model="adaptive",
                y_true=1,
                prediction=1,
            )
        ]
    )
    with pytest.raises(ValueError, match="paired"):
        block_bootstrap_delta(df, "adaptive")
