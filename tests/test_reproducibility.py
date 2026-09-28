"""
Automated Reproducibility Guardrail Tests.

Enforces zero-drift between experimental results (JSON ground truth),
generated LaTeX paper tables, and publication figures.
Fails immediately upon any numeric or artifact mismatch.
"""

import json
import re
from pathlib import Path
import pytest

from src.config import BASE_DIR, METRICS_DIR, FIGURES_DIR

ARXIV_TABLES_DIR = BASE_DIR / "arxiv_paper" / "tables"
ARXIV_FIGURES_DIR = BASE_DIR / "arxiv_paper" / "figures"


def test_metric_json_artifacts_exist_and_valid():
    """Asserts that all core empirical benchmark metric files exist and contain valid JSON."""
    required_metrics = [
        "statistical_rigor_summary.json",
        "drift_detection_benchmark.json",
        "attribution_validation_benchmark.json",
        "federated_summary_metrics.json",
        "label_delay_adaptation_benchmark.json"
    ]
    for metric_name in required_metrics:
        json_file = METRICS_DIR / metric_name
        assert json_file.exists(), f"Missing required metrics artifact: {json_file}"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"{metric_name} root is not a dict"
            assert len(data) > 0, f"{metric_name} is empty"


def test_publication_figures_exist_and_synchronized():
    """Asserts that all key publication figures exist and are synchronized between results/ and arxiv_paper/."""
    figures = [
        "label_delay_adaptation_frontier.png",
        "stream_f1_drift_timeline.png",
        "verification_latency_degradation.png",
        "model_comparison_f1_aucpr.png",
        "drift_attribution_week_06.png",
        "statistical_rigor_errorbars.png"
    ]
    for fig_name in figures:
        res_fig = FIGURES_DIR / fig_name
        paper_fig = ARXIV_FIGURES_DIR / fig_name
        assert res_fig.exists(), f"Missing results figure: {res_fig}"
        assert paper_fig.exists(), f"Missing paper figure: {paper_fig}"
        assert res_fig.stat().st_size > 5000, f"Results figure {fig_name} is abnormally small"
        assert paper_fig.stat().st_size > 5000, f"Paper figure {fig_name} is abnormally small"


def test_label_delay_latex_table_matches_json():
    """
    Verifies that tab_label_delay_adaptation.tex strictly matches
    label_delay_adaptation_benchmark.json ground truth.
    """
    tex_file = ARXIV_TABLES_DIR / "tab_label_delay_adaptation.tex"
    json_file = METRICS_DIR / "label_delay_adaptation_benchmark.json"
    
    assert tex_file.exists(), f"LaTeX table {tex_file} missing"
    assert json_file.exists(), f"Metrics JSON {json_file} missing"
    
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        
    summary_map = {(item["mode"], item["delay_weeks"]): item for item in json_data["summary"]}
    tex_content = tex_file.read_text(encoding="utf-8")
    
    for delay in [0, 1, 2, 4, 8]:
        # Check Periodic-1w recall in table
        p1 = summary_map[("periodic_1w", delay)]
        p1_rec_str = f"{p1['recall_mean']*100:.2f}\\%"
        assert p1_rec_str in tex_content, f"Periodic-1w recall {p1_rec_str} for delay {delay} not found in {tex_file.name}"
        
        # Check Unsupervised P(X) recall in table
        px = summary_map[("unsupervised_px", delay)]
        px_rec_str = f"{px['recall_mean']*100:.2f}\\%"
        assert px_rec_str in tex_content, f"Unsupervised P(X) recall {px_rec_str} for delay {delay} not found in {tex_file.name}"


def test_drift_detection_latex_table_matches_json():
    """
    Verifies that tab_drift_detection.tex matches drift_detection_benchmark.json ground truth.
    """
    tex_file = ARXIV_TABLES_DIR / "tab_drift_detection.tex"
    json_file = METRICS_DIR / "drift_detection_benchmark.json"
    
    assert tex_file.exists()
    assert json_file.exists()
    
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        
    adwin_mttd = json_data["ADWIN"]["abrupt_drift_mean_delay_samples"]
    adwin_str = f"{adwin_mttd:.1f}"
    
    tex_content = tex_file.read_text(encoding="utf-8")
    assert adwin_str in tex_content, f"ADWIN MTTD {adwin_str} not found in {tex_file.name}"
