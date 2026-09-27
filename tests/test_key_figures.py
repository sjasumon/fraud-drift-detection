"""
Unit tests for the publication key figures generator (src/generate_key_figures.py).
Validates strict absence of silent synthetic-data fallbacks and guarantees that
figure generation refuses to plot placeholder data when artifacts are missing.
"""

import pytest
from pathlib import Path
import pandas as pd

from src.generate_key_figures import parse_ieee_26w_metrics, plot_verification_latency_degradation


def test_parse_ieee_26w_metrics_missing_log_raises(monkeypatch, tmp_path):
    """Verifies that missing pipeline log immediately raises FileNotFoundError without synthetic fallback."""
    fake_root = tmp_path / "nonexistent_root"
    monkeypatch.setattr("src.generate_key_figures.ROOT_DIR", fake_root)

    with pytest.raises(FileNotFoundError, match="refusing to plot synthetic placeholder data"):
        parse_ieee_26w_metrics()


def test_parse_ieee_26w_metrics_valid_execution():
    """Verifies that real pipeline log parses exactly 26 sequential weeks with expected metrics."""
    df = parse_ieee_26w_metrics()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 26
    assert list(df["week"]) == list(range(1, 27))
    assert "static_f1" in df.columns
    assert "adaptive_f1" in df.columns
    assert "is_retrain" in df.columns
    assert df["is_retrain"].sum() == 22  # Exactly 22 GPU retrains


def test_plot_verification_latency_degradation_missing_json_raises(monkeypatch, tmp_path):
    """Verifies that missing delayed feedback benchmark JSON raises FileNotFoundError."""
    fake_root = tmp_path / "nonexistent_root"
    monkeypatch.setattr("src.generate_key_figures.ROOT_DIR", fake_root)

    with pytest.raises(FileNotFoundError, match="refusing to plot synthetic placeholder data"):
        plot_verification_latency_degradation()
