"""
Unit tests for non-stationary multi-armed bandit benchmark suite (src/bandit/benchmark.py).
Tests simulation convergence, statistical hypothesis testing (paired t-test),
and visualization rendering under controlled change-points.
"""

import numpy as np
from pathlib import Path

from src.bandit.benchmark import run_bandit_simulation, generate_bandit_plots


def test_run_bandit_simulation_fast(tmp_path: Path, monkeypatch):
    """Verifies that run_bandit_simulation executes across multiple seeds and agents."""
    monkeypatch.setattr("src.bandit.benchmark.METRICS_DIR", tmp_path)
    monkeypatch.setattr("src.bandit.benchmark.FIGURES_DIR", tmp_path)

    result = run_bandit_simulation(seeds=[42, 101], total_steps=60, n_arms=3)

    assert "summary" in result
    assert "regrets" in result

    summary = result["summary"]
    agents = summary["agents"]
    assert "standard_ucb" in agents
    assert "sliding_window_ucb" in agents
    assert "discounted_ucb" in agents
    assert "adwin_ucb" in agents

    # Verify statistical metrics
    for k, d in agents.items():
        assert "final_regret" in d
        assert "mean" in d["final_regret"]
        assert "std" in d["final_regret"]
        assert "ci_95" in d["final_regret"]
        assert len(d["final_regret"]["raw"]) == 2

    # Verify paired hypothesis tests
    paired = summary["paired_tests"]
    assert "adwin_vs_standard_ucb" in paired
    assert "t_stat" in paired["adwin_vs_standard_ucb"]
    assert "p_value" in paired["adwin_vs_standard_ucb"]

    # Verify output files
    summary_file = tmp_path / "bandit_benchmark_summary.json"
    plot_file = tmp_path / "bandit_cumulative_regret.png"
    assert summary_file.exists()
    assert plot_file.exists()


def test_generate_bandit_plots_custom(tmp_path: Path, monkeypatch):
    """Verifies custom plot generation with arbitrary regrets and summary."""
    monkeypatch.setattr("src.bandit.benchmark.FIGURES_DIR", tmp_path)

    all_regrets = {
        "standard_ucb": [np.array([1.0, 2.5, 4.0]), np.array([1.2, 2.8, 4.5])],
        "sliding_window_ucb": [np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.2, 3.2])],
        "discounted_ucb": [np.array([1.0, 1.8, 2.8]), np.array([1.0, 1.9, 2.9])],
        "adwin_ucb": [np.array([1.0, 1.5, 2.0]), np.array([1.0, 1.6, 2.2])],
    }
    summary = {
        "agents": {
            "standard_ucb": {"final_regret": {"mean": 4.25, "std": 0.35}},
            "sliding_window_ucb": {"final_regret": {"mean": 3.1, "std": 0.14}},
            "discounted_ucb": {"final_regret": {"mean": 2.85, "std": 0.07}},
            "adwin_ucb": {"final_regret": {"mean": 2.1, "std": 0.14}},
        },
        "paired_tests": {
            "adwin_vs_standard_ucb": {"p_value": 0.0005, "pct_regret_reduction": 50.59}
        },
    }

    out = generate_bandit_plots(
        all_regrets,
        summary,
        change_points=[2],
        total_steps=3,
        output_filename="test_plot.png",
    )
    assert out.exists()
