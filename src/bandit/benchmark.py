"""
Empirical Benchmark Suite for Non-Stationary Multi-Armed Bandits.
Compares:
1. Standard UCB1 (Stationary control)
2. Sliding-Window UCB / M-UCB (Garivier & Moulines, 2011)
3. Discounted UCB (Kocsis & Szepesvári, 2006)
4. ADWIN-UCB (Proposed Concept-Drift-Aware Adaptive Bandit)

Evaluates:
- Cumulative pseudo-regret across time
- Optimal arm selection percentage
- Convergence velocity following abrupt concept drift shifts
- Statistical significance (mean ± std across multiple seeds)
- High-resolution publication visualization with change-point annotations
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.config import RESULTS_DIR, METRICS_DIR, FIGURES_DIR
from src.bandit.environment import NonStationaryBanditEnv
from src.bandit.agents import (
    StandardUCBAgent,
    SlidingWindowUCBAgent,
    DiscountedUCBAgent,
    AdwinUCBAgent
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fraud_drift.bandit")

DEFAULT_SEEDS = [42, 101, 202, 303, 404]


def run_bandit_simulation(
    seeds: List[int] = DEFAULT_SEEDS,
    total_steps: int = 10000,
    n_arms: int = 5
) -> Dict[str, Any]:
    """
    Runs multi-seed simulation across all 4 bandit agents on NonStationaryBanditEnv.
    """
    logger.info("=" * 80)
    logger.info(f"STARTING NON-STATIONARY BANDIT BENCHMARK ({len(seeds)} Seeds, {total_steps} Steps)")
    logger.info("Agents: Standard UCB1 vs Sliding-Window UCB vs Discounted UCB vs ADWIN-UCB")
    logger.info("=" * 80)

    agent_factories = {
        "standard_ucb": lambda: StandardUCBAgent(n_arms=n_arms, c=1.0),
        "sliding_window_ucb": lambda: SlidingWindowUCBAgent(n_arms=n_arms, window_size=600, c=1.0),
        "discounted_ucb": lambda: DiscountedUCBAgent(n_arms=n_arms, gamma=0.995, c=1.0),
        "adwin_ucb": lambda: AdwinUCBAgent(n_arms=n_arms, delta=0.002, c=1.0)
    }

    # Tracking containers: agent -> metric -> list of arrays across seeds
    all_regrets: Dict[str, List[np.ndarray]] = {k: [] for k in agent_factories}
    all_rewards: Dict[str, List[float]] = {k: [] for k in agent_factories}
    all_optimal_pct: Dict[str, List[float]] = {k: [] for k in agent_factories}
    all_final_regret: Dict[str, List[float]] = {k: [] for k in agent_factories}
    all_drift_counts: Dict[str, List[int]] = {k: [] for k in agent_factories}

    change_points = []

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\n--- [Bandit Run {seed_idx + 1}/{len(seeds)}] Initializing with seed={seed} ---")
        t0 = time.time()

        for agent_key, factory in agent_factories.items():
            env = NonStationaryBanditEnv(n_arms=n_arms, random_seed=seed)
            if not change_points:
                change_points = env.change_points

            agent = factory()
            agent.reset()

            cum_regret = np.zeros(total_steps, dtype=float)
            total_rew = 0.0
            optimal_pulls = 0
            running_regret = 0.0

            for t in range(total_steps):
                arm = agent.select_arm()
                reward, instant_regret, info = env.step(arm)
                agent.update(arm, reward)

                running_regret += instant_regret
                cum_regret[t] = running_regret
                total_rew += reward
                if arm == info["optimal_arm"]:
                    optimal_pulls += 1

            all_regrets[agent_key].append(cum_regret)
            all_rewards[agent_key].append(round(total_rew, 1))
            all_optimal_pct[agent_key].append(round((optimal_pulls / total_steps) * 100, 2))
            all_final_regret[agent_key].append(round(running_regret, 2))

            drift_count = len(getattr(agent, "drift_events", []))
            all_drift_counts[agent_key].append(drift_count)

        duration = time.time() - t0
        logger.info(
            f"Seed {seed} finished in {duration:.1f}s | Final Regret: "
            f"UCB1={all_final_regret['standard_ucb'][-1]:.1f}, "
            f"SW-UCB={all_final_regret['sliding_window_ucb'][-1]:.1f}, "
            f"D-UCB={all_final_regret['discounted_ucb'][-1]:.1f}, "
            f"ADWIN-UCB={all_final_regret['adwin_ucb'][-1]:.1f}"
        )

    # Compute Statistical Summaries
    summary: Dict[str, Any] = {
        "metadata": {
            "seeds": seeds,
            "total_steps": total_steps,
            "n_arms": n_arms,
            "change_points": change_points
        },
        "agents": {},
        "paired_tests": {}
    }

    for k in agent_factories:
        reg_arr = np.array(all_final_regret[k])
        rew_arr = np.array(all_rewards[k])
        opt_arr = np.array(all_optimal_pct[k])

        summary["agents"][k] = {
            "final_regret": {
                "mean": round(float(np.mean(reg_arr)), 2),
                "std": round(float(np.std(reg_arr, ddof=1)), 2),
                "ci_95": [
                    round(float(np.mean(reg_arr) - 2.776 * np.std(reg_arr, ddof=1) / np.sqrt(len(seeds))), 2),
                    round(float(np.mean(reg_arr) + 2.776 * np.std(reg_arr, ddof=1) / np.sqrt(len(seeds))), 2)
                ],
                "raw": all_final_regret[k]
            },
            "total_reward": {
                "mean": round(float(np.mean(rew_arr)), 2),
                "std": round(float(np.std(rew_arr, ddof=1)), 2)
            },
            "optimal_arm_pct": {
                "mean": round(float(np.mean(opt_arr)), 2),
                "std": round(float(np.std(opt_arr, ddof=1)), 2)
            },
            "mean_drift_events": round(float(np.mean(all_drift_counts[k])), 1)
        }

    # Hypothesis Testing (ADWIN-UCB vs Baselines)
    for baseline in ["standard_ucb", "sliding_window_ucb", "discounted_ucb"]:
        t_res = stats.ttest_rel(all_final_regret["adwin_ucb"], all_final_regret[baseline])
        p_val = float(t_res.pvalue)
        summary["paired_tests"][f"adwin_vs_{baseline}"] = {
            "mean_regret_reduction": round(float(np.mean(all_final_regret[baseline]) - np.mean(all_final_regret["adwin_ucb"])), 2),
            "pct_regret_reduction": round(float((np.mean(all_final_regret[baseline]) - np.mean(all_final_regret["adwin_ucb"])) / np.mean(all_final_regret[baseline]) * 100), 2),
            "t_stat": round(float(t_res.statistic), 4),
            "p_value": float(f"{p_val:.6e}"),
            "significant_001": bool(p_val < 0.001)
        }

    logger.info("\n" + "=" * 80)
    logger.info("NON-STATIONARY BANDIT STATISTICAL RIGOR SUMMARY:")
    logger.info("=" * 80)
    for k, d in summary["agents"].items():
        logger.info(
            f"   {k.upper():<20}: Regret = {d['final_regret']['mean']:.1f} ± {d['final_regret']['std']:.1f} | "
            f"Optimal Arm = {d['optimal_arm_pct']['mean']:.1f}% ± {d['optimal_arm_pct']['std']:.1f}% | "
            f"Total Reward = {d['total_reward']['mean']:.1f}"
        )
    for k, p in summary["paired_tests"].items():
        logger.info(
            f"   Paired t-test ({k}): Regret reduction = -{p['mean_regret_reduction']:.1f} "
            f"(-{p['pct_regret_reduction']:.1f}%), p={p['p_value']:.6e} -> Sig (p < 0.001): {p['significant_001']}"
        )

    # Save summary JSON
    summary_path = METRICS_DIR / "bandit_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Bandit summary JSON saved to {summary_path}")

    # Generate Publication Chart
    chart_path = generate_bandit_plots(all_regrets, summary, change_points, total_steps)
    logger.info(f"Bandit visualization saved to {chart_path}")

    return {"summary": summary, "regrets": all_regrets}


def generate_bandit_plots(
    all_regrets: Dict[str, List[np.ndarray]],
    summary: Dict[str, Any],
    change_points: List[int],
    total_steps: int,
    output_filename: str = "bandit_cumulative_regret.png"
) -> Path:
    """
    Generates high-resolution publication plot with regret curves and error bars.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Color palette
    colors = {
        "standard_ucb": "#d62728",        # Red (fails under drift)
        "sliding_window_ucb": "#1f77b4",  # Blue (M-UCB)
        "discounted_ucb": "#ff7f0e",      # Orange (D-UCB)
        "adwin_ucb": "#2ca02c"            # Green (ADWIN-UCB proposed)
    }
    labels = {
        "standard_ucb": "Standard UCB1 (Stationary Control)",
        "sliding_window_ucb": "Sliding-Window UCB (M-UCB, W=600)",
        "discounted_ucb": "Discounted UCB (D-UCB, γ=0.995)",
        "adwin_ucb": "ADWIN-UCB (Drift-Adaptive Proposed)"
    }

    t_axis = np.arange(1, total_steps + 1)

    # 1. Left Subplot: Cumulative Regret Curves over Time
    for k in colors:
        arrs = np.array(all_regrets[k])  # shape: (n_seeds, total_steps)
        mean_reg = np.mean(arrs, axis=0)
        std_reg = np.std(arrs, axis=0)

        ax1.plot(t_axis, mean_reg, label=labels[k], color=colors[k], lw=2.2)
        ax1.fill_between(t_axis, mean_reg - std_reg, mean_reg + std_reg, color=colors[k], alpha=0.15)

    # Annotate Concept Drift Change Points
    for cp in change_points:
        ax1.axvline(x=cp, color="#555555", linestyle=":", lw=1.5, alpha=0.85)
        ax1.text(cp + 60, ax1.get_ylim()[1] * 0.05 if ax1.get_ylim()[1] > 0 else 50,
                 f"Drift Shock\n(t={cp:,})", rotation=90, fontsize=8.5, color="#333333", fontweight="bold")

    ax1.set_title("Non-Stationary Multi-Armed Bandit Pseudo-Regret\n(Mean ± Std across 5 Random Seeds)",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Time Step (t)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Pseudo-Regret", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. Right Subplot: Final Cumulative Regret Comparison with Significance
    agent_order = ["standard_ucb", "sliding_window_ucb", "discounted_ucb", "adwin_ucb"]
    bar_labels = ["Standard\nUCB1", "Sliding-Window\n(M-UCB)", "Discounted\n(D-UCB)", "ADWIN-UCB\n(Proposed)"]
    bar_colors = [colors[k] for k in agent_order]

    means = [summary["agents"][k]["final_regret"]["mean"] for k in agent_order]
    stds = [summary["agents"][k]["final_regret"]["std"] for k in agent_order]

    bars = ax2.bar(bar_labels, means, yerr=stds, capsize=6, color=bar_colors, alpha=0.88, edgecolor="black", width=0.55)
    ax2.set_title("Final Cumulative Regret at t=10,000\n(Lower is Better)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Total Cumulative Regret", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bar values
    for bar, m, s in zip(bars, means, stds):
        h = bar.get_height()
        ax2.annotate(
            f"{m:.1f}\n(±{s:.1f})",
            xy=(bar.get_x() + bar.get_width() / 2, h + 20),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Bracket showing ADWIN-UCB vs Standard UCB1 reduction
    p_ucb = summary["paired_tests"]["adwin_vs_standard_ucb"]["p_value"]
    pct_red = summary["paired_tests"]["adwin_vs_standard_ucb"]["pct_regret_reduction"]
    sig_text = f"-{pct_red:.1f}% Regret\n*** p < 0.001" if p_ucb < 0.001 else f"p = {p_ucb:.4f}"

    max_h = max(means)
    ax2.set_ylim(0, max_h * 1.30)
    ax2.plot([0, 0, 3, 3], [max_h * 1.05, max_h * 1.15, max_h * 1.15, means[3] + 60], lw=1.3, c="k")
    ax2.text(1.5, max_h * 1.17, sig_text, ha="center", va="bottom", color="#d62728", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    chart_path = FIGURES_DIR / output_filename
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def main():
    parser = argparse.ArgumentParser(description="Non-Stationary Multi-Armed Bandit Benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Random seeds")
    parser.add_argument("--steps", type=int, default=10000, help="Total simulation steps")
    parser.add_argument("--arms", type=int, default=5, help="Number of bandit arms")
    args = parser.parse_args()

    run_bandit_simulation(seeds=args.seeds, total_steps=args.steps, n_arms=args.arms)


if __name__ == "__main__":
    main()
