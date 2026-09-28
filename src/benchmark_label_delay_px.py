"""
Decisive Benchmark: Label-Delay-Aware Stream Adaptation via Unsupervised P(X) Drift Triggers.

Directly evaluates the core research question (RQ1):
Under verification latency (where ground-truth labels are delayed by delta in {0, 1, 2, 4, 8} weeks),
how does unsupervised P(X) feature drift monitoring (PSI) compare against:
1. Supervised error-monitoring (ADWIN / DDM on matured delayed labels),
2. Cadence-matched Periodic-1w retraining (blind weekly upper bound), and
3. Frozen Static baseline?

Evaluated across 5 independent random seeds with 95% Confidence Intervals (Student's t, df=4).
Produces:
- results/metrics/label_delay_adaptation_benchmark.json
- results/metrics/label_delay_adaptation_benchmark.csv
- results/figures/label_delay_adaptation_frontier.png
- arxiv_paper/figures/label_delay_adaptation_frontier.png
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc

from src.config import (
    BASE_DIR, METRICS_DIR, FIGURES_DIR, RANDOM_SEED
)
from src.drift_detector import DriftMonitor
from src.drift_explainer import calculate_psi

logger = logging.getLogger("fraud_drift.label_delay_px")


def generate_streaming_benchmark_data(
    n_weeks: int = 16,
    tx_per_week: int = 1500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a controlled non-stationary transaction stream with multi-phase concept drift:
    - Phase 0 (Weeks 0-3): Baseline distribution.
    - Phase 1 (Weeks 4-7): Sudden micro-structuring drift (low amounts, high velocity).
    - Phase 2 (Weeks 8-11): High-ticket burst drift.
    - Phase 3 (Weeks 12-15): Mule account turnover drift.
    """
    rng = np.random.default_rng(seed)
    records = []
    
    for w in range(n_weeks):
        n = tx_per_week
        # Fraud base rate ~3.5%
        is_fraud = (rng.uniform(0, 1, n) < 0.035).astype(int)
        
        if w < 4:
            # Baseline
            amt = np.where(is_fraud, rng.normal(300, 50, n), rng.normal(80, 20, n))
            vel = np.where(is_fraud, rng.poisson(8, n), rng.poisson(2, n))
            bal_ratio = np.where(is_fraud, rng.beta(0.5, 0.5, n), rng.beta(2, 2, n))
        elif w < 8:
            # Drift Phase 1: Sudden micro-structuring
            amt = np.where(is_fraud, rng.normal(45, 10, n), rng.normal(80, 20, n))
            vel = np.where(is_fraud, rng.poisson(18, n), rng.poisson(2, n))
            bal_ratio = np.where(is_fraud, rng.beta(0.2, 0.8, n), rng.beta(2, 2, n))
        elif w < 12:
            # Drift Phase 2: High-ticket burst
            amt = np.where(is_fraud, rng.normal(1200, 200, n), rng.normal(80, 20, n))
            vel = np.where(is_fraud, rng.poisson(4, n), rng.poisson(2, n))
            bal_ratio = np.where(is_fraud, rng.beta(0.9, 0.1, n), rng.beta(2, 2, n))
        else:
            # Drift Phase 3: Mule turnover
            amt = np.where(is_fraud, rng.normal(500, 100, n), rng.normal(80, 20, n))
            vel = np.where(is_fraud, rng.poisson(12, n), rng.poisson(2, n))
            bal_ratio = np.where(is_fraud, rng.beta(0.1, 0.9, n), rng.beta(2, 2, n))
            
        amt = np.maximum(1.0, amt)
        vel = np.maximum(0, vel)
        bal_ratio = np.clip(bal_ratio, 0.0, 1.0)
        
        df_w = pd.DataFrame({
            "week": w,
            "isFraud": is_fraud,
            "amt": amt,
            "vel": vel,
            "bal_ratio": bal_ratio
        })
        records.append(df_w)
        
    return pd.concat(records, ignore_index=True)


def fit_booster(df_train: pd.DataFrame, features: List[str], seed: int) -> xgb.XGBClassifier:
    """Trains standardized XGBoost classifier matching production parameters."""
    X = df_train[features].values
    y = df_train["isFraud"].values
    n_pos = int(np.sum(y))
    spw = (len(y) - n_pos) / max(1, n_pos)
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1
    )
    model.fit(X, y)
    return model


def run_single_simulation(
    df: pd.DataFrame,
    features: List[str],
    mode: str,
    delay: int,
    seed: int,
    n_weeks: int = 16,
    warmup_weeks: int = 4
) -> Dict[str, Any]:
    """Runs a single streaming simulation under given delay and model mode."""
    baseline_df = df[df["week"] < warmup_weeks]
    model = fit_booster(baseline_df, features, seed)
    
    drift_mon = DriftMonitor(ddm_warning_level=2.0, ddm_drift_level=2.5)
    
    oos_preds = []
    oos_probs = []
    oos_true = []
    retrains = 0
    model_trained_data = baseline_df
    last_matured_week_used = warmup_weeks - 1
    drift_pending = False
    
    for w in range(warmup_weeks, n_weeks):
        df_w = df[df["week"] == w]
        X_w = df_w[features].values
        y_w = df_w["isFraud"].values
        
        # 1. Immediate Test-Before-Learn Inference
        probs = model.predict_proba(X_w)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        oos_preds.extend(preds)
        oos_probs.extend(probs)
        oos_true.extend(y_w)
        
        # 2. Check maturity of delayed ground truth
        mat_w = w - delay
        should_retrain = False
        
        if mode == "static":
            should_retrain = False
        elif mode == "periodic_1w":
            if mat_w >= warmup_weeks and mat_w > last_matured_week_used:
                should_retrain = True
        elif mode == "supervised":
            if mat_w >= warmup_weeks:
                df_mat = df[df["week"] == mat_w]
                m_probs = model.predict_proba(df_mat[features].values)[:, 1]
                m_preds = (m_probs >= 0.5).astype(int)
                m_true = df_mat["isFraud"].values
                for idx in range(len(m_true)):
                    ev = drift_mon.update(
                        y_true=int(m_true[idx]),
                        y_pred=int(m_preds[idx]),
                        slice_index=w,
                        step_number=idx,
                        trans_id=f"{mat_w}_{idx}"
                    )
                    if ev["any_drift"]:
                        should_retrain = True
        elif mode == "unsupervised_px":
            # Compute PSI on arrival features X_w against model's training distribution
            psis = [calculate_psi(model_trained_data[col].values, df_w[col].values) for col in features]
            if max(psis) > 0.025:
                drift_pending = True
            
            # Retrain if drift detected and new matured labels are available
            if drift_pending and mat_w >= warmup_weeks and mat_w > last_matured_week_used:
                should_retrain = True
                
        # 3. Retrain strictly on matured labels (zero lookahead leakage)
        if should_retrain:
            chunks = [baseline_df, df[(df["week"] >= warmup_weeks) & (df["week"] <= mat_w)]]
            matured_data = pd.concat(chunks, ignore_index=True)
            model = fit_booster(matured_data, features, seed)
            retrains += 1
            model_trained_data = matured_data
            last_matured_week_used = mat_w
            drift_pending = False
            if mode == "supervised":
                drift_mon.reset_detectors()
                
    oos_preds = np.array(oos_preds)
    oos_probs = np.array(oos_probs)
    oos_true = np.array(oos_true)
    
    tp = int(np.sum((oos_preds == 1) & (oos_true == 1)))
    fn = int(np.sum((oos_preds == 0) & (oos_true == 1)))
    fp = int(np.sum((oos_preds == 1) & (oos_true == 0)))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    p_curve, r_curve, _ = precision_recall_curve(oos_true, oos_probs)
    aucpr = float(auc(r_curve, p_curve))
    
    return {
        "mode": mode,
        "delay": delay,
        "seed": seed,
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "aucpr": float(aucpr),
        "retrains": int(retrains),
        "total_oos_frauds": int(tp + fn),
        "detected_frauds": int(tp)
    }


def run_benchmark(
    seeds: List[int] = [42, 43, 44, 45, 46],
    delays: List[int] = [0, 1, 2, 4, 8],
    modes: List[str] = ["supervised", "unsupervised_px", "periodic_1w", "static"]
) -> Dict[str, Any]:
    """Runs the full multi-seed benchmark across latency regimes."""
    logger.info(f"Starting decisive label-delay benchmark across {len(seeds)} seeds and {len(delays)} delays...")
    
    raw_results = []
    features = ["amt", "vel", "bal_ratio"]
    
    for seed in seeds:
        df = generate_streaming_benchmark_data(n_weeks=16, tx_per_week=1500, seed=seed)
        for mode in modes:
            for d in delays:
                res = run_single_simulation(
                    df=df,
                    features=features,
                    mode=mode,
                    delay=d,
                    seed=seed
                )
                raw_results.append(res)
                
    df_res = pd.DataFrame(raw_results)
    
    # Compute aggregate statistics with 95% CIs
    grouped = df_res.groupby(["mode", "delay"])
    summary = []
    
    for (mode, delay), group in grouped:
        n = len(group)
        rec_mean = float(group["recall"].mean())
        rec_std = float(group["recall"].std()) if n > 1 else 0.0
        
        auc_mean = float(group["aucpr"].mean())
        auc_std = float(group["aucpr"].std()) if n > 1 else 0.0
        
        ret_mean = float(group["retrains"].mean())
        ret_std = float(group["retrains"].std()) if n > 1 else 0.0
        
        # 95% CI via Student's t distribution
        ci_half = float(stats.t.ppf(0.975, df=n-1) * (rec_std / np.sqrt(n))) if (n > 1 and rec_std > 0) else 0.0
        
        summary.append({
            "mode": mode,
            "delay_weeks": delay,
            "n_seeds": n,
            "recall_mean": rec_mean,
            "recall_std": rec_std,
            "recall_ci_95": [max(0.0, rec_mean - ci_half), min(1.0, rec_mean + ci_half)],
            "aucpr_mean": auc_mean,
            "aucpr_std": auc_std,
            "retrains_mean": ret_mean,
            "retrains_std": ret_std
        })
        
    output_dict = {
        "metadata": {
            "title": "Decisive Benchmark: Label Verification Latency vs Adaptation Frontier",
            "research_question": "RQ1: Unsupervised P(X) Drift Triggers vs Supervised Latency Collapse",
            "n_seeds": len(seeds),
            "seeds": seeds,
            "delays_evaluated": delays,
            "stream_weeks": 16,
            "warmup_weeks": 4
        },
        "summary": summary,
        "raw_runs": raw_results
    }
    
    # Save JSON and CSV
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = METRICS_DIR / "label_delay_adaptation_benchmark.json"
    out_csv = METRICS_DIR / "label_delay_adaptation_benchmark.csv"
    
    with open(out_json, "w") as f:
        json.dump(output_dict, f, indent=2)
        
    df_res.to_csv(out_csv, index=False)
    logger.info(f"Saved benchmark metrics to {out_json} and {out_csv}")
    
    return output_dict


def plot_adaptation_frontier(json_data: Dict[str, Any], output_path: Path):
    """
    Renders a publication-grade 300 DPI dual-panel chart:
    Panel A: Recall vs Label Delay (Weeks) with shaded 95% CIs.
    Panel B: Cumulative Retrain Count (Compute Cost) vs Label Delay.
    """
    summary = json_data["summary"]
    df_sum = pd.DataFrame(summary)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    styles = {
        "unsupervised_px": {
            "label": "Proposed: Unsupervised P(X) Triggered",
            "color": "#1b9e77",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.5
        },
        "supervised": {
            "label": "Supervised Error Triggered (ADWIN/DDM)",
            "color": "#d95f02",
            "marker": "s",
            "linestyle": "-",
            "linewidth": 2.0
        },
        "periodic_1w": {
            "label": "Cadence-Matched Periodic-1w (Upper Bound)",
            "color": "#7570b3",
            "marker": "^",
            "linestyle": "--",
            "linewidth": 1.8
        },
        "static": {
            "label": "Static Baseline (Frozen)",
            "color": "#666666",
            "marker": "x",
            "linestyle": ":",
            "linewidth": 1.5
        }
    }
    
    # ---------------- PANEL A: RECALL VS DELAY ----------------
    for mode in ["periodic_1w", "unsupervised_px", "supervised", "static"]:
        sub = df_sum[df_sum["mode"] == mode].sort_values("delay_weeks")
        if len(sub) == 0:
            continue
            
        x = sub["delay_weeks"].values
        y = sub["recall_mean"].values * 100
        ci_low = np.array([c[0] for c in sub["recall_ci_95"].values]) * 100
        ci_high = np.array([c[1] for c in sub["recall_ci_95"].values]) * 100
        
        cfg = styles.get(mode, {"label": mode, "color": "black", "marker": "o", "linestyle": "-"})
        
        ax1.plot(x, y, label=cfg["label"], color=cfg["color"], marker=cfg["marker"],
                 linestyle=cfg["linestyle"], linewidth=cfg["linewidth"], markersize=7)
        ax1.fill_between(x, ci_low, ci_high, color=cfg["color"], alpha=0.15)
        
    ax1.set_title("(a) Out-of-Sample Recall vs. Verification Latency (5 Seeds, 95% CI)",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Label Verification Latency $\\delta$ (Weeks)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Out-of-Sample Recall (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks([0, 1, 2, 4, 8])
    ax1.set_ylim(60.0, 95.0)
    ax1.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # ---------------- PANEL B: RETRAIN COMPUTE COST ----------------
    for mode in ["periodic_1w", "unsupervised_px", "supervised", "static"]:
        sub = df_sum[df_sum["mode"] == mode].sort_values("delay_weeks")
        if len(sub) == 0:
            continue
            
        x = sub["delay_weeks"].values
        y = sub["retrains_mean"].values
        std = sub["retrains_std"].values
        
        cfg = styles.get(mode, {"label": mode, "color": "black", "marker": "o", "linestyle": "-"})
        
        ax2.plot(x, y, label=cfg["label"], color=cfg["color"], marker=cfg["marker"],
                 linestyle=cfg["linestyle"], linewidth=cfg["linewidth"], markersize=7)
        ax2.errorbar(x, y, yerr=std, color=cfg["color"], fmt="none", capsize=3, alpha=0.7)
        
    ax2.set_title("(b) Retraining Compute Overhead vs. Latency (Retrain Count)",
                  fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Label Verification Latency $\\delta$ (Weeks)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Number of Retrains", fontsize=11, fontweight="bold")
    ax2.set_xticks([0, 1, 2, 4, 8])
    ax2.set_ylim(-0.5, 13.0)
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated decisive plot at {output_path}")


def main():
    seeds = [42, 43, 44, 45, 46]
    delays = [0, 1, 2, 4, 8]
    data = run_benchmark(seeds=seeds, delays=delays)
    
    plot_file = FIGURES_DIR / "label_delay_adaptation_frontier.png"
    plot_adaptation_frontier(data, plot_file)
    
    # Also mirror to arxiv_paper/figures
    paper_fig = BASE_DIR / "arxiv_paper" / "figures" / "label_delay_adaptation_frontier.png"
    paper_fig.parent.mkdir(parents=True, exist_ok=True)
    plot_adaptation_frontier(data, paper_fig)
    print("Decisive experiment successfully generated and plotted.")


if __name__ == "__main__":
    main()
