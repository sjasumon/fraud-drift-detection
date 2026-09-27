"""
Empirical Card/UID Memorization vs. Concept Drift Generalization Audit.
Strict Leak-Free Preprocessing & Sequential Time-Slice Partitioning.

Evaluates:
1. Method A: Clean Complete Cases Only (card1, card2, addr1 non-null).
   - Cohort 1: Repeat / Baseline-Seen Cards (seen in Weeks 1-5)
   - Cohort 2: Baseline-Unseen Cards (not seen in Weeks 1-5)
   - Cohort 3: Strict Stream-Unseen Cards (Zero-Memory Arrival Test)
2. Method B: Full Population with Collision-Free Tokenization (MISSING_<trans_id>).
   - Cohort 1: Repeat / Baseline-Seen Entities
   - Cohort 2: Baseline-Unseen Entities
   - Cohort 3: Strict Stream-Unseen Entities

Supports both:
- Configuration A: Standard Features (card identifiers included in numerical features)
- Configuration B: Feature-Blind / Identifier-Exclusion (card1, card2, card3, card5, addr1, addr2 excluded)
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Set, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score

from src.config import TRAINING_WINDOW_RATIO, RANDOM_SEED
from src.ingestion import DatasetIngestion
from src.preprocessing import Preprocessor
from src.static_model import StaticBaselineModel
from src.adaptive_model import DriftAdaptiveModel
from src.drift_detector import DriftMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fraud_drift.uid_audit")


def compute_metrics(d: Dict[str, list]) -> Dict[str, Any]:
    y = np.array(d["y"])
    sp = np.array(d["stat_p"])
    ap = np.array(d["adapt_p"])
    sprob = np.array(d["stat_prob"])
    aprob = np.array(d["adapt_prob"])

    total_tx = len(y)
    total_fraud = int(y.sum()) if total_tx > 0 else 0
    if total_fraud == 0 or total_tx == 0:
        return {}

    stat_tp = int(((y == 1) & (sp == 1)).sum())
    adapt_tp = int(((y == 1) & (ap == 1)).sum())
    stat_fp = int(((y == 0) & (sp == 1)).sum())
    adapt_fp = int(((y == 0) & (ap == 1)).sum())

    stat_recall = (stat_tp / total_fraud) * 100
    adapt_recall = (adapt_tp / total_fraud) * 100

    stat_prec = (stat_tp / (stat_tp + stat_fp)) * 100 if (stat_tp + stat_fp) > 0 else 0.0
    adapt_prec = (adapt_tp / (adapt_tp + adapt_fp)) * 100 if (adapt_tp + adapt_fp) > 0 else 0.0

    stat_auc = float(average_precision_score(y, sprob)) if (0 < total_fraud < total_tx) else 0.0
    adapt_auc = float(average_precision_score(y, aprob)) if (0 < total_fraud < total_tx) else 0.0

    return {
        "total_tx": total_tx,
        "fraud_tx": total_fraud,
        "prevalence_pct": (total_fraud / total_tx) * 100,
        "stat_tp": stat_tp,
        "adapt_tp": adapt_tp,
        "stat_recall": stat_recall,
        "adapt_recall": adapt_recall,
        "recall_gain": adapt_recall - stat_recall,
        "stat_prec": stat_prec,
        "adapt_prec": adapt_prec,
        "stat_auc": stat_auc,
        "adapt_auc": adapt_auc,
        "auc_gain": adapt_auc - stat_auc
    }


def print_audit_report(results: Dict[str, Any], title: str):
    print("\n" + "=" * 92)
    print(f"AUDIT REPORT: {title}")
    print("=" * 92)

    g = results.get("global_oos", {})
    if g:
        print(f"\n[GLOBAL OUT-OF-SAMPLE BENCHMARK (All {g['total_tx']:,} Transactions, {g['fraud_tx']:,} Frauds)]")
        print(f"   Static Model   : Recall = {g['stat_recall']:.2f}% ({g['stat_tp']:,}/{g['fraud_tx']:,}) | Precision = {g['stat_prec']:.2f}% | AUC-PR = {g['stat_auc']:.4f}")
        print(f"   Adaptive Model : Recall = {g['adapt_recall']:.2f}% ({g['adapt_tp']:,}/{g['fraud_tx']:,}) | Precision = {g['adapt_prec']:.2f}% | AUC-PR = {g['adapt_auc']:.4f}")
        print(f"   Net Gain       : Recall = +{g['recall_gain']:.2f}% (+{g['adapt_tp'] - g['stat_tp']:,} frauds caught) | AUC-PR = +{g['auc_gain']:.4f}")

    cl_seen_m = results.get("clean", {}).get("seen", {})
    cl_unseen_m = results.get("clean", {}).get("unseen", {})
    cl_cov_tx = cl_seen_m.get("total_tx", 0) + cl_unseen_m.get("total_tx", 0)
    g_total_tx = g.get("total_tx", 0)
    cl_cov_pct = (cl_cov_tx / g_total_tx * 100) if g_total_tx > 0 else 0.0
    print("\n" + "-" * 92)
    print("METHOD A: CLEAN COMPLETE CASES ONLY (card1, card2, addr1 all strictly non-null)")
    print(f"Coverage: {cl_cov_tx:,} transactions ({cl_cov_pct:.2f}% of stream; excludes {100 - cl_cov_pct:.2f}% missing cases)")
    print("-" * 92)
    for name, key in [
        ("1. Repeat / Baseline-Seen Cards (seen in Weeks 1-5)", "seen"),
        ("2. Baseline-Unseen Cards (not in Weeks 1-5)", "unseen"),
        ("3. Strict Stream-Unseen Cards (Zero-Memory Arrival Test)", "strict")
    ]:
        m = results.get("clean", {}).get(key, {})
        print(f"   {name}:")
        if not m:
            print("      No transactions or zero fraud observed in this cohort.")
            continue
        print(f"      Transactions: {m['total_tx']:,} | Frauds: {m['fraud_tx']:,} ({m['prevalence_pct']:.2f}%)")
        print(f"      Static Model   : Recall = {m['stat_recall']:.2f}% ({m['stat_tp']:,}/{m['fraud_tx']:,}) | Precision = {m['stat_prec']:.2f}% | AUC-PR = {m['stat_auc']:.4f}")
        print(f"      Adaptive Model : Recall = {m['adapt_recall']:.2f}% ({m['adapt_tp']:,}/{m['fraud_tx']:,}) | Precision = {m['adapt_prec']:.2f}% | AUC-PR = {m['adapt_auc']:.4f}")
        print(f"      Net Advantage  : Recall Gain = +{m['recall_gain']:.2f}% | AUC-PR Gain = +{m['auc_gain']:.4f}")

    print("\n" + "-" * 92)
    print("METHOD B: FULL POPULATION WITH COLLISION-FREE TOKENIZATION (MISSING_<trans_id>)")
    print(f"Coverage: 100% of out-of-sample stream ({g_total_tx:,} transactions, 0 spurious collisions)")
    print("-" * 92)
    for name, key in [
        ("1. Repeat / Baseline-Seen Entities", "seen"),
        ("2. Baseline-Unseen Entities (New cards + unique anon transactions)", "unseen"),
        ("3. Strict Stream-Unseen Entities (Mixed arrival population)", "strict")
    ]:
        m = results.get("collision_free", {}).get(key, {})
        print(f"   {name}:")
        if not m:
            print("      No transactions or zero fraud observed in this cohort.")
            continue
        print(f"      Transactions: {m['total_tx']:,} | Frauds: {m['fraud_tx']:,} ({m['prevalence_pct']:.2f}%)")
        print(f"      Static Model   : Recall = {m['stat_recall']:.2f}% ({m['stat_tp']:,}/{m['fraud_tx']:,}) | Precision = {m['stat_prec']:.2f}% | AUC-PR = {m['stat_auc']:.4f}")
        print(f"      Adaptive Model : Recall = {m['adapt_recall']:.2f}% ({m['adapt_tp']:,}/{m['fraud_tx']:,}) | Precision = {m['adapt_prec']:.2f}% | AUC-PR = {m['adapt_auc']:.4f}")
        print(f"      Net Advantage  : Recall Gain = +{m['recall_gain']:.2f}% | AUC-PR Gain = +{m['auc_gain']:.4f}")

    print("\n" + "-" * 92)
    print("MATHEMATICAL RECONCILIATION VERIFICATION:")
    cl_tp_sum = cl_seen_m.get("adapt_tp", 0) + cl_unseen_m.get("adapt_tp", 0)
    cl_fraud_sum = cl_seen_m.get("fraud_tx", 0) + cl_unseen_m.get("fraud_tx", 0)
    cf_seen_m = results.get("collision_free", {}).get("seen", {})
    cf_unseen_m = results.get("collision_free", {}).get("unseen", {})
    cf_tp_sum = cf_seen_m.get("adapt_tp", 0) + cf_unseen_m.get("adapt_tp", 0)
    cf_fraud_sum = cf_seen_m.get("fraud_tx", 0) + cf_unseen_m.get("fraud_tx", 0)

    cl_weighted_recall = (cl_tp_sum / cl_fraud_sum * 100) if cl_fraud_sum > 0 else 0.0
    cf_weighted_recall = (cf_tp_sum / cf_fraud_sum * 100) if cf_fraud_sum > 0 else 0.0

    print(f"   Method A Cohorts 1+2 Frauds: {cl_fraud_sum:,} | Adaptive Caught: {cl_tp_sum:,} -> Weighted Recall: {cl_weighted_recall:.4f}%")
    print(f"   Method B Cohorts 1+2 Frauds: {cf_fraud_sum:,} (matches global {g.get('fraud_tx', 0):,}) | Adaptive Caught: {cf_tp_sum:,} (matches global {g.get('adapt_tp', 0):,}) -> Weighted Recall: {cf_weighted_recall:.4f}%")
    print("=" * 92 + "\n")


def run_single_audit(
    df_raw: pd.DataFrame,
    exclude_card_identifiers: bool,
    title: str,
    random_seed: int = RANDOM_SEED,
    preprocessor: Any = None,
    raw_slices: Any = None
) -> Dict[str, Any]:
    logger.info(f"\n>>> Running Audit Configuration: {title} (exclude_card_identifiers={exclude_card_identifiers}, seed={random_seed})")

    # 1. Zero-Lookahead Preprocessor Initialization & Partitioning
    if preprocessor is None:
        preprocessor = Preprocessor(exclude_card_identifiers=exclude_card_identifiers)
    if raw_slices is None:
        raw_slices = preprocessor.partition_time_slices(df_raw)

    n_train_slices = max(1, int(len(raw_slices) * TRAINING_WINDOW_RATIO))
    train_raw = pd.concat([s["data"] for s in raw_slices[:n_train_slices]], ignore_index=True)

    # 2. Fit Preprocessor STRICTLY on Baseline Training Partition (Weeks 1-5)
    preprocessor.fit(train_raw)
    imbalance_info = preprocessor.imbalance_weights
    train_df = preprocessor.transform(train_raw)

    # Extract Baseline Card/Entity UIDs
    base_clean_uids = set(train_raw.loc[train_raw["is_clean"], "uid_clean"].unique())
    base_cf_uids = set(train_raw["uid_cf"].unique())

    logger.info(
        f"Baseline window fitted: {len(train_df):,} transactions, {train_df['isFraud'].sum():,} frauds. "
        f"scale_pos_weight={imbalance_info['scale_pos_weight']:.2f}, feature_count={len(preprocessor.feature_names)}. "
        f"Base clean cards: {len(base_clean_uids):,}, Base CF entities: {len(base_cf_uids):,}."
    )

    # 3. Model Initialization on NVIDIA GPU
    static_model = StaticBaselineModel(preprocessor.feature_names, imbalance_info["scale_pos_weight"], random_seed=random_seed)
    static_model.fit(train_df)

    adaptive_model = DriftAdaptiveModel(preprocessor.feature_names, imbalance_info["scale_pos_weight"], random_seed=random_seed)
    adaptive_model.fit_initial(train_df)
    drift_monitor = DriftMonitor()

    # 4. Warmup Drift Monitor across Baseline Window (Weeks 1-5, slices 0-4)
    logger.info("Warming up streaming drift monitor across baseline window (Weeks 1-5)...")
    for s in raw_slices[:n_train_slices]:
        slice_data = preprocessor.transform(s["data"])
        y_true = slice_data["isFraud"].values
        adapt_preds, _ = adaptive_model.predict(slice_data)
        slice_drift = False
        for i in range(len(slice_data)):
            alert = drift_monitor.update(
                int(y_true[i]), int(adapt_preds[i]),
                s["slice_index"], int(slice_data["step"].iloc[i]),
                slice_data["trans_id"].iloc[i]
            )
            if alert["any_drift"]:
                slice_drift = True
        adaptive_model.update_with_slice(slice_data, slice_drift, s["slice_index"])
        if slice_drift:
            drift_monitor.reset_detectors()

    logger.info(f"Baseline warmup complete. Drift alerts in warmup: {len(drift_monitor.drift_events)}.")

    # 5. Out-of-Sample Sequential Streaming Evaluation (Weeks 6-26, slices 5-25)
    logger.info("Evaluating sequential streaming slices (Weeks 6-26) with zero temporal lookahead...")
    start_t = time.time()

    # Tracking containers for:
    # A. Collision-Free Full Population (Method B)
    cf_seen = {"y": [], "stat_p": [], "adapt_p": [], "stat_prob": [], "adapt_prob": []}
    cf_unseen = {"y": [], "stat_p": [], "adapt_p": [], "stat_prob": [], "adapt_prob": []}
    cf_strict = {"y": [], "stat_p": [], "adapt_p": [], "stat_prob": [], "adapt_prob": []}

    # B. Clean Complete Cases Only (Method A)
    cl_seen = {"y": [], "stat_p": [], "adapt_p": [], "stat_prob": [], "adapt_prob": []}
    cl_unseen = {"y": [], "stat_p": [], "adapt_p": [], "stat_prob": [], "adapt_prob": []}
    cl_strict = {"y": [], "stat_p": [], "adapt_p": [], "stat_prob": [], "adapt_prob": []}

    cum_cf_uids = set(base_cf_uids)
    cum_cl_uids = set(base_clean_uids)

    global_y = []
    global_stat_p, global_adapt_p = [], []
    global_stat_prob, global_adapt_prob = [], []

    for s in raw_slices[n_train_slices:]:
        slice_data = preprocessor.transform(s["data"])
        y_true = slice_data["isFraud"].values
        uid_clean_vals = slice_data["uid_clean"].values
        uid_cf_vals = slice_data["uid_cf"].values
        is_clean_vals = slice_data["is_clean"].values

        # Predictions (out-of-sample before model update)
        stat_preds, stat_probs = static_model.predict(slice_data)
        adapt_preds, adapt_probs = adaptive_model.predict(slice_data)

        # Drift monitoring & model adaptation
        slice_drift = False
        for i in range(len(slice_data)):
            alert = drift_monitor.update(
                int(y_true[i]), int(adapt_preds[i]),
                s["slice_index"], int(slice_data["step"].iloc[i]),
                slice_data["trans_id"].iloc[i]
            )
            if alert["any_drift"]:
                slice_drift = True

        adaptive_model.update_with_slice(slice_data, slice_drift, s["slice_index"])
        if slice_drift:
            drift_monitor.reset_detectors()

        for i in range(len(y_true)):
            yt = int(y_true[i])
            sp = int(stat_preds[i])
            ap = int(adapt_preds[i])
            sprob = float(stat_probs[i])
            aprob = float(adapt_probs[i])

            global_y.append(yt)
            global_stat_p.append(sp)
            global_adapt_p.append(ap)
            global_stat_prob.append(sprob)
            global_adapt_prob.append(aprob)

            ucf = uid_cf_vals[i]
            is_cl = is_clean_vals[i]
            ucl = uid_clean_vals[i]

            # Method B: Collision-Free Full Population
            if ucf in base_cf_uids:
                cf_seen["y"].append(yt)
                cf_seen["stat_p"].append(sp)
                cf_seen["adapt_p"].append(ap)
                cf_seen["stat_prob"].append(sprob)
                cf_seen["adapt_prob"].append(aprob)
            else:
                cf_unseen["y"].append(yt)
                cf_unseen["stat_p"].append(sp)
                cf_unseen["adapt_p"].append(ap)
                cf_unseen["stat_prob"].append(sprob)
                cf_unseen["adapt_prob"].append(aprob)

            if ucf not in cum_cf_uids:
                cf_strict["y"].append(yt)
                cf_strict["stat_p"].append(sp)
                cf_strict["adapt_p"].append(ap)
                cf_strict["stat_prob"].append(sprob)
                cf_strict["adapt_prob"].append(aprob)

            # Method A: Clean Complete Cases Only
            if is_cl:
                if ucl in base_clean_uids:
                    cl_seen["y"].append(yt)
                    cl_seen["stat_p"].append(sp)
                    cl_seen["adapt_p"].append(ap)
                    cl_seen["stat_prob"].append(sprob)
                    cl_seen["adapt_prob"].append(aprob)
                else:
                    cl_unseen["y"].append(yt)
                    cl_unseen["stat_p"].append(sp)
                    cl_unseen["adapt_p"].append(ap)
                    cl_unseen["stat_prob"].append(sprob)
                    cl_unseen["adapt_prob"].append(aprob)

                if ucl not in cum_cl_uids:
                    cl_strict["y"].append(yt)
                    cl_strict["stat_p"].append(sp)
                    cl_strict["adapt_p"].append(ap)
                    cl_strict["stat_prob"].append(sprob)
                    cl_strict["adapt_prob"].append(aprob)

        cum_cf_uids.update(uid_cf_vals)
        cum_cl_uids.update(uid_clean_vals[is_clean_vals])

    duration = time.time() - start_t
    logger.info(f"Stream evaluation completed in {duration:.2f}s.")

    results = {
        "global_oos": compute_metrics({
            "y": global_y, "stat_p": global_stat_p, "adapt_p": global_adapt_p,
            "stat_prob": global_stat_prob, "adapt_prob": global_adapt_prob
        }),
        "clean": {
            "seen": compute_metrics(cl_seen),
            "unseen": compute_metrics(cl_unseen),
            "strict": compute_metrics(cl_strict),
        },
        "collision_free": {
            "seen": compute_metrics(cf_seen),
            "unseen": compute_metrics(cf_unseen),
            "strict": compute_metrics(cf_strict),
        }
    }

    print_audit_report(results, title)
    return results


def run(args_list: Optional[List[str]] = None):
    default_path = "data/ieee/train_transaction.csv" if Path("data/ieee/train_transaction.csv").exists() else "ieee/train_transaction.csv"
    parser = argparse.ArgumentParser(description="UID Generalization vs Memorization Audit")
    parser.add_argument("--dataset", type=str, default=default_path, help="Path to raw CSV dataset")
    parser.add_argument("--mode", type=str, choices=["both", "standard", "feature-blind"], default="both",
                        help="Audit mode: 'both' (default), 'standard' (with IDs), or 'feature-blind' (without card/addr IDs)")
    args = parser.parse_args(args_list)

    data_path = Path(args.dataset)
    if not data_path.exists():
        fallback_path = Path("ieee/train_transaction.csv") if "data" in str(data_path) else Path("data/ieee/train_transaction.csv")
        if fallback_path.exists():
            data_path = fallback_path
        else:
            raise FileNotFoundError(f"IEEE-CIS dataset not found at {data_path} or {fallback_path}")

    logger.info(f"Loading IEEE-CIS raw dataset from {data_path} for UID Generalization Audit...")
    ingestor = DatasetIngestion(data_path)
    df_raw, _ = ingestor.load_and_validate()

    # Precompute collision-free UID annotations directly on raw data
    valid_mask = df_raw["card2"].notna() & df_raw["addr1"].notna()
    uid_clean = pd.Series(index=df_raw.index, dtype=object)
    uid_clean[valid_mask] = (
        df_raw.loc[valid_mask, "card1"].astype(int).astype(str) + "_" +
        df_raw.loc[valid_mask, "card2"].astype(int).astype(str) + "_" +
        df_raw.loc[valid_mask, "addr1"].astype(int).astype(str)
    )
    uid_cf = uid_clean.copy()
    missing_mask = ~valid_mask
    uid_cf[missing_mask] = "MISSING_" + df_raw.loc[missing_mask, "TransactionID"].astype(str)

    df_raw["uid_clean"] = uid_clean.values
    df_raw["uid_cf"] = uid_cf.values
    df_raw["is_clean"] = valid_mask.values

    clean_total = int(valid_mask.sum())
    missing_total = int(missing_mask.sum())
    logger.info(
        f"Population Summary: {len(df_raw):,} total rows. "
        f"Clean complete cases: {clean_total:,} ({(clean_total/len(df_raw))*100:.2f}%). "
        f"Missing card2/addr1: {missing_total:,} ({(missing_total/len(df_raw))*100:.2f}%)."
    )

    if args.mode in ("standard", "both"):
        run_single_audit(
            df_raw=df_raw,
            exclude_card_identifiers=False,
            title="STANDARD FEATURES (Raw Card/Address Identifiers Included)"
        )

    if args.mode in ("feature-blind", "both"):
        run_single_audit(
            df_raw=df_raw,
            exclude_card_identifiers=True,
            title="FEATURE-BLIND / IDENTIFIER-EXCLUDED (card1, card2, card3, card5, addr1, addr2 EXCLUDED)"
        )


if __name__ == "__main__":
    run()
