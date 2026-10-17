"""Entity-proxy cohort audit backed by the same chronological pipeline as other runs."""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from src.config import TRAINING_WINDOW_RATIO, RANDOM_SEED, RESULTS_DIR
from src.preprocessing import Preprocessor
from src.ingestion import DatasetIngestion

logger = logging.getLogger(__name__)


def annotate_entities(df):
    df = df.copy()
    required = ["card1", "card2", "addr1"]
    if not set(required).issubset(df):
        raise ValueError("Entity audit requires card1, card2 and addr1")
    valid = df[required].notna().all(axis=1)
    clean = pd.Series(None, index=df.index, dtype=object)
    clean.loc[valid] = df.loc[valid, required].apply(
        lambda row: "_".join(str(int(v)) for v in row), axis=1
    )
    df["uid_clean"] = clean
    df["uid_cf"] = clean.where(valid, "MISSING_" + df.trans_id.astype(str))
    df["is_clean"] = valid
    return df


def compute_metrics(d):
    y = np.asarray(d["y"])
    sp = np.asarray(d["stat_p"])
    ap = np.asarray(d["adapt_p"])
    if not len(y):
        return {}
    total = int(y.sum())
    out = dict(
        total_tx=len(y),
        fraud_tx=total,
        prevalence_pct=float(y.mean() * 100),
        recall_status="defined" if total else "undefined_no_fraud",
    )
    for prefix, pred, prob in [
        ("stat", sp, np.asarray(d["stat_prob"])),
        ("adapt", ap, np.asarray(d["adapt_prob"])),
    ]:
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        out.update(
            {
                prefix + "_tp": tp,
                prefix + "_fp": fp,
                prefix + "_recall": 100 * tp / total if total else 0.0,
                prefix + "_prec": 100 * tp / (tp + fp) if tp + fp else 0.0,
                prefix + "_auc": float(average_precision_score(y, prob))
                if total
                else None,
            }
        )
    out["recall_gain"] = out["adapt_recall"] - out["stat_recall"] if total else None
    out["auc_gain"] = out["adapt_auc"] - out["stat_auc"] if total else None
    return out


def _emit(text=""):
    print(text)
    logger.info(text)


def print_audit_report(results, title):
    _emit("AUDIT REPORT: " + title)
    _emit(
        "Entity keys are proxies. Missing-ID tokens identify transactions, not new people."
    )
    for family in ["global_oos", "clean", "collision_free"]:
        groups = (
            {"global": results[family]} if family == "global_oos" else results[family]
        )
        for cohort, m in groups.items():
            if not m:
                _emit(f"{family}/{cohort}: empty")
                continue
            recall = (
                f"{m['adapt_recall']:.2f}%" if m["fraud_tx"] else "undefined (no fraud)"
            )
            _emit(
                f"{family}/{cohort}: {m['total_tx']} transactions, {m['fraud_tx']} frauds; adaptive recall {recall}"
            )
    group = results.get("collision_free", {})
    total = sum(group.get(c, {}).get("fraud_tx", 0) for c in ["seen", "unseen"])
    caught = sum(group.get(c, {}).get("adapt_tp", 0) for c in ["seen", "unseen"])
    _emit(
        f"Weighted Recall: {100 * caught / total if total else 0.0:.4f}%"
        + (" (undefined; zero fraud denominator)" if not total else "")
    )


def run_single_audit(
    df_raw,
    exclude_card_identifiers,
    title,
    random_seed=RANDOM_SEED,
    preprocessor=None,
    raw_slices=None,
    output_dir=None,
    data_kind="unverified",
):
    from src.pipeline import run_pipeline

    raw = df_raw.copy()
    if not {"uid_clean", "uid_cf", "is_clean"}.issubset(raw):
        raw = annotate_entities(raw)
    prep = preprocessor or Preprocessor(
        exclude_card_identifiers=exclude_card_identifiers
    )
    prep.exclude_card_identifiers = exclude_card_identifiers
    slices = raw_slices if raw_slices is not None else prep.partition_time_slices(raw)
    n_train = max(1, int(len(slices) * TRAINING_WINDOW_RATIO))
    result = run_pipeline(
        "provided_entity_frame.csv",
        raw_dataframe=raw,
        raw_slices=slices,
        preprocessor=prep,
        output_dir=output_dir or RESULTS_DIR / "cohort_runs",
        n_train_slices=n_train,
        exclude_card_identifiers=exclude_card_identifiers,
        seed=random_seed,
        data_kind=data_kind,
        strategies=("static", "adaptive"),
    )
    pred = pd.read_csv(result["run_dir"] / "predictions.csv", dtype={"trans_id": str})
    static = pred[pred.model == "static"].set_index("trans_id")
    adaptive = pred[pred.model == "adaptive"].set_index("trans_id")
    initial = pd.concat([s["data"] for s in slices[:n_train]], ignore_index=True)
    stream = pd.concat([s["data"] for s in slices[n_train:]], ignore_index=True).copy()
    base_cf = set(initial.uid_cf)
    base_clean = set(initial.loc[initial.is_clean, "uid_clean"])
    seen_cf = set(base_cf)
    seen_clean = set(base_clean)
    rows = []
    for tx in stream.to_dict("records"):
        identity = str(tx["trans_id"])
        st = static.loc[identity]
        ad = adaptive.loc[identity]
        u = tx["uid_cf"]
        clean = bool(tx["is_clean"])
        uc = tx["uid_clean"]
        rows.append(
            dict(
                trans_id=identity,
                y=int(st.y_true),
                stat_p=int(st.prediction),
                adapt_p=int(ad.prediction),
                stat_prob=float(st.probability),
                adapt_prob=float(ad.probability),
                cf_seen=u in base_cf,
                cf_strict=u not in seen_cf,
                clean=clean,
                cl_seen=clean and uc in base_clean,
                cl_strict=clean and uc not in seen_clean,
            )
        )
        seen_cf.add(u)
        if clean:
            seen_clean.add(uc)
    frame = pd.DataFrame(rows)

    def metric(mask):
        return compute_metrics(
            frame.loc[
                mask, ["y", "stat_p", "adapt_p", "stat_prob", "adapt_prob"]
            ].to_dict("list")
        )

    results = dict(
        global_oos=metric(np.ones(len(frame), dtype=bool)),
        clean=dict(
            seen=metric(frame.clean & frame.cl_seen),
            unseen=metric(frame.clean & ~frame.cl_seen),
            strict=metric(frame.cl_strict),
        ),
        collision_free=dict(
            seen=metric(frame.cf_seen),
            unseen=metric(~frame.cf_seen),
            strict=metric(frame.cf_strict),
        ),
        run_id=result["run_id"],
        run_dir=str(result["run_dir"]),
    )
    frame.to_csv(result["run_dir"] / "cohort_predictions.csv", index=False)
    (result["run_dir"] / "cohort_summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    print_audit_report(results, title)
    return results


def run(args_list=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--mode", choices=["standard", "feature-blind", "both"], default="both"
    )
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args(args_list)
    raw, _ = DatasetIngestion(Path(a.dataset)).load_and_validate()
    raw = annotate_entities(raw)
    for blind in [False, True] if a.mode == "both" else [a.mode == "feature-blind"]:
        r = run_single_audit(
            raw,
            blind,
            "FEATURE BLIND" if blind else "STANDARD",
            random_seed=a.seed,
            output_dir=a.output_dir,
            data_kind="synthetic" if a.synthetic else "real",
        )
        print(Path(r["run_dir"]) / "cohort_summary.json")


if __name__ == "__main__":
    run()
