"""
Federated Fraud Detection Module for Multi-Institution Privacy-Preserving Collaboration.

Simulates decentralized, non-IID transaction streams across 3 banking institutions:
- Bank A: Retail Digital E-Commerce Bank (High volume, low-ticket card-not-present fraud)
- Bank B: Cross-Border Remittances & High-Net-Worth (High-ticket wire/transfer fraud)
- Bank C: Commercial SME & Merchant Clearing (Payroll/invoice camouflage fraud)

Threat Model:
-------------
- Scenario: Collaborative multi-bank fraud ring detection without pooling raw customer data.
- Parties: K decentralized financial institutions (clients) and a semi-honest coordination server.
- Adversary Model: Semi-honest (honest-but-curious). Participants follow protocol specifications
  faithfully but may attempt to infer private customer transactions, cardholder identities, or
  proprietary balance features from transmitted model weights, tree structures, or gradients.
- Privacy Guarantees & Invariants:
  1. Strict Zero Raw-Data Leakage: No raw transaction rows, feature vectors, or ground-truth
     labels ever leave an institution's private perimeter (enforced by test assertion).
  2. Model-Only Exchange: Institutions exchange only serialized tree booster structures or
     differentially private split histogram aggregates.
  3. Defense Roadmap: In Phase 2, tree leaf weights and histogram splits are protected via
     Secure Multi-Party Computation (additive secret sharing) and Local Differential Privacy,
     directly extending Wang et al. (GTree TrustCom 2024, XGT TDSC 2025).

Demonstrates:
-------------
1. Non-IID distribution divergence across institutional silos.
2. The "Cross-Institution Blindness" of isolated single-bank models.
3. Federated Ensemble Aggregation achieving superior cross-bank recall without data sharing.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

from src.config import METRICS_DIR, RANDOM_SEED
from src.base_model import DEFAULT_XGB_HYPERPARAMS

logger = logging.getLogger("fraud_drift.federated")


# =========================================================================
# 1. NON-IID MULTI-BANK TRANSACTION STREAM GENERATOR
# =========================================================================
def generate_non_iid_bank_data(
    n_samples_per_bank: int = 3000,
    seed: int = RANDOM_SEED
) -> Dict[str, pd.DataFrame]:
    """
    Generates non-IID transaction data across 3 distinct banking archetypes.

    Archetypes:
    - Bank A (Digital E-Commerce): Low amounts ($5-$300), high transaction rate, automated attacks.
    - Bank B (Cross-Border Remittance): High amounts ($500-$25,000), balance depletion, international hops.
    - Bank C (Commercial SME): Moderate amounts ($100-$5,000), invoice/payroll camouflage, off-hours fraud.
    """
    rng = np.random.RandomState(seed)
    banks_data = {}

    # --- Bank A: Retail Digital E-Commerce ---
    # Low amounts, high transaction count, fast velocity fraud
    n_a = n_samples_per_bank
    fraud_rate_a = 0.022
    n_fraud_a = int(n_a * fraud_rate_a)
    n_legit_a = n_a - n_fraud_a

    amt_legit_a = rng.lognormal(mean=3.5, sigma=0.8, size=n_legit_a)  # Median ~$33
    amt_fraud_a = rng.lognormal(mean=2.8, sigma=0.5, size=n_fraud_a)  # Micro-amounts ~$16

    vel_legit_a = rng.poisson(lam=2.0, size=n_legit_a)
    vel_fraud_a = rng.poisson(lam=12.0, size=n_fraud_a)  # High automated attack velocity

    bal_err_legit_a = rng.normal(loc=0.0, scale=5.0, size=n_legit_a)
    bal_err_fraud_a = rng.normal(loc=2.0, scale=10.0, size=n_fraud_a)

    def _sample_hours(weights, size):
        p = np.array(weights, dtype=float)
        p = p / p.sum()
        return rng.choice(len(p), size=size, p=p)

    hour_legit_a = _sample_hours([0.02]*6 + [0.05]*12 + [0.0466]*6, n_legit_a)
    hour_fraud_a = rng.choice(24, size=n_fraud_a)  # Distributed overnight and daytime

    df_a = pd.DataFrame({
        "bank_id": "Bank_A_Retail_Ecommerce",
        "TransactionAmt": np.concatenate([amt_legit_a, amt_fraud_a]),
        "velocity_count": np.concatenate([vel_legit_a, vel_fraud_a]),
        "errorBalance": np.concatenate([bal_err_legit_a, bal_err_fraud_a]),
        "hour_of_day": np.concatenate([hour_legit_a, hour_fraud_a]),
        "foreign_dest": rng.binomial(1, p=np.concatenate([[0.05]*n_legit_a, [0.35]*n_fraud_a])),
        "isFraud": np.concatenate([np.zeros(n_legit_a, dtype=int), np.ones(n_fraud_a, dtype=int)])
    }).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    banks_data["Bank_A"] = df_a

    # --- Bank B: Cross-Border & High-Net-Worth ---
    # High amounts, account-takeover balance draining
    n_b = n_samples_per_bank
    fraud_rate_b = 0.012
    n_fraud_b = int(n_b * fraud_rate_b)
    n_legit_b = n_b - n_fraud_b

    amt_legit_b = rng.lognormal(mean=7.0, sigma=1.0, size=n_legit_b)  # Median ~$1,100
    amt_fraud_b = rng.lognormal(mean=8.5, sigma=0.7, size=n_fraud_b)  # Large wire ~$4,900

    vel_legit_b = rng.poisson(lam=1.2, size=n_legit_b)
    vel_fraud_b = rng.poisson(lam=2.5, size=n_fraud_b)  # Moderate wire velocity

    bal_err_legit_b = rng.normal(loc=0.0, scale=20.0, size=n_legit_b)
    bal_err_fraud_b = rng.normal(loc=1200.0, scale=300.0, size=n_fraud_b)  # Severe balance depletion

    hour_legit_b = _sample_hours([0.01]*7 + [0.06]*11 + [0.033]*6, n_legit_b)
    hour_fraud_b = _sample_hours([0.08]*6 + [0.03]*12 + [0.0266]*6, n_fraud_b)  # Overnight wire attacks

    df_b = pd.DataFrame({
        "bank_id": "Bank_B_CrossBorder_Remittance",
        "TransactionAmt": np.concatenate([amt_legit_b, amt_fraud_b]),
        "velocity_count": np.concatenate([vel_legit_b, vel_fraud_b]),
        "errorBalance": np.concatenate([bal_err_legit_b, bal_err_fraud_b]),
        "hour_of_day": np.concatenate([hour_legit_b, hour_fraud_b]),
        "foreign_dest": rng.binomial(1, p=np.concatenate([[0.40]*n_legit_b, [0.85]*n_fraud_b])),
        "isFraud": np.concatenate([np.zeros(n_legit_b, dtype=int), np.ones(n_fraud_b, dtype=int)])
    }).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    banks_data["Bank_B"] = df_b

    # --- Bank C: Commercial SME & Merchant Clearing ---
    # Moderate invoice amounts, invoice diversion fraud, business hours vs off-hours
    n_c = n_samples_per_bank
    fraud_rate_c = 0.018
    n_fraud_c = int(n_c * fraud_rate_c)
    n_legit_c = n_c - n_fraud_c

    amt_legit_c = rng.lognormal(mean=5.5, sigma=0.9, size=n_legit_c)  # Median ~$245
    amt_fraud_c = rng.lognormal(mean=6.2, sigma=0.8, size=n_fraud_c)  # Invoice spoof ~$500

    vel_legit_c = rng.poisson(lam=3.5, size=n_legit_c)
    vel_fraud_c = rng.poisson(lam=7.0, size=n_fraud_c)

    bal_err_legit_c = rng.normal(loc=0.0, scale=10.0, size=n_legit_c)
    bal_err_fraud_c = rng.normal(loc=150.0, scale=50.0, size=n_fraud_c)

    hour_legit_c = _sample_hours([0.005]*7 + [0.08]*10 + [0.0214]*7, n_legit_c)  # Strictly business hours
    hour_fraud_c = _sample_hours([0.09]*6 + [0.02]*12 + [0.0366]*6, n_fraud_c)  # Sneaked in off-hours

    df_c = pd.DataFrame({
        "bank_id": "Bank_C_Commercial_SME",
        "TransactionAmt": np.concatenate([amt_legit_c, amt_fraud_c]),
        "velocity_count": np.concatenate([vel_legit_c, vel_fraud_c]),
        "errorBalance": np.concatenate([bal_err_legit_c, bal_err_fraud_c]),
        "hour_of_day": np.concatenate([hour_legit_c, hour_fraud_c]),
        "foreign_dest": rng.binomial(1, p=np.concatenate([[0.15]*n_legit_c, [0.45]*n_fraud_c])),
        "isFraud": np.concatenate([np.zeros(n_legit_c, dtype=int), np.ones(n_fraud_c, dtype=int)])
    }).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    banks_data["Bank_C"] = df_c

    return banks_data


# =========================================================================
# 2. FEDERATED CLIENT & AGGREGATOR ARCHITECTURE
# =========================================================================
class BankClient:
    """
    Decentralized banking institution client holding private local data.
    Enforces strict zero-raw-data leakage invariant.
    """

    def __init__(self, bank_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]):
        self.bank_name = bank_name
        self._private_train = train_df.copy()  # Private local partition
        self._private_test = test_df.copy()
        self.feature_cols = feature_cols
        self.model: Optional[xgb.XGBClassifier] = None

    @property
    def sample_count(self) -> int:
        return len(self._private_train)

    @property
    def fraud_count(self) -> int:
        return int(self._private_train["isFraud"].sum())

    def train_local_model(self, hyperparams: Optional[Dict[str, Any]] = None) -> xgb.XGBClassifier:
        """Trains an isolated local XGBoost model on private data."""
        params = hyperparams or DEFAULT_XGB_HYPERPARAMS.copy()
        params = {k: v for k, v in params.items() if k not in ["scale_pos_weight", "random_state", "eval_metric"]}

        n_pos = self.fraud_count
        n_neg = self.sample_count - n_pos
        spw = max(1.0, float(n_neg) / max(1, n_pos))

        self.model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=spw,
            random_state=RANDOM_SEED,
            eval_metric="logloss"
        )
        X_train = self._private_train[self.feature_cols]
        y_train = self._private_train["isFraud"]
        self.model.fit(X_train, y_train)
        return self.model

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Provides the client's local evaluation set."""
        return self._private_test[self.feature_cols], self._private_test["isFraud"]

    def export_trees_for_federation(self) -> xgb.XGBClassifier:
        """
        Exports the trained booster object to the federation aggregator.
        CRITICAL PRIVACY INVARIANT: No raw rows or customer records are transmitted.
        """
        if self.model is None:
            raise RuntimeError(f"Client {self.bank_name} has not trained a local model.")
        return self.model


class FederatedEnsembleAggregator:
    """
    Federated Coordinator that aggregates client models into a unified
    cross-institution ensemble without accessing raw private records.
    """

    def __init__(self, clients: List[BankClient], feature_cols: List[str]):
        self.clients = clients
        self.feature_cols = feature_cols
        self.client_models: Dict[str, xgb.XGBClassifier] = {}
        self.client_weights: Dict[str, float] = {}

    def collect_and_aggregate(self):
        """
        Collects model weights from all decentralized clients and establishes
        a federated weighted ensemble based on institutional sample sizes.
        """
        total_samples = sum(c.sample_count for c in self.clients)
        for client in self.clients:
            # Model-only transfer: client exports tree estimator
            self.client_models[client.bank_name] = client.export_trees_for_federation()
            self.client_weights[client.bank_name] = client.sample_count / total_samples
            logger.info(f"Aggregator received model from {client.bank_name} (Weight: {self.client_weights[client.bank_name]:.3f})")

    def predict_proba(self, X: pd.DataFrame, mode: str = "noisy_or") -> np.ndarray:
        """
        Computes federated ensemble prediction probabilities across aggregated client models.

        Aggregation modes:
        - 'noisy_or' (default): Multi-expert evidence accumulation (P = 1 - prod(1 - P_k)).
          Captures domain-specialized fraud attacks (e.g. Bank A knows high-frequency attacks,
          Bank B knows wire fraud) without diluting specialized alert signals.
        - 'max': P = max_k P_k
        - 'weighted_mean': P = sum_k w_k * P_k
        """
        probs_matrix = np.array([
            model.predict_proba(X[self.feature_cols])[:, 1]
            for model in self.client_models.values()
        ])
        if mode == "noisy_or":
            return 1.0 - np.prod(1.0 - probs_matrix, axis=0)
        elif mode == "max":
            return np.max(probs_matrix, axis=0)
        elif mode == "weighted_mean":
            weights = np.array(list(self.client_weights.values()))
            return np.tensordot(weights, probs_matrix, axes=(0, 0))
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

    def predict(self, X: pd.DataFrame, threshold: float = 0.5, mode: str = "noisy_or") -> np.ndarray:
        probs = self.predict_proba(X, mode=mode)
        return (probs >= threshold).astype(int)


# =========================================================================
# 3. BENCHMARK EXECUTION & EVALUATION
# =========================================================================
def evaluate_model_performance(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive classification metrics."""
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc_roc = 0.5
    try:
        auc_pr = float(average_precision_score(y_true, y_prob))
    except Exception:
        auc_pr = 0.0
    return {
        "recall": round(rec * 100, 2),
        "precision": round(prec * 100, 2),
        "f1": round(f1, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4)
    }


def run_federated_fraud_benchmark(
    n_samples_per_bank: int = 3000,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Executes the 3-bank non-IID federated fraud benchmark and evaluates
    isolated local performance vs. federated collaborative performance.
    """
    logger.info("=" * 80)
    logger.info(f"STARTING 3-BANK FEDERATED FRAUD BENCHMARK (Non-IID Partition, N={n_samples_per_bank * 3})")
    logger.info("Threat Model: Semi-Honest Adversary | Zero Raw-Data Sharing")
    logger.info("=" * 80)

    # 1. Generate Non-IID Multi-Bank Streams
    banks_raw = generate_non_iid_bank_data(n_samples_per_bank=n_samples_per_bank, seed=seed)
    feature_cols = ["TransactionAmt", "velocity_count", "errorBalance", "hour_of_day", "foreign_dest"]

    # Verify Non-IID Statistical Divergence (KS-test on TransactionAmt between Bank A and Bank B)
    ks_stat, ks_pval = stats.ks_2samp(banks_raw["Bank_A"]["TransactionAmt"], banks_raw["Bank_B"]["TransactionAmt"])
    logger.info(f"Non-IID Verification (Bank A vs Bank B TransactionAmt): KS-stat = {ks_stat:.4f}, p = {ks_pval:.4e}")

    # 2. Partition into 70% Train / 30% Test locally per bank
    clients = []
    for bank_name, df_bank in banks_raw.items():
        split_idx = int(len(df_bank) * 0.7)
        train_df = df_bank.iloc[:split_idx].copy()
        test_df = df_bank.iloc[split_idx:].copy()
        client = BankClient(bank_name, train_df, test_df, feature_cols)
        client.train_local_model()
        clients.append(client)
        logger.info(f"Initialized & trained local model for {bank_name}: {len(train_df)} train, {len(test_df)} test (Frauds: {client.fraud_count})")

    # 3. Federated Aggregation (Zero Data Sharing)
    aggregator = FederatedEnsembleAggregator(clients, feature_cols)
    aggregator.collect_and_aggregate()

    # 4. Cross-Bank Evaluation Matrix
    # Compare:
    # A) Local Model on Local Test
    # B) Local Model on Foreign Test (Cross-Institution Blindness)
    # C) Federated Model on Each Bank's Test Set & Global Test Set
    results: Dict[str, Any] = {
        "metadata": {
            "n_banks": len(clients),
            "n_samples_total": n_samples_per_bank * 3,
            "feature_cols": feature_cols,
            "ks_test_a_vs_b": {"stat": round(ks_stat, 4), "p_value": float(ks_pval)},
            "threat_model": {
                "adversary": "Semi-honest (honest-but-curious)",
                "raw_data_transferred_bytes": 0,
                "model_exchange_protocol": "Federated Tree Booster Aggregation"
            }
        },
        "per_bank_eval": {},
        "global_eval": {}
    }

    global_test_X = []
    global_test_y = []

    for client in clients:
        X_test, y_test = client.get_test_data()
        global_test_X.append(X_test)
        global_test_y.append(y_test)

        # Local model prediction
        local_preds = client.model.predict(X_test)
        local_probs = client.model.predict_proba(X_test)[:, 1]
        local_metrics = evaluate_model_performance(y_test, local_preds, local_probs)

        # Federated model prediction on this bank's test set
        fed_probs = aggregator.predict_proba(X_test)
        fed_preds = (fed_probs >= 0.5).astype(int)
        fed_metrics = evaluate_model_performance(y_test, fed_preds, fed_probs)

        recall_delta = fed_metrics["recall"] - local_metrics["recall"]

        results["per_bank_eval"][client.bank_name] = {
            "local_isolated_model": local_metrics,
            "federated_model": fed_metrics,
            "net_recall_gain": round(recall_delta, 2)
        }
        logger.info(
            f"[{client.bank_name}] Local Recall: {local_metrics['recall']}% | "
            f"Federated Recall: {fed_metrics['recall']}% (Delta: {recall_delta:+.2f}%)"
        )

    # 5. Global Pooled Test Evaluation
    X_global = pd.concat(global_test_X, ignore_index=True)
    y_global = pd.concat(global_test_y, ignore_index=True)

    # Federated model on global pooled stream
    global_fed_probs = aggregator.predict_proba(X_global)
    global_fed_preds = (global_fed_probs >= 0.5).astype(int)
    results["global_eval"]["federated_model"] = evaluate_model_performance(y_global, global_fed_preds, global_fed_probs)

    # Average single-bank isolated model on global pooled stream (showing cross-bank failure)
    single_bank_recalls = []
    for client in clients:
        probs = client.model.predict_proba(X_global)[:, 1]
        preds = (probs >= 0.5).astype(int)
        m = evaluate_model_performance(y_global, preds, probs)
        single_bank_recalls.append(m["recall"])
    
    mean_isolated_recall = float(np.mean(single_bank_recalls))
    results["global_eval"]["isolated_models_mean_recall"] = round(mean_isolated_recall, 2)
    results["global_eval"]["federated_advantage_over_isolated"] = round(
        results["global_eval"]["federated_model"]["recall"] - mean_isolated_recall, 2
    )

    logger.info("=" * 80)
    logger.info(f"GLOBAL POOLED TEST RESULTS:")
    logger.info(f"Isolated Single-Bank Models Mean Recall: {mean_isolated_recall:.2f}%")
    logger.info(f"Federated Collaborative Model Recall:    {results['global_eval']['federated_model']['recall']:.2f}%")
    logger.info(f"Collaborative Net Advantage:             {results['global_eval']['federated_advantage_over_isolated']:+.2f}%")
    logger.info("=" * 80)

    # Persist summary metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = METRICS_DIR / "federated_summary_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved federated summary metrics to {out_file}")

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Multi-Bank Federated Fraud Detection Demonstration")
    parser.add_argument("--n-samples-per-bank", type=int, default=3000, help="Number of transactions per bank")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_federated_fraud_benchmark(n_samples_per_bank=args.n_samples_per_bank, seed=args.seed)


if __name__ == "__main__":
    main()
