"""
Synthetic financial transaction stream generator matching the PaySim schema.
Generates a multi-month (24-week / 6-month) transaction stream with realistic
temporal dynamics, class imbalance, and documented concept drift phases.

Satisfies:
- FR-1.1: Standard columns, transaction id, features, binary label, time step.
- FR-1.2: Genuine time dimension spanning 24 weeks (4,032 hours).
- FRD Risk Mitigation: Controlled concept drifts (Sudden, Gradual, Evolutionary).
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    RANDOM_SEED,
    TOTAL_SIMULATION_WEEKS,
    DEFAULT_HOURS_PER_WEEK,
)

logger = logging.getLogger("fraud_drift.data_generator")


def generate_paysim_stream(
    total_weeks: int = TOTAL_SIMULATION_WEEKS,
    tx_per_week: int = 4500,
    fraud_rate: float = 0.015,
    seed: int = RANDOM_SEED,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    Generates a 24-week synthetic financial transaction stream with controlled concept drift.

    Temporal Structure:
    - Total Hours: 24 weeks * 168 hours = 4,032 steps.
    - Total Transactions: ~108,000.
    - Class Imbalance: ~1.5% fraudulent transactions.

    Concept Drift Phases:
    - Phase 0 (Weeks 1-5, Training Window): Baseline Account Draining.
      Large TRANSFER and CASH_OUT during business hours draining oldbalanceOrg to 0.
    - Phase 1 (Weeks 6-10, Sudden Drift): Micro-Structuring / Smurfing.
      Abrupt shift to rapid low-value transfers ($500-$3,500) overnight (01:00-05:00).
    - Phase 2 (Weeks 11-16, Gradual Drift): Merchant & Camouflage Infiltration.
      Gradual transition into PAYMENT and DEBIT channels targeting merchant endpoints.
    - Phase 3 (Weeks 17-24, Complex Drift): Rapid Mule Routing.
      Turnover through newly created accounts with zero prior balance history.
    """
    np.random.seed(seed)
    total_steps = total_weeks * DEFAULT_HOURS_PER_WEEK
    records = []

    tx_counter = 1
    cust_counter = 1000

    for week in range(total_weeks):
        start_step = week * DEFAULT_HOURS_PER_WEEK + 1
        end_step = (week + 1) * DEFAULT_HOURS_PER_WEEK

        # Determine drift phase
        if week < 5:
            drift_phase = "Phase_0_Baseline"
        elif week < 10:
            drift_phase = "Phase_1_Sudden_MicroStructuring"
        elif week < 16:
            drift_phase = "Phase_2_Gradual_MerchantCamouflage"
        else:
            drift_phase = "Phase_3_Mule_Network"

        # Generate transactions for the week
        n_tx = int(np.random.normal(tx_per_week, tx_per_week * 0.05))

        for _ in range(n_tx):
            # Step in hours
            step = int(np.random.randint(start_step, end_step + 1))
            hour_of_day = (step - 1) % 24
            day_of_week = ((step - 1) // 24) % 7

            # Base probability of transaction by hour (diurnal cycle: peak 10:00-18:00)
            is_fraud = 1 if np.random.rand() < fraud_rate else 0

            # -------------------------------------------------------------
            # LEGITIMATE TRANSACTIONS (Background Distribution)
            # -------------------------------------------------------------
            if is_fraud == 0:
                tx_types = ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"]
                tx_type_probs = [0.40, 0.30, 0.18, 0.09, 0.03]
                tx_type = np.random.choice(tx_types, p=tx_type_probs)

                if tx_type == "PAYMENT":
                    amount = float(np.random.exponential(scale=65.0) + 5.0)
                    oldbalanceOrg = float(np.random.exponential(scale=1500.0) + amount)
                    newbalanceOrig = max(0.0, oldbalanceOrg - amount)
                    oldbalanceDest = float(np.random.exponential(scale=5000.0))
                    newbalanceDest = oldbalanceDest + amount
                    dest_prefix = "M"  # Merchant
                elif tx_type == "CASH_OUT":
                    amount = float(np.random.lognormal(mean=7.5, sigma=1.0))
                    oldbalanceOrg = float(amount + np.random.exponential(scale=1000.0))
                    newbalanceOrig = max(0.0, oldbalanceOrg - amount)
                    oldbalanceDest = float(np.random.exponential(scale=20000.0))
                    newbalanceDest = oldbalanceDest + amount
                    dest_prefix = "C"
                elif tx_type == "CASH_IN":
                    amount = float(np.random.lognormal(mean=7.8, sigma=0.9))
                    oldbalanceOrg = float(np.random.exponential(scale=10000.0))
                    newbalanceOrig = oldbalanceOrg + amount
                    oldbalanceDest = float(
                        max(0.0, np.random.exponential(scale=15000.0) - amount)
                    )
                    newbalanceDest = max(0.0, oldbalanceDest - amount)
                    dest_prefix = "C"
                elif tx_type == "TRANSFER":
                    amount = float(np.random.lognormal(mean=8.2, sigma=1.2))
                    oldbalanceOrg = float(amount + np.random.exponential(scale=5000.0))
                    newbalanceOrig = max(0.0, oldbalanceOrg - amount)
                    oldbalanceDest = float(np.random.exponential(scale=15000.0))
                    newbalanceDest = oldbalanceDest + amount
                    dest_prefix = "C"
                else:  # DEBIT
                    amount = float(np.random.uniform(10.0, 500.0))
                    oldbalanceOrg = float(amount + np.random.exponential(scale=2000.0))
                    newbalanceOrig = max(0.0, oldbalanceOrg - amount)
                    oldbalanceDest = float(np.random.exponential(scale=5000.0))
                    newbalanceDest = oldbalanceDest + amount
                    dest_prefix = "C"

                is_flagged = 1 if (amount > 200000 and tx_type == "TRANSFER") else 0

            # -------------------------------------------------------------
            # FRAUDULENT TRANSACTIONS (Evolving with Concept Drift)
            # -------------------------------------------------------------
            else:
                if drift_phase == "Phase_0_Baseline":
                    # Classic account drainage: High amount, TRANSFER/CASH_OUT, old balance cleared to zero
                    tx_type = np.random.choice(["TRANSFER", "CASH_OUT"], p=[0.55, 0.45])
                    amount = float(np.random.uniform(80000.0, 450000.0))
                    oldbalanceOrg = amount
                    newbalanceOrig = 0.0  # Perfect drainage signature
                    oldbalanceDest = float(
                        np.random.choice(
                            [0.0, np.random.uniform(100.0, 5000.0)], p=[0.7, 0.3]
                        )
                    )
                    newbalanceDest = oldbalanceDest + amount
                    dest_prefix = "C"

                elif drift_phase == "Phase_1_Sudden_MicroStructuring":
                    # Sudden drift: Attackers evade high-value rules by using smurfing/micro-transfers
                    # Predominantly overnight (00:00 - 05:00) with small amounts ($800 - $4,200)
                    tx_type = np.random.choice(["TRANSFER", "CASH_OUT"], p=[0.70, 0.30])
                    amount = float(np.random.uniform(800.0, 4200.0))
                    oldbalanceOrg = float(amount + np.random.uniform(1000.0, 20000.0))
                    # Crucially: does NOT drain to zero!
                    newbalanceOrig = oldbalanceOrg - amount
                    oldbalanceDest = 0.0
                    newbalanceDest = amount
                    dest_prefix = "C"
                    # Skew time towards overnight (hours 00:00 - 05:00 and 22:00 - 23:00)
                    hour_of_day = int(np.random.choice([0, 1, 2, 3, 4, 5, 22, 23]))
                    # Account for 1-indexed steps: hour_of_day = (step - 1) % 24
                    step = ((step - 1) // 24) * 24 + hour_of_day + 1

                elif drift_phase == "Phase_2_Gradual_MerchantCamouflage":
                    # Gradual drift: Blending into PAYMENT and DEBIT channels with merchant destinations
                    # Amount shifts to medium ($1,500 - $9,500)
                    tx_type = np.random.choice(
                        ["PAYMENT", "TRANSFER", "DEBIT"], p=[0.60, 0.25, 0.15]
                    )
                    amount = float(np.random.uniform(1500.0, 9500.0))
                    oldbalanceOrg = float(amount + np.random.uniform(500.0, 5000.0))
                    newbalanceOrig = oldbalanceOrg - amount
                    oldbalanceDest = float(np.random.uniform(1000.0, 50000.0))
                    newbalanceDest = oldbalanceDest + amount
                    dest_prefix = "M" if tx_type == "PAYMENT" else "C"

                else:  # Phase_3_Mule_Network
                    # Mule networks: Instant turnarounds, newly generated accounts with zero balance
                    tx_type = np.random.choice(
                        ["TRANSFER", "CASH_OUT", "CASH_IN"], p=[0.50, 0.35, 0.15]
                    )
                    amount = float(np.random.uniform(12000.0, 85000.0))
                    oldbalanceOrg = float(
                        amount + np.random.choice([0.0, 500.0], p=[0.8, 0.2])
                    )
                    newbalanceOrig = max(0.0, oldbalanceOrg - amount)
                    oldbalanceDest = 0.0
                    newbalanceDest = amount
                    dest_prefix = "C"

                is_flagged = 1 if (amount > 200000 and tx_type == "TRANSFER") else 0

            # Identifier generation
            orig_id = f"C{cust_counter + np.random.randint(0, 5000)}"
            dest_id = f"{dest_prefix}{cust_counter + np.random.randint(5000, 15000)}"
            tx_id = f"TX_{tx_counter:08d}"

            records.append(
                {
                    "trans_id": tx_id,
                    "step": step,
                    "type": tx_type,
                    "amount": round(amount, 2),
                    "nameOrig": orig_id,
                    "oldbalanceOrg": round(oldbalanceOrg, 2),
                    "newbalanceOrig": round(newbalanceOrig, 2),
                    "nameDest": dest_id,
                    "oldbalanceDest": round(oldbalanceDest, 2),
                    "newbalanceDest": round(newbalanceDest, 2),
                    "isFraud": is_fraud,
                    "isFlaggedFraud": is_flagged,
                    "drift_phase": drift_phase,
                    "week": week + 1,
                }
            )
            tx_counter += 1

    df = pd.DataFrame(records)
    # Sort strictly by time step (FR-2.4)
    df = df.sort_values(by=["step", "trans_id"]).reset_index(drop=True)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    out_file = RAW_DATA_DIR / "paysim_synthetic_24w.csv"
    logger.info(f"Generating synthetic financial transaction stream to {out_file}...")
    df = generate_paysim_stream(output_path=out_file)
    logger.info(
        f"Generated {len(df):,} transactions across {TOTAL_SIMULATION_WEEKS} weeks."
    )
    logger.info(
        f"Class Balance: {df['isFraud'].value_counts(normalize=True).to_dict()}"
    )
