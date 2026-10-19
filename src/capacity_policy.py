"""
Review Capacity Allocation Policies (Finding R06).
Formally implements and separates three operational decision mechanisms:
1. Retrospective Daily Batch Ranking (retrospective ranking at day boundaries)
2. Expected-Volume Thresholding (calibrated threshold targeting expected alert volume)
3. Causal Online Review Allocation (real-time quota tracker with restart persistence)
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Policy identification constants
POLICY_RETROSPECTIVE_BATCH = "retrospective_daily_batch_ranking"
POLICY_EXPECTED_VOLUME = "expected_volume_thresholding"
POLICY_CAUSAL_ONLINE = "causal_online_review_allocation"


class QuotaResult:
    """Evaluation result for a single transaction under causal quota policy."""
    def __init__(self, is_alert: bool, reason: str, remaining_quota: int, day: int):
        self.is_alert = is_alert
        self.reason = reason
        self.remaining_quota = remaining_quota
        self.day = day

    def __iter__(self):
        return iter((self.is_alert, self.reason))

    def __repr__(self):
        return f"QuotaResult(is_alert={self.is_alert}, reason='{self.reason}', remaining={self.remaining_quota}, day={self.day})"


class OnlineDailyQuotaAllocator:
    """
    Causal online review capacity state manager (Finding R06).
    Enforces a strict daily quota (e.g. K=5, 100, 500 alerts/day) in causal real time.
    Advances day boundaries automatically based on transaction timestamps/steps.
    Guarantees deterministic tie-handling and supports restart persistence.
    """

    def __init__(
        self,
        daily_budget: int = 100,
        budget: Optional[int] = None,
        hours_per_day: float = 24.0,
        decision_threshold: float = 0.5,
        threshold: Optional[float] = None,
        state_path: Optional[Union[str, Path]] = None,
    ):
        if budget is not None:
            daily_budget = budget
        if threshold is not None:
            decision_threshold = threshold
        if daily_budget <= 0:
            raise ValueError(f"daily_budget must be a positive integer, got {daily_budget}")
        self.daily_budget = int(daily_budget)
        self.hours_per_day = float(hours_per_day)
        self.decision_threshold = float(decision_threshold)
        self.state_path = Path(state_path) if state_path else None

        # Causal state variables
        self.current_day: Optional[int] = None
        self.alerts_today: int = 0
        self.total_alerts_emitted: int = 0
        self.total_evaluations: int = 0
        self.total_excess_demand: int = 0
        self.day_seen_decisions: Dict[str, Tuple[bool, str]] = {}

        if self.state_path and self.state_path.exists():
            self._load_state()

    @property
    def daily_remaining(self) -> int:
        return max(0, self.daily_budget - self.alerts_today)

    @property
    def day_allocated(self) -> int:
        return self.alerts_today

    def get_day(self, step: float) -> int:
        """Determines discrete day index from step/hour causal timestamp."""
        return int(math.floor(step / self.hours_per_day))

    def evaluate(
        self,
        trans_id: str,
        probability: float,
        step: Optional[float] = None,
        day: Optional[int] = None,
    ) -> QuotaResult:
        """
        Evaluates a single incoming transaction causally against the daily review quota.
        Returns QuotaResult(is_alert, reason, remaining_quota, day).
        """
        self.total_evaluations += 1
        if day is None:
            day = self.get_day(step if step is not None else 0.0)

        # Advance day boundary
        if self.current_day is None or day > self.current_day:
            self.current_day = day
            self.alerts_today = 0
            self.day_seen_decisions.clear()

        # Check for duplicate trans_id within current day (idempotent replay)
        if trans_id in self.day_seen_decisions:
            prior_alert, _ = self.day_seen_decisions[trans_id]
            reason = "admitted_prior" if prior_alert else "rejected_prior"
            return QuotaResult(prior_alert, reason, self.daily_remaining, day)

        above_threshold = bool(probability >= self.decision_threshold)
        if not above_threshold:
            self.day_seen_decisions[trans_id] = (False, "below_threshold")
            res = QuotaResult(False, "below_threshold", self.daily_remaining, day)
        elif self.alerts_today < self.daily_budget:
            self.alerts_today += 1
            self.total_alerts_emitted += 1
            self.day_seen_decisions[trans_id] = (True, "admitted")
            res = QuotaResult(True, "admitted", self.daily_remaining, day)
        else:
            self.total_excess_demand += 1
            self.day_seen_decisions[trans_id] = (False, "quota_exhausted")
            res = QuotaResult(False, "quota_exhausted", self.daily_remaining, day)

        if self.state_path:
            self._save_state()
        return res

    def process_transaction(
        self,
        trans_id: str,
        probability: float,
        day: Optional[int] = None,
        step: Optional[float] = None,
    ) -> QuotaResult:
        """Alias for evaluate."""
        return self.evaluate(trans_id=trans_id, probability=probability, step=step, day=day)

    def _save_state(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.to_dict(), indent=2))

    def _load_state(self):
        if self.state_path and self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            self.daily_budget = data.get("daily_budget", self.daily_budget)
            self.hours_per_day = data.get("hours_per_day", self.hours_per_day)
            self.decision_threshold = data.get("decision_threshold", self.decision_threshold)
            self.current_day = data.get("current_day")
            self.alerts_today = data.get("alerts_today", 0)
            self.total_alerts_emitted = data.get("total_alerts_emitted", 0)
            self.total_evaluations = data.get("total_evaluations", 0)
            self.total_excess_demand = data.get("total_excess_demand", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes causal quota state for restart persistence."""
        return {
            "daily_budget": self.daily_budget,
            "hours_per_day": self.hours_per_day,
            "decision_threshold": self.decision_threshold,
            "current_day": self.current_day,
            "alerts_today": self.alerts_today,
            "total_alerts_emitted": self.total_alerts_emitted,
            "total_evaluations": self.total_evaluations,
            "total_excess_demand": self.total_excess_demand,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OnlineDailyQuotaAllocator":
        """Restores causal quota state from persisted dictionary."""
        alloc = cls(
            daily_budget=data["daily_budget"],
            hours_per_day=data["hours_per_day"],
            decision_threshold=data["decision_threshold"],
        )
        alloc.current_day = data.get("current_day")
        alloc.alerts_today = data.get("alerts_today", 0)
        alloc.total_alerts_emitted = data.get("total_alerts_emitted", 0)
        alloc.total_evaluations = data.get("total_evaluations", 0)
        alloc.total_excess_demand = data.get("total_excess_demand", 0)
        return alloc


def allocate_batch_ranking(
    df: pd.DataFrame,
    budget: int = 100,
    daily_budget: Optional[int] = None,
    day_col: str = "day",
    step_col: str = "step",
    prob_col: str = "probability",
    id_col: str = "trans_id",
    label_col: str = "y_true",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Retrospective Daily Batch Ranking (Policy 1).
    Ranks transactions retrospectively per day by probability descending, breaking ties by trans_id.
    Selects exactly min(budget, N_day) alerts per day.
    """
    if daily_budget is not None:
        budget = daily_budget
    if budget <= 0:
        raise ValueError(f"budget must be a positive integer, got {budget}")

    frame = df.copy()
    if day_col not in frame.columns:
        if step_col in frame.columns:
            frame[day_col] = (frame[step_col] // 24).astype(int)
        else:
            frame[day_col] = 0

    if label_col not in frame.columns:
        for c in ["is_fraud", "isFraud"]:
            if c in frame.columns:
                label_col = c
                break

    selected_rows = []
    total_frauds = int(frame[label_col].sum()) if label_col in frame.columns else 0

    for day_val, day_group in frame.groupby(day_col):
        # Deterministic sorting on probability descending, trans_id ascending
        top = day_group.sort_values(
            [prob_col, id_col], ascending=[False, True]
        ).head(budget)
        selected_rows.append(top)

    if selected_rows:
        alerts_df = pd.concat(selected_rows, ignore_index=True)
    else:
        alerts_df = pd.DataFrame(columns=frame.columns)

    alerts_df["is_alert"] = 1
    total_alerts = len(alerts_df)
    detected_fraud = int(alerts_df[label_col].sum()) if label_col in alerts_df.columns else 0
    precision = float(detected_fraud / total_alerts) if total_alerts > 0 else 0.0
    recall = float(detected_fraud / total_frauds) if total_frauds > 0 else 0.0

    summary = {
        "policy": POLICY_RETROSPECTIVE_BATCH,
        "daily_budget": budget,
        "total_days": frame[day_col].nunique(),
        "total_transactions": len(frame),
        "total_alerts": total_alerts,
        "detected_fraud": detected_fraud,
        "total_frauds": total_frauds,
        "precision": precision,
        "recall": recall,
        "operational_assumption": "Retrospective end-of-day batch ranking; requires collecting full daily volume before decision",
    }

    return alerts_df, summary


def allocate_online_stream(
    df: pd.DataFrame,
    daily_budget: int = 100,
    budget: Optional[int] = None,
    decision_threshold: float = 0.5,
    threshold: Optional[float] = None,
    step_col: str = "step",
    day_col: str = "day",
    prob_col: str = "probability",
    id_col: str = "trans_id",
    label_col: str = "y_true",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Causal Online Review Allocation (Policy 3).
    Evaluates transactions strictly in temporal order without lookahead.
    Enforces daily quota and records excess demand.
    """
    if budget is not None:
        daily_budget = budget
    if threshold is not None:
        decision_threshold = threshold

    allocator = OnlineDailyQuotaAllocator(
        daily_budget=daily_budget,
        decision_threshold=decision_threshold,
    )

    if label_col not in df.columns:
        for c in ["is_fraud", "isFraud"]:
            if c in df.columns:
                label_col = c
                break

    # Determine sorting order
    sort_cols = [c for c in [day_col, step_col, id_col] if c in df.columns]
    if sort_cols:
        sorted_df = df.sort_values(sort_cols, ascending=[True] * len(sort_cols))
    else:
        sorted_df = df.copy()

    decisions = []
    reasons = []

    for _, row in sorted_df.iterrows():
        day_val = int(row[day_col]) if day_col in row else None
        step_val = float(row[step_col]) if step_col in row else None
        res = allocator.evaluate(
            trans_id=str(row[id_col]),
            probability=float(row[prob_col]),
            step=step_val,
            day=day_val,
        )
        decisions.append(int(res.is_alert))
        reasons.append(res.reason)

    result_df = sorted_df.copy()
    result_df["online_alert"] = decisions
    result_df["online_reason"] = reasons
    result_df["is_alert"] = decisions

    alerts_df = result_df[result_df["online_alert"] == 1]
    total_alerts = len(alerts_df)
    total_frauds = int(df[label_col].sum()) if label_col in df.columns else 0
    detected_fraud = int(alerts_df[label_col].sum()) if label_col in alerts_df.columns else 0

    precision = float(detected_fraud / total_alerts) if total_alerts > 0 else 0.0
    recall = float(detected_fraud / total_frauds) if total_frauds > 0 else 0.0

    summary = {
        "policy": POLICY_CAUSAL_ONLINE,
        "daily_budget": daily_budget,
        "decision_threshold": decision_threshold,
        "total_evaluations": allocator.total_evaluations,
        "total_alerts": total_alerts,
        "total_excess_demand": allocator.total_excess_demand,
        "detected_fraud": detected_fraud,
        "total_frauds": total_frauds,
        "precision": precision,
        "recall": recall,
        "operational_assumption": "Strict causal online stream allocation; no future day lookahead; capped at daily_budget per 24h",
    }

    return alerts_df, summary
