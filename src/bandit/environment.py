"""
Non-Stationary Multi-Armed Bandit Environment.
Simulates piecewise-stationary Bernoulli and Gaussian reward processes with
abrupt and recurring concept drift regimes (representing adversarial fraud tactics).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class NonStationaryBanditEnv:
    """
    Piecewise-stationary multi-armed bandit environment with scheduled concept drift.
    """

    def __init__(
        self,
        n_arms: int = 5,
        drift_schedule: Optional[List[Dict[str, Any]]] = None,
        reward_distribution: str = "bernoulli",
        noise_std: float = 0.1,
        random_seed: int = 42,
    ):
        self.n_arms = n_arms
        self.reward_distribution = reward_distribution.lower()
        self.noise_std = noise_std
        self.rng = np.random.default_rng(random_seed)

        # Default multi-phase non-stationary schedule (10,000 steps, 4 phases)
        # Reflects initial regime, abrupt attack shift, second attack shift, and cyclical recurrence
        if drift_schedule is None:
            self.drift_schedule = [
                {
                    "end_step": 2500,
                    "means": [0.80, 0.30, 0.20, 0.40, 0.15],
                    "name": "Phase 1: Baseline",
                },
                {
                    "end_step": 5000,
                    "means": [0.20, 0.25, 0.85, 0.30, 0.10],
                    "name": "Phase 2: Abrupt Drift to Arm 2",
                },
                {
                    "end_step": 7500,
                    "means": [0.15, 0.20, 0.25, 0.30, 0.90],
                    "name": "Phase 3: Abrupt Drift to Arm 4",
                },
                {
                    "end_step": 10000,
                    "means": [0.85, 0.20, 0.15, 0.35, 0.25],
                    "name": "Phase 4: Cyclical Recurrence to Arm 0",
                },
            ]
        else:
            self.drift_schedule = drift_schedule

        self.current_step = 0
        self.current_phase_idx = 0
        self.change_points = [p["end_step"] for p in self.drift_schedule[:-1]]

    def reset(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        self.current_step = 0
        self.current_phase_idx = 0

    @property
    def current_means(self) -> List[float]:
        """Returns the true arm reward means at the current time step."""
        for idx, phase in enumerate(self.drift_schedule):
            if self.current_step < phase["end_step"]:
                self.current_phase_idx = idx
                return phase["means"]
        return self.drift_schedule[-1]["means"]

    @property
    def optimal_arm(self) -> int:
        """Returns the index of the optimal arm at the current time step."""
        return int(np.argmax(self.current_means))

    @property
    def max_mean(self) -> float:
        """Returns the expected reward of the optimal arm."""
        return float(np.max(self.current_means))

    def step(self, arm: int) -> Tuple[float, float, Dict[str, Any]]:
        """
        Executes an action by pulling arm `arm`.
        Returns:
            reward: Realized reward in [0, 1]
            instant_regret: Difference between optimal mean and chosen arm mean
            info: Metadata including current step, optimal arm, and phase
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Arm index {arm} out of range [0, {self.n_arms - 1}]")

        means = self.current_means
        chosen_mean = means[arm]
        opt_mean = float(np.max(means))
        instant_regret = opt_mean - chosen_mean

        if self.reward_distribution == "bernoulli":
            reward = 1.0 if self.rng.random() < chosen_mean else 0.0
        elif self.reward_distribution == "gaussian":
            raw_r = self.rng.normal(chosen_mean, self.noise_std)
            reward = float(np.clip(raw_r, 0.0, 1.0))
        else:
            raise ValueError(f"Unknown reward distribution: {self.reward_distribution}")

        self.current_step += 1

        info = {
            "step": self.current_step,
            "chosen_arm": arm,
            "chosen_mean": chosen_mean,
            "optimal_arm": int(np.argmax(means)),
            "optimal_mean": opt_mean,
            "instant_regret": instant_regret,
            "phase": self.drift_schedule[self.current_phase_idx]["name"],
        }

        return reward, instant_regret, info
