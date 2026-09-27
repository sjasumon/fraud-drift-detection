"""
Multi-Armed Bandit Agents for Stationary and Non-Stationary Environments.
Implements:
1. StandardUCBAgent (UCB1 stationary control)
2. SlidingWindowUCBAgent (SW-UCB / M-UCB benchmark; Garivier & Moulines, 2011)
3. DiscountedUCBAgent (D-UCB benchmark; Kocsis & Szepesvári, 2006)
4. AdwinUCBAgent (Proposed Concept-Drift-Aware Adaptive Bandit with ADWIN)
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np
from river import drift


class BaseBanditAgent:
    """Abstract base class for multi-armed bandit agents."""

    def __init__(self, n_arms: int, name: str):
        self.n_arms = n_arms
        self.name = name
        self.t = 0
        self.total_pulls = np.zeros(n_arms, dtype=int)
        self.total_rewards = np.zeros(n_arms, dtype=float)

    def select_arm(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float):
        raise NotImplementedError

    def reset(self):
        self.t = 0
        self.total_pulls.fill(0)
        self.total_rewards.fill(0.0)


# =========================================================================
# 1. STANDARD UCB1 (STATIONARY CONTROL BASELINE)
# =========================================================================
class StandardUCBAgent(BaseBanditAgent):
    """
    Standard UCB1 (Auer et al., 2002).
    Optimal for stationary bandits; suffers linear regret under concept drift.
    """

    def __init__(self, n_arms: int, c: float = 1.0):
        super().__init__(n_arms, name="Standard UCB1 (Stationary)")
        self.c = c

    def select_arm(self) -> int:
        self.t += 1
        # Initial round-robin exploration
        for arm in range(self.n_arms):
            if self.total_pulls[arm] == 0:
                return arm

        ucb_values = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            mean_a = self.total_rewards[a] / self.total_pulls[a]
            bonus = self.c * math.sqrt(2.0 * math.log(self.t) / self.total_pulls[a])
            ucb_values[a] = mean_a + bonus

        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += reward


# =========================================================================
# 2. SLIDING-WINDOW UCB (M-UCB / SW-UCB BASELINE)
# =========================================================================
class SlidingWindowUCBAgent(BaseBanditAgent):
    """
    Sliding-Window UCB (Garivier & Moulines, 2011).
    Maintains a fixed recent window tau of pulls and rewards.
    Discarding observations older than tau enables adaptation to drift,
    at the cost of persistent exploration during stationary periods.
    """

    def __init__(self, n_arms: int, window_size: int = 600, c: float = 1.0):
        super().__init__(n_arms, name=f"Sliding-Window UCB (W={window_size})")
        self.window_size = window_size
        self.c = c
        self.history: List[Tuple[int, float]] = []

    def reset(self):
        super().reset()
        self.history = []

    def select_arm(self) -> int:
        self.t += 1

        # Count pulls and rewards strictly within the sliding window
        window_pulls = np.zeros(self.n_arms, dtype=int)
        window_rewards = np.zeros(self.n_arms, dtype=float)

        for arm, rew in self.history[-self.window_size:]:
            window_pulls[arm] += 1
            window_rewards[arm] += rew

        # Explore unpulled arms in current window
        for arm in range(self.n_arms):
            if window_pulls[arm] == 0:
                return arm

        eff_t = min(self.t, self.window_size)
        ucb_values = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            mean_a = window_rewards[a] / window_pulls[a]
            bonus = self.c * math.sqrt(2.0 * math.log(eff_t) / window_pulls[a])
            ucb_values[a] = mean_a + bonus

        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += reward
        self.history.append((arm, reward))
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size:]


# =========================================================================
# 3. DISCOUNTED UCB (D-UCB BASELINE)
# =========================================================================
class DiscountedUCBAgent(BaseBanditAgent):
    """
    Discounted UCB (Kocsis & Szepesvári, 2006).
    Applies exponential decay discount factor gamma to past rewards.
    """

    def __init__(self, n_arms: int, gamma: float = 0.995, c: float = 1.0):
        super().__init__(n_arms, name=f"Discounted UCB (γ={gamma})")
        self.gamma = gamma
        self.c = c
        self.discounted_pulls = np.zeros(n_arms, dtype=float)
        self.discounted_rewards = np.zeros(n_arms, dtype=float)

    def reset(self):
        super().reset()
        self.discounted_pulls.fill(0.0)
        self.discounted_rewards.fill(0.0)

    def select_arm(self) -> int:
        self.t += 1
        for arm in range(self.n_arms):
            if self.discounted_pulls[arm] < 1.0:
                return arm

        n_gamma = float(np.sum(self.discounted_pulls))
        ucb_values = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            mean_a = self.discounted_rewards[a] / self.discounted_pulls[a]
            bonus = self.c * math.sqrt(2.0 * math.log(max(1.0, n_gamma)) / self.discounted_pulls[a])
            ucb_values[a] = mean_a + bonus

        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += reward

        # Apply exponential decay to all arms
        self.discounted_pulls *= self.gamma
        self.discounted_rewards *= self.gamma

        # Increment chosen arm
        self.discounted_pulls[arm] += 1.0
        self.discounted_rewards[arm] += reward


# =========================================================================
# 4. ADWIN-UCB (PROPOSED CONCEPT-DRIFT-ADAPTIVE BANDIT)
# =========================================================================
class AdwinUCBAgent(BaseBanditAgent):
    """
    Concept-Drift-Aware Adaptive Multi-Armed Bandit (ADWIN-UCB).
    Instruments each arm with an independent ADWIN drift detector.
    Accumulates full historical memory during stationary periods for minimal regret.
    Upon confirmed statistical drift in an arm's reward distribution, resets
    that arm's memory to allow immediate re-estimation of the new concept.
    """

    def __init__(self, n_arms: int, delta: float = 0.002, c: float = 1.0):
        super().__init__(n_arms, name="ADWIN-UCB (Drift-Adaptive)")
        self.delta = delta
        self.c = c
        self.detectors = [drift.ADWIN(delta=delta) for _ in range(n_arms)]
        self.active_pulls = np.zeros(n_arms, dtype=int)
        self.active_rewards = np.zeros(n_arms, dtype=float)
        self.drift_events: List[Dict[str, Any]] = []

    def reset(self):
        super().reset()
        self.detectors = [drift.ADWIN(delta=self.delta) for _ in range(self.n_arms)]
        self.active_pulls.fill(0)
        self.active_rewards.fill(0.0)
        self.drift_events = []

    def select_arm(self) -> int:
        self.t += 1

        # Check for uninitialized arms
        for arm in range(self.n_arms):
            if self.active_pulls[arm] == 0:
                return arm

        total_active = max(1, int(np.sum(self.active_pulls)))
        ucb_values = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            mean_a = self.active_rewards[a] / self.active_pulls[a]
            bonus = self.c * math.sqrt(2.0 * math.log(total_active) / self.active_pulls[a])
            ucb_values[a] = mean_a + bonus

        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += reward
        self.active_pulls[arm] += 1
        self.active_rewards[arm] += reward

        # Update ADWIN detector for this arm
        self.detectors[arm].update(reward)

        # Check if ADWIN flags concept drift in the arm's reward distribution
        if self.detectors[arm].drift_detected:
            # Concept drift confirmed: reset memory for this arm
            est_mean = self.detectors[arm].estimation if hasattr(self.detectors[arm], "estimation") else reward
            self.drift_events.append({
                "step": self.t,
                "arm": arm,
                "prior_pulls": int(self.active_pulls[arm]),
                "prior_mean": round(float(self.active_rewards[arm] / max(1, self.active_pulls[arm])), 4),
                "adwin_est": round(float(est_mean), 4)
            })

            # Reset arm statistics to allow rapid re-learning
            self.active_pulls[arm] = 1
            self.active_rewards[arm] = float(est_mean)
            self.detectors[arm] = drift.ADWIN(delta=self.delta)
