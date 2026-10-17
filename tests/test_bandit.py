"""
Unit tests for Non-Stationary Multi-Armed Bandit Environment and Adaptive Agents.
Tests environment transitions, agent selection rules, and ADWIN drift adaptation resets.
"""

import pytest

from src.bandit.environment import NonStationaryBanditEnv
from src.bandit.agents import (
    StandardUCBAgent,
    SlidingWindowUCBAgent,
    DiscountedUCBAgent,
    AdwinUCBAgent,
)


def test_env_phase_progression():
    schedule = [
        {"end_step": 10, "means": [0.9, 0.1], "name": "P1"},
        {"end_step": 20, "means": [0.1, 0.9], "name": "P2"},
    ]
    env = NonStationaryBanditEnv(n_arms=2, drift_schedule=schedule, random_seed=42)
    assert env.optimal_arm == 0
    assert env.current_means == [0.9, 0.1]

    for _ in range(10):
        env.step(0)

    assert env.current_step == 10
    assert env.optimal_arm == 1
    assert env.current_means == [0.1, 0.9]


def test_standard_ucb_round_robin():
    agent = StandardUCBAgent(n_arms=3)
    # First 3 steps with updates should pull arms 0, 1, 2
    pulled = []
    for _ in range(3):
        a = agent.select_arm()
        pulled.append(a)
        agent.update(a, 0.5)
    assert pulled == [0, 1, 2]


def test_sliding_window_ucb_history_pruning():
    agent = SlidingWindowUCBAgent(n_arms=2, window_size=50)
    for t in range(200):
        arm = agent.select_arm()
        agent.update(arm, 1.0)
    assert len(agent.history) <= 100


def test_discounted_ucb_decay():
    agent = DiscountedUCBAgent(n_arms=2, gamma=0.9)
    arm = agent.select_arm()
    agent.update(arm, 1.0)
    prior_pull = agent.discounted_pulls[arm]

    # Select another arm and update
    arm2 = agent.select_arm()
    agent.update(arm2, 0.0)

    # First arm's discounted pulls should have decayed by gamma
    assert pytest.approx(agent.discounted_pulls[arm], rel=1e-3) == prior_pull * 0.9


def test_adwin_ucb_drift_adaptation_reset():
    agent = AdwinUCBAgent(n_arms=2, delta=0.01)

    # Arm 0 emits 100 consecutive 1.0 rewards (high mean)
    for _ in range(100):
        agent.update(0, 1.0)

    prior_pulls = agent.active_pulls[0]
    assert prior_pulls >= 100

    # Arm 0 abruptly shifts to 0.0 rewards (abrupt negative shift)
    drift_detected = False
    for _ in range(100):
        agent.update(0, 0.0)
        if len(agent.drift_events) > 0:
            drift_detected = True
            break

    assert drift_detected is True
    # Arm 0 active pulls should have been reset following drift detection
    assert agent.active_pulls[0] < prior_pulls
