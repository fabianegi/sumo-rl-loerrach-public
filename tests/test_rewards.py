"""Tests for src/environment/rewards.py."""
from __future__ import annotations

import pytest

from src.environment.rewards import diff_waiting_time_reward, pressure_reward


class MockTrafficSignal:
    """Minimal mock of sumo-rl's TrafficSignal for unit testing rewards."""

    def __init__(
        self,
        incoming_density: list[float],
        outgoing_density: list[float],
        waiting_times: list[float] | None = None,
        last_measure: float = 0.0,
    ) -> None:
        self._incoming = incoming_density
        self._outgoing = outgoing_density
        self._waiting = waiting_times if waiting_times is not None else [0.0]
        self.last_measure = last_measure

    def get_lanes_density(self) -> list[float]:
        return self._incoming

    def get_out_lanes_density(self) -> list[float]:
        return self._outgoing

    def get_accumulated_waiting_time_per_lane(self) -> list[float]:
        return self._waiting


def test_pressure_reward_returns_negative_or_zero() -> None:
    """Pressure reward should always be <= 0 (negative absolute value)."""
    # Imbalanced: more incoming than outgoing
    ts = MockTrafficSignal(
        incoming_density=[0.8, 0.6],
        outgoing_density=[0.2, 0.1],
    )
    assert pressure_reward(ts) <= 0

    # Reversed imbalance
    ts2 = MockTrafficSignal(
        incoming_density=[0.1, 0.1],
        outgoing_density=[0.9, 0.8],
    )
    assert pressure_reward(ts2) <= 0


def test_pressure_reward_zero_when_balanced() -> None:
    """Pressure = 0 when incoming == outgoing density."""
    ts = MockTrafficSignal(
        incoming_density=[0.5, 0.3],
        outgoing_density=[0.5, 0.3],
    )
    assert pressure_reward(ts) == pytest.approx(0.0)


def test_pressure_reward_more_negative_when_imbalance_increases() -> None:
    """Larger density gap produces more negative reward."""
    ts_small = MockTrafficSignal(
        incoming_density=[0.5, 0.5],
        outgoing_density=[0.4, 0.4],
    )
    ts_large = MockTrafficSignal(
        incoming_density=[0.9, 0.9],
        outgoing_density=[0.1, 0.1],
    )
    assert pressure_reward(ts_large) < pressure_reward(ts_small)


def test_diff_waiting_time_positive_when_improving() -> None:
    """diff-waiting-time should be positive when waiting time decreases."""
    # last_measure=5.0, current waiting = 200/100 = 2.0 → reward = 5.0 - 2.0 = 3.0
    ts = MockTrafficSignal(
        incoming_density=[0.5],
        outgoing_density=[0.5],
        waiting_times=[100.0, 100.0],
        last_measure=5.0,
    )
    reward = diff_waiting_time_reward(ts)
    assert reward > 0
    assert reward == pytest.approx(3.0)
    # Verify state update
    assert ts.last_measure == pytest.approx(2.0)


def test_diff_waiting_time_negative_when_worsening() -> None:
    """diff-waiting-time should be negative when waiting time increases."""
    # last_measure=1.0, current waiting = 500/100 = 5.0 → reward = 1.0 - 5.0 = -4.0
    ts = MockTrafficSignal(
        incoming_density=[0.5],
        outgoing_density=[0.5],
        waiting_times=[300.0, 200.0],
        last_measure=1.0,
    )
    reward = diff_waiting_time_reward(ts)
    assert reward < 0
    assert reward == pytest.approx(-4.0)
