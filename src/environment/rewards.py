"""Custom reward functions for sumo-rl traffic signal control."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sumo_rl import TrafficSignal


def pressure_reward(traffic_signal: TrafficSignal) -> float:
    """PressLight-inspired pressure reward.

    Reward = -|sum(incoming_density) - sum(outgoing_density)|
    Minimizes pressure differential for throughput maximization.

    Args:
        traffic_signal: sumo-rl TrafficSignal object.

    Returns:
        Negative absolute pressure value.

    Reference:
        Wei et al. (2019). PressLight. KDD.
    """
    incoming = sum(traffic_signal.get_lanes_density())
    outgoing = sum(traffic_signal.get_out_lanes_density())
    return -abs(incoming - outgoing)


def diff_waiting_time_reward(traffic_signal: TrafficSignal) -> float:
    """Default sumo-rl reward: change in cumulative waiting time.

    Reward = W_{t-1} - W_t (positive when waiting time decreases).

    Args:
        traffic_signal: sumo-rl TrafficSignal object.

    Returns:
        Difference in cumulative waiting time.
    """
    current_wt = sum(traffic_signal.get_accumulated_waiting_time_per_lane()) / 100.0
    reward = traffic_signal.last_measure - current_wt
    traffic_signal.last_measure = current_wt
    return reward
