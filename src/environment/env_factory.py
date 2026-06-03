"""Factory functions for creating configured sumo-rl environments."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
from sumo_rl import SumoEnvironment

from src.environment.rewards import diff_waiting_time_reward, pressure_reward

logger = logging.getLogger(__name__)

# Map string names to reward callables
REWARD_MAP: dict[str, str | Callable] = {
    "diff-waiting-time": diff_waiting_time_reward,
    "pressure": pressure_reward,
}


def make_env(
    net_file: Path,
    route_file: Path,
    use_gui: bool = False,
    reward_fn: str = "diff-waiting-time",
    delta_time: int = 5,
    min_green: int = 10,
    yellow_time: int = 3,
    num_seconds: int = 3600,
    seed: int = 42,
    tripinfo_output: Path | None = None,
) -> gym.Env:
    """Create a configured sumo-rl Gymnasium environment.

    Args:
        net_file: Path to SUMO .net.xml network file.
        route_file: Path to SUMO .rou.xml route file.
        use_gui: Whether to launch sumo-gui.
        reward_fn: Reward function name ("diff-waiting-time" or "pressure").
        delta_time: Seconds between RL decisions.
        min_green: Minimum green time in seconds.
        yellow_time: Yellow phase duration in seconds.
        num_seconds: Total simulation seconds per episode.
        seed: Random seed for SUMO.
        tripinfo_output: If set, SUMO writes a tripinfo XML to this path -
            used by audit-grade throughput counting (one row per completed
            vehicle). Replaces the per-step ``getArrivedNumber()``
            accumulation that systematically under-counts under
            ``delta_time > 1`` (Audit B4, 2026-05-19).

    Returns:
        Configured Gymnasium environment.
    """
    reward = REWARD_MAP.get(reward_fn, reward_fn)
    logger.info("Creating env: reward=%s, gui=%s, seed=%d", reward_fn, use_gui, seed)

    extra_cmd = None
    if tripinfo_output is not None:
        # SUMO will overwrite/create the file on episode end.
        tripinfo_output.parent.mkdir(parents=True, exist_ok=True)
        extra_cmd = f"--tripinfo-output {tripinfo_output}"

    env = SumoEnvironment(
        net_file=str(net_file),
        route_file=str(route_file),
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        min_green=min_green,
        reward_fn=reward,
        sumo_seed=seed,
        single_agent=True,
        time_to_teleport=-1,
        additional_sumo_cmd=extra_cmd,
    )
    return env


def make_eval_env(
    net_file: Path,
    route_file: Path,
    seed: int = 0,
    tripinfo_output: Path | None = None,
) -> gym.Env:
    """Create environment configured for evaluation (no exploration, deterministic).

    Args:
        net_file: Path to SUMO .net.xml network file.
        route_file: Path to SUMO .rou.xml route file.
        seed: Random seed.
        tripinfo_output: Forwarded to ``make_env``; see there.

    Returns:
        Evaluation-configured Gymnasium environment.
    """
    return make_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        reward_fn="diff-waiting-time",
        seed=seed,
        tripinfo_output=tripinfo_output,
    )
