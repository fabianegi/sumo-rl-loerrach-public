"""Evaluate training checkpoints to show agent improvement over time.

Finds checkpoint files, runs quick evaluations, and generates a
multi-panel snapshot figure.

Usage:
    python scripts/evaluate_checkpoints.py --checkpoint-dir models/checkpoints/ --prefix dqn_diff-waiting-time
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

from src.config.settings import CONFIG
from src.environment.env_factory import make_eval_env
from src.utils.plotting import plot_training_snapshots

logger = logging.getLogger(__name__)


def detect_algorithm(name: str) -> type:
    """Detect algorithm from checkpoint name."""
    if "ppo" in name.lower():
        return PPO
    return DQN


def find_checkpoints(checkpoint_dir: Path, prefix: str) -> list[tuple[int, Path]]:
    """Find checkpoint files matching prefix, sorted by timestep.

    Returns:
        List of (timestep, path) tuples sorted by timestep.
    """
    escaped_prefix = re.escape(prefix)
    pattern = re.compile(rf"{escaped_prefix}_(\d+)\.zip$")
    results: list[tuple[int, Path]] = []

    for f in checkpoint_dir.glob(f"{prefix}_*.zip"):
        m = pattern.match(f.name)
        if m:
            results.append((int(m.group(1)), f))

    return sorted(results)


def quick_eval(
    model_path: Path,
    n_episodes: int = 5,
    seed_start: int = 100,
) -> dict:
    """Run quick evaluation of a checkpoint.

    Returns:
        Dict with avg_waiting_time, std_waiting_time, avg_reward.
    """
    algo_cls = detect_algorithm(model_path.stem)
    waiting_times: list[float] = []
    rewards: list[float] = []

    for i in range(n_episodes):
        seed = seed_start + i
        env = make_eval_env(
            net_file=CONFIG.paths.net_file,
            route_file=CONFIG.paths.route_file,
            reward_fn="diff-waiting-time",
            seed=seed,
        )
        model = algo_cls.load(str(model_path), env=env)
        obs, _info = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated

        # Extract waiting time from final metrics
        sumo_env = env.unwrapped
        if hasattr(sumo_env, "metrics") and sumo_env.metrics:
            df = pd.DataFrame(sumo_env.metrics)
            if "system_mean_waiting_time" in df.columns:
                waiting_times.append(float(df["system_mean_waiting_time"].mean()))

        rewards.append(ep_reward)
        env.close()

    result = {"avg_reward": float(np.mean(rewards))}
    if waiting_times:
        result["avg_waiting_time"] = float(np.mean(waiting_times))
        result["std_waiting_time"] = float(np.std(waiting_times))
    return result


def main() -> None:
    """Entry point for checkpoint evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate training checkpoints")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Directory with checkpoint .zip files")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Checkpoint name prefix (e.g. dqn_diff-waiting-time)")
    parser.add_argument("--n-episodes", type=int, default=5, help="Episodes per checkpoint")
    parser.add_argument("--baseline-wt", type=float, default=None,
                        help="Baseline avg waiting time for reference line")
    parser.add_argument("--output", type=Path, default=None, help="Output plot path")
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.checkpoint_dir, args.prefix)
    if not checkpoints:
        print(f"No checkpoints found with prefix '{args.prefix}' in {args.checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        print(f"  {step:>10d} steps: {path.name}")

    checkpoint_metrics: dict[int, dict] = {}
    for step, path in checkpoints:
        print(f"\nEvaluating {path.name} ({args.n_episodes} episodes)...")
        metrics = quick_eval(path, n_episodes=args.n_episodes)
        checkpoint_metrics[step] = metrics

        if "avg_waiting_time" in metrics:
            print(f"  Ø Wartezeit: {metrics['avg_waiting_time']:.1f}s "
                  f"(±{metrics.get('std_waiting_time', 0):.1f}s)")
        print(f"  Ø Reward: {metrics['avg_reward']:.1f}")

    # Generate plot
    output = args.output or Path("results") / "plots" / f"training_snapshots_{args.prefix}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_training_snapshots(checkpoint_metrics, output, baseline_wt=args.baseline_wt)
    print(f"\nSnapshot plot saved: {output}")


if __name__ == "__main__":
    main()
