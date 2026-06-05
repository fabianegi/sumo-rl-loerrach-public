"""Standalone DQN training script. Run: python src/training/train_dqn.py"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.config.settings import CONFIG
from src.environment.env_factory import make_env
from src.training.callbacks import (
    MetricsCacheWrapper,
    ProjectCheckpointCallback,
    TrafficMetricsCallback,
)
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DQN training."""
    parser = argparse.ArgumentParser(
        description="Train DQN agent for traffic signal control",
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gui", action="store_true", help="Launch SUMO GUI")
    parser.add_argument(
        "--reward",
        type=str,
        default="diff-waiting-time",
        choices=["diff-waiting-time", "pressure"],
    )
    parser.add_argument(
        "--net-file", type=Path, default=None,
        help="SUMO network file (default: from settings)",
    )
    parser.add_argument(
        "--route-file", type=Path, default=None,
        help="SUMO route file (default: from settings)",
    )
    return parser.parse_args()


def train_dqn(
    total_timesteps: int,
    seed: int,
    learning_rate: float | None,
    use_gui: bool,
    reward_fn: str,
    net_file: Path | None = None,
    route_file: Path | None = None,
) -> Path:
    """Train a DQN agent and save the model.

    Args:
        total_timesteps: Number of training steps.
        seed: Random seed for reproducibility.
        learning_rate: DQN learning rate (None = use CONFIG default).
        use_gui: Whether to show SUMO GUI during training.
        reward_fn: Which reward function to use.
        net_file: SUMO network file (None = CONFIG default).
        route_file: SUMO route file (None = CONFIG default).

    Returns:
        Path to saved model checkpoint.
    """
    seed_everything(seed)

    lr = learning_rate if learning_rate is not None else CONFIG.dqn.learning_rate
    nf = net_file or CONFIG.paths.net_file
    rf = route_file or CONFIG.paths.route_file

    logger.info("Creating SUMO environment: net=%s, route=%s", nf.name, rf.name)
    env = make_env(
        net_file=nf,
        route_file=rf,
        use_gui=use_gui,
        reward_fn=reward_fn,
        delta_time=CONFIG.sumo.delta_time,
        min_green=CONFIG.sumo.min_green,
        yellow_time=CONFIG.sumo.yellow_time,
        num_seconds=CONFIG.sumo.num_seconds,
        seed=seed,
    )
    env = MetricsCacheWrapper(env)
    env = Monitor(env)

    logger.info(
        "DQN: lr=%s, buffer=%d, batch=%d, gamma=%.3f, timesteps=%d, reward=%s",
        lr, CONFIG.dqn.buffer_size, CONFIG.dqn.batch_size,
        CONFIG.dqn.gamma, total_timesteps, reward_fn,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=lr,
        buffer_size=CONFIG.dqn.buffer_size,
        learning_starts=CONFIG.dqn.learning_starts,
        batch_size=CONFIG.dqn.batch_size,
        gamma=CONFIG.dqn.gamma,
        exploration_fraction=CONFIG.dqn.exploration_fraction,
        exploration_final_eps=CONFIG.dqn.exploration_final_eps,
        target_update_interval=CONFIG.dqn.target_update_interval,
        train_freq=CONFIG.dqn.train_freq,
        verbose=1,
        tensorboard_log=str(CONFIG.training.tensorboard_log),
        seed=seed,
        device=CONFIG.training.device,
    )

    # Model name encodes reward + timesteps + seed for traceability
    model_name = f"dqn_{reward_fn}_{total_timesteps}steps_seed{seed}"
    log_dir = CONFIG.training.log_dir / model_name

    callbacks = [
        TrafficMetricsCallback(log_dir=log_dir, verbose=1),
        ProjectCheckpointCallback(
            save_dir=CONFIG.paths.models_dir,
            save_freq=CONFIG.training.checkpoint_freq,
            name_prefix=f"dqn_{reward_fn}",
        ),
    ]

    logger.info("Starting DQN training: %d timesteps", total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=model_name,
    )

    # Save final model
    model_path = CONFIG.paths.models_dir / f"{model_name}.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    logger.info("Final model saved: %s", model_path)

    env.close()
    return model_path


def main() -> None:
    """Entry point for DQN training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    model_path = train_dqn(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        use_gui=args.gui,
        reward_fn=args.reward,
        net_file=args.net_file,
        route_file=args.route_file,
    )
    print(f"Training complete. Model saved: {model_path}")


if __name__ == "__main__":
    main()
