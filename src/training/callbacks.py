"""Custom SB3 callbacks for training monitoring."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import gymnasium as gym
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class MetricsCacheWrapper(gym.Wrapper):
    """Caches sumo-rl episode metrics before reset() clears them.

    sumo-rl stores per-step metrics in env.metrics and clears them on
    reset(). This wrapper saves the completed episode's data so
    TrafficMetricsCallback can read it after DummyVecEnv auto-resets.

    Must be placed INSIDE the Monitor wrapper (closer to the raw env):
        env = make_env(...)
        env = MetricsCacheWrapper(env)
        env = Monitor(env)
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.last_episode_metrics: list[dict] = []

    def reset(self, **kwargs):
        """Cache metrics from completed episode, then delegate reset."""
        if hasattr(self.env, "metrics") and self.env.metrics:
            self.last_episode_metrics = list(self.env.metrics)
        return self.env.reset(**kwargs)


class TrafficMetricsCallback(BaseCallback):
    """Logs traffic KPIs per episode to CSV and TensorBoard.

    Writes ``log_dir/traffic_metrics.csv`` with columns:
    episode, timestep, reward_sum, avg_waiting_time, max_queue, avg_speed.

    Traffic KPIs require a MetricsCacheWrapper in the env chain.
    Without it, only episode reward is logged.
    """

    def __init__(self, log_dir: Path, verbose: int = 0) -> None:
        """Initialize TrafficMetricsCallback.

        Args:
            log_dir: Directory for metric logs.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self._csv_path: Path | None = None
        self._csv_writer = None
        self._csv_file = None
        self._episode_count = 0
        self._episode_reward = 0.0
        self._metrics_cache: MetricsCacheWrapper | None = None

    def _on_training_start(self) -> None:
        """Open CSV file and locate MetricsCacheWrapper in env chain."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / "traffic_metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="")  # noqa: SIM115
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "episode", "timestep", "reward_sum",
            "avg_waiting_time", "max_queue", "avg_speed",
        ])

        self._metrics_cache = self._find_metrics_cache()
        if self._metrics_cache is None:
            logger.warning(
                "MetricsCacheWrapper not found - traffic KPIs will not be logged. "
                "Wrap env with MetricsCacheWrapper before Monitor."
            )

    def _find_metrics_cache(self) -> MetricsCacheWrapper | None:
        """Walk the env wrapper chain to find MetricsCacheWrapper."""
        try:
            env = self.training_env.envs[0]
            while env is not None:
                if isinstance(env, MetricsCacheWrapper):
                    return env
                env = getattr(env, "env", None)
        except (AttributeError, IndexError):
            pass
        return None

    def _on_step(self) -> bool:
        """Accumulate reward; on episode end, extract metrics and log.

        Returns:
            Whether training should continue.
        """
        self._episode_reward += float(self.locals["rewards"][0])

        if self.locals["dones"][0]:
            self._episode_count += 1

            # Extract traffic metrics from cached episode data
            avg_wt = max_q = avg_spd = None
            if self._metrics_cache and self._metrics_cache.last_episode_metrics:
                try:
                    df = pd.DataFrame(self._metrics_cache.last_episode_metrics)
                    if "system_mean_waiting_time" in df.columns:
                        avg_wt = float(df["system_mean_waiting_time"].mean())
                    if "system_total_stopped" in df.columns:
                        max_q = int(df["system_total_stopped"].max())
                    if "system_mean_speed" in df.columns:
                        avg_spd = float(df["system_mean_speed"].mean())
                except (KeyError, ValueError) as exc:
                    logger.debug("Could not extract traffic metrics: %s", exc)

            # CSV row
            if self._csv_writer:
                self._csv_writer.writerow([
                    self._episode_count,
                    self.num_timesteps,
                    round(self._episode_reward, 2),
                    round(avg_wt, 2) if avg_wt is not None else "",
                    max_q if max_q is not None else "",
                    round(avg_spd, 2) if avg_spd is not None else "",
                ])
                self._csv_file.flush()

            # TensorBoard
            self.logger.record("traffic/episode_reward", self._episode_reward)
            if avg_wt is not None:
                self.logger.record("traffic/avg_waiting_time", avg_wt)
            if max_q is not None:
                self.logger.record("traffic/max_queue", float(max_q))
            if avg_spd is not None:
                self.logger.record("traffic/avg_speed", avg_spd)

            if self.verbose >= 1:
                wt_str = f"{avg_wt:.1f}s" if avg_wt is not None else "N/A"
                logger.info(
                    "Episode %d (step %d): reward=%.1f, avg_wait=%s, max_queue=%s",
                    self._episode_count,
                    self.num_timesteps,
                    self._episode_reward,
                    wt_str,
                    max_q if max_q is not None else "N/A",
                )

            self._episode_reward = 0.0

        return True

    def _on_training_end(self) -> None:
        """Close CSV file."""
        if self._csv_file:
            self._csv_file.close()
            logger.info(
                "Traffic metrics saved to %s (%d episodes)",
                self._csv_path,
                self._episode_count,
            )


class ProjectCheckpointCallback(BaseCallback):
    """Saves model checkpoints at regular intervals.

    Named ProjectCheckpointCallback to avoid conflict with SB3's
    built-in CheckpointCallback.
    """

    def __init__(
        self,
        save_dir: Path,
        save_freq: int = 50_000,
        name_prefix: str = "checkpoint",
        verbose: int = 0,
    ) -> None:
        """Initialize ProjectCheckpointCallback.

        Args:
            save_dir: Directory to save checkpoints.
            save_freq: Save every N steps.
            name_prefix: Prefix for checkpoint filenames.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.name_prefix = name_prefix

    def _on_training_start(self) -> None:
        """Create save directory."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Save checkpoint every save_freq steps.

        Returns:
            Whether training should continue.
        """
        if self.n_calls % self.save_freq == 0:
            path = self.save_dir / f"{self.name_prefix}_{self.num_timesteps}"
            self.model.save(str(path))
            logger.info(
                "Checkpoint saved: %s.zip (%d steps)", path, self.num_timesteps
            )
        return True
