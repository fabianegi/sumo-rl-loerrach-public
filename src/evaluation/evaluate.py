"""Main evaluation script: loads trained agents, runs N episodes, computes KPIs."""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd

from src.config.settings import CONFIG
from src.environment.env_factory import make_eval_env
from src.evaluation.metrics import compute_kpis, parse_tripinfo
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def evaluate_agent(
    model_path: Path,
    net_file: Path,
    route_file: Path,
    n_episodes: int = 30,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate a trained agent over multiple episodes.

    Args:
        model_path: Path to saved SB3 model (.zip).
        net_file: SUMO network file.
        route_file: SUMO route file.
        n_episodes: Number of evaluation episodes.
        seeds: List of seeds for each episode.

    Returns:
        DataFrame with KPIs per episode (same schema as baseline).
    """
    # Detect algorithm from filename convention: *_dqn_*.zip or *_ppo_*.zip
    model_name = model_path.stem.lower()
    if "ppo" in model_name:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        logger.info("Loaded PPO model from %s", model_path)
    else:
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
        logger.info("Loaded DQN model from %s", model_path)

    if seeds is None:
        seeds = list(range(n_episodes))

    results: list[dict] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="eval_tripinfo_"))

    for i, seed in enumerate(seeds[:n_episodes]):
        seed_everything(seed)
        logger.info("Eval episode %d/%d (seed=%d)", i + 1, n_episodes, seed)

        tripinfo_path = tmpdir / f"tripinfo_seed{seed}.xml"
        env = make_eval_env(
            net_file=net_file, route_file=route_file, seed=seed,
            tripinfo_output=tripinfo_path,
        )
        obs, _info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

        episode_df = pd.DataFrame(env.metrics)
        kpi_step = compute_kpis(episode_df)

        env.close()  # finalises tripinfo file
        ti = parse_tripinfo(tripinfo_path)

        results.append({
            "episode": i,
            "seed": seed,
            "avg_waiting_time": kpi_step.avg_waiting_time,
            "max_queue_length": kpi_step.max_queue_length,
            "total_throughput": ti["completed_vehicles"],
            "avg_speed": kpi_step.avg_speed,
            "tripinfo_mean_waiting_time": ti["mean_waiting_time"],
            "tripinfo_mean_duration": ti["mean_duration"],
            "tripinfo_mean_time_loss": ti["mean_time_loss"],
        })

        logger.info(
            "  -> step_wait=%.2f, queue=%d, throughput=%d, speed=%.1f m/s | "
            "tripinfo_wait=%.1fs, time_loss=%.1fs",
            kpi_step.avg_waiting_time, kpi_step.max_queue_length,
            ti["completed_vehicles"], kpi_step.avg_speed,
            ti["mean_waiting_time"], ti["mean_time_loss"],
        )

    return pd.DataFrame(results)


def main() -> None:
    """Entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent vs baseline")
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to saved SB3 model (.zip)",
    )
    parser.add_argument(
        "--net-file", type=Path, default=CONFIG.paths.net_file,
        help="SUMO network file",
    )
    parser.add_argument(
        "--route-file", type=Path, default=CONFIG.paths.route_file,
        help="SUMO route file",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=CONFIG.training.n_eval_episodes,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Output CSV path (default: results/csv/eval_<model>.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    results = evaluate_agent(
        model_path=args.model,
        net_file=args.net_file,
        route_file=args.route_file,
        n_episodes=args.n_episodes,
    )

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = CONFIG.paths.csv_dir / f"eval_{args.model.stem}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    logger.info("Results saved to %s", output_csv)


if __name__ == "__main__":
    main()
