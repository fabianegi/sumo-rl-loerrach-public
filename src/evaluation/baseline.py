"""Fixed-time baseline simulation and KPI measurement."""
from __future__ import annotations

import argparse
import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from sumo_rl import SumoEnvironment

from src.config.settings import CONFIG
from src.evaluation.metrics import KPIResult, compute_kpis, parse_tripinfo
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def run_fixed_time_baseline(
    net_file: Path,
    route_file: Path,
    cycle_time: int = 90,
    n_episodes: int = 30,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run fixed-time traffic signal baseline.

    Uses sumo-rl with fixed_ts=True so the TLS follows its pre-programmed
    cycle from the .net.xml (no RL intervention).

    Args:
        net_file: SUMO network file.
        route_file: SUMO route file.
        cycle_time: Total cycle time in seconds (informational, timing from .net.xml).
        n_episodes: Number of evaluation runs.
        seeds: Random seeds for each run.

    Returns:
        DataFrame with columns: episode, seed, avg_waiting_time,
        max_queue_length, total_throughput, avg_speed.
    """
    if seeds is None:
        seeds = list(range(n_episodes))

    results: list[dict] = []

    tmpdir = Path(tempfile.mkdtemp(prefix="baseline_tripinfo_"))

    for i, seed in enumerate(seeds[:n_episodes]):
        seed_everything(seed)
        logger.info("Baseline episode %d/%d (seed=%d)", i + 1, n_episodes, seed)

        tripinfo_path = tmpdir / f"tripinfo_seed{seed}.xml"
        env = SumoEnvironment(
            net_file=str(net_file),
            route_file=str(route_file),
            use_gui=False,
            num_seconds=3600,
            delta_time=5,
            yellow_time=3,
            min_green=10,
            reward_fn="diff-waiting-time",
            sumo_seed=seed,
            single_agent=True,
            fixed_ts=True,
            time_to_teleport=-1,
            additional_sumo_cmd=f"--tripinfo-output {tripinfo_path}",
        )

        _obs, _info = env.reset()
        done = False
        while not done:
            _obs, _reward, terminated, truncated, _info = env.step(None)
            done = terminated or truncated

        # sumo-rl step metrics still feed per-step KPIs (waiting/queue/speed).
        episode_df = pd.DataFrame(env.metrics)
        kpi_step = compute_kpis(episode_df)

        env.close()  # finalises tripinfo file

        # Audit-grade throughput + per-vehicle waiting from tripinfo (Audit F1).
        ti = parse_tripinfo(tripinfo_path)

        kpi = KPIResult(
            avg_waiting_time=kpi_step.avg_waiting_time,
            max_queue_length=kpi_step.max_queue_length,
            total_throughput=ti["completed_vehicles"],
            avg_speed=kpi_step.avg_speed,
        )

        results.append({
            "episode": i,
            "seed": seed,
            "avg_waiting_time": kpi.avg_waiting_time,
            "max_queue_length": kpi.max_queue_length,
            "total_throughput": kpi.total_throughput,
            "avg_speed": kpi.avg_speed,
            # Audit-grade per-vehicle metrics (tripinfo definitions)
            "tripinfo_mean_waiting_time": ti["mean_waiting_time"],
            "tripinfo_mean_duration": ti["mean_duration"],
            "tripinfo_mean_time_loss": ti["mean_time_loss"],
        })

        logger.info(
            "  -> step_wait=%.2f, queue=%d, throughput=%d, speed=%.1f m/s | "
            "tripinfo_wait=%.1fs, time_loss=%.1fs",
            kpi.avg_waiting_time, kpi.max_queue_length, kpi.total_throughput,
            kpi.avg_speed, ti["mean_waiting_time"], ti["mean_time_loss"],
        )

    return pd.DataFrame(results)


def measure_kpis(sumo_output_file: Path) -> dict[str, float]:
    """Extract KPIs from a SUMO tripinfo XML output file.

    Args:
        sumo_output_file: Path to SUMO tripinfo output XML.

    Returns:
        Dict with keys: avg_waiting_time, avg_time_loss, throughput, avg_duration.
    """
    tree = ET.parse(sumo_output_file)
    root = tree.getroot()

    waiting_times: list[float] = []
    time_losses: list[float] = []
    durations: list[float] = []

    for trip in root.iter("tripinfo"):
        waiting_times.append(float(trip.get("waitingTime", 0)))
        time_losses.append(float(trip.get("timeLoss", 0)))
        durations.append(float(trip.get("duration", 0)))

    n = len(waiting_times)
    if n == 0:
        return {"avg_waiting_time": 0.0, "avg_time_loss": 0.0, "throughput": 0, "avg_duration": 0.0}

    return {
        "avg_waiting_time": sum(waiting_times) / n,
        "avg_time_loss": sum(time_losses) / n,
        "throughput": n,
        "avg_duration": sum(durations) / n,
    }


def main() -> None:
    """CLI entry point: run the fixed-time baseline and save KPIs to CSV."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run fixed-time signal baseline")
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
        help="Number of baseline runs (seeds 0..N-1)",
    )
    parser.add_argument("--cycle-time", type=int, default=90, help="Cycle time (s)")
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Output CSV path (default: results/csv/baseline_fixed_time.csv)",
    )
    args = parser.parse_args()

    results = run_fixed_time_baseline(
        net_file=args.net_file,
        route_file=args.route_file,
        cycle_time=args.cycle_time,
        n_episodes=args.n_episodes,
    )

    output_csv = args.output_csv or CONFIG.paths.csv_dir / "baseline_fixed_time.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    logger.info("Baseline results saved to %s", output_csv)


if __name__ == "__main__":
    main()
