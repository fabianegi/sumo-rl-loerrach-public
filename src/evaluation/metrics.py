"""KPI computation for traffic signal evaluation."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class KPIResult:
    """Container for evaluation KPIs."""

    avg_waiting_time: float
    max_queue_length: float
    total_throughput: int
    avg_speed: float
    total_co2: float | None = None


def compute_kpis(episode_data: pd.DataFrame) -> KPIResult:
    """Compute KPIs from sumo-rl step metrics.

    Args:
        episode_data: DataFrame from env.metrics with columns like
            system_mean_waiting_time, system_total_stopped,
            system_mean_speed, and optionally arrived_vehicles.

    Returns:
        KPIResult with computed metrics.
    """
    avg_wt = episode_data["system_mean_waiting_time"].mean()
    max_queue = int(episode_data["system_total_stopped"].max())
    avg_speed = episode_data["system_mean_speed"].mean()

    # Throughput: sum of arrived vehicles per step (added by caller)
    if "arrived_vehicles" in episode_data.columns:
        throughput = int(episode_data["arrived_vehicles"].sum())
    else:
        throughput = 0

    return KPIResult(
        avg_waiting_time=avg_wt,
        max_queue_length=max_queue,
        total_throughput=throughput,
        avg_speed=avg_speed,
    )


def parse_tripinfo(path: Path) -> dict[str, float]:
    """Parse a SUMO ``--tripinfo-output`` XML and return audit-grade KPIs.

    One ``<tripinfo>`` element is written per *completed* vehicle, so the
    element count is the true throughput - independent of how often
    ``traci.simulation.getArrivedNumber()`` is sampled (Audit B4).

    Args:
        path: Path to the tripinfo XML produced by SUMO.

    Returns:
        Dict with:
            ``completed_vehicles``  - int, true throughput
            ``mean_waiting_time``   - float, seconds per completed vehicle
                (tripinfo ``waitingTime`` attribute, defined as time below
                0.1 m/s - SUMO standard, comparable to HBS waiting time)
            ``mean_duration``       - float, seconds per completed vehicle
                (total trip time)
            ``mean_time_loss``      - float, seconds per completed vehicle
                (additional time vs. free-flow)
    """
    root = ET.parse(path).getroot()
    trips = root.findall("tripinfo")
    n = len(trips)
    if n == 0:
        return {
            "completed_vehicles": 0,
            "mean_waiting_time": 0.0,
            "mean_duration": 0.0,
            "mean_time_loss": 0.0,
        }

    def _avg(attr: str) -> float:
        return sum(float(t.get(attr, 0.0)) for t in trips) / n

    return {
        "completed_vehicles": n,
        "mean_waiting_time": _avg("waitingTime"),
        "mean_duration": _avg("duration"),
        "mean_time_loss": _avg("timeLoss"),
    }


def compute_summary_statistics(kpi_results: list[KPIResult]) -> pd.DataFrame:
    """Compute mean, std, min, max across multiple episodes.

    Args:
        kpi_results: List of KPIResult from multiple evaluation episodes.

    Returns:
        Summary DataFrame with metrics as rows and statistics as columns.
    """
    records = [asdict(k) for k in kpi_results]
    df = pd.DataFrame(records).drop(columns=["total_co2"], errors="ignore")
    summary = df.agg(["mean", "std", "min", "max"]).T
    summary.columns = ["mean", "std", "min", "max"]
    return summary
