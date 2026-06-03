"""Consistent plot generation for all project figures."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Consistent style
plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE = (10, 6)
DPI = 150
COLORS = {"baseline": "#888888", "dqn": "#2196F3", "ppo": "#FF9800"}

# Phase colors for signal plan visualization
PHASE_COLORS = {
    "N-S green": "#4CAF50",
    "E-W green": "#2196F3",
    "yellow": "#FFC107",
}


def plot_learning_curve(
    log_file: Path,
    output_path: Path,
    title: str = "Lernkurve",
    window: int = 50,
) -> None:
    """Plot reward and waiting time over training episodes.

    Reads the CSV output of TrafficMetricsCallback. Primary y-axis shows
    episode reward with rolling average; secondary y-axis shows average
    waiting time.

    Args:
        log_file: Path to traffic_metrics.csv.
        output_path: Where to save the plot.
        title: Plot title.
        window: Rolling average window size.
    """
    df = pd.read_csv(log_file)

    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    # Episode reward (left y-axis)
    color_reward = "#2196F3"
    ax1.plot(
        df["episode"], df["reward_sum"],
        alpha=0.25, color=color_reward, linewidth=0.5,
    )
    if len(df) >= window:
        rolling = df["reward_sum"].rolling(window=window, min_periods=1).mean()
        ax1.plot(
            df["episode"], rolling,
            color=color_reward, linewidth=2,
            label=f"Reward (Ø {window} Episoden)",
        )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episodenbelohnung", color=color_reward)
    ax1.tick_params(axis="y", labelcolor=color_reward)

    # Average waiting time (right y-axis)
    if "avg_waiting_time" in df.columns and df["avg_waiting_time"].notna().any():
        ax2 = ax1.twinx()
        color_wt = "#F44336"
        wt = pd.to_numeric(df["avg_waiting_time"], errors="coerce")
        ax2.plot(
            df["episode"], wt,
            alpha=0.25, color=color_wt, linewidth=0.5,
        )
        if len(df) >= window:
            rolling_wt = wt.rolling(window=window, min_periods=1).mean()
            ax2.plot(
                df["episode"], rolling_wt,
                color=color_wt, linewidth=2,
                label=f"Wartezeit (Ø {window} Ep.)",
            )
        ax2.set_ylabel("Ø Wartezeit [s]", color=color_wt)
        ax2.tick_params(axis="y", labelcolor=color_wt)
        ax2.legend(loc="upper right")

    ax1.legend(loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Learning curve saved: %s", output_path)


def plot_kpi_comparison(
    baseline_kpis: pd.DataFrame,
    agent_kpis: dict[str, pd.DataFrame],
    kpi_name: str,
    output_path: Path,
) -> None:
    """Boxplot comparing a KPI across baseline and RL agents.

    Args:
        baseline_kpis: DataFrame with baseline KPI values (one row per seed).
        agent_kpis: Dict mapping agent name (e.g. "DQN") to KPI DataFrame.
        kpi_name: Column name to compare (e.g. "avg_waiting_time").
        output_path: Where to save the plot.
    """
    # Build long-form DataFrame for seaborn
    records: list[dict] = []
    for _, row in baseline_kpis.iterrows():
        records.append({"Variante": "Fixed-Time", "value": row[kpi_name]})
    for name, df in agent_kpis.items():
        for _, row in df.iterrows():
            records.append({"Variante": name, "value": row[kpi_name]})

    plot_df = pd.DataFrame(records)

    # German KPI labels
    kpi_labels = {
        "avg_waiting_time": "Ø Wartezeit [s]",
        "max_queue": "Max. Rückstaulänge [Fzg]",
        "throughput": "Durchsatz [Fzg/h]",
        "avg_speed": "Ø Geschwindigkeit [m/s]",
    }
    ylabel = kpi_labels.get(kpi_name, kpi_name)

    # Color palette
    palette: dict[str, str] = {"Fixed-Time": COLORS["baseline"]}
    for name in agent_kpis:
        key = name.lower().split("_")[0]
        palette[name] = COLORS.get(key, "#9C27B0")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(
        data=plot_df, x="Variante", y="value",
        palette=palette, ax=ax, width=0.5,
    )
    sns.stripplot(
        data=plot_df, x="Variante", y="value",
        color="black", alpha=0.3, size=4, ax=ax,
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(f"Vergleich: {ylabel}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("KPI comparison saved: %s", output_path)


def plot_queue_timeseries(
    baseline_queues: NDArray[np.float64],
    agent_queues: NDArray[np.float64],
    output_path: Path,
    agent_name: str = "DQN",
) -> None:
    """Time series of queue length: baseline vs agent over one episode.

    Args:
        baseline_queues: Queue length array over simulation steps.
        agent_queues: Queue length array over simulation steps.
        output_path: Where to save the plot.
        agent_name: Name of the RL agent for legend.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Time axis in seconds (delta_time=5s per RL step)
    time_bl = np.arange(len(baseline_queues)) * 5
    time_ag = np.arange(len(agent_queues)) * 5

    ax.plot(time_bl, baseline_queues, color=COLORS["baseline"], linewidth=1.5,
            label="Fixed-Time", alpha=0.8)
    ax.plot(time_ag, agent_queues, color=COLORS.get(agent_name.lower(), "#2196F3"),
            linewidth=1.5, label=agent_name, alpha=0.8)

    # Shaded improvement area (where agent < baseline)
    min_len = min(len(baseline_queues), len(agent_queues))
    bl_short = baseline_queues[:min_len]
    ag_short = agent_queues[:min_len]
    time_short = np.arange(min_len) * 5
    ax.fill_between(
        time_short, ag_short, bl_short,
        where=ag_short < bl_short,
        alpha=0.15, color="#4CAF50", label="Verbesserung",
    )

    ax.set_xlabel("Simulationszeit [s]")
    ax.set_ylabel("Rückstaulänge [Fzg]")
    ax.set_title(f"Rückstau: Fixed-Time vs. {agent_name}")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Queue timeseries saved: %s", output_path)


def plot_demand_profile(
    hourly_counts: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Plot 24-hour traffic demand profile as bar chart.

    Highlights the rush-hour window (17:00-18:00) used for simulation.

    Args:
        hourly_counts: 24 values (vehicles per hour).
        output_path: Where to save the plot.
    """
    hours = np.arange(24)
    colors = ["#F44336" if h == 17 else "#2196F3" for h in hours]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(hours, hourly_counts, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate rush hour bar
    rush_idx = 17
    ax.annotate(
        f"Rush Hour\n{int(hourly_counts[rush_idx])} Fzg/h",
        xy=(rush_idx, hourly_counts[rush_idx]),
        xytext=(rush_idx - 4, hourly_counts[rush_idx] * 1.15),
        arrowprops={"arrowstyle": "->", "color": "#F44336"},
        fontsize=10, color="#F44336", fontweight="bold",
    )

    ax.set_xlabel("Uhrzeit [h]")
    ax.set_ylabel("Verkehrsstärke [Fzg/h]")
    ax.set_title("Tagesganglinie (Basler Straße / Lörrach Mitte)")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours], rotation=45)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Demand profile saved: %s", output_path)


def plot_signal_plan(
    agent_phases: list[tuple[float, int]],
    fixed_phases: list[tuple[float, int]],
    output_path: Path,
) -> None:
    """Gantt-chart-style signal plan comparison: fixed-time vs RL agent.

    Each entry is (start_time, phase_index) where phases are:
    0 = N-S green, 1 = N-S yellow, 2 = E-W green, 3 = E-W yellow.

    Args:
        agent_phases: List of (sim_time, phase_id) for the RL agent.
        fixed_phases: List of (sim_time, phase_id) for fixed-time control.
        output_path: Where to save the plot.
    """
    phase_map = {0: "N-S green", 1: "yellow", 2: "E-W green", 3: "yellow"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)

    for ax, phases, label in [
        (axes[0], fixed_phases, "Fixed-Time (90s Zyklus)"),
        (axes[1], agent_phases, "RL-Agent"),
    ]:
        for i in range(len(phases)):
            start = phases[i][0]
            end = phases[i + 1][0] if i + 1 < len(phases) else 3600.0
            phase_name = phase_map.get(phases[i][1], "yellow")
            color = PHASE_COLORS.get(phase_name, "#999999")
            ax.barh(0, end - start, left=start, height=0.6, color=color,
                    edgecolor="white", linewidth=0.3)
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=11)
        ax.set_xlim(0, 3600)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[1].set_xlabel("Simulationszeit [s]")
    fig.suptitle("Signalplan: Fixed-Time vs. RL-Agent", fontsize=13, y=1.02)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PHASE_COLORS["N-S green"], label="N-S Grün (Basler Str.)"),
        Patch(facecolor=PHASE_COLORS["E-W green"], label="E-W Grün (Querstraße)"),
        Patch(facecolor=PHASE_COLORS["yellow"], label="Gelb"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=9, ncol=3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Signal plan saved: %s", output_path)


def plot_training_snapshots(
    checkpoint_metrics: dict[int, dict],
    output_path: Path,
    baseline_wt: float | None = None,
) -> None:
    """Multi-panel figure showing agent performance at training checkpoints.

    Args:
        checkpoint_metrics: Dict mapping timestep → {"avg_waiting_time": mean,
            "std_waiting_time": std, "avg_reward": mean}.
        output_path: Where to save the plot.
        baseline_wt: Baseline average waiting time (drawn as horizontal line).
    """
    steps = sorted(checkpoint_metrics.keys())
    n = len(steps)
    if n == 0:
        logger.warning("No checkpoint metrics to plot")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, step in zip(axes, steps, strict=False):
        m = checkpoint_metrics[step]
        avg_wt = m.get("avg_waiting_time", 0)
        std_wt = m.get("std_waiting_time", 0)

        ax.bar(
            ["Agent"], [avg_wt], yerr=[std_wt],
            color="#2196F3", capsize=8, width=0.4,
        )

        if baseline_wt is not None:
            ax.axhline(y=baseline_wt, color=COLORS["baseline"],
                       linestyle="--", linewidth=2, label="Fixed-Time")
            ax.legend(fontsize=9)

        # Format step label
        if step >= 1_000_000:
            step_label = f"{step / 1_000_000:.1f}M"
        elif step >= 1_000:
            step_label = f"{step // 1_000}k"
        else:
            step_label = str(step)

        ax.set_title(f"{step_label} Steps", fontsize=11)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Ø Wartezeit [s]")
    fig.suptitle("Agenten-Verbesserung über das Training", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training snapshots saved: %s", output_path)
