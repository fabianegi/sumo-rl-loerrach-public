"""Generate all presentation/documentation figures from the OSM evaluation CSVs.

Reads the 30-seed baseline and DQN evaluation CSVs (same schema as produced by
``src/evaluation/baseline.py`` / ``src/evaluation/evaluate.py``) plus the
training metrics log, and writes the figure set used in the slides and the
documentation portfolio:

* KPI boxplots (throughput, time-loss, max queue, speed, per-step + per-trip
  waiting) with Mann-Whitney U significance star and rank-biserial effect size,
  labelled "DQN (OSM-Netz)" vs. "Festzeit-Baseline".
* A selection-bias footnote under the throughput plot (honest science: the
  baseline leaves ~124 vehicles stuck in the queue, so tripinfo means only
  cover completed trips).
* The gridlock-frequency bar chart (per-step waiting > 200 s ⇒ quasi-gridlock).
* The DQN learning curve over 4 166 episodes.

Usage:
    python scripts/generate_plots.py
    python scripts/generate_plots.py --gridlock-threshold 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.settings import CONFIG
from src.evaluation.statistical_tests import mann_whitney_u
from src.utils.plotting import plot_learning_curve

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150
C_BASELINE = "#888888"
C_DQN = "#2196F3"
C_GRIDLOCK_BL = "#E53935"  # red
C_GRIDLOCK_DQN = "#43A047"  # green

BASELINE_LABEL = "Festzeit-Baseline"
DQN_LABEL = "DQN (OSM-Netz)"

# Per-step waiting threshold (s) above which an episode counts as quasi-gridlock.
GRIDLOCK_THRESHOLD = 200.0


def _stars(p: float) -> str:
    """Significance stars for a p-value (n.s. if not significant)."""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _kpi_boxplot(
    baseline: np.ndarray,
    agent: np.ndarray,
    *,
    title: str,
    ylabel: str,
    lower_is_better: bool,
    output_path: Path,
    footnote: str | None = None,
) -> None:
    """Two-group boxplot with significance bracket and effect size r."""
    res = mann_whitney_u(baseline, agent)
    fig, ax = plt.subplots(figsize=(7.5, 6))

    box = ax.boxplot(
        [baseline, agent],
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.6},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    for patch, color in zip(box["boxes"], (C_BASELINE, C_DQN), strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    # Jittered raw seed points
    rng = np.random.default_rng(0)
    for i, data in enumerate((baseline, agent), start=1):
        x = rng.normal(i, 0.045, size=len(data))
        ax.scatter(x, data, color="black", alpha=0.35, s=14, zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{BASELINE_LABEL}\n(N=30)", f"{DQN_LABEL}\n(N=30)"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, pad=14)

    # Significance bracket
    ymax = max(baseline.max(), agent.max())
    ymin = min(baseline.min(), agent.min())
    span = ymax - ymin if ymax > ymin else 1.0
    bar_y = ymax + span * 0.08
    ax.plot(
        [1, 1, 2, 2],
        [bar_y, bar_y + span * 0.02, bar_y + span * 0.02, bar_y],
        color="black",
        linewidth=1.2,
    )
    rel = (agent.mean() - baseline.mean()) / baseline.mean() * 100
    ax.text(
        1.5,
        bar_y + span * 0.03,
        f"{_stars(res.p_value)}  (p = {res.p_value:.1e}, r = {res.effect_size:+.2f})",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_ylim(top=bar_y + span * 0.22)

    direction = "↓ besser" if lower_is_better else "↑ besser"
    ax.text(
        0.5,
        bar_y + span * 0.13,
        f"Δ = {rel:+.1f} %   ({direction})",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )

    if footnote:
        fig.subplots_adjust(bottom=0.22)
        fig.text(
            0.5,
            0.015,
            footnote,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#444444",
            style="italic",
            wrap=True,
        )

    fig.tight_layout(rect=(0, 0.06, 1, 1) if footnote else None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "[KPI] %s: baseline %.2f → DQN %.2f (Δ %.1f %%, p=%.2e, r=%.2f) → %s",
        title,
        baseline.mean(),
        agent.mean(),
        rel,
        res.p_value,
        res.effect_size,
        output_path.name,
    )


def _gridlock_chart(
    baseline_wt: np.ndarray,
    agent_wt: np.ndarray,
    *,
    threshold: float,
    output_path: Path,
) -> tuple[int, int]:
    """Bar chart: number of seeds ending in quasi-gridlock."""
    n = len(baseline_wt)
    bl_gridlock = int((baseline_wt > threshold).sum())
    dqn_gridlock = int((agent_wt > threshold).sum())

    fig, ax = plt.subplots(figsize=(7.5, 6))
    bars = ax.bar(
        [BASELINE_LABEL, DQN_LABEL],
        [bl_gridlock, dqn_gridlock],
        color=[C_GRIDLOCK_BL, C_GRIDLOCK_DQN],
        width=0.5,
        edgecolor="white",
        linewidth=1.0,
    )
    for bar, val in zip(bars, (bl_gridlock, dqn_gridlock), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.3,
            f"{val}/{n}",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )

    ax.set_ylabel("Seeds in Quasi-Gridlock")
    ax.set_ylim(0, n)
    ax.set_title(
        f"Gridlock-Häufigkeit bei medium-Last ({n} Seeds)",
        fontsize=13,
        pad=12,
    )
    ax.text(
        0.5,
        -0.13,
        f"Quasi-Gridlock = Per-Step-Wartezeit > {threshold:.0f} s "
        f"(System kollabiert). DQN halbiert die Kollaps-Häufigkeit.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "[GRIDLOCK] baseline %d/%d vs DQN %d/%d (threshold %.0f s) → %s",
        bl_gridlock,
        n,
        dqn_gridlock,
        n,
        threshold,
        output_path.name,
    )
    return bl_gridlock, dqn_gridlock


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Generate OSM evaluation figures")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=CONFIG.paths.csv_dir / "baseline_osm_medium.csv",
    )
    parser.add_argument(
        "--agent-csv",
        type=Path,
        default=CONFIG.paths.csv_dir / "eval_dqn_osm_medium_3M.csv",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=CONFIG.paths.logs_dir
        / "dqn_diff-waiting-time_3000000steps_seed42"
        / "traffic_metrics.csv",
    )
    parser.add_argument("--plots-dir", type=Path, default=CONFIG.paths.plots_dir)
    parser.add_argument("--gridlock-threshold", type=float, default=GRIDLOCK_THRESHOLD)
    args = parser.parse_args()

    bl = pd.read_csv(args.baseline_csv)
    ag = pd.read_csv(args.agent_csv)
    logger.info("Baseline N=%d, DQN N=%d", len(bl), len(ag))

    n_extra = round(float(ag["total_throughput"].mean() - bl["total_throughput"].mean()))
    bias_note = (
        "Selektionsverzerrung: tripinfo-Mittel (Wartezeit/Zeitverlust) zählen "
        f"nur abgeschlossene Trips. Baseline bringt ~{n_extra} Fz weniger ins "
        "Ziel (im Stau) → die schwersten Fälle fehlen in ihren Mitteln. "
        "Unverzerrter Leitindikator ist der Durchsatz."
    )

    # --- KPI boxplots -----------------------------------------------------
    _kpi_boxplot(
        bl["total_throughput"].to_numpy(float),
        ag["total_throughput"].to_numpy(float),
        title="Durchsatz (tripinfo) - abgeschlossene Fahrten / Episode",
        ylabel="Durchsatz [Fz / Episode]",
        lower_is_better=False,
        output_path=args.plots_dir / "kpi_osm_throughput.png",
        footnote=bias_note,
    )
    _kpi_boxplot(
        bl["tripinfo_mean_time_loss"].to_numpy(float),
        ag["tripinfo_mean_time_loss"].to_numpy(float),
        title="Ø Zeitverlust pro Fahrzeug (tripinfo) - gering selektionsverzerrt",
        ylabel="Ø Zeitverlust [s/Fz]",
        lower_is_better=True,
        output_path=args.plots_dir / "kpi_osm_timeloss.png",
    )
    _kpi_boxplot(
        bl["max_queue_length"].to_numpy(float),
        ag["max_queue_length"].to_numpy(float),
        title="Maximaler Rückstau pro Episode",
        ylabel="Max. Rückstau [Fz]",
        lower_is_better=True,
        output_path=args.plots_dir / "kpi_osm_maxqueue.png",
    )
    _kpi_boxplot(
        bl["avg_speed"].to_numpy(float),
        ag["avg_speed"].to_numpy(float),
        title="Ø Geschwindigkeit (Per-Step-System)",
        ylabel="Ø Geschwindigkeit [m/s]",
        lower_is_better=False,
        output_path=args.plots_dir / "kpi_osm_speed.png",
    )
    _kpi_boxplot(
        bl["avg_waiting_time"].to_numpy(float),
        ag["avg_waiting_time"].to_numpy(float),
        title="Ø Wartezeit (Per-Step-System) - bimodale Festzeit-Verteilung",
        ylabel="Ø Wartezeit [Per-Step-System]",
        lower_is_better=True,
        output_path=args.plots_dir / "kpi_osm_waiting_perstep.png",
    )
    _kpi_boxplot(
        bl["tripinfo_mean_waiting_time"].to_numpy(float),
        ag["tripinfo_mean_waiting_time"].to_numpy(float),
        title="Ø Wartezeit pro Fahrzeug (tripinfo) - selektionsverzerrt",
        ylabel="Ø Wartezeit [s/Fz]",
        lower_is_better=True,
        output_path=args.plots_dir / "kpi_osm_waiting_pertrip.png",
        footnote=bias_note,
    )

    # --- Gridlock chart ---------------------------------------------------
    _gridlock_chart(
        bl["avg_waiting_time"].to_numpy(float),
        ag["avg_waiting_time"].to_numpy(float),
        threshold=args.gridlock_threshold,
        output_path=args.plots_dir / "gridlock_osm.png",
    )

    # --- Learning curve ---------------------------------------------------
    if args.metrics_csv.exists():
        plot_learning_curve(
            args.metrics_csv,
            args.plots_dir / "learning_curve_osm_dqn_3M.png",
            title="DQN-Lernkurve (OSM-Netz, 3 Mio. Steps / 4 166 Episoden)",
            window=50,
        )
    else:
        logger.warning("metrics CSV not found, skipping learning curve: %s", args.metrics_csv)

    logger.info("All figures written to %s", args.plots_dir)


if __name__ == "__main__":
    main()
