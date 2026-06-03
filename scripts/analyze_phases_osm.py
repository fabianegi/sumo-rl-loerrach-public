"""Compare the DQN agent's learned green-phase durations vs. the fixed program.

Answers the slide question *"what did the agent learn?"* on the real OSM network
(TLS 1628110071). The imported OSM default program is a static 90 s cycle with
three green phases:

* green[0] ``GGggrrrGGg`` - B317 main axis        (fixed 38 s)
* green[1] ``GGGGrrrrrr`` - secondary/protected     (fixed  6 s)
* green[2] ``rrrrGGgGrr`` - cross street (Querstr.)  (fixed 37 s)

The DQN agent re-selects a green phase every ``delta_time`` seconds. This script
runs N agent episodes, measures the mean *contiguous* green duration the agent
holds each phase, and plots it against the fixed program. There is **no**
time-of-day split - the OSM demand is a single constant-rate "medium" scenario
(see results/OSM_NETWORK_RESULTS.md), so any "morning/noon/evening" breakdown
would be fabricated.

Usage:
    python scripts/analyze_phases_osm.py \
        --model models/checkpoints/dqn_diff-waiting-time_3000000steps_seed42.zip
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

from src.config.settings import CONFIG
from src.environment.env_factory import make_eval_env

logger = logging.getLogger(__name__)

DPI = 150
C_FIXED = "#888888"
C_DQN = "#2196F3"

# Phase index -> (short label, fixed-program green duration [s]) for TLS 1628110071.
PHASE_LABELS = {
    0: "B317 Haupt\n(GGggrrrGGg)",
    1: "Nebenphase\n(GGGGrrrrrr)",
    2: "Quer / Riehenstr.\n(rrrrGGgGrr)",
}
FIXED_GREEN_S = {0: 38.0, 1: 6.0, 2: 37.0}
_FIXED_TOTAL_GREEN = sum(FIXED_GREEN_S.values())
FIXED_SHARE_PCT = {p: g / _FIXED_TOTAL_GREEN * 100 for p, g in FIXED_GREEN_S.items()}


def _agent_phase_log(model_path: Path, seed: int, net: Path, route: Path) -> list[int]:
    """Run one deterministic agent episode; return the green-phase index per step."""
    env = make_eval_env(net_file=net, route_file=route, seed=seed)
    model = DQN.load(str(model_path), env=env)
    obs, _ = env.reset()
    log: list[int] = []
    done = False
    while not done:
        ts = next(iter(env.unwrapped.traffic_signals.values()))
        log.append(int(ts.green_phase))
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc
    env.close()
    return log


def _green_runs(log: list[int]) -> dict[int, list[int]]:
    """Contiguous run lengths (in RL steps) per phase index from a per-step log."""
    runs: dict[int, list[int]] = {}
    if not log:
        return runs
    cur, length = log[0], 1
    for p in log[1:]:
        if p == cur:
            length += 1
        else:
            runs.setdefault(cur, []).append(length)
            cur, length = p, 1
    runs.setdefault(cur, []).append(length)
    return runs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Analyze learned green-phase durations")
    parser.add_argument(
        "--model",
        type=Path,
        default=CONFIG.paths.models_dir / "dqn_diff-waiting-time_3000000steps_seed42.zip",
    )
    parser.add_argument(
        "--net-file",
        type=Path,
        default=CONFIG.paths.sumo_config_dir / "loerrach_osm.net.xml",
    )
    parser.add_argument(
        "--route-file",
        type=Path,
        default=CONFIG.paths.sumo_config_dir / "loerrach_osm_medium.rou.xml",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,2,3",
        help="Comma-separated free-flow eval seeds to average over",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG.paths.plots_dir / "phases_osm_dqn_vs_fixed.png",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    dt = CONFIG.sumo.delta_time

    pooled_runs: dict[int, list[int]] = {0: [], 1: [], 2: []}
    pooled_steps: dict[int, int] = {0: 0, 1: 0, 2: 0}
    total_steps = 0
    for seed in seeds:
        logger.info("Running agent episode (seed=%d) ...", seed)
        log = _agent_phase_log(args.model, seed, args.net_file, args.route_file)
        for p, runs in _green_runs(log).items():
            pooled_runs.setdefault(p, []).extend(runs)
        for p in pooled_steps:
            pooled_steps[p] += log.count(p)
        total_steps += len(log)
        share = {p: f"{log.count(p) / len(log) * 100:.0f}%" for p in sorted(set(log))}
        logger.info("  seed %d green-time share: %s", seed, share)

    # Mean contiguous green duration (s) and green-time share (%) per phase.
    dqn_dur = {p: (float(np.mean(r)) * dt if r else 0.0) for p, r in pooled_runs.items()}
    dqn_n = {p: len(r) for p, r in pooled_runs.items()}
    dqn_share = {p: pooled_steps[p] / total_steps * 100 for p in (0, 1, 2)}

    logger.info("\n=== Gelernte Grünphasen (DQN, gepoolt über %d Seeds) ===", len(seeds))
    for p in (0, 1, 2):
        logger.info(
            "  %-26s Anteil: Fix %4.1f%% vs DQN %4.1f%% | Ø Dauer: Fix %3.0fs vs "
            "DQN %4.1fs (n=%d Aktivierungen)",
            PHASE_LABELS[p].replace("\n", " "),
            FIXED_SHARE_PCT[p],
            dqn_share[p],
            FIXED_GREEN_S[p],
            dqn_dur[p],
            dqn_n[p],
        )

    # --- two-panel grouped bar chart -------------------------------------
    phases = [0, 1, 2]
    labels = [PHASE_LABELS[p] for p in phases]
    x = np.arange(len(phases))
    width = 0.38

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))

    # Panel A: green-time share (%) - the robust "reallocation" story.
    a1 = ax_a.bar(
        x - width / 2,
        [FIXED_SHARE_PCT[p] for p in phases],
        width,
        label="Festzeit (OSM-Default)",
        color=C_FIXED,
        edgecolor="white",
    )
    a2 = ax_a.bar(
        x + width / 2,
        [dqn_share[p] for p in phases],
        width,
        label="DQN (OSM-Netz)",
        color=C_DQN,
        edgecolor="white",
    )
    for bars in (a1, a2):
        for bar in bars:
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{bar.get_height():.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=8.5)
    ax_a.set_ylabel("Grünzeit-Anteil [%]")
    ax_a.set_title("(a) Verteilung der Grünzeit", fontsize=11.5)
    ax_a.legend(loc="upper right", fontsize=9)

    # Panel B: mean contiguous green duration (s).
    b1 = ax_b.bar(
        x - width / 2,
        [FIXED_GREEN_S[p] for p in phases],
        width,
        label="Festzeit (OSM-Default)",
        color=C_FIXED,
        edgecolor="white",
    )
    b2 = ax_b.bar(
        x + width / 2,
        [dqn_dur[p] for p in phases],
        width,
        label="DQN (OSM-Netz)",
        color=C_DQN,
        edgecolor="white",
    )
    for bars in (b1, b2):
        for bar in bars:
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{bar.get_height():.0f}s",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=8.5)
    ax_b.set_ylabel("Ø Grünphasendauer [s]")
    ax_b.set_title("(b) Ø Dauer einer Grünphase", fontsize=11.5)
    ax_b.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Was hat der Agent gelernt? Festzeit vs. DQN - OSM-Netz, medium-Last "
        f"(gepoolt über {len(seeds)} Seeds)",
        fontsize=13,
        y=1.0,
    )
    fig.text(
        0.5,
        -0.02,
        "Der Agent verlagert Grünzeit auf die B317-Hauptachse und verwirft die "
        "Nebenphase fast vollständig. Kein Tagesgang - konstante medium-Last "
        "(1 Szenario).",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("\nPhase comparison plot saved: %s", args.output)


if __name__ == "__main__":
    main()
