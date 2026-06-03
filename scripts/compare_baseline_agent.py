"""Compare a fixed-time baseline against an RL agent over N evaluation seeds.

Reads two evaluation CSVs (baseline + agent, same KPI schema as produced by
``src/evaluation/baseline.py`` and ``src/evaluation/evaluate.py``), computes
per-KPI mean ± std, the relative change, and a Mann-Whitney U test with
rank-biserial effect size, then writes a Markdown comparison table.

The table honestly distinguishes **per-step system quantities** (sumo-rl
``env.metrics``, not per-vehicle) from **per-vehicle tripinfo metrics**
(SUMO ``--tripinfo-output``), per Audit findings B4/B5/F1.

Usage:
    python scripts/compare_baseline_agent.py \
        --baseline-csv results/csv/baseline_osm_medium.csv \
        --agent-csv results/csv/eval_dqn_osm_medium_3M.csv \
        --output-md results/csv/compare_osm_medium.md \
        --title "OSM-Netz (medium), DQN 3M Steps"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.statistical_tests import mann_whitney_u

# KPI metadata: column -> (label, unit, lower_is_better, metric_type)
KPIS: dict[str, tuple[str, str, bool, str]] = {
    "avg_waiting_time": ("Ø Wartezeit", "Per-Step-System", True, "per-step"),
    "max_queue_length": ("Max. Rückstau", "Fz", True, "per-step"),
    "avg_speed": ("Ø Geschwindigkeit", "m/s", False, "per-step"),
    "total_throughput": ("Durchsatz (tripinfo)", "Fz/Episode", False, "tripinfo"),
    "tripinfo_mean_waiting_time": ("Ø Wartezeit/Fz", "s", True, "tripinfo"),
    "tripinfo_mean_duration": ("Ø Reisezeit/Fz", "s", True, "tripinfo"),
    "tripinfo_mean_time_loss": ("Ø Zeitverlust/Fz", "s", True, "tripinfo"),
}


def _fmt(x: float) -> str:
    """Format a number compactly (more decimals for small magnitudes)."""
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def compare(baseline: pd.DataFrame, agent: pd.DataFrame, title: str) -> str:
    """Build the Markdown comparison table for all shared KPI columns."""
    n_bl, n_ag = len(baseline), len(agent)
    lines = [
        f"## KPI-Vergleich - {title}",
        "",
        f"Festzeit-Baseline (N={n_bl}) vs. DQN-Agent (N={n_ag}), "
        "Mann-Whitney U (zweiseitig, alpha=0,05), r = rank-biserial.",
        "",
        "| KPI | Typ | Festzeit | DQN | Δ rel. | p-Wert | r | sig. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for col, (label, unit, lower_better, mtype) in KPIS.items():
        if col not in baseline.columns or col not in agent.columns:
            continue
        bl = baseline[col].dropna().to_numpy(dtype=float)
        ag = agent[col].dropna().to_numpy(dtype=float)
        if len(bl) == 0 or len(ag) == 0:
            continue

        bl_m, bl_s = bl.mean(), bl.std()
        ag_m, ag_s = ag.mean(), ag.std()
        rel = (ag_m - bl_m) / bl_m * 100 if bl_m != 0 else float("nan")

        res = mann_whitney_u(bl, ag)
        improved = (ag_m < bl_m) if lower_better else (ag_m > bl_m)
        sig_mark = "ja" if res.significant and improved else (
            "ungünstig" if res.significant else "-"
        )
        p_str = f"{res.p_value:.2e}" if res.p_value < 1e-3 else f"{res.p_value:.4f}"
        lines.append(
            f"| {label} [{unit}] | {mtype} | {_fmt(bl_m)} ± {_fmt(bl_s)} | "
            f"{_fmt(ag_m)} ± {_fmt(ag_s)} | {rel:+.1f} % | {p_str} | "
            f"{res.effect_size:+.2f} | {sig_mark} |"
        )

    lines += [
        "",
        "> **Lesehilfe:** *Per-Step*-Größen sind sumo-rl-Systemwerte "
        "über alle Fahrzeuge je RL-Schritt - **nicht** Sekunden pro Fahrzeug. "
        "Absolut belastbar sind nur die *tripinfo*-Größen (pro abgeschlossenem "
        "Fahrzeug). ja = signifikante Verbesserung, ungünstig = signifikant, aber "
        "Richtung ungünstig, - = nicht signifikant.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs RL agent KPIs")
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--agent-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Szenario")
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_csv)
    agent = pd.read_csv(args.agent_csv)
    md = compare(baseline, agent, args.title)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\n[written] {args.output_md}")


if __name__ == "__main__":
    main()
