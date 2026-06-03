"""Audit the reported OSM results by recomputing every headline number.

This is the answer to anyone who doubts the figures: it reloads the raw 30-seed
evaluation CSVs and **independently recomputes** the KPI means, the Mann-Whitney
U p-values, the rank-biserial effect sizes, the relative deltas and the gridlock
counts, then checks them against the values claimed in the documentation /
slides. It also verifies that the required artifacts (model checkpoint, network,
route file, training log) exist and that each group has exactly N = 30 seeds.

Exit code 0 = all checks passed; 1 = at least one check failed.

Usage:
    python scripts/audit_results.py
"""
from __future__ import annotations

import sys

import pandas as pd

from src.config.settings import CONFIG
from src.evaluation.statistical_tests import mann_whitney_u

GRIDLOCK_THRESHOLD = 200.0

# (column, label, lower_is_better, expected_rel_pct, expected_p_max)
EXPECTED_KPIS = [
    ("avg_waiting_time", "Ø Wartezeit (per-step)", True, -67.5, 1e-5),
    ("max_queue_length", "Max. Rückstau", True, -59.0, 1e-3),
    ("avg_speed", "Ø Geschwindigkeit", False, +54.1, 1e-5),
    ("total_throughput", "Durchsatz", False, +26.2, 1e-3),
    ("tripinfo_mean_waiting_time", "Ø Wartezeit/Fz", True, -55.6, 1e-5),
    ("tripinfo_mean_duration", "Ø Reisezeit/Fz", True, -3.1, 5e-2),
    ("tripinfo_mean_time_loss", "Ø Zeitverlust/Fz", True, -18.3, 1e-5),
]
EXPECTED_GRIDLOCK = {"baseline": 16, "dqn": 7, "n": 30}
EXPECTED_COMPLETION = {"loaded": 681, "baseline_pct": 69.5, "dqn_pct": 87.7}


class Audit:
    """Collects PASS/FAIL checks and prints a report."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        self.passed += ok
        self.failed += not ok
        print(f"  [{mark}] {name}" + (f"  - {detail}" if detail else ""))

    def near(self, name: str, got: float, want: float, tol: float, unit: str = "") -> None:
        ok = abs(got - want) <= tol
        self.check(name, ok, f"recomputed {got:+.1f}{unit}, expected {want:+.1f}{unit} (±{tol}{unit})")


def main() -> int:
    bl_csv = CONFIG.paths.csv_dir / "baseline_osm_medium.csv"
    ag_csv = CONFIG.paths.csv_dir / "eval_dqn_osm_medium_3M.csv"
    model = CONFIG.paths.models_dir / "dqn_diff-waiting-time_3000000steps_seed42.zip"
    net = CONFIG.paths.sumo_config_dir / "loerrach_osm.net.xml"
    route = CONFIG.paths.sumo_config_dir / "loerrach_osm_medium.rou.xml"
    metrics = (
        CONFIG.paths.logs_dir
        / "dqn_diff-waiting-time_3000000steps_seed42"
        / "traffic_metrics.csv"
    )

    a = Audit()

    print("\n=== 1. Artefakte vorhanden ===")
    for name, p in [
        ("Baseline-CSV", bl_csv), ("DQN-Eval-CSV", ag_csv),
        ("Modell-Checkpoint (3M)", model), ("OSM-Netz", net),
        ("Route (medium)", route), ("Trainings-Log", metrics),
    ]:
        a.check(name, p.exists(), str(p.relative_to(CONFIG.paths.results_dir.parent)))

    if not (bl_csv.exists() and ag_csv.exists()):
        print("\nFEHLER: Eval-CSVs fehlen, Abbruch.")
        return 1

    bl = pd.read_csv(bl_csv)
    ag = pd.read_csv(ag_csv)

    print("\n=== 2. Stichprobengröße (N=30 Seeds je Gruppe) ===")
    a.check("Baseline N=30", len(bl) == 30, f"N={len(bl)}")
    a.check("DQN N=30", len(ag) == 30, f"N={len(ag)}")
    a.check("Eval-Seeds = 0..29 (Baseline)", sorted(bl["seed"]) == list(range(30)))
    a.check("Eval-Seeds = 0..29 (DQN)", sorted(ag["seed"]) == list(range(30)))

    print("\n=== 3. KPI-Statistik neu berechnet vs. dokumentiert ===")
    all_sig = True
    for col, label, lower, exp_rel, exp_pmax in EXPECTED_KPIS:
        b = bl[col].to_numpy(float)
        g = ag[col].to_numpy(float)
        rel = (g.mean() - b.mean()) / b.mean() * 100
        res = mann_whitney_u(b, g)
        improved = (g.mean() < b.mean()) if lower else (g.mean() > b.mean())
        a.near(f"{label}: Δ rel.", rel, exp_rel, 0.6, "%")
        a.check(
            f"{label}: signifikant (p<{exp_pmax:g})",
            res.p_value < exp_pmax,
            f"p={res.p_value:.2e}, r={res.effect_size:+.2f}",
        )
        all_sig = all_sig and res.significant and improved
    a.check("Alle 7 KPIs signifikant & richtungsrichtig", all_sig)

    print("\n=== 4. Gridlock-Häufigkeit (per-step Wartezeit > 200 s) ===")
    bl_grid = int((bl["avg_waiting_time"] > GRIDLOCK_THRESHOLD).sum())
    ag_grid = int((ag["avg_waiting_time"] > GRIDLOCK_THRESHOLD).sum())
    a.check("Baseline 16/30", bl_grid == EXPECTED_GRIDLOCK["baseline"], f"{bl_grid}/30")
    a.check("DQN 7/30", ag_grid == EXPECTED_GRIDLOCK["dqn"], f"{ag_grid}/30")

    print("\n=== 5. Abschlussquote (681 geladene Trips) ===")
    bl_pct = bl["total_throughput"].mean() / EXPECTED_COMPLETION["loaded"] * 100
    ag_pct = ag["total_throughput"].mean() / EXPECTED_COMPLETION["loaded"] * 100
    a.near("Baseline Abschlussquote", bl_pct, EXPECTED_COMPLETION["baseline_pct"], 1.0, "%")
    a.near("DQN Abschlussquote", ag_pct, EXPECTED_COMPLETION["dqn_pct"], 1.0, "%")

    print("\n=== 6. Selektionsverzerrung - Plausibilitätsindiz ===")
    bl_wt_std = bl["tripinfo_mean_waiting_time"].std()
    bl_ps_std = bl["avg_waiting_time"].std()
    a.check(
        "Festzeit tripinfo-Wartezeit-Std << per-step-Std (Bias-Indiz)",
        bl_wt_std < 1.0 < bl_ps_std,
        f"tripinfo std={bl_wt_std:.2f}s vs per-step std={bl_ps_std:.0f}",
    )
    n_extra = ag["total_throughput"].mean() - bl["total_throughput"].mean()
    a.near("~124 Fz mehr im Ziel (DQN-Baseline)", n_extra, 124.0, 5.0, " Fz")

    print(f"\n{'=' * 52}")
    total = a.passed + a.failed
    print(f"ERGEBNIS: {a.passed}/{total} Checks bestanden, {a.failed} fehlgeschlagen.")
    if a.failed == 0:
        print("OK - Alle Audit-Checks bestanden, die berichteten Zahlen sind reproduzierbar.")
        return 0
    print("FEHLER - Mindestens ein Check ist fehlgeschlagen.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
