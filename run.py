#!/usr/bin/env python3
"""Canonical, OS-agnostic operator entrypoint for the SUMO-RL Lörrach project.

One interface for the whole workflow::

    python run.py check       # environment doctor (Python, imports, SUMO, data)
    python run.py list        # valid intersections / scenarios / algos / rewards
    python run.py demo        # before/after vs fixed-time baseline (no SUMO, ~seconds)
    python run.py compare     # regenerate DQN vs PPO vs baseline figures
    python run.py train       # train an agent (live SUMO)
    python run.py evaluate    # evaluate a trained model (live SUMO)

This module is a *thin* wrapper: it imports and reuses the existing project
functions under ``src/`` and ``scripts/`` - it never re-implements training,
evaluation, reward or environment logic.

``demo``/``compare``/``check``/``list`` need no SUMO and no trained model - they
replay the audited evaluation CSVs committed under ``results/csv/``. ``train``
and ``evaluate`` run live SUMO and therefore require SUMO (shipped via pip as
``eclipse-sumo``; see requirements.txt) and, for ``evaluate``, a model file.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Headless-safe matplotlib backend BEFORE any project module imports pyplot.
import matplotlib

matplotlib.use("Agg")

# Repo root = directory of this file. Ensures `import src...` works regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# SUMO bootstrap (non-fatal)
# --------------------------------------------------------------------------- #
def bootstrap_sumo() -> str | None:
    """Resolve SUMO_HOME from the pip ``sumo`` package and expose ``tools``.

    Mirrors the standard SUMO Python stanza but driven by the pip install:
    if ``SUMO_HOME`` is not already set, take it from ``import sumo`` and append
    ``<SUMO_HOME>/tools`` to ``sys.path`` so ``traci``/``sumolib`` import cleanly.

    Returns the resolved SUMO_HOME, or ``None`` if SUMO cannot be located. Never
    raises - the no-SUMO commands (demo/compare/check/list) must still work.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        try:
            import sumo  # noqa: PLC0415 (pip eclipse-sumo)

            sumo_home = sumo.SUMO_HOME
            os.environ["SUMO_HOME"] = sumo_home
        except Exception:  # noqa: BLE001 - any failure → SUMO simply unavailable
            return None

    tools = Path(sumo_home) / "tools"
    if tools.is_dir() and str(tools) not in sys.path:
        sys.path.append(str(tools))
    return sumo_home


# --------------------------------------------------------------------------- #
# Static option metadata (single source of truth for `list`, `train`, `evaluate`)
# --------------------------------------------------------------------------- #
INTERSECTIONS = {
    "single": "Synthetisches 4-Arm-Kreuz (Methodenbeweis), TLS 'C'",
    "corridor": "3-Kreuzungs-Korridor (nur 'medium')",
    "real": "Manuell konstruierte Realgeometrie (Basler Str. x Obere Riehenstr.)",
    "osm": "OSM-Import der Realkreuzung - Headline-Netz (DQN 3M)",
}
SCENARIOS = ("low", "medium", "high")
ALGOS = ("dqn", "ppo")
REWARDS = ("diff-waiting-time", "pressure")

# Headline audit-evidence CSVs that `demo` replays (committed under results/csv/).
DEMO_BASELINE_CSV = "baseline_osm_medium.csv"
DEMO_AGENT_CSV = "eval_dqn_osm_medium_3M.csv"

# KPI columns → pretty keys understood by plotting.plot_kpi_comparison's label map.
_PRETTY_RENAME = {"total_throughput": "throughput", "max_queue_length": "max_queue"}
DEMO_KPIS = ("throughput", "avg_waiting_time", "max_queue", "avg_speed")


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def _mark(ok: bool) -> str:
    return "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"


def cmd_check(_args: argparse.Namespace) -> int:
    """Environment doctor. Prints a pass/fail checklist; exits 0 only if ready."""
    print("=== run.py check - SUMO-RL Lörrach environment doctor ===\n")
    failures = 0

    # 1. Python version
    py_ok = sys.version_info[:2] == (3, 11)
    failures += not py_ok
    pv = ".".join(map(str, sys.version_info[:3]))
    print(f"  {_mark(py_ok)} Python {pv}  (erwartet: 3.11.x)")

    # 2. Core imports (+ version-sensitive gymnasium pin)
    core = ["stable_baselines3", "sumo_rl", "torch", "pandas", "matplotlib", "scipy", "seaborn"]
    for mod in core:
        try:
            __import__(mod)
            print(f"  {_mark(True)} import {mod}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  {_mark(False)} import {mod}  ({exc})")

    try:
        import gymnasium

        gym_ok = gymnasium.__version__ == "0.29.1"
        failures += not gym_ok
        suffix = "" if gym_ok else "  (erwartet 0.29.1 - sumo-rl 1.4.5 bricht mit >=1.0)"
        print(f"  {_mark(gym_ok)} import gymnasium {gymnasium.__version__}{suffix}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"  {_mark(False)} import gymnasium  ({exc})")

    # 3. SUMO resolution (needed for train/evaluate; demo/compare work without it)
    sumo_home = bootstrap_sumo()
    if sumo_home:
        try:
            import traci  # noqa: F401
            import sumolib  # noqa: F401

            print(f"  {_mark(True)} SUMO via SUMO_HOME={sumo_home}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  {_mark(False)} SUMO_HOME set but traci/sumolib import failed ({exc})")
    else:
        failures += 1
        print(f"  {_mark(False)} SUMO not found (pip install eclipse-sumo). "
              "Hinweis: demo/compare laufen trotzdem.")

    # 4. Demo evidence CSVs present
    from src.config.settings import CONFIG

    for name in (DEMO_BASELINE_CSV, DEMO_AGENT_CSV):
        p = CONFIG.paths.csv_dir / name
        ok = p.exists()
        failures += not ok
        print(f"  {_mark(ok)} results/csv/{name}")

    print()
    if failures:
        print(f"FAIL - {failures} Problem(e). Siehe README → Troubleshooting.")
        return 1
    print("OK - Umgebung bereit. Nächster Schritt: python run.py demo")
    return 0


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def cmd_list(_args: argparse.Namespace) -> int:
    """Print valid intersections, scenarios, algorithms and reward functions."""
    print("Intersections (--intersection):")
    for name, desc in INTERSECTIONS.items():
        print(f"  {name:9s} - {desc}")
    print("\nScenarios (--scenario):  " + ", ".join(SCENARIOS))
    print("  Hinweis: nicht jede Kombination existiert (osm/corridor: nur 'medium').")
    print("\nAlgorithms (--algo):     " + ", ".join(ALGOS))
    print("Rewards (--reward):      " + ", ".join(REWARDS))
    return 0


# --------------------------------------------------------------------------- #
# Shared helpers for demo / compare
# --------------------------------------------------------------------------- #
def _silence_plot_warnings() -> None:
    """Hide a pre-existing seaborn FutureWarning from plotting.py (presentation only)."""
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)


def _load_csv(name: str):
    import pandas as pd

    from src.config.settings import CONFIG

    path = CONFIG.paths.csv_dir / name
    if not path.exists():
        raise SystemExit(f"Fehlende Belegdatei: {path}\nBitte aus dem Repo (results/csv/) wiederherstellen.")
    return pd.read_csv(path)


def _pretty(df):
    """Rename KPI columns so plotting.plot_kpi_comparison shows German labels."""
    return df.rename(columns=_PRETTY_RENAME)


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
def cmd_demo(_args: argparse.Namespace) -> int:
    """Replay the audited OSM result: print before/after table + save a figure."""
    logging.basicConfig(level=logging.WARNING)
    _silence_plot_warnings()
    from src.config.settings import CONFIG
    from src.utils.plotting import plot_kpi_comparison

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from compare_baseline_agent import compare  # noqa: PLC0415

    baseline = _load_csv(DEMO_BASELINE_CSV)
    agent = _load_csv(DEMO_AGENT_CSV)
    title = "OSM-Netz (medium), DQN 3M Steps"

    print(compare(baseline, agent, title))
    print()

    pb, pa = _pretty(baseline), _pretty(agent)
    hero = CONFIG.paths.results_dir / "demo_comparison.png"
    plot_kpi_comparison(pb, {"DQN": pa}, "throughput", hero)
    for kpi in DEMO_KPIS:
        if kpi == "throughput":
            continue
        plot_kpi_comparison(pb, {"DQN": pa}, kpi, CONFIG.paths.plots_dir / f"demo_{kpi}.png")

    print(f"[Figur] {hero}")
    print(f"[Figuren] {CONFIG.paths.plots_dir}/demo_*.png")
    print("\nHeadline: Durchsatz +26,2 % (p=2,6e-05, r=+0,63) gegenüber Festzeit-Baseline.")
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def _write_compare_md(baseline, agent, title: str, out_name: str) -> None:
    from src.config.settings import CONFIG

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from compare_baseline_agent import compare  # noqa: PLC0415

    md = compare(baseline, agent, title)
    out = CONFIG.paths.csv_dir / out_name
    out.write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"[geschrieben] {out}\n")


def cmd_compare(_args: argparse.Namespace) -> int:
    """Regenerate the publication comparison figures + markdown (CSV-driven)."""
    logging.basicConfig(level=logging.WARNING)
    _silence_plot_warnings()
    from src.config.settings import CONFIG
    from src.utils.plotting import plot_kpi_comparison

    plots = CONFIG.paths.plots_dir

    # (1) OSM headline: DQN 3M vs fixed-time baseline.
    osm_bl = _load_csv(DEMO_BASELINE_CSV)
    osm_ag = _load_csv(DEMO_AGENT_CSV)
    _write_compare_md(osm_bl, osm_ag, "OSM-Netz (medium), DQN 3M Steps", "compare_osm_medium.md")
    pb, pa = _pretty(osm_bl), _pretty(osm_ag)
    for kpi in DEMO_KPIS:
        plot_kpi_comparison(pb, {"DQN": pa}, kpi, plots / f"compare_osm_{kpi}.png")

    # (2) Synthetic Methodenbeweis: DQN vs PPO vs baseline (1M, saturation scenario).
    try:
        syn_bl = _load_csv("baseline_fixed_time_saturation.csv")
        syn_dqn = _load_csv("eval_dqn_dwt_1M_saturation.csv")
        syn_ppo = _load_csv("eval_ppo_dwt_1M_saturation.csv")
    except SystemExit as exc:
        print(f"[übersprungen] Synthetik-Vergleich: {exc}")
        return 0

    agents = {"DQN": _pretty(syn_dqn), "PPO": _pretty(syn_ppo)}
    pbl = _pretty(syn_bl)
    for kpi in DEMO_KPIS:
        plot_kpi_comparison(pbl, agents, kpi, plots / f"compare_synth_{kpi}.png")
    print(f"[Figuren] {plots}/compare_osm_*.png  &  {plots}/compare_synth_*.png")
    return 0


# --------------------------------------------------------------------------- #
# Network/scenario resolution for live commands
# --------------------------------------------------------------------------- #
def resolve_net_route(intersection: str, scenario: str):
    """Map (intersection, scenario) → (net_file, route_file) via CONFIG.paths.

    Raises SystemExit with the available route files if the combination has no
    committed demand file.
    """
    from src.config.settings import CONFIG

    net_map = {
        "single": CONFIG.paths.net_file,
        "corridor": CONFIG.paths.corridor_net_file,
        "real": CONFIG.paths.real_net_file,
        "osm": CONFIG.paths.osm_net_file,
    }
    if intersection not in net_map:
        raise SystemExit(f"Unbekannte Kreuzung '{intersection}'. Gültig: {', '.join(net_map)}")
    net = net_map[intersection]
    if not net.exists():
        raise SystemExit(f"Netzdatei fehlt: {net}")

    cand = net.parent / f"{net.stem}_{scenario}.rou.xml"
    if cand.exists():
        return net, cand
    # single + medium ships as the bare loerrach.rou.xml
    if intersection == "single" and scenario == "medium" and CONFIG.paths.route_file.exists():
        return net, CONFIG.paths.route_file

    avail = sorted(p.name for p in net.parent.glob(f"{net.stem}_*.rou.xml"))
    raise SystemExit(
        f"Keine Routendatei für {intersection}/{scenario}.\n"
        f"Verfügbar für dieses Netz: {avail or '(keine)'}"
    )


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    """Wrap the existing DQN/PPO training functions behind one CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if bootstrap_sumo() is None:
        raise SystemExit("SUMO nicht gefunden. `pip install -r requirements.txt` (enthält eclipse-sumo).")

    net, route = resolve_net_route(args.intersection, args.scenario)
    print(f"Training {args.algo.upper()} | {args.intersection}/{args.scenario} | "
          f"{args.timesteps} Steps | seed {args.seed} | reward {args.reward} | gui {args.gui}")

    if args.algo == "dqn":
        from src.training.train_dqn import train_dqn

        model_path = train_dqn(
            total_timesteps=args.timesteps,
            seed=args.seed,
            learning_rate=None,
            use_gui=args.gui,
            reward_fn=args.reward,
            net_file=net,
            route_file=route,
        )
    else:
        from src.training.train_ppo import train_ppo

        model_path = train_ppo(
            total_timesteps=args.timesteps,
            seed=args.seed,
            learning_rate=None,
            n_steps=None,
            use_gui=args.gui,
            reward_fn=args.reward,
            net_file=net,
            route_file=route,
        )
    print(f"\nFertig. Modell gespeichert: {model_path}")
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def _latest_model(models_dir: Path) -> Path | None:
    zips = sorted(models_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a trained model live (loads SB3 policy, runs N headless episodes)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if bootstrap_sumo() is None:
        raise SystemExit("SUMO nicht gefunden. `pip install -r requirements.txt` (enthält eclipse-sumo).")

    from src.config.settings import CONFIG
    from src.evaluation.evaluate import evaluate_agent

    model_path = args.model or _latest_model(CONFIG.paths.models_dir)
    if model_path is None:
        raise SystemExit(
            "Kein Modell gefunden in models/checkpoints/.\n"
            "Die trainierten Gewichte liegen auf dem Trainings-PC und sind nicht im Repo.\n"
            "→ `python run.py train ...` für ein frisches Modell, oder eine .zip dort ablegen.\n"
            "→ Für das dokumentierte Ergebnis ohne Modell: `python run.py demo`."
        )
    model_path = Path(model_path)
    if not model_path.exists():
        raise SystemExit(f"Modell nicht gefunden: {model_path}")

    net, route = resolve_net_route(args.intersection, args.scenario)
    print(f"Evaluiere {model_path.name} | {args.intersection}/{args.scenario} | {args.n_episodes} Episoden")

    results = evaluate_agent(
        model_path=model_path,
        net_file=net,
        route_file=route,
        n_episodes=args.n_episodes,
    )

    out_csv = args.output_csv or (CONFIG.paths.csv_dir / f"eval_{model_path.stem}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)

    print("\nZusammenfassung (Mittelwert über Episoden):")
    print(f"  Durchsatz (tripinfo): {results['total_throughput'].mean():.1f} Fz/Episode")
    print(f"  Ø Wartezeit (per-step): {results['avg_waiting_time'].mean():.2f}")
    print(f"  Max. Rückstau: {results['max_queue_length'].mean():.1f}")
    print(f"\n[geschrieben] {out_csv}")
    return 0


# --------------------------------------------------------------------------- #
# argparse wiring
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="SUMO-RL Lörrach - ein Befehl von git clone zu sichtbaren Ergebnissen.",
    )
    parser.add_argument("--list", action="store_true", help="Gültige Werte auflisten (wie `list`).")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check", help="Umgebungs-Doktor (Python, Imports, SUMO, Daten).")
    sub.add_parser("list", help="Gültige Kreuzungen/Szenarien/Algorithmen/Rewards.")
    sub.add_parser("demo", help="Before/After vs. Festzeit-Baseline (ohne SUMO, ~Sekunden).")
    sub.add_parser("compare", help="DQN vs. PPO vs. Baseline Figuren neu erzeugen.")

    p_train = sub.add_parser("train", help="Agent trainieren (live SUMO).")
    p_train.add_argument("--algo", choices=ALGOS, default="dqn")
    p_train.add_argument("--intersection", choices=list(INTERSECTIONS), default="single")
    p_train.add_argument("--scenario", choices=SCENARIOS, default="medium")
    p_train.add_argument("--timesteps", type=int, default=500_000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--reward", choices=REWARDS, default="diff-waiting-time")
    gui = p_train.add_mutually_exclusive_group()
    gui.add_argument("--gui", dest="gui", action="store_true", help="SUMO-GUI anzeigen.")
    gui.add_argument("--no-gui", dest="gui", action="store_false", help="Headless (Standard).")
    p_train.set_defaults(gui=False)

    p_eval = sub.add_parser("evaluate", help="Trainiertes Modell evaluieren (live SUMO).")
    p_eval.add_argument("--model", type=Path, default=None, help="Pfad zur .zip (Standard: neuestes Modell).")
    p_eval.add_argument("--intersection", choices=list(INTERSECTIONS), default="osm")
    p_eval.add_argument("--scenario", choices=SCENARIOS, default="medium")
    p_eval.add_argument("--n-episodes", dest="n_episodes", type=int, default=30)
    p_eval.add_argument("--output-csv", dest="output_csv", type=Path, default=None)
    return parser


DISPATCH = {
    "check": cmd_check,
    "list": cmd_list,
    "demo": cmd_demo,
    "compare": cmd_compare,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
}


def main(argv: list[str] | None = None) -> int:
    # Resolve SUMO_HOME up front: sumo_rl (imported transitively by several
    # commands) validates SUMO_HOME at import time. Non-fatal if SUMO is absent.
    bootstrap_sumo()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        if getattr(args, "list", False):
            return cmd_list(args)
        parser.print_help()
        return 0
    return DISPATCH[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
