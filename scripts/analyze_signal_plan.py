"""Analyze a trained agent's signal plan and compare to fixed-time.

Runs one evaluation episode, records phase decisions at each RL step,
computes timing statistics, and generates a comparison plot.

Usage:
    python scripts/analyze_signal_plan.py --model models/checkpoints/dqn_diff-waiting-time_500000steps_seed42.zip
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from stable_baselines3 import DQN, PPO

from src.config.settings import CONFIG
from src.environment.env_factory import make_eval_env
from src.utils.plotting import plot_signal_plan

logger = logging.getLogger(__name__)


def detect_algorithm(model_path: Path) -> type:
    """Detect whether model is DQN or PPO from filename."""
    name = model_path.stem.lower()
    if name.startswith("ppo"):
        return PPO
    return DQN


def run_episode_with_phases(
    model_path: Path,
    seed: int = 0,
    net_file: Path | None = None,
    route_file: Path | None = None,
) -> list[tuple[float, int]]:
    """Run one evaluation episode and record phase at each RL step.

    Returns:
        List of (sim_time, phase_index) tuples.
    """
    nf = net_file or CONFIG.paths.net_file
    rf = route_file or CONFIG.paths.route_file

    env = make_eval_env(
        net_file=nf,
        route_file=rf,
        reward_fn="diff-waiting-time",
        seed=seed,
    )

    algo_cls = detect_algorithm(model_path)
    model = algo_cls.load(str(model_path), env=env)

    obs, _info = env.reset()
    phases: list[tuple[float, int]] = []
    done = False

    while not done:
        # Get current simulation time and phase from the underlying sumo-rl env
        sumo_env = env.unwrapped
        sim_time = sumo_env.sim_step if hasattr(sumo_env, "sim_step") else 0
        # sumo-rl traffic signal phase
        ts = next(iter(sumo_env.traffic_signals.values())) if hasattr(sumo_env, "traffic_signals") else None
        phase = ts.green_phase if ts is not None else 0
        phases.append((float(sim_time), int(phase)))

        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

    env.close()
    return phases


def generate_fixed_phases(cycle: int = 90, ns_green: int = 42, yellow: int = 3) -> list[tuple[float, int]]:
    """Generate fixed-time phase sequence for 3600s.

    Phase mapping: 0=N-S green, 1=yellow, 2=E-W green, 3=yellow.
    """
    phases: list[tuple[float, int]] = []
    t = 0.0
    while t < 3600:
        phases.append((t, 0))           # N-S green
        t += ns_green
        phases.append((t, 1))           # yellow
        t += yellow
        phases.append((t, 2))           # E-W green
        t += ns_green
        phases.append((t, 3))           # yellow
        t += yellow
    return phases


def compute_phase_stats(phases: list[tuple[float, int]], total_time: float = 3600.0) -> dict:
    """Compute statistics from phase log.

    Returns:
        Dict with green times, switch count, longest green, phase distribution.
    """
    ns_green_total = 0.0
    ew_green_total = 0.0
    yellow_total = 0.0
    switches = 0
    longest_ns = 0.0
    longest_ew = 0.0

    for i in range(len(phases)):
        start = phases[i][0]
        end = phases[i + 1][0] if i + 1 < len(phases) else total_time
        duration = end - start
        phase = phases[i][1]

        if phase == 0:
            ns_green_total += duration
            longest_ns = max(longest_ns, duration)
        elif phase == 2:
            ew_green_total += duration
            longest_ew = max(longest_ew, duration)
        else:
            yellow_total += duration

        if i > 0 and phases[i][1] != phases[i - 1][1]:
            switches += 1

    return {
        "ns_green_total": ns_green_total,
        "ew_green_total": ew_green_total,
        "yellow_total": yellow_total,
        "ns_green_pct": ns_green_total / total_time * 100,
        "ew_green_pct": ew_green_total / total_time * 100,
        "yellow_pct": yellow_total / total_time * 100,
        "ns_green_avg": ns_green_total / max(1, sum(1 for _, p in phases if p == 0)),
        "ew_green_avg": ew_green_total / max(1, sum(1 for _, p in phases if p == 2)),
        "switches_per_hour": switches,
        "longest_ns_green": longest_ns,
        "longest_ew_green": longest_ew,
    }


def main() -> None:
    """Entry point for signal plan analysis."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Analyze trained agent signal plan")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model .zip")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")
    parser.add_argument("--net-file", type=Path, default=None, help="SUMO network file")
    parser.add_argument("--route-file", type=Path, default=None, help="SUMO route file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("results") / "signal_plans"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model.stem

    print(f"Running evaluation episode with model: {args.model}")
    agent_phases = run_episode_with_phases(
        args.model, seed=args.seed, net_file=args.net_file, route_file=args.route_file,
    )

    # Save phase log to CSV
    csv_path = output_dir / f"signal_plan_{model_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sim_time", "phase"])
        writer.writerows(agent_phases)
    print(f"Phase log saved: {csv_path}")

    # Generate fixed-time reference
    fixed_phases = generate_fixed_phases()

    # Compute statistics
    agent_stats = compute_phase_stats(agent_phases)
    fixed_stats = compute_phase_stats(fixed_phases)

    # Print comparison
    print("\n=== Signalplan-Vergleich ===")
    print(f"{'':20s} {'Fixed-Time':>15s} {'RL-Agent':>15s}")
    print("-" * 52)
    print(f"{'N-S Grün (gesamt)':20s} {fixed_stats['ns_green_total']:>12.0f}s {agent_stats['ns_green_total']:>12.0f}s")
    print(f"{'E-W Grün (gesamt)':20s} {fixed_stats['ew_green_total']:>12.0f}s {agent_stats['ew_green_total']:>12.0f}s")
    print(f"{'Gelb (gesamt)':20s} {fixed_stats['yellow_total']:>12.0f}s {agent_stats['yellow_total']:>12.0f}s")
    print(f"{'N-S Grün (%)':20s} {fixed_stats['ns_green_pct']:>12.1f}% {agent_stats['ns_green_pct']:>12.1f}%")
    print(f"{'E-W Grün (%)':20s} {fixed_stats['ew_green_pct']:>12.1f}% {agent_stats['ew_green_pct']:>12.1f}%")
    print(f"{'N-S Grün (Ø)':20s} {fixed_stats['ns_green_avg']:>12.1f}s {agent_stats['ns_green_avg']:>12.1f}s")
    print(f"{'E-W Grün (Ø)':20s} {fixed_stats['ew_green_avg']:>12.1f}s {agent_stats['ew_green_avg']:>12.1f}s")
    print(f"{'Umschaltungen/h':20s} {fixed_stats['switches_per_hour']:>12d} {agent_stats['switches_per_hour']:>12d}")
    print(f"{'Längste N-S Grün':20s} {fixed_stats['longest_ns_green']:>12.0f}s {agent_stats['longest_ns_green']:>12.0f}s")
    print(f"{'Längste E-W Grün':20s} {fixed_stats['longest_ew_green']:>12.0f}s {agent_stats['longest_ew_green']:>12.0f}s")

    # Generate plot
    plot_path = output_dir / f"signal_plan_{model_name}.png"
    plot_signal_plan(agent_phases, fixed_phases, plot_path)
    print(f"\nSignalplan-Plot: {plot_path}")


if __name__ == "__main__":
    main()
