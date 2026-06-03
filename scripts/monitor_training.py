"""Live training dashboard - watches TrafficMetricsCallback CSV output.

Usage:
    python scripts/monitor_training.py --log-dir logs/ppo_diff-waiting-time_500000steps_seed42/
    python scripts/monitor_training.py --log-dir logs/ --refresh 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_latest_csv(log_dir: Path) -> Path | None:
    """Find the most recent traffic_metrics.csv in log_dir tree."""
    csvs = sorted(log_dir.rglob("traffic_metrics.csv"), key=lambda p: p.stat().st_mtime)
    return csvs[-1] if csvs else None


def update_dashboard(csv_path: Path, fig: plt.Figure, axes: list, window: int = 50) -> int:
    """Read CSV and update the 3-subplot dashboard. Returns row count."""
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return 0

    if len(df) == 0:
        return 0

    for ax in axes:
        ax.clear()

    # Subplot 1: Episode reward
    axes[0].plot(df["episode"], df["reward_sum"], alpha=0.3, color="#2196F3", linewidth=0.5)
    if len(df) >= window:
        rolling = df["reward_sum"].rolling(window=window, min_periods=1).mean()
        axes[0].plot(df["episode"], rolling, color="#2196F3", linewidth=2,
                     label=f"Ø {window} Ep.")
    axes[0].set_ylabel("Episodenbelohnung")
    axes[0].set_title("Reward pro Episode")
    axes[0].legend(loc="upper left")

    # Subplot 2: Average waiting time
    if "avg_waiting_time" in df.columns:
        wt = pd.to_numeric(df["avg_waiting_time"], errors="coerce")
        axes[1].plot(df["episode"], wt, alpha=0.3, color="#F44336", linewidth=0.5)
        if len(df) >= window:
            rolling_wt = wt.rolling(window=window, min_periods=1).mean()
            axes[1].plot(df["episode"], rolling_wt, color="#F44336", linewidth=2)
        axes[1].set_ylabel("Ø Wartezeit [s]")
        axes[1].set_title("Durchschn. Wartezeit")

    # Subplot 3: Max queue
    if "max_queue" in df.columns:
        mq = pd.to_numeric(df["max_queue"], errors="coerce")
        axes[2].plot(df["episode"], mq, alpha=0.3, color="#FF9800", linewidth=0.5)
        if len(df) >= window:
            rolling_mq = mq.rolling(window=window, min_periods=1).mean()
            axes[2].plot(df["episode"], rolling_mq, color="#FF9800", linewidth=2)
        axes[2].set_ylabel("Max. Rückstau [Fzg]")
        axes[2].set_title("Max. Rückstaulänge")

    axes[2].set_xlabel("Episode")
    fig.tight_layout()
    return len(df)


def print_summary(csv_path: Path) -> None:
    """Print summary statistics to terminal."""
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return

    if len(df) == 0:
        return

    last_50 = df.tail(50)
    print(f"\n--- Training Status ({len(df)} Episoden) ---")
    print(f"  Letzte Episode:   {int(df['episode'].iloc[-1])}")
    print(f"  Letzter Timestep: {int(df['timestep'].iloc[-1])}")
    print(f"  Reward (Ø50):     {last_50['reward_sum'].mean():.1f}")

    if "avg_waiting_time" in df.columns:
        wt = pd.to_numeric(last_50["avg_waiting_time"], errors="coerce")
        if wt.notna().any():
            print(f"  Wartezeit (Ø50):  {wt.mean():.1f}s")
    if "max_queue" in df.columns:
        mq = pd.to_numeric(last_50["max_queue"], errors="coerce")
        if mq.notna().any():
            print(f"  Max Queue (Ø50):  {mq.mean():.0f}")


def main() -> None:
    """Entry point for live training monitor."""
    parser = argparse.ArgumentParser(description="Live training dashboard")
    parser.add_argument("--log-dir", type=Path, required=True, help="Log directory to watch")
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval in seconds")
    parser.add_argument("--window", type=int, default=50, help="Rolling average window")
    args = parser.parse_args()

    csv_path = find_latest_csv(args.log_dir)
    if csv_path is None:
        print(f"No traffic_metrics.csv found in {args.log_dir}")
        print("Start training first, then re-run this monitor.")
        return

    print(f"Monitoring: {csv_path}")
    print(f"Refresh: every {args.refresh}s | Window: {args.window} episodes")
    print("Press Ctrl+C to stop.\n")

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f"Training Monitor: {csv_path.parent.name}", fontsize=13)

    try:
        while True:
            n_rows = update_dashboard(csv_path, fig, list(axes), window=args.window)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            print_summary(csv_path)
            if n_rows > 0 and n_rows % 100 == 0:
                snapshot = csv_path.parent / f"training_progress_{n_rows}ep.png"
                fig.savefig(snapshot, dpi=150, bbox_inches="tight")
                print(f"  Snapshot saved: {snapshot}")
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
