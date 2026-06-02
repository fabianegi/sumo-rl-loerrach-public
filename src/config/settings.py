"""Central configuration for sumo-rl-loerrach project.

All magic numbers, paths, and hyperparameters live here.
Import this module everywhere instead of hardcoding values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Project root (2 levels up from this file: src/config/ -> src/ -> root)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class PathConfig:
    """All file paths, relative to PROJECT_ROOT."""

    data_dir: Path = PROJECT_ROOT / "data"
    sumo_config_dir: Path = PROJECT_ROOT / "data" / "sumo_config"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models" / "checkpoints"
    results_dir: Path = PROJECT_ROOT / "results"
    plots_dir: Path = PROJECT_ROOT / "results" / "plots"
    csv_dir: Path = PROJECT_ROOT / "results" / "csv"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # SUMO simulation files (single intersection)
    net_file: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach.net.xml"
    route_file: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach.rou.xml"
    sumo_cfg: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach.sumocfg"
    additional_file: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach.add.xml"

    # Corridor network (3 intersections)
    corridor_net_file: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_corridor.net.xml"
    corridor_route_file: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_corridor_medium.rou.xml"

    # Real intersection: Basler Str. x Obere Riehenstr. (Lörrach-Stetten)
    # Geometry derived from OSM (Overpass 2026-05-13); see DATA_SOURCES_RESEARCH.md
    real_net_file: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_real.net.xml"
    real_route_file_low: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_real_low.rou.xml"
    real_route_file_medium: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_real_medium.rou.xml"
    real_route_file_high: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_real_high.rou.xml"
    real_sumo_cfg: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_real.sumocfg"

    # Real OSM-imported geometry: Basler Str. x Obere Riehenstr. (Audit F3)
    # netconvert --osm-files (Overpass 2026-05-20); single TLS OSM-ID 1628110071.
    # This is the headline network for the DQN-3M result (results/OSM_NETWORK_RESULTS.md).
    osm_net_file: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_osm.net.xml"
    osm_route_file_medium: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_osm_medium.rou.xml"
    osm_sumo_cfg: Path = PROJECT_ROOT / "data" / "sumo_config" / "loerrach_osm.sumocfg"
    osm_gui_settings: Path = PROJECT_ROOT / "data" / "sumo_config" / "demo_gui_settings.xml"


@dataclass(frozen=True)
class SumoConfig:
    """SUMO simulation parameters."""

    intersection_id: str = "C"  # TLS ID from SUMO network (central junction)
    num_seconds: int = 3600  # Simulation duration per episode (s)
    begin_time: int = 0  # Simulation start time (s)
    warmup_steps: int = 300  # Warmup before RL starts (s)

    # Traffic signal parameters
    delta_time: int = 5  # RL decision frequency (s)
    yellow_time: int = 3  # Yellow phase duration (s)
    min_green: int = 10  # Minimum green time (s)
    max_green: int = 60  # Maximum green time (s)

    # Rendering
    use_gui: bool = False  # Set True for debugging only


@dataclass(frozen=True)
class DQNConfig:
    """DQN hyperparameters (stable-baselines3)."""

    learning_rate: float = 1e-3
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    batch_size: int = 64
    gamma: float = 0.99
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.05
    target_update_interval: int = 1_000
    train_freq: int = 4
    total_timesteps: int = 1_000_000


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters (stable-baselines3)."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    clip_range: float = 0.2
    total_timesteps: int = 1_000_000


@dataclass(frozen=True)
class TrainingConfig:
    """Training and evaluation settings."""

    seed: int = 42
    n_eval_episodes: int = 30  # For statistical validity
    eval_seeds: list[int] = field(default_factory=lambda: list(range(30)))
    demand_scenarios: list[str] = field(
        default_factory=lambda: ["low", "medium", "high"]
    )
    device: str = "auto"  # "cpu", "cuda", "mps", "auto"
    checkpoint_freq: int = 50_000  # Save model every N steps
    log_dir: Path = PROJECT_ROOT / "logs"
    tensorboard_log: Path = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class ProjectConfig:
    """Master config - instantiate this."""

    paths: PathConfig = field(default_factory=PathConfig)
    sumo: SumoConfig = field(default_factory=SumoConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Singleton for import everywhere
CONFIG: Final[ProjectConfig] = ProjectConfig()
