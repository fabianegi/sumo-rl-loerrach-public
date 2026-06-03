"""Generate demand-profile figures for documentation section 3.

Reads the disaggregated hourly demand profiles
(``data/processed/demand_profiles/``) and renders two figures into
``docs/figures/``:

* ``demand_profile_weekday.png`` -- medium weekday daily profile (absolute
  veh/h), via the shared ``plot_demand_profile`` utility.
* ``demand_scenarios.png`` -- low/medium/high scenario overlay.

Run: ``PYTHONPATH=. python scripts/generate_demand_plots.py``
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from src.config.settings import CONFIG, PROJECT_ROOT
from src.utils.plotting import plot_demand_profile

logger = logging.getLogger(__name__)

PROFILE_DIR = CONFIG.paths.processed_dir / "demand_profiles"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

SCENARIOS: dict[str, str] = {
    "low": "Low (x0,60)",
    "medium": "Medium (x1,00)",
    "high": "High (x1,35)",
}
COLORS: dict[str, str] = {"low": "#4CAF50", "medium": "#2196F3", "high": "#F44336"}


def _load_profile(scenario: str) -> pd.Series:
    """Return hourly ``total_veh_per_hour`` indexed 0..23 for a scenario."""
    df = pd.read_csv(PROFILE_DIR / f"demand_profile_{scenario}.csv")
    return df.set_index("hour")["total_veh_per_hour"].reindex(range(24)).astype(float)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    profiles = {name: _load_profile(name) for name in SCENARIOS}
    for name, series in profiles.items():
        logger.info("%s: peak %d veh/h at %02d:00", name, int(series.max()), int(series.idxmax()))

    # Figure 3.1 -- weekday profile (medium), reuse project utility
    plot_demand_profile(
        profiles["medium"].to_numpy(),
        FIG_DIR / "demand_profile_weekday.png",
    )

    # Figure 3.2 -- scenario overlay
    hours = list(range(24))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(16, 18, color="#FFC107", alpha=0.15, label="Nachmittagsspitze 16-18 h")
    for name, label in SCENARIOS.items():
        ax.plot(hours, profiles[name].to_numpy(), marker="o", ms=3, lw=1.8,
                color=COLORS[name], label=label)
    ax.set_xlabel("Uhrzeit [h]")
    ax.set_ylabel("Verkehrsstaerke [Fzg/h]")
    ax.set_title("Nachfrageszenarien -- Low / Medium / High (Basler Strasse)")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "demand_scenarios.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scenario figure saved: %s", out)


if __name__ == "__main__":
    main()
