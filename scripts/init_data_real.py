"""Idempotent data initialization for the REAL intersection
(Basler Str. × Obere Riehenstraße, Lörrach-Stetten).

Generates `.rou.xml` files for low/medium/high demand scenarios.
After Audit F2 (2026-05-19) the hourly distribution is derived from the
**measured BASt 2023 hourly series** (`data/raw/bast/...`), not the
HBS-2015 literature array. HBS remains as fallback.

Run after `netconvert` has produced `loerrach_real.net.xml`.

Usage:
    python scripts/init_data_real.py                 # only missing files
    python scripts/init_data_real.py --force         # regenerate all
    python scripts/init_data_real.py --use-hbs       # force HBS profile
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Calibration constants ---

DTV = 8037  # MobiData BW, Lörrach-Stetten 2024 (real BASt 2023 Zst 8570 ≈ 8291)

# HBS-2015 hourly distribution - kept as fallback only (Audit B2).
HBS_HOURLY_PCT = [
    1.5, 0.8, 0.5, 0.4, 0.6, 1.5,
    4.0, 7.5, 7.0, 5.5, 5.0, 5.5,
    6.0, 5.5, 5.5, 6.5, 8.0, 9.6,
    7.5, 5.0, 3.5, 2.5, 2.0, 1.6,
]

BAST_CSV = PROJECT_ROOT / "data" / "raw" / "bast" / "b317_zst8570_8921_2023.csv"
BAST_ZST_PRIMARY = "8570"  # Lörrach-Stetten ehem. B317; verified DTV ≈ 8291


def load_bast_hourly_pct(csv_path: Path = BAST_CSV, zst: str = BAST_ZST_PRIMARY) -> list[float]:
    """Return a 24-element list with the measured hourly share (in %) for
    BASt counting station ``zst``, averaged across all days in the file.

    BASt format: ``Stunde`` is 1..24 (1 = 00:00-01:00). We map back to 0..23.
    Raises ``FileNotFoundError`` if the CSV is missing so callers can fall
    back to ``HBS_HOURLY_PCT`` if desired.
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    totals: dict[int, int] = defaultdict(int)
    with csv_path.open(encoding="latin-1") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)
        idx = {col.strip(): i for i, col in enumerate(header)}
        zst_col = idx["Zst"]
        h_col = idx["Stunde"]
        r1_col, r2_col = idx["KFZ_R1"], idx["KFZ_R2"]
        for row in reader:
            if len(row) < max(zst_col, h_col, r1_col, r2_col) + 1:
                continue
            if row[zst_col].strip() != zst:
                continue
            try:
                h = int(row[h_col]) - 1  # BASt 1..24 → 0..23
                kfz = int(row[r1_col]) + int(row[r2_col])
            except ValueError:
                continue
            if 0 <= h <= 23:
                totals[h] += kfz

    total_all = sum(totals.values())
    if total_all == 0:
        raise ValueError(f"No KFZ data found for Zst {zst} in {csv_path}")
    return [100.0 * totals[h] / total_all for h in range(24)]


def get_hourly_pct(use_real: bool = True) -> tuple[list[float], str]:
    """Return (hourly_pct, provenance_label). Falls back to HBS if CSV missing."""
    if use_real:
        try:
            return load_bast_hourly_pct(), f"BASt 2023 Zst {BAST_ZST_PRIMARY} (gemessen)"
        except (FileNotFoundError, ValueError) as exc:
            print(f"[warn] real BASt profile unavailable ({exc}); fallback to HBS-2015")
    return HBS_HOURLY_PCT, "HBS-2015 (Literatur)"

BASLER_SHARE = 0.60   # N-S axis (Basler Straße, main)
QUER_SHARE = 0.40     # E-W axis (Obere Riehenstr., cross)

DIR_SPLIT = {"N": 0.30, "S": 0.30, "E": 0.20, "W": 0.20}

# Turning ratios
STRAIGHT = 0.60
LEFT = 0.20
RIGHT = 0.20

# Scenario scaling factors relative to the peak hour (17:00 in HBS profile)
SCENARIOS = {"low": 0.60, "medium": 1.00, "high": 1.35}

OUTPUT_DIR = PROJECT_ROOT / "data" / "sumo_config"

# 12 OD pairs for the real-network edges.
# Right-hand-drive cardinal directions (N=top, E=right, S=bottom, W=left).
# Vehicle entering at N is travelling south:
#   - straight = continue south  → C2S
#   - left turn (heading south, left=east)  → C2E
#   - right turn (heading south, right=west) → C2W
OD_PAIRS = [
    # (flow_id, route_edges, origin_direction_key, turn_fraction)
    ("N_S_flow", "N2C C2S", "N", STRAIGHT),  # N straight to S
    ("N_E_flow", "N2C C2E", "N", LEFT),       # N left to E
    ("N_W_flow", "N2C C2W", "N", RIGHT),      # N right to W
    ("S_N_flow", "S2C C2N", "S", STRAIGHT),
    ("S_E_flow", "S2C C2E", "S", RIGHT),
    ("S_W_flow", "S2C C2W", "S", LEFT),
    ("E_W_flow", "E2C C2W", "E", STRAIGHT),
    ("E_S_flow", "E2C C2S", "E", LEFT),
    ("E_N_flow", "E2C C2N", "E", RIGHT),
    ("W_E_flow", "W2C C2E", "W", STRAIGHT),
    ("W_N_flow", "W2C C2N", "W", LEFT),
    ("W_S_flow", "W2C C2S", "W", RIGHT),
]

ROUTE_DEFS = """\
    <!-- Routes (12 OD pairs across the 4-arm intersection) -->
    <route id="N_S" edges="N2C C2S"/>
    <route id="N_E" edges="N2C C2E"/>
    <route id="N_W" edges="N2C C2W"/>
    <route id="S_N" edges="S2C C2N"/>
    <route id="S_E" edges="S2C C2E"/>
    <route id="S_W" edges="S2C C2W"/>
    <route id="E_W" edges="E2C C2W"/>
    <route id="E_S" edges="E2C C2S"/>
    <route id="E_N" edges="E2C C2N"/>
    <route id="W_E" edges="W2C C2E"/>
    <route id="W_N" edges="W2C C2N"/>
    <route id="W_S" edges="W2C C2S"/>
"""


def compute_flow_vph(scenario: str, hourly_pct: list[float]) -> tuple[dict[str, int], int]:
    """Return ``({flow_id: vehicles_per_hour}, peak_hour_index)`` for the
    given scenario, using ``hourly_pct`` (24-element list, percent shares).

    Demand at peak = DTV × max(hourly_pct) × scenario_factor, distributed
    across approaches via ``DIR_SPLIT`` × turn fractions.
    """
    peak_hour = max(range(24), key=hourly_pct.__getitem__)
    peak_pct = hourly_pct[peak_hour] / 100.0
    scaled_total = DTV * peak_pct * SCENARIOS[scenario]

    flow_vph: dict[str, int] = {}
    for flow_id, _edges, origin_dir, turn_frac in OD_PAIRS:
        approach_share = DIR_SPLIT[origin_dir]
        veh = scaled_total * approach_share * turn_frac
        flow_vph[flow_id] = max(1, round(veh))
    return flow_vph, peak_hour


def render_route_file(
    scenario: str,
    flow_vph: dict[str, int],
    peak_hour: int,
    provenance: str,
) -> str:
    total = sum(flow_vph.values())
    flows_xml = "\n".join(
        f'    <flow id="{fid}" type="car" route="{fid[:-5]}"'
        f' begin="0" end="3600" vehsPerHour="{vph}"'
        f' departLane="best" departSpeed="max"/>'
        for fid, vph in flow_vph.items()
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by scripts/init_data_real.py - verankertes Modell
     Basler Str. × Obere Riehenstr., Lörrach-Stetten.
     Scenario: {scenario} (total ≈ {total} veh/h at peak hour {peak_hour}:00)
     Calibration: DTV {DTV} Kfz/24h × hourly profile [{provenance}]
                  × DIR_SPLIT × turn fractions.
-->
<routes>
    <vType id="car" vClass="passenger" length="5.0" minGap="2.5"
           maxSpeed="13.89" accel="2.6" decel="4.5" sigma="0.5"/>

{ROUTE_DEFS}
    <!-- Flows (constant rate over 3600s episode) -->
{flows_xml}
</routes>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if files exist")
    parser.add_argument("--use-hbs", action="store_true",
                        help="Force HBS-2015 literature profile (legacy)")
    args = parser.parse_args()

    hourly_pct, provenance = get_hourly_pct(use_real=not args.use_hbs)
    print(f"[profile] using {provenance}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        out = OUTPUT_DIR / f"loerrach_real_{scenario}.rou.xml"
        if out.exists() and not args.force:
            print(f"[skip]   {out.name} (exists; use --force to regenerate)")
            continue
        flow_vph, peak_hour = compute_flow_vph(scenario, hourly_pct)
        total = sum(flow_vph.values())
        out.write_text(
            render_route_file(scenario, flow_vph, peak_hour, provenance),
            encoding="utf-8",
        )
        print(f"[write]  {out.name}  →  {total} veh/h (peak h={peak_hour}:00)")


if __name__ == "__main__":
    main()
