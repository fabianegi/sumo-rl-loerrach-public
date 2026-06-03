"""Idempotent data initialization for sumo-rl-loerrach.

Generates all derived data files from calibration parameters:
1. Synthetic BASt hourly CSV (DTV 8037, HBS-2015 profile, seed=42)
2. Demand profile CSVs (low / medium / high)
3. SUMO .rou.xml files (low / medium / high)

Usage:
    python scripts/init_data.py          # only missing files
    python scripts/init_data.py --force  # regenerate all
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Calibration constants (from docs/documentation/03_data_and_environment.md) ---

DTV = 8037  # Daily Traffic Volume (MobiData BW, Loerrach-Stetten 2024)

# HBS-2015 hourly distribution for urban Hauptverkehrsstrasse (% of DTV per hour)
HBS_HOURLY_PCT = [
    1.5, 0.8, 0.5, 0.4, 0.6, 1.5,  # 00-05
    4.0, 7.5, 7.0, 5.5, 5.0, 5.5,  # 06-11
    6.0, 5.5, 5.5, 6.5, 8.0, 9.6,  # 12-17
    7.5, 5.0, 3.5, 2.5, 2.0, 1.6,  # 18-23
]

# Day-of-week factors (BASt convention: 1=Mon ... 7=Sun)
DOW_FACTORS = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.85, 7: 0.65}

# Direction split
BASLER_SHARE = 0.60  # N-S axis (Basler Strasse)
QUER_SHARE = 0.40    # E-W axis (Querstrasse)
# Within each axis: slight N/S and E/W asymmetry
DIR_SPLIT = {"basler_n": 0.30, "basler_s": 0.30, "quer_e": 0.20, "quer_w": 0.20}

# Turning ratios
STRAIGHT = 0.60
LEFT = 0.20
RIGHT = 0.20

# Scenario scaling factors
SCENARIOS = {"low": 0.60, "medium": 1.00, "high": 1.35}

# R1/R2 directional split for BASt CSV (52/48 slight asymmetry)
R1_SHARE = 0.52
R2_SHARE = 0.48

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
NC = "\033[0m"

# --- Output paths ---

BAST_CSV = PROJECT_ROOT / "data" / "raw" / "bast" / "b317_loerrach_stundenwerte_2023_synthetic.csv"
DEMAND_DIR = PROJECT_ROOT / "data" / "processed" / "demand_profiles"
SUMO_DIR = PROJECT_ROOT / "data" / "sumo_config"

# 12 OD pairs: (flow_id, route_edges, direction_key, turn_fraction)
# direction_key maps to DIR_SPLIT for the origin direction
OD_PAIRS = [
    # From North (basler_n)
    ("N_S_flow", "N_C C_S", "basler_n", STRAIGHT),  # straight
    ("N_E_flow", "N_C C_E", "basler_n", RIGHT),      # right turn
    ("N_W_flow", "N_C C_W", "basler_n", LEFT),       # left turn
    # From South (basler_s)
    ("S_N_flow", "S_C C_N", "basler_s", STRAIGHT),
    ("S_W_flow", "S_C C_W", "basler_s", RIGHT),
    ("S_E_flow", "S_C C_E", "basler_s", LEFT),
    # From East (quer_e)
    ("E_W_flow", "E_C C_W", "quer_e", STRAIGHT),
    ("E_N_flow", "E_C C_N", "quer_e", RIGHT),
    ("E_S_flow", "E_C C_S", "quer_e", LEFT),
    # From West (quer_w)
    ("W_E_flow", "W_C C_E", "quer_w", STRAIGHT),
    ("W_S_flow", "W_C C_S", "quer_w", RIGHT),
    ("W_N_flow", "W_C C_N", "quer_w", LEFT),
]

# Route definitions (id -> edges) matching existing loerrach.rou.xml
ROUTES = [
    ("N_S", "N_C C_S"),
    ("N_E", "N_C C_E"),
    ("N_W", "N_C C_W"),
    ("S_N", "S_C C_N"),
    ("S_W", "S_C C_W"),
    ("S_E", "S_C C_E"),
    ("E_W", "E_C C_W"),
    ("E_N", "E_C C_N"),
    ("E_S", "E_C C_S"),
    ("W_E", "W_C C_E"),
    ("W_S", "W_C C_S"),
    ("W_N", "W_C C_N"),
]

# --- Corridor route definitions ---
# Through-traffic traverses all 3 intersections; local traffic enters/exits at flanking junctions
# Demand split: 70% through-traffic on main axis, 15% enter/exit at each flanking junction
CORRIDOR_THROUGH_SHARE = 0.70
CORRIDOR_FLANK_SHARE = 0.15  # per flanking junction
CORRIDOR_CROSS_RATIO = 0.30  # flanking cross-street demand = 30% of center

CORRIDOR_ROUTES = [
    # Through-traffic N→S and S→N (full corridor)
    ("cor_N_S", "N_Jn Jn_C C_Js Js_S"),
    ("cor_S_N", "S_Js Js_C C_Jn Jn_N"),
    # Center cross-traffic (at C)
    ("cor_E_W", "E_C C_W"),
    ("cor_W_E", "W_C C_E"),
    # Flanking cross-traffic at J_N (Tumringer Str.)
    ("cor_Wn_En", "Wn_Jn Jn_En"),
    ("cor_En_Wn", "En_Jn Jn_Wn"),
    # Flanking cross-traffic at J_S (Brühlstr.)
    ("cor_Ws_Es", "Ws_Js Js_Es"),
    ("cor_Es_Ws", "Es_Js Js_Ws"),
    # N entering at J_N, turning onto cross streets
    ("cor_N_Wn", "N_Jn Jn_Wn"),
    ("cor_N_En", "N_Jn Jn_En"),
    # S entering at J_S, turning onto cross streets
    ("cor_S_Ws", "S_Js Js_Ws"),
    ("cor_S_Es", "S_Js Js_Es"),
    # From flanking cross streets, merging onto main axis
    ("cor_Wn_S", "Wn_Jn Jn_C C_Js Js_S"),
    ("cor_En_S", "En_Jn Jn_C C_Js Js_S"),
    ("cor_Ws_N", "Ws_Js Js_C C_Jn Jn_N"),
    ("cor_Es_N", "Es_Js Js_C C_Jn Jn_N"),
    # Center turns (N/S entering at corridor ends, turning at C)
    ("cor_N_E", "N_Jn Jn_C C_E"),
    ("cor_N_W", "N_Jn Jn_C C_W"),
    ("cor_S_E", "S_Js Js_C C_E"),
    ("cor_S_W", "S_Js Js_C C_W"),
    # From center cross streets into main axis
    ("cor_E_N", "E_C C_Jn Jn_N"),
    ("cor_E_S", "E_C C_Js Js_S"),
    ("cor_W_N", "W_C C_Jn Jn_N"),
    ("cor_W_S", "W_C C_Js Js_S"),
]


def should_create(path: Path, *, force: bool) -> bool:
    """Check if a file should be created (missing or --force)."""
    if force:
        return True
    return not path.exists()


# ---- Step 1: Synthetic BASt CSV ----

def generate_synthetic_bast(*, force: bool) -> str:
    """Generate synthetic BASt hourly traffic data for 2023.

    Returns:
        Status string: "CREATED (N rows)" or "SKIPPED (exists)".
    """
    if not should_create(BAST_CSV, force=force):
        return f"{YELLOW}SKIPPED{NC} (exists)"

    rng = np.random.default_rng(42)
    hourly_factors = np.array(HBS_HOURLY_PCT) / 100.0

    rows: list[dict] = []
    start = datetime.date(2023, 1, 1)

    for day_offset in range(365):
        date = start + datetime.timedelta(days=day_offset)
        # BASt Wotag: 1=Mon ... 7=Sun (Python weekday: 0=Mon ... 6=Sun)
        wotag = date.weekday() + 1
        dow_factor = DOW_FACTORS[wotag]
        date_str = date.strftime("%d.%m.%Y")

        for hour in range(24):
            base_count = DTV * hourly_factors[hour] * dow_factor
            # Add ±5% Gaussian noise
            noise = rng.normal(1.0, 0.05)
            total = max(0, round(base_count * noise))
            r1 = round(total * R1_SHARE)
            r2 = total - r1

            rows.append({
                "Datum": date_str,
                "Wotag": wotag,
                "Stunde": hour,
                "KFZ_R1": r1,
                "KFZ_R2": r2,
                "KFZ_Gesamt": total,
            })

    df = pd.DataFrame(rows)
    BAST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BAST_CSV, sep=";", index=False)
    return f"{GREEN}CREATED{NC} ({len(df)} rows)"


# ---- Step 2: Demand Profile CSVs ----

def generate_demand_profiles(*, force: bool) -> dict[str, str]:
    """Generate 24-hour demand profiles for low/medium/high scenarios.

    Returns:
        Dict mapping scenario name to status string.
    """
    statuses: dict[str, str] = {}

    # Check if any need creating (we need the BASt CSV for all)
    paths = {s: DEMAND_DIR / f"demand_profile_{s}.csv" for s in SCENARIOS}
    needs_work = any(should_create(p, force=force) for p in paths.values())

    if not needs_work:
        for scenario in SCENARIOS:
            statuses[scenario] = f"{YELLOW}SKIPPED{NC} (exists)"
        return statuses

    # Load synthetic BASt data
    bast_df = pd.read_csv(BAST_CSV, sep=";")

    # Filter weekdays only (Wotag 1-5) and compute hourly averages
    weekday_df = bast_df[bast_df["Wotag"] <= 5]
    hourly_avg = weekday_df.groupby("Stunde")["KFZ_Gesamt"].mean()

    DEMAND_DIR.mkdir(parents=True, exist_ok=True)

    for scenario, scale in SCENARIOS.items():
        path = paths[scenario]
        if not should_create(path, force=force):
            statuses[scenario] = f"{YELLOW}SKIPPED{NC} (exists)"
            continue

        rows: list[dict] = []
        for hour in range(24):
            total = round(hourly_avg.iloc[hour] * scale)
            rows.append({
                "hour": hour,
                "total_veh_per_hour": total,
                "basler_n": round(total * DIR_SPLIT["basler_n"]),
                "basler_s": round(total * DIR_SPLIT["basler_s"]),
                "quer_e": round(total * DIR_SPLIT["quer_e"]),
                "quer_w": round(total * DIR_SPLIT["quer_w"]),
            })

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        statuses[scenario] = f"{GREEN}CREATED{NC} (24 rows)"

    return statuses


# ---- Step 3: SUMO .rou.xml files ----

def _build_rou_xml(demand_profile: pd.DataFrame) -> str:
    """Build .rou.xml content from a demand profile DataFrame.

    Uses hour 17 (rush-hour peak, 17:00-18:00) for the 1h simulation.
    """
    rush_hour = demand_profile[demand_profile["hour"] == 17].iloc[0]
    dir_flows = {
        "basler_n": int(rush_hour["basler_n"]),
        "basler_s": int(rush_hour["basler_s"]),
        "quer_e": int(rush_hour["quer_e"]),
        "quer_w": int(rush_hour["quer_w"]),
    }
    total = int(rush_hour["total_veh_per_hour"])

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- Generated by scripts/init_data.py - {total} veh/h rush-hour -->",
        "<routes>",
        '    <!-- Vehicle type: standard passenger car -->',
        '    <vType id="car" vClass="passenger" length="5.0" minGap="2.5"',
        '           maxSpeed="13.89" accel="2.6" decel="4.5" sigma="0.5"/>',
        "",
        "    <!-- Routes (all 12 OD pairs) -->",
    ]

    for route_id, edges in ROUTES:
        lines.append(f'    <route id="{route_id}" edges="{edges}"/>')

    lines.append("")
    lines.append("    <!-- Flows (rush-hour 17:00-18:00) -->")

    for flow_id, _edges, dir_key, turn_frac in OD_PAIRS:
        # Route ID matches the first part of the flow ID (e.g., N_S_flow -> N_S)
        route_id = flow_id.replace("_flow", "")
        vph = round(dir_flows[dir_key] * turn_frac)
        lines.append(
            f'    <flow id="{flow_id}" type="car" route="{route_id}" '
            f'begin="0" end="3600" vehsPerHour="{vph}" '
            f'departLane="best" departSpeed="max"/>'
        )

    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def generate_rou_xml(*, force: bool) -> dict[str, str]:
    """Generate SUMO .rou.xml files for low/medium/high scenarios.

    Returns:
        Dict mapping scenario name to status string.
    """
    statuses: dict[str, str] = {}
    paths = {s: SUMO_DIR / f"loerrach_{s}.rou.xml" for s in SCENARIOS}

    for scenario in SCENARIOS:
        path = paths[scenario]
        if not should_create(path, force=force):
            statuses[scenario] = f"{YELLOW}SKIPPED{NC} (exists)"
            continue

        profile_csv = DEMAND_DIR / f"demand_profile_{scenario}.csv"
        if not profile_csv.exists():
            statuses[scenario] = f"\033[91mERROR{NC} (profile CSV missing)"
            continue

        profile_df = pd.read_csv(profile_csv)
        xml_content = _build_rou_xml(profile_df)

        path.write_text(xml_content, encoding="utf-8")

        # Extract total for report
        rush = profile_df[profile_df["hour"] == 17].iloc[0]
        total = int(rush["total_veh_per_hour"])
        statuses[scenario] = f"{GREEN}CREATED{NC} (12 flows, {total} veh/h)"

    return statuses


# ---- Step 4: Corridor .rou.xml files ----

def _build_corridor_rou_xml(demand_profile: pd.DataFrame) -> str:
    """Build corridor .rou.xml from demand profile.

    Uses hour 17 (rush-hour). Distributes traffic across through-routes,
    center cross-traffic, and flanking cross-traffic.
    """
    rush = demand_profile[demand_profile["hour"] == 17].iloc[0]
    ns_total = int(rush["basler_n"]) + int(rush["basler_s"])
    ew_center = int(rush["quer_e"]) + int(rush["quer_w"])
    total = int(rush["total_veh_per_hour"])

    # Through-traffic on main axis (70% of N-S)
    through_ns = round(ns_total * CORRIDOR_THROUGH_SHARE / 2)  # per direction
    # Center cross-traffic (same as single intersection)
    center_ew = round(ew_center / 2)  # per direction
    # Flanking cross-traffic (30% of center)
    flank_ew = round(center_ew * CORRIDOR_CROSS_RATIO)
    # Center turns from main axis
    center_turn = round(ns_total * 0.05)  # 5% of N-S traffic turns at C
    # Flanking turns
    flank_turn = round(ns_total * 0.03)  # 3% turns at flanking

    # Flow assignments: (route_id, vehsPerHour)
    flows = [
        # Through-traffic
        ("cor_N_S", through_ns),
        ("cor_S_N", through_ns),
        # Center cross-traffic
        ("cor_E_W", center_ew),
        ("cor_W_E", center_ew),
        # Flanking cross-traffic
        ("cor_Wn_En", flank_ew),
        ("cor_En_Wn", flank_ew),
        ("cor_Ws_Es", flank_ew),
        ("cor_Es_Ws", flank_ew),
        # Flanking turns from main axis
        ("cor_N_Wn", flank_turn),
        ("cor_N_En", flank_turn),
        ("cor_S_Ws", flank_turn),
        ("cor_S_Es", flank_turn),
        # From flanking cross to through
        ("cor_Wn_S", flank_turn),
        ("cor_En_S", flank_turn),
        ("cor_Ws_N", flank_turn),
        ("cor_Es_N", flank_turn),
        # Center turns
        ("cor_N_E", center_turn),
        ("cor_N_W", center_turn),
        ("cor_S_E", center_turn),
        ("cor_S_W", center_turn),
        # From center cross to main axis
        ("cor_E_N", center_turn),
        ("cor_E_S", center_turn),
        ("cor_W_N", center_turn),
        ("cor_W_S", center_turn),
    ]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- Generated by scripts/init_data.py - corridor {total} veh/h rush-hour -->",
        "<routes>",
        '    <vType id="car" vClass="passenger" length="5.0" minGap="2.5"',
        '           maxSpeed="13.89" accel="2.6" decel="4.5" sigma="0.5"/>',
        "",
        "    <!-- Corridor routes -->",
    ]
    for route_id, edges in CORRIDOR_ROUTES:
        lines.append(f'    <route id="{route_id}" edges="{edges}"/>')

    lines.append("")
    lines.append("    <!-- Flows (rush-hour 17:00-18:00) -->")

    for route_id, vph in flows:
        if vph > 0:
            lines.append(
                f'    <flow id="{route_id}_flow" type="car" route="{route_id}" '
                f'begin="0" end="3600" vehsPerHour="{vph}" '
                f'departLane="best" departSpeed="max"/>'
            )

    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def generate_corridor_rou_xml(*, force: bool) -> dict[str, str]:
    """Generate corridor .rou.xml files for low/medium/high scenarios.

    Returns:
        Dict mapping scenario name to status string.
    """
    statuses: dict[str, str] = {}
    paths = {s: SUMO_DIR / f"loerrach_corridor_{s}.rou.xml" for s in SCENARIOS}

    for scenario in SCENARIOS:
        path = paths[scenario]
        if not should_create(path, force=force):
            statuses[scenario] = f"{YELLOW}SKIPPED{NC} (exists)"
            continue

        profile_csv = DEMAND_DIR / f"demand_profile_{scenario}.csv"
        if not profile_csv.exists():
            statuses[scenario] = f"\033[91mERROR{NC} (profile CSV missing)"
            continue

        profile_df = pd.read_csv(profile_csv)
        xml_content = _build_corridor_rou_xml(profile_df)
        path.write_text(xml_content, encoding="utf-8")

        rush = profile_df[profile_df["hour"] == 17].iloc[0]
        total = int(rush["total_veh_per_hour"])
        statuses[scenario] = f"{GREEN}CREATED{NC} (corridor, {total} veh/h)"

    return statuses


# ---- Main ----

def main() -> None:
    """Entry point: generate all derived data files."""
    parser = argparse.ArgumentParser(
        description="Initialize all derived data files (idempotent)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all files even if they exist",
    )
    args = parser.parse_args()

    print("=== sumo-rl-loerrach Data Initialization ===")
    print(f"Project root: {PROJECT_ROOT}")
    if args.force:
        print("Mode: --force (regenerating all)")
    else:
        print("Mode: skip existing files")
    print()

    # Step 1: Synthetic BASt CSV
    print("--- Step 1: Synthetic BASt CSV ---")
    bast_status = generate_synthetic_bast(force=args.force)

    # Step 2: Demand profiles (depends on Step 1)
    print("--- Step 2: Demand Profile CSVs ---")
    profile_statuses = generate_demand_profiles(force=args.force)

    # Step 3: SUMO .rou.xml (depends on Step 2)
    print("--- Step 3: SUMO .rou.xml files ---")
    rou_statuses = generate_rou_xml(force=args.force)

    # Step 4: Corridor .rou.xml (depends on Step 2)
    print("--- Step 4: Corridor .rou.xml files ---")
    corridor_statuses = generate_corridor_rou_xml(force=args.force)

    # Report
    print()
    print("=== Data Initialization Report ===")
    print(f"  b317_synthetic.csv          {bast_status}")
    for scenario in SCENARIOS:
        print(f"  demand_profile_{scenario:<7s}.csv  {profile_statuses[scenario]}")
    for scenario in SCENARIOS:
        print(f"  loerrach_{scenario}.rou.xml      {rou_statuses[scenario]}")
    for scenario in SCENARIOS:
        print(f"  corridor_{scenario}.rou.xml      {corridor_statuses[scenario]}")
    print()


if __name__ == "__main__":
    main()
