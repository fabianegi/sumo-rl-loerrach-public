"""Render a presentation demo: fixed-time baseline vs. trained DQN agent.

Part 5 of the audit/presentation work. Drives ``sumo-gui`` through TraCI on the
real OSM-imported network (``loerrach_osm``) and captures screenshots of the
*same* demand under two controllers:

* **baseline** - the OSM-default fixed-time signal program (``fixed_ts``),
* **agent**    - the DQN agent trained for 3M steps (``diff-waiting-time``).

Both runs use the identical route file and SUMO seed, so the visual difference
in queue length / standing vehicles is attributable to the controller alone.

For each capture time the script records the active and halted vehicle counts,
writes per-side PNGs, builds labelled side-by-side composites, and assembles an
animated GIF (a stand-in for the screen-recording in ``presentation/DEMO_PLAN.md``
since ffmpeg is not installed).

Usage::

    python scripts/render_demo.py                 # full run (baseline + agent + composites + gif)
    python scripts/render_demo.py --skip-agent     # re-render baseline only
    python scripts/render_demo.py --zoom 1300      # tune camera framing

The view is centred on the signalised junction (OSM TLS ``1628110071``) and
coloured by speed via ``data/sumo_config/demo_gui_settings.xml`` (red = standing,
green = free flow).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Real TraCI is required for GUI rendering; libsumo cannot drive sumo-gui.
os.environ.pop("LIBSUMO_AS_TRACI", None)

# Allow `python scripts/render_demo.py` from any CWD: put the project root (not
# the scripts/ dir) on sys.path so `import src...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Make SUMO's python tools (traci, sumolib) importable.
if "SUMO_HOME" in os.environ:
    sys.path.insert(0, str(Path(os.environ["SUMO_HOME"]) / "tools"))

import traci

from src.config.settings import CONFIG

logger = logging.getLogger(__name__)

# Camera target: signalised junction (OSM TLS 1628110071), net coordinates.
JUNCTION_X: float = 431.4
JUNCTION_Y: float = 331.2
DEMO_SCHEME: str = "by speed demo"  # defined in demo_gui_settings.xml
DEFAULT_VIEW: str = "View #0"


def _count_vehicles(conn: traci.connection.Connection) -> tuple[int, int]:
    """Return (active, halted) vehicle counts for the current step.

    Args:
        conn: Active TraCI connection.

    Returns:
        Tuple of total active vehicles and the subset standing (< 0.1 m/s).
    """
    ids = conn.vehicle.getIDList()
    halted = sum(1 for v in ids if conn.vehicle.getSpeed(v) < 0.1)
    return len(ids), halted


def _aim_camera(conn: traci.connection.Connection, zoom: float) -> None:
    """Point the GUI camera at the junction with the speed colour scheme.

    Args:
        conn: Active TraCI connection to a sumo-gui instance.
        zoom: SUMO zoom factor (larger = closer).
    """
    # Apply "real world" first (green map background + high-quality vehicle
    # polygons), then overlay our by-speed colour scheme. sumo-rl already sets
    # "real world" for the agent; doing it here too keeps both panels identical.
    conn.gui.setSchema(DEFAULT_VIEW, "real world")
    conn.gui.setSchema(DEFAULT_VIEW, DEMO_SCHEME)
    conn.gui.setOffset(DEFAULT_VIEW, JUNCTION_X, JUNCTION_Y)
    conn.gui.setZoom(DEFAULT_VIEW, zoom)


def render_baseline(
    times: list[int],
    out_dir: Path,
    *,
    seed: int,
    zoom: float,
    window: tuple[int, int],
) -> dict[int, dict]:
    """Render the fixed-time baseline and capture frames at ``times``.

    Uses plain ``sumo-gui`` via TraCI; the network's OSM-default signal program
    is used unchanged (no RL, no Webster tuning).

    Args:
        times: Sorted simulation times (s) at which to capture screenshots.
        out_dir: Directory for ``baseline_t####.png`` files.
        seed: SUMO seed (matched to the agent run for a fair comparison).
        zoom: Camera zoom factor.
        window: GUI window size (width, height) in pixels.

    Returns:
        Mapping of capture time to ``{"png", "active", "halted"}``.
    """
    cfg = CONFIG.paths.osm_sumo_cfg
    gui = CONFIG.paths.osm_gui_settings
    logger.info("Baseline: launching sumo-gui (seed=%d, zoom=%g)", seed, zoom)
    traci.start(
        [
            "sumo-gui",
            "-c", str(cfg),
            "--gui-settings-file", str(gui),
            "--window-size", f"{window[0]},{window[1]}",
            "--seed", str(seed),
            "--start", "--quit-on-end", "--delay", "0",
            "--no-step-log", "true", "--no-warnings", "true",
        ]
    )
    _aim_camera(traci, zoom)
    captured = _capture_loop(traci, times, out_dir, prefix="baseline")
    traci.close()
    return captured


def render_agent(
    times: list[int],
    out_dir: Path,
    *,
    seed: int,
    zoom: float,
    window: tuple[int, int],
    model_path: Path,
) -> dict[int, dict]:
    """Render the trained DQN agent and capture frames at ``times``.

    Builds a ``SumoEnvironment`` (sumo-rl) in GUI mode, loads the SB3 model, and
    drives signal decisions with ``model.predict`` (deterministic).

    Args:
        times: Sorted simulation times (s) at which to capture screenshots.
        out_dir: Directory for ``agent_t####.png`` files.
        seed: SUMO seed (matched to the baseline run).
        zoom: Camera zoom factor.
        window: GUI window size (width, height) in pixels.
        model_path: Path to the trained SB3 DQN ``.zip``.

    Returns:
        Mapping of capture time to ``{"png", "active", "halted"}``.
    """
    from stable_baselines3 import DQN
    from sumo_rl import SumoEnvironment

    from src.environment.rewards import diff_waiting_time_reward

    gui = CONFIG.paths.osm_gui_settings
    logger.info("Agent: building GUI env + loading %s", model_path.name)
    env = SumoEnvironment(
        net_file=str(CONFIG.paths.osm_net_file),
        route_file=str(CONFIG.paths.osm_route_file_medium),
        use_gui=True,
        num_seconds=CONFIG.sumo.num_seconds,
        delta_time=CONFIG.sumo.delta_time,
        yellow_time=CONFIG.sumo.yellow_time,
        min_green=CONFIG.sumo.min_green,
        reward_fn=diff_waiting_time_reward,
        sumo_seed=seed,
        single_agent=True,
        time_to_teleport=-1,
        additional_sumo_cmd=(
            f"--gui-settings-file {gui} "
            f"--window-size {window[0]},{window[1]} --delay 0"
        ),
    )
    model = DQN.load(str(model_path))
    obs, _ = env.reset()
    _aim_camera(env.sumo, zoom)

    targets = sorted(times)
    out_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[int, dict] = {}
    ti = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        now = int(env.sumo.simulation.getTime())
        while ti < len(targets) and now >= targets[ti]:
            t = targets[ti]
            png = out_dir / f"agent_t{t:04d}.png"
            active, halted = _count_vehicles(env.sumo)
            env.sumo.gui.screenshot(DEFAULT_VIEW, str(png))
            captured[t] = {"png": png.name, "active": active, "halted": halted}
            logger.info("  agent  t=%4d  active=%3d halted=%3d", t, active, halted)
            ti += 1
    env.close()
    return captured


def _capture_loop(
    conn: traci.connection.Connection,
    times: list[int],
    out_dir: Path,
    *,
    prefix: str,
) -> dict[int, dict]:
    """Step a plain TraCI simulation 1 s at a time, capturing at ``times``.

    Args:
        conn: Active TraCI connection.
        times: Sorted capture times (s).
        out_dir: Output directory for ``{prefix}_t####.png``.
        prefix: Filename prefix ("baseline").

    Returns:
        Mapping of capture time to ``{"png", "active", "halted"}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(times)
    captured: dict[int, dict] = {}
    ti = 0
    end = CONFIG.sumo.num_seconds
    for t in range(1, end + 1):
        if conn.simulation.getMinExpectedNumber() <= 0 and t > targets[-1]:
            break
        conn.simulationStep()
        if ti < len(targets) and t >= targets[ti]:
            tt = targets[ti]
            png = out_dir / f"{prefix}_t{tt:04d}.png"
            active, halted = _count_vehicles(conn)
            conn.gui.screenshot(DEFAULT_VIEW, str(png))
            captured[tt] = {"png": png.name, "active": active, "halted": halted}
            logger.info("  %s t=%4d  active=%3d halted=%3d", prefix, tt, active, halted)
            ti += 1
    # Flush the final queued screenshot (rendered on the following step).
    if ti > 0:
        conn.simulationStep()
    return captured


def _load_font(size: int):
    """Load a TrueType font, falling back to the PIL default.

    Args:
        size: Desired point size.

    Returns:
        A PIL ImageFont instance.
    """
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def compose(
    baseline: dict[int, dict],
    agent: dict[int, dict],
    out_dir: Path,
) -> list[Path]:
    """Build labelled side-by-side composites for each shared capture time.

    Args:
        baseline: Result mapping from :func:`render_baseline`.
        agent: Result mapping from :func:`render_agent`.
        out_dir: Directory for ``compare_t####.png`` files.

    Returns:
        List of composite PNG paths, ordered by time.
    """
    from PIL import Image, ImageDraw

    title_font = _load_font(34)
    sub_font = _load_font(26)
    pad, banner = 12, 96
    composites: list[Path] = []

    for t in sorted(set(baseline) & set(agent)):
        b, a = baseline[t], agent[t]
        bi = Image.open(out_dir / Path(b["png"]).name).convert("RGB")
        ai = Image.open(out_dir / Path(a["png"]).name).convert("RGB")
        h = min(bi.height, ai.height)
        bi = bi.resize((int(bi.width * h / bi.height), h))
        ai = ai.resize((int(ai.width * h / ai.height), h))

        w = bi.width + ai.width + pad
        canvas = Image.new("RGB", (w, h + banner), "white")
        canvas.paste(bi, (0, banner))
        canvas.paste(ai, (bi.width + pad, banner))

        d = ImageDraw.Draw(canvas)
        d.text((16, 10), "Festzeit-Baseline", font=title_font, fill=(180, 0, 0))
        d.text(
            (16, 52),
            f"t = {t} s   ·   {b['active']} Fzg aktiv, {b['halted']} stehend",
            font=sub_font, fill=(60, 60, 60),
        )
        d.text((bi.width + pad + 16, 10), "DQN-Agent (3M)", font=title_font, fill=(0, 130, 0))
        d.text(
            (bi.width + pad + 16, 52),
            f"t = {t} s   ·   {a['active']} Fzg aktiv, {a['halted']} stehend",
            font=sub_font, fill=(60, 60, 60),
        )
        d.line([(bi.width + pad // 2, banner), (bi.width + pad // 2, h + banner)],
               fill=(200, 200, 200), width=2)

        path = out_dir / f"compare_t{t:04d}.png"
        canvas.save(path)
        composites.append(path)
        logger.info("  composite -> %s", path.name)
    return composites


def make_gif(
    composites: list[Path],
    out_path: Path,
    *,
    fps: int = 6,
    max_width: int = 1600,
    hold_last_s: float = 1.5,
) -> None:
    """Assemble composites into a looping animated GIF.

    Args:
        composites: Ordered composite PNG paths.
        out_path: Destination ``.gif`` path.
        fps: Frames per second for the body of the animation.
        max_width: Downscale frames wider than this (keeps file size sane).
        hold_last_s: Extra seconds to hold the final (peak-contrast) frame
            before the loop restarts.
    """
    from PIL import Image

    if not composites:
        logger.warning("No composites to animate; skipping GIF.")
        return
    frames = []
    for p in composites:
        im = Image.open(p).convert("RGB")
        if im.width > max_width:
            h = round(im.height * max_width / im.width)
            im = im.resize((max_width, h), Image.LANCZOS)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))

    step_ms = int(1000 / fps)
    durations = [step_ms] * len(frames)
    durations[-1] += int(hold_last_s * 1000)  # linger on the final frame
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = out_path.stat().st_size / 1024
    logger.info("GIF (%d frames, %.0f KB) -> %s", len(frames), size_kb, out_path)


def main() -> None:
    """CLI entry point: render baseline + agent, compose, animate."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Seed 13 is one of the 16/30 eval seeds where the fixed-time baseline
    # collapses into gridlock (≈234 standing vehicles by t=3400) while the agent
    # keeps the junction flowing (≈3 standing) - i.e. it illustrates the headline
    # gridlock-avoidance finding and the divergence is visible for most of the
    # animation. SUMO junction congestion here is bistable in the seed (some seeds
    # flow under both controllers, e.g. seed 2/42; some gridlock under both, e.g.
    # seed 1). This single run is illustrative; the rigorous claim rests on the
    # 30-seed aggregate in results/OSM_NETWORK_RESULTS.md.
    parser.add_argument("--seed", type=int, default=13, help="SUMO seed for both runs.")
    parser.add_argument("--zoom", type=float, default=600.0, help="Camera zoom factor.")
    parser.add_argument("--width", type=int, default=1280, help="GUI window width (px).")
    parser.add_argument("--height", type=int, default=900, help="GUI window height (px).")
    parser.add_argument(
        "--interval", type=int, default=100,
        help="Seconds between captured frames (smaller = smoother animation).",
    )
    parser.add_argument("--gif-fps", type=int, default=6, help="Animation frames per second.")
    parser.add_argument(
        "--gif-width", type=int, default=1600,
        help="Max GIF width in px (downscaled to keep file size sane).",
    )
    parser.add_argument(
        "--model", type=Path,
        default=CONFIG.paths.models_dir / "dqn_diff-waiting-time_3000000steps_seed42.zip",
        help="Trained SB3 DQN model .zip.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=CONFIG.paths.results_dir.parent / "presentation" / "screenshots",
        help="Output directory.",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    window = (args.width, args.height)
    # Capture from the first interval up to just before episode end.
    times = list(range(args.interval, CONFIG.sumo.num_seconds, args.interval))
    logger.info("Capture times (s): %s", times)

    metrics_path = out_dir / "demo_metrics.json"
    metrics: dict[str, dict] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    if not args.skip_baseline:
        metrics["baseline"] = {
            str(k): v for k, v in render_baseline(
                times, out_dir, seed=args.seed, zoom=args.zoom, window=window
            ).items()
        }
    if not args.skip_agent:
        if not args.model.exists():
            raise FileNotFoundError(f"Model not found: {args.model}")
        metrics["agent"] = {
            str(k): v for k, v in render_agent(
                times, out_dir, seed=args.seed, zoom=args.zoom,
                window=window, model_path=args.model,
            ).items()
        }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics -> %s", metrics_path)

    if "baseline" in metrics and "agent" in metrics:
        baseline = {int(k): v for k, v in metrics["baseline"].items()}
        agent = {int(k): v for k, v in metrics["agent"].items()}
        composites = compose(baseline, agent, out_dir)
        make_gif(
            composites, out_dir / "demo.gif",
            fps=args.gif_fps, max_width=args.gif_width,
        )

    logger.info("Done. Artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
