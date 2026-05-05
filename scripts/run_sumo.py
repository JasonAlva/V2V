"""
run_sumo.py — Phase 1: SUMO simulation via TraCI
Runs the SUMO scenario and saves per-frame ground-truth state JSONs to
data/frames/frame_XXXXXX.json for every step of the simulation.

Usage:
    python scripts/run_sumo.py \
        --cfg sumo_scenario/scenario.sumocfg \
        --ego EGO_1 \
        --coop COOP_2 \
        --out data/frames \
        [--gui]

Fixes vs original:
  1. _get_vehicle_state() called only ONCE per vehicle in save_frame (was twice).
  2. step_length default unified to 0.05 s in both run() and argparse.
  3. Removed unused _vehicle_ids() helper.
  4. traci.close() now guarded so it only fires if traci.start() succeeded.
  5. Type hints use Optional[]/List[] (typing module) for Python 3.9 compat.
  6. all_vehicles built from a pre-fetched state cache — no double TraCI calls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

# ── SUMO / TraCI setup ──────────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME is not set. "
        "Export it to your SUMO installation directory, e.g.:\n"
        "  export SUMO_HOME=/usr/share/sumo"
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci          # noqa: E402  (must come after sys.path update)
import traci.constants as tc  # noqa: E402  (imported for downstream use)


# ── Camera / sensor geometry ─────────────────────────────────────────────────
CAMERA_INTRINSICS: Dict[str, float] = {
    "fx": 600.0,
    "fy": 600.0,
    "cx": 400.0,
    "cy": 300.0,
    "width": 800,
    "height": 600,
}
CAMERA_MOUNT: Dict[str, float] = {
    "x_offset": 0.0,
    "y_offset": 0.0,
    "z_offset": 1.5,
    "pitch_deg": -5.0,
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_vehicle_state(veh_id: str) -> Optional[Dict]:
    """
    Pull all ground-truth state for one vehicle via TraCI.
    Returns None if the vehicle is not currently in the simulation.

    heading_rad uses standard math convention (0 = east, CCW positive),
    converted from SUMO's convention (0 = north, CW positive).
    """
    if veh_id not in traci.vehicle.getIDList():
        return None

    x, y   = traci.vehicle.getPosition(veh_id)
    angle  = traci.vehicle.getAngle(veh_id)       # SUMO degrees, 0=north CW
    speed  = traci.vehicle.getSpeed(veh_id)
    length = traci.vehicle.getLength(veh_id)
    width  = traci.vehicle.getWidth(veh_id)
    height = traci.vehicle.getHeight(veh_id)
    vtype  = traci.vehicle.getTypeID(veh_id)
    lane_id = traci.vehicle.getLaneID(veh_id)
    road_id = traci.vehicle.getRoadID(veh_id)

    # SUMO: 0° = north, clockwise  →  math: 0° = east, counter-clockwise
    heading_rad = math.radians(90.0 - angle)

    return {
        "id": veh_id,
        "x": x,
        "y": y,
        "heading_deg": angle,        # raw SUMO convention (kept for reference)
        "heading_rad": heading_rad,  # standard math convention
        "speed_ms": speed,
        "length": length,
        "width": width,
        "height": height,
        "type": vtype,
        "lane_id": lane_id,
        "road_id": road_id,
    }


def _get_neighbourhood(
    ego_id: str,
    radius: float = 80.0,
    state_cache: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """
    Return ground-truth state for every vehicle within *radius* metres of ego.

    Accepts an optional pre-fetched state_cache (vid -> state dict) so the
    caller can avoid redundant TraCI calls when building all_vehicles.
    """
    if ego_id not in traci.vehicle.getIDList():
        return []

    ego_x, ego_y = traci.vehicle.getPosition(ego_id)
    neighbours: List[Dict] = []

    for vid in traci.vehicle.getIDList():
        if vid == ego_id:
            continue
        vx, vy = traci.vehicle.getPosition(vid)
        dist = math.hypot(vx - ego_x, vy - ego_y)
        if dist <= radius:
            # FIX 1: reuse cached state instead of calling _get_vehicle_state again
            state = (state_cache or {}).get(vid) or _get_vehicle_state(vid)
            if state:
                neighbour = dict(state)          # shallow copy to avoid mutation
                neighbour["dist_to_ego"] = dist
                neighbours.append(neighbour)

    return neighbours


def _frame_path(out_dir: str, step: int) -> str:
    return os.path.join(out_dir, f"frame_{step:06d}.json")


def _fetch_all_states() -> Dict[str, Dict]:
    """
    Fetch TraCI state for every vehicle currently in the simulation.
    Returns a dict keyed by vehicle ID so callers can reuse results without
    making duplicate TraCI queries.
    """
    cache: Dict[str, Dict] = {}
    for vid in traci.vehicle.getIDList():
        state = _get_vehicle_state(vid)
        if state is not None:
            cache[vid] = state
    return cache


def save_frame(
    out_dir: str,
    step: int,
    sim_time: float,
    ego_state: Optional[Dict],
    coop_state: Optional[Dict],
    gt_objects: List[Dict],
    all_vehicle_states: List[Dict],   # FIX 1: accept pre-built list
) -> None:
    """Serialise the full simulation state for this timestep to JSON."""
    frame = {
        "step": step,
        "sim_time": sim_time,
        "camera_intrinsics": CAMERA_INTRINSICS,
        "camera_mount": CAMERA_MOUNT,
        "ego": ego_state,
        "coop": coop_state,
        "gt_objects": gt_objects,
        "all_vehicles": all_vehicle_states,
    }
    with open(_frame_path(out_dir, step), "w") as fh:
        json.dump(frame, fh, indent=2)


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(
    cfg_path: str,
    ego_id: str,
    coop_id: str,
    out_dir: str,
    use_gui: bool = False,
    step_length: float = 0.05,       # FIX 2: unified default (was 0.1 here, 0.05 in CLI)
    detection_radius: float = 80.0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", cfg_path,
        "--step-length", str(step_length),
        "--collision.action", "warn",
        "--no-step-log", "true",
    ]
    if use_gui:
        sumo_cmd.append("--start")

    print(f"[run_sumo] Starting SUMO: {' '.join(sumo_cmd)}")

    # FIX 4: track whether TraCI started so finally block doesn't call
    # traci.close() on a connection that was never opened.
    traci_started = False
    step = 0
    saved = 0

    try:
        traci.start(sumo_cmd)
        traci_started = True

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            # FIX 1 & 6: fetch all states once per step into a cache dict.
            state_cache = _fetch_all_states()

            ego_state  = state_cache.get(ego_id)
            coop_state = state_cache.get(coop_id)

            if ego_state is not None:
                gt_objects = _get_neighbourhood(
                    ego_id,
                    radius=detection_radius,
                    state_cache=state_cache,
                )
                save_frame(
                    out_dir,
                    step,
                    sim_time,
                    ego_state,
                    coop_state,
                    gt_objects,
                    all_vehicle_states=list(state_cache.values()),  # reuse cache
                )
                saved += 1

                if step % 50 == 0:
                    n_vehs = len(state_cache)
                    print(
                        f"[run_sumo] step={step:6d}  t={sim_time:.1f}s  "
                        f"vehicles={n_vehs}  ego=✓  "
                        f"coop={'✓' if coop_state else '✗'}"
                    )

            step += 1

    finally:
        if traci_started:          # FIX 4: only close if open
            traci.close()
        print(f"[run_sumo] Done. {saved} frames saved to '{out_dir}'.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 – SUMO TraCI simulation loop")
    p.add_argument(
        "--cfg",
        default="sumo_scenario/scenario.sumocfg",
        help="Path to .sumocfg file",
    )
    p.add_argument(
        "--ego",
        default="EGO_1",
        help="SUMO vehicle ID of the ego agent",
    )
    p.add_argument(
        "--coop",
        default="COOP_2",
        help="SUMO vehicle ID of the cooperative agent (V2V peer)",
    )
    p.add_argument(
        "--out",
        default="data/frames",
        help="Output directory for frame JSONs",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Launch sumo-gui instead of headless sumo",
    )
    p.add_argument(
        "--step-length",
        type=float,
        default=0.05,               # FIX 2: matches run() default
        help="Simulation step length in seconds (default 0.05 s = 20 Hz)",
    )
    p.add_argument(
        "--radius",
        type=float,
        default=80.0,
        help="Neighbourhood detection radius around ego (metres)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        cfg_path=args.cfg,
        ego_id=args.ego,
        coop_id=args.coop,
        out_dir=args.out,
        use_gui=args.gui,
        step_length=args.step_length,
        detection_radius=args.radius,
    )