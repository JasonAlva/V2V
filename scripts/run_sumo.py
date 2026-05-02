"""
run_sumo.py — Phase 1: SUMO simulation via TraCI
Runs the SUMO scenario and saves per-frame ground-truth state JSONs to
data/frames/frame_XXXXXX.json for every step of the simulation.

Usage:
    python scripts/run_sumo.py \
        --cfg sumo_scenario/scenario.sumocfg \
        --ego ego_vehicle \
        --coop coop_vehicle \
        --out data/frames \
        [--gui]
"""

import argparse
import json
import math
import os
import sys

# ── SUMO / TraCI setup ──────────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME is not set. "
        "Export it to your SUMO installation directory, e.g.:\n"
        "  export SUMO_HOME=/usr/share/sumo"
    )
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402  (must come after sys.path update)
import traci.constants as tc  # noqa: E402


# ── Camera / sensor geometry ─────────────────────────────────────────────────
# Intrinsics used by generate_images.py; stored in state so every downstream
# script can reconstruct the projection without extra config files.
CAMERA_INTRINSICS = {
    "fx": 600.0,   # focal length x  (pixels)
    "fy": 600.0,   # focal length y  (pixels)
    "cx": 400.0,   # principal point x
    "cy": 300.0,   # principal point y
    "width": 800,
    "height": 600,
}
# Camera mounted on the roof, looking forward (ego-vehicle frame)
CAMERA_MOUNT = {
    "x_offset": 0.0,   # metres forward of vehicle centre
    "y_offset": 0.0,   # metres left
    "z_offset": 1.5,   # metres above ground
    "pitch_deg": -5.0, # slight downward tilt
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vehicle_ids(veh_id: str) -> list[str]:
    """Return [veh_id] when the vehicle is present, else []."""
    return [veh_id] if veh_id in traci.vehicle.getIDList() else []


def _get_vehicle_state(veh_id: str) -> dict | None:
    """
    Pull all ground-truth state for one vehicle via TraCI.
    Returns None if the vehicle is not currently in the simulation.
    """
    if veh_id not in traci.vehicle.getIDList():
        return None

    x, y = traci.vehicle.getPosition(veh_id)        # SUMO XY (metres)
    angle = traci.vehicle.getAngle(veh_id)           # degrees, 0 = north, CW
    speed = traci.vehicle.getSpeed(veh_id)           # m/s
    length = traci.vehicle.getLength(veh_id)         # m
    width = traci.vehicle.getWidth(veh_id)           # m
    height = traci.vehicle.getHeight(veh_id)         # m
    vtype = traci.vehicle.getTypeID(veh_id)
    lane_id = traci.vehicle.getLaneID(veh_id)
    road_id = traci.vehicle.getRoadID(veh_id)

    # SUMO angle: 0° = north, clockwise → convert to standard math angle
    # (0° = east, counter-clockwise) for downstream geometry.
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


def _get_neighbourhood(ego_id: str, radius: float = 80.0) -> list[dict]:
    """
    Return ground-truth state for every vehicle within `radius` metres of ego.
    These are the "ground-truth objects" that perception should detect.
    """
    if ego_id not in traci.vehicle.getIDList():
        return []

    ego_x, ego_y = traci.vehicle.getPosition(ego_id)
    neighbours = []
    for vid in traci.vehicle.getIDList():
        if vid == ego_id:
            continue
        vx, vy = traci.vehicle.getPosition(vid)
        dist = math.hypot(vx - ego_x, vy - ego_y)
        if dist <= radius:
            state = _get_vehicle_state(vid)
            if state:
                state["dist_to_ego"] = dist
                neighbours.append(state)

    return neighbours


def _frame_path(out_dir: str, step: int) -> str:
    return os.path.join(out_dir, f"frame_{step:06d}.json")


def save_frame(
    out_dir: str,
    step: int,
    sim_time: float,
    ego_state: dict | None,
    coop_state: dict | None,
    gt_objects: list[dict],
) -> None:
    """Serialise the full simulation state for this timestep to JSON."""
    frame = {
        "step": step,
        "sim_time": sim_time,
        "camera_intrinsics": CAMERA_INTRINSICS,
        "camera_mount": CAMERA_MOUNT,
        "ego": ego_state,
        "coop": coop_state,
        "gt_objects": gt_objects,   # ground-truth neighbours for eval
        "all_vehicles": [           # full census (useful for visualise.py)
            _get_vehicle_state(vid)
            for vid in traci.vehicle.getIDList()
            if _get_vehicle_state(vid) is not None
        ],
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
    step_length: float = 0.1,
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
    traci.start(sumo_cmd)

    step = 0
    saved = 0

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            ego_state = _get_vehicle_state(ego_id)
            coop_state = _get_vehicle_state(coop_id)

            # Only record frames where at least the ego vehicle is present
            if ego_state is not None:
                gt_objects = _get_neighbourhood(ego_id, radius=detection_radius)
                save_frame(out_dir, step, sim_time, ego_state, coop_state, gt_objects)
                saved += 1

                if step % 50 == 0:
                    n_vehs = len(traci.vehicle.getIDList())
                    print(
                        f"[run_sumo] step={step:6d}  t={sim_time:.1f}s  "
                        f"vehicles={n_vehs}  ego={'✓' if ego_state else '✗'}  "
                        f"coop={'✓' if coop_state else '✗'}"
                    )

            step += 1

    finally:
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
        default="ego_vehicle",
        help="SUMO vehicle ID of the ego agent",
    )
    p.add_argument(
        "--coop",
        default="coop_vehicle",
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
        default=0.05,
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