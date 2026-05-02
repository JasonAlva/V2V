"""Run SUMO and generate outputs live (images, detections, fusion, BEV)."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import concurrent.futures
import threading
import time
from pathlib import Path
from typing import Any

import cv2

import fuse_ego_coop
import generate_images
import project_3d
import run_detection
import visualize
import ws_server


if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME is not set. "
        "Export it to your SUMO installation directory, e.g.:\n"
        "  export SUMO_HOME=/usr/share/sumo"
    )

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci  # noqa: E402


CAMERA_INTRINSICS = {
    "fx": 600.0,
    "fy": 600.0,
    "cx": 400.0,
    "cy": 300.0,
    "width": 800,
    "height": 600,
}
CAMERA_MOUNT = {
    "x_offset": 0.0,
    "y_offset": 0.0,
    "z_offset": 1.5,
    "pitch_deg": -5.0,
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
}


def _load_roles(live_dir: Path) -> dict[str, Any] | None:
    """Load vehicle roles from roles.json. Returns dict with ego, coops, coop_radius."""
    roles_file = live_dir / "roles.json"
    if roles_file.exists():
        try:
            return json.loads(roles_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _write_roles_atomic(live_dir: Path, ego_id: str, coop_ids: list[str], coop_radius: float = 80.0) -> None:
    """Atomically write roles.json to avoid partial reads during dynamic updates."""
    roles = {
        "ego": ego_id,
        "coops": coop_ids,
        "coop_radius": coop_radius,
    }
    roles_file = live_dir / "roles.json"
    tmp_file = live_dir / "roles.json.tmp"
    tmp_file.write_text(json.dumps(roles, indent=2), encoding="utf-8")
    try:
        tmp_file.replace(roles_file)
    except OSError:
        roles_file.write_text(json.dumps(roles, indent=2), encoding="utf-8")
        tmp_file.unlink(missing_ok=True)


def update_roles(ego_id: str, coop_ids: list[str], coop_radius: float = 80.0) -> None:
    """
    Helper function that any script can call to update vehicle roles dynamically.
    This enables changing ego/coop assignments at runtime without restarting the pipeline.

    Args:
        ego_id: Vehicle ID to become the primary observer
        coop_ids: List of cooperative vehicle IDs
        coop_radius: Radius for neighborhood detection (default 80m)
    """
    repo_root = Path(__file__).resolve().parent.parent
    live_dir = repo_root / "data/live"
    live_dir.mkdir(parents=True, exist_ok=True)
    _write_roles_atomic(live_dir, ego_id, coop_ids, coop_radius)


def assign_coops_by_proximity(
    frame_data: dict[str, Any],
    ego_id: str,
    radius_m: float,
    max_coops: int = 3,
) -> list[str]:
    """Return nearest `max_coops` vehicle IDs to `ego_id` within `radius_m` from frame_data.

    Also writes the resulting coops list to `data/live/roles.json` atomically via update_roles().
    """
    all_vehicles = frame_data.get("all_vehicles") or []
    ego_obj = next((v for v in all_vehicles if str(v.get("id")) == str(ego_id)), None)
    if ego_obj is None:
        return []

    ex = float(ego_obj.get("x", 0.0))
    ey = float(ego_obj.get("y", 0.0))

    candidates: list[tuple[float, str]] = []
    for v in all_vehicles:
        vid = v.get("id")
        if vid is None:
            continue
        if str(vid) == str(ego_id):
            continue
        try:
            vx = float(v.get("x", 0.0))
            vy = float(v.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        dist = math.hypot(vx - ex, vy - ey)
        if dist <= float(radius_m):
            candidates.append((dist, str(vid)))

    candidates.sort(key=lambda t: t[0])
    selected = [vid for _, vid in candidates[: int(max_coops)]]

    # FIX #6: Log exceptions instead of silently swallowing them.
    try:
        update_roles(ego_id=str(ego_id), coop_ids=selected, coop_radius=float(radius_m))
    except Exception as exc:
        print(f"[live] Warning: failed to persist roles for ego={ego_id}: {exc}")

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO + outputs live.")

    parser.add_argument("--cfg", default="sumo_scenario/scenario.sumocfg", help="SUMO config path")
    parser.add_argument("--ego", default="ego_vehicle", help="Ego vehicle ID")
    parser.add_argument("--coop", default="coop_vehicle", help="Cooperative vehicle ID")

    parser.add_argument("--frames", default="data/frames", help="Frame output dir")
    parser.add_argument("--images", default="data/images", help="Image output dir")
    parser.add_argument("--detections", default="data/images/detections", help="Detection output dir")
    parser.add_argument("--lifted", default="data/fused/lifted", help="Lifted 3D output dir")
    parser.add_argument("--fused", default="data/fused", help="Fused output dir")
    parser.add_argument("--bev", default="data/fused/bev", help="BEV output dir")
    parser.add_argument("--video", default="data/fused/fused_bev_live.mp4", help="BEV video path")
    parser.add_argument("--live", default="data/live", help="Live manifest output dir")

    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name/path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    parser.add_argument("--step-length", type=float, default=0.05, help="SUMO step length")
    parser.add_argument("--duration-sec", type=float, default=0, help="Simulation duration in seconds (0 = until vehicles end; default: 0)")
    parser.add_argument("--radius", type=float, default=80.0, help="Neighborhood radius in meters")
    parser.add_argument("--no-coop-images", action="store_true", help="Render only ego images")

    parser.add_argument("--skip-detection", action="store_true", help="Skip YOLO stage")
    parser.add_argument("--skip-video", action="store_true", help="Skip MP4 export")
    parser.add_argument("--display", action="store_true", help="Show live BEV window")
    parser.add_argument("--clean", action="store_true", help="Delete prior outputs before running")

    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws", action="store_true", help="Disable the WebSocket broadcast server")
    parser.add_argument(
        "--all-dashboards",
        action="store_true",
        help="Render synthetic dashboard images for every active vehicle each step",
    )

    parser.add_argument("--iou", type=float, default=0.1, help="GT matching IoU threshold")
    parser.add_argument("--max-match-dist", type=float, default=6.0, help="Max world-distance for ego/coop match")
    parser.add_argument("--ego-weight", type=float, default=0.55, help="Position blending weight for ego")
    parser.add_argument("--scale", type=float, default=5.0, help="Pixels per meter in BEV")
    parser.add_argument("--range-m", type=float, default=70.0, help="Forward range shown in BEV")
    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=0,
        help="Maximum number of vehicles kept per frame for live outputs (0 = no limit)",
    )

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--auto-coop", dest="auto_coop", action="store_true", help="Enable automatic coop assignment (default)")
    g.add_argument("--no-auto-coop", dest="auto_coop", action="store_false", help="Disable automatic coop assignment")
    parser.set_defaults(auto_coop=True)
    parser.add_argument("--max-coops", type=int, default=3, help="Maximum number of automatic cooperative vehicles to assign")
    parser.add_argument("--coop-radius", type=float, default=80.0, help="Radius (m) used for automatic coop assignment")
    parser.add_argument("--workers", type=int, default=1, help="Number of background worker threads for heavy rendering (default 1)")

    return parser.parse_args()


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def _write_latest_atomic(live_d: Path, payload: dict) -> None:
    """Write latest.json as safely as possible.

    Tries an atomic rename (tmp → latest.json) first. On Windows the rename
    can fail with WinError 5 (Access is denied) if Vite's static server has
    the file open at that instant, so we fall back to a direct write in that
    case. The fallback is slightly less atomic but avoids crashing the
    pipeline — and Vite re-reads the file on the next poll anyway.

    FIX #7: tmp file is always cleaned up even if the direct-write fallback fails.
    """
    text = json.dumps(payload, indent=2)
    dest = live_d / "latest.json"
    tmp = live_d / "latest.json.tmp"
    
    try:
        tmp.write_text(text, encoding="utf-8")
    except OSError:
        # If we can't write tmp file, try direct write
        try:
            dest.write_text(text, encoding="utf-8")
        except OSError:
            # Silently ignore if we can't write anything
            pass
        return
    
    try:
        tmp.replace(dest)
    except OSError:
        try:
            dest.write_text(text, encoding="utf-8")
        finally:
            # Always clean up the tmp file regardless of whether the direct write
            # succeeded, failed, or raised — avoids leaking stale .tmp files.
            tmp.unlink(missing_ok=True)


def _safe_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)
    return safe or "veh"


def _get_vehicle_state(veh_id: str) -> dict | None:
    if veh_id not in traci.vehicle.getIDList():
        return None

    x, y = traci.vehicle.getPosition(veh_id)
    angle = traci.vehicle.getAngle(veh_id)
    speed = traci.vehicle.getSpeed(veh_id)
    length = traci.vehicle.getLength(veh_id)
    width = traci.vehicle.getWidth(veh_id)
    height = traci.vehicle.getHeight(veh_id)
    vtype = traci.vehicle.getTypeID(veh_id)
    lane_id = traci.vehicle.getLaneID(veh_id)
    road_id = traci.vehicle.getRoadID(veh_id)

    heading_rad = math.radians(90.0 - angle)

    return {
        "id": veh_id,
        "x": x,
        "y": y,
        "heading_deg": angle,
        "heading_rad": heading_rad,
        "speed_ms": speed,
        "length": length,
        "width": width,
        "height": height,
        "type": vtype,
        "lane_id": lane_id,
        "road_id": road_id,
    }


def _get_neighbourhood(ego_id: str, radius: float = 80.0) -> list[dict[str, Any]]:
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


def _build_frame_payload(
    step: int,
    sim_time: float,
    ego_state: dict | None,
    coop_state: dict | None,
    gt_objects: list[dict[str, Any]],
    all_vehicles: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "step": step,
        "sim_time": sim_time,
        "camera_intrinsics": CAMERA_INTRINSICS,
        "camera_mount": CAMERA_MOUNT,
        "ego": ego_state,
        "coop": coop_state,
        "gt_objects": gt_objects,
        "all_vehicles": all_vehicles,
    }


def _limit_vehicle_states(
    all_states: list[dict[str, Any]],
    ego_state: dict[str, Any],
    coop_state: dict[str, Any] | None,
    max_vehicles: int,
) -> list[dict[str, Any]]:
    if max_vehicles <= 0 or len(all_states) <= max_vehicles:
        return all_states

    ego_id = str(ego_state.get("id", ""))
    # FIX #2: Guard against coop_state being None before calling .get().
    coop_id = str(coop_state.get("id", "")) if coop_state is not None else ""

    def dist_to_ego(state: dict[str, Any]) -> float:
        return math.hypot(float(state["x"]) - float(ego_state["x"]), float(state["y"]) - float(ego_state["y"]))

    sorted_states = sorted(all_states, key=dist_to_ego)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for state in sorted_states:
        sid = str(state.get("id", ""))
        if sid in selected_ids:
            continue
        selected.append(state)
        selected_ids.add(sid)
        if len(selected) >= max_vehicles:
            break

    for must_keep in (ego_id, coop_id):
        if not must_keep or must_keep in selected_ids:
            continue
        found = next((s for s in all_states if str(s.get("id", "")) == must_keep), None)
        if found is None:
            continue
        if len(selected) < max_vehicles:
            selected.append(found)
            selected_ids.add(must_keep)
        else:
            farthest_idx = max(
                range(len(selected)),
                key=lambda i: math.hypot(
                    float(selected[i]["x"]) - float(ego_state["x"]),
                    float(selected[i]["y"]) - float(ego_state["y"]),
                ),
            )
            selected_ids.discard(str(selected[farthest_idx].get("id", "")))
            selected[farthest_idx] = found
            selected_ids.add(must_keep)

    return selected


def _write_detection_payload(
    out_dir: Path,
    frame_index: int,
    view: str,
    image_path: Path,
    model_name: str,
    conf: float,
    detections: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "frame_index": frame_index,
        "view": view,
        "image_file": str(image_path),
        "model": model_name,
        "conf_threshold": conf,
        "detections": detections,
    }
    out_file = out_dir / f"frame_{frame_index:06d}_{view}.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _lift_frame(
    frame: dict[str, Any],
    det_payload: dict[str, Any],
    view: str,
    iou_threshold: float,
) -> dict[str, Any] | None:
    observer = frame.get("ego") if view == "ego" else frame.get("coop")
    if observer is None:
        return None

    all_objects = frame.get("all_vehicles") or []
    gt_candidates = [obj for obj in all_objects if obj.get("id") != observer.get("id")]

    lifted = []
    for det in det_payload.get("detections", []):
        bbox = [float(v) for v in det.get("bbox", [])]
        if len(bbox) != 4:
            continue

        best_iou = 0.0
        best_obj = None
        for obj in gt_candidates:
            gt_bbox = project_3d._bbox_from_obj(obj, frame, observer)
            if gt_bbox is None:
                continue
            score = project_3d._iou(bbox, gt_bbox)
            if score > best_iou:
                best_iou = score
                best_obj = obj

        if best_obj is not None and best_iou >= iou_threshold:
            local = project_3d._world_to_local(float(best_obj["x"]), float(best_obj["y"]), observer)
            lifted.append(
                {
                    "track_id": best_obj.get("id"),
                    "source_view": view,
                    "observer_id": observer.get("id"),
                    "class_name": det.get("class_name", "object"),
                    "confidence": float(det.get("confidence", 0.0)),
                    "bbox_2d": bbox,
                    "iou_gt": best_iou,
                    "matched_gt": True,
                    "position_world": {"x": float(best_obj["x"]), "y": float(best_obj["y"])},
                    "position_observer": local,
                    "size_m": {
                        "length": float(best_obj.get("length", 4.5)),
                        "width": float(best_obj.get("width", 2.0)),
                        "height": float(best_obj.get("height", 1.5)),
                    },
                }
            )
        else:
            estimate = project_3d._estimate_from_bbox(bbox, frame, observer)
            lifted.append(
                {
                    "track_id": None,
                    "source_view": view,
                    "observer_id": observer.get("id"),
                    "class_name": det.get("class_name", "object"),
                    "confidence": float(det.get("confidence", 0.0)),
                    "bbox_2d": bbox,
                    "iou_gt": best_iou,
                    "matched_gt": False,
                    "position_world": estimate["position_world"],
                    "position_observer": estimate["position_observer"],
                    "size_m": {"length": 4.5, "width": 2.0, "height": 1.6},
                }
            )

    return {
        "frame_index": int(det_payload.get("frame_index", 0)),
        "sim_time": float(frame.get("sim_time", 0.0)),
        "view": view,
        "observer_id": observer.get("id"),
        "objects_3d": lifted,
    }


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    cfg_path = cfg_path.resolve()
    frames_dir = repo_root / args.frames
    images_dir = repo_root / args.images
    detections_dir = repo_root / args.detections
    lifted_dir = repo_root / args.lifted
    fused_dir = repo_root / args.fused
    bev_dir = repo_root / args.bev
    live_dir = repo_root / args.live
    video_path = repo_root / args.video

    if args.clean:
        remove_path(frames_dir)
        remove_path(images_dir)
        remove_path(detections_dir)
        remove_path(lifted_dir)
        remove_path(bev_dir)
        remove_path(video_path)
        remove_path(live_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)
    lifted_dir.mkdir(parents=True, exist_ok=True)
    fused_dir.mkdir(parents=True, exist_ok=True)
    bev_dir.mkdir(parents=True, exist_ok=True)
    live_dir.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    roles_data = _load_roles(live_dir)
    if roles_data is None:
        _write_roles_atomic(live_dir, args.ego, [args.coop], args.coop_radius)
        print(f"[live] Created roles.json with ego={args.ego}, coop={args.coop}, radius={args.coop_radius}")
        roles_data = _load_roles(live_dir)

    model = None
    if not args.skip_detection:
        model = run_detection._try_load_model(args.model)
        if model is None:
            print("[live] YOLO unavailable; writing empty detections.")

    sumo_binary = "sumo-gui" if args.gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c",
        str(cfg_path),
        "--step-length",
        str(args.step_length),
        "--collision.action",
        "warn",
        "--no-step-log",
        "true",
    ]
    if args.gui:
        sumo_cmd.append("--start")

    video_writer = None
    video_writer_lock = threading.Lock()
    if args.video and not args.skip_video:
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (1200, 700),
        )

    if not args.no_ws:
        ws_server.start_background_server(host="0.0.0.0", port=args.ws_port)

    print(f"[live] Starting SUMO: {' '.join(sumo_cmd)}")
    traci.start(sumo_cmd)

    step = 0
    saved = 0
    last_frame_time = time.perf_counter()
    prev_ego: str | None = None

    _heavy_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers))

    def _heavy_work(
        fp: dict[str, Any],
        # FIX #4: Accept s and st as explicit value parameters (not closed-over
        # loop variables) so rapid loop iterations cannot mutate them mid-flight.
        s: int,
        st: float,
        mdl: Any,
        live_d: Path,
        img_d: Path,
        det_d: Path,
        lift_d: Path,
        fus_d: Path,
        bev_d: Path,
    ) -> None:
        """Runs in background thread: images -> YOLO -> lift -> fuse -> BEV -> latest.json."""
        try:
            manifest = generate_images.render_single_frame(
                fp,
                img_d,
                render_coop=not args.no_coop_images,
                observer_ids=[
                    v.get("id") for v in (fp.get("all_vehicles") or []) if v.get("id")
                ] if args.all_dashboards else None,
            )

            det_payloads: dict[str, dict[str, Any]] = {}
            if manifest is not None and not args.skip_detection:
                for view in ("ego", "coop"):
                    image_value = manifest.get(f"{view}_image")
                    if not image_value:
                        continue
                    image_path = Path(image_value)
                    if not image_path.exists():
                        continue
                    detections = run_detection._infer_one(mdl, image_path, args.conf) if mdl else []
                    det_payloads[view] = _write_detection_payload(
                        det_d, s, view, image_path, args.model, args.conf, detections
                    )

            lifted_payloads: dict[str, dict[str, Any]] = {}
            for view in ("ego", "coop"):
                det_pl = det_payloads.get(view, {"frame_index": s, "detections": []})
                lifted = _lift_frame(fp, det_pl, view, args.iou)
                if lifted is None:
                    continue
                lifted_payloads[view] = lifted
                (lift_d / f"frame_{s:06d}_{view}_3d.json").write_text(
                    json.dumps(lifted, indent=2), encoding="utf-8"
                )

            fused = fuse_ego_coop._fuse_frame(
                frame_index=s,
                sim_time=float(st),
                ego_objs=list(lifted_payloads.get("ego", {}).get("objects_3d", [])),
                coop_objs=list(lifted_payloads.get("coop", {}).get("objects_3d", [])),
                max_match_dist=args.max_match_dist,
                ego_weight=args.ego_weight,
            )
            (fus_d / f"frame_{s:06d}_fused.json").write_text(
                json.dumps(fused, indent=2), encoding="utf-8"
            )

            ego_img = img_d / f"frame_{s:06d}_ego.png"
            ego_obs = fp.get("ego")
            canvas = visualize._compose_frame(
                fused, fp,
                ego_img if ego_img.exists() else None,
                ego_obs, "EGO", args.scale, args.range_m, True,
            )
            bev_path = bev_d / f"frame_{s:06d}_bev.png"
            cv2.imwrite(str(bev_path), canvas)
            with video_writer_lock:
                if video_writer is not None:
                    video_writer.write(canvas)

            # Per-observer BEV
            observer_ids_local: list[str] = []
            if args.all_dashboards:
                observer_ids_local = [
                    v.get("id") for v in (fp.get("all_vehicles") or []) if v.get("id") is not None
                ]
            else:
                if fp.get("ego") is not None:
                    observer_ids_local.append(str(fp["ego"]["id"]))
                if fp.get("coop") is not None:
                    observer_ids_local.append(str(fp["coop"]["id"]))

            vehicle_by_id = {
                str(v.get("id")): v for v in (fp.get("all_vehicles") or []) if v.get("id") is not None
            }
            bev_by_observer: dict[str, str] = {}
            # FIX #8: Consistently guard manifest and its "dashboards" key.
            dashboards: dict[str, Any] = (manifest or {}).get("dashboards") or {}
            for observer_id in observer_ids_local:
                observer = vehicle_by_id.get(str(observer_id))
                if observer is None:
                    continue
                dash_path = None
                dash_value = dashboards.get(str(observer_id))
                if dash_value:
                    candidate = Path(dash_value)
                    if candidate.exists():
                        dash_path = candidate
                observer_label = "EGO" if observer_id == (fp.get("ego") or {}).get("id") else "OBS"
                obs_canvas = visualize._compose_frame(
                    fused, fp, dash_path, observer, observer_label,
                    args.scale, args.range_m, True,
                )
                obs_bev_name = f"frame_{s:06d}_obs_{_safe_id(str(observer_id))}_bev.png"
                cv2.imwrite(str(bev_d / obs_bev_name), obs_canvas)
                bev_by_observer[str(observer_id)] = f"/fused/bev/{obs_bev_name}"

            live_payload_update = {
                "frame_index": s,
                "sim_time": float(st),
                "vehicles": [v.get("id") for v in (fp.get("all_vehicles") or []) if v.get("id")],
                "ego_id": (fp.get("ego") or {}).get("id"),
                "coop_id": (fp.get("coop") or {}).get("id"),
                "frame_json": f"/frames/frame_{s:06d}.json",
                "bev_image": f"/fused/bev/frame_{s:06d}_bev.png",
                "fused_json": f"/fused/frame_{s:06d}_fused.json",
                "bev_by_observer": bev_by_observer,
                "dashboards": {
                    k: f"/images/{Path(v).name}"
                    for k, v in dashboards.items()
                },
                "updated_at": time.time(),
            }
            _write_latest_atomic(live_d, live_payload_update)

            if not args.no_ws:
                ws_server.push_frame(live_payload_update)

            if args.display:
                cv2.imshow("SUMO V2V Live BEV", canvas)
                cv2.waitKey(1)

        except Exception as exc:
            print(f"[live][bg] frame {s} error: {exc}")

    try:
        max_sim_time = args.duration_sec if args.duration_sec > 0 else float('inf')
        step_count = 0
        while True:
            # Check duration limit first (if set)
            sim_time = traci.simulation.getTime()
            min_expected = traci.simulation.getMinExpectedNumber()
            
            if args.duration_sec > 0 and sim_time >= max_sim_time:
                print(f"[live] Loop exit: duration reached at sim_time={sim_time:.2f}s", flush=True)
                break
            
            # Otherwise exit when no more vehicles expected (if duration not set)
            if args.duration_sec <= 0 and min_expected <= 0:
                print(f"[live] Loop exit: no vehicles (duration not set)", flush=True)
                break
            
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            step_count += 1

            # Read current roles
            roles_data = _load_roles(live_dir)
            if roles_data is not None:
                current_ego = roles_data.get("ego", args.ego)
                current_coops: list[str] = roles_data.get("coops") or [args.coop]
                current_coop = current_coops[0] if current_coops else args.coop
                current_coop_radius = roles_data.get("coop_radius", args.coop_radius)
            else:
                # FIX #1: Always initialise current_coops so it is never undefined
                # when referenced later in the need_reassign check.
                current_ego = args.ego
                current_coops = [args.coop]
                current_coop = args.coop
                current_coop_radius = args.coop_radius

            ego_state = _get_vehicle_state(current_ego)
            coop_state = _get_vehicle_state(current_coop) if current_coop else None
            if ego_state is None:
                step += 1
                continue

            all_vehicle_states = [
                state
                for state in (_get_vehicle_state(vid) for vid in traci.vehicle.getIDList())
                if state is not None
            ]
            all_vehicle_states = _limit_vehicle_states(
                all_vehicle_states, ego_state, coop_state, args.max_vehicles
            )

            gt_objects = _get_neighbourhood(current_ego, radius=current_coop_radius)

            frame_payload = _build_frame_payload(
                step, sim_time, ego_state, coop_state, gt_objects, all_vehicle_states
            )

            # Auto-assign coops
            if args.auto_coop:
                try:
                    vehicle_ids = {str(v.get("id")) for v in all_vehicle_states}
                except Exception:
                    vehicle_ids = set()
                need_reassign = (
                    prev_ego is None
                    or str(prev_ego) != str(current_ego)
                    or any(
                        c and str(c) not in vehicle_ids
                        for c in current_coops
                    )
                )
                if need_reassign:
                    new_coops = assign_coops_by_proximity(
                        frame_payload, current_ego, args.coop_radius, args.max_coops
                    )
                    current_coops = new_coops
                    current_coop = current_coops[0] if current_coops else None

            prev_ego = current_ego

            # Write frame JSON immediately
            frame_path = frames_dir / f"frame_{step:06d}.json"
            frame_path.write_text(json.dumps(frame_payload, indent=2), encoding="utf-8")

            # Push lightweight position update over WS immediately (before heavy work)
            quick_payload = {
                "frame_index": step,
                "sim_time": float(sim_time),
                "vehicles": [v.get("id") for v in (frame_payload.get("all_vehicles") or []) if v.get("id")],
                "ego_id": (frame_payload.get("ego") or {}).get("id"),
                "coop_id": (frame_payload.get("coop") or {}).get("id"),
                "frame_json": f"/frames/frame_{step:06d}.json",
                "bev_image": "",
                "fused_json": "",
                "bev_by_observer": {},
                "dashboards": {},
                "updated_at": time.time(),
            }
            _write_latest_atomic(live_dir, quick_payload)
            if not args.no_ws:
                ws_server.push_frame(quick_payload)

            # Submit heavy work to background thread (non-blocking).
            # FIX #4: Pass step and sim_time as explicit positional arguments so
            # each submitted task captures its own snapshot of these values, not a
            # reference to the loop variable that will be mutated next iteration.
            _heavy_executor.submit(
                _heavy_work,
                frame_payload, step, sim_time, model,
                live_dir, images_dir, detections_dir,
                lifted_dir, fused_dir, bev_dir,
            )

            saved += 1
            if saved % 25 == 0:
                now = time.perf_counter()
                fps = 25.0 / max(1e-6, now - last_frame_time)
                last_frame_time = now
                print(f"[live] {saved} frames stepped ({fps:.1f} fps SUMO)")

            step += 1

    finally:
        # FIX #10: Shut down executor and close traci in a nested try/finally so
        # that traci.close() is always called even if executor.shutdown() raises.
        try:
            print("[live] Waiting for background renders to finish...")
            _heavy_executor.shutdown(wait=True)
        finally:
            traci.close()

        # FIX #5: Release the video writer under the lock so no background thread
        # can write to it after release, and only if it was successfully created.
        with video_writer_lock:
            if video_writer is not None:
                video_writer.release()

        if args.display:
            cv2.destroyAllWindows()
        print(f"[live] Done. {saved} frames written to {bev_dir}")


if __name__ == "__main__":
    main()