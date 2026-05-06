"""
generate_image.py  —  Phase 2: Synthetic Perception
=====================================================
Converts per-frame SUMO vehicle-state dicts (Phase 1 output) into
synthetic dashcam images using OpenCV / Pillow.

Pipeline position:
  Phase 1 (SUMO / dummy data)  →  [THIS FILE]  →  Phase 3 (V2V fusion)

Public API
----------
  render_frame(frame_data)          → np.ndarray  (BGR, H×W×3)
  render_frame_rgb(frame_data)      → np.ndarray  (RGB, H×W×3)
  frames_to_pil(frame_data_list)    → list[PIL.Image]
  save_gif(pil_frames, path, fps)   → str  (saved path)
  save_frames(pil_frames, out_dir)  → list[str]
  show_sample_frames(pil_frames, …) → None  (matplotlib display)

Frame-data schema (same as Phase 1 output)
-------------------------------------------
  {
    "timestamp": float,               # seconds
    "vehicles": [
      {
        "id":    str,                 # "ego" | "veh_XX" | "ped_XX"
        "x":     float,              # lateral metres (+right)
        "y":     float,              # depth metres ahead of ego
        "speed": float,              # m/s
        "angle": float,              # heading degrees (0 = North)
        "lane":  int,                # -1 = pedestrian / not in lane
        "color": (R, G, B)           # display colour
      }, …
    ]
  }
"""

import cv2
import numpy as np
import math
import os
import json
import argparse
import re
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Camera & canvas parameters
# ──────────────────────────────────────────────────────────────────────────────

IMG_W          = 800        # canvas width  (pixels)
IMG_H          = 600        # canvas height (pixels)

FOCAL_LENGTH   = 600        # perspective strength
CAM_HEIGHT     = 1.4        # camera height above road (metres)
HORIZON_Y      = int(IMG_H * 0.42)   # pixel row of the horizon
FOV_HALF_DEG   = 45.0       # half horizontal FOV (degrees)

# ──────────────────────────────────────────────────────────────────────────────
# Road geometry
# ──────────────────────────────────────────────────────────────────────────────

LANE_WIDTH  = 3.5           # metres per lane
N_LANES     = 3
ROAD_WIDTH  = LANE_WIDTH * N_LANES

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette  (all stored as RGB tuples; converted to BGR when passed to cv2)
# ──────────────────────────────────────────────────────────────────────────────

SKY_TOP     = (100, 160, 220)
SKY_BOT     = (180, 210, 240)
ROAD_COLOR  = ( 60,  60,  65)
CURB_COLOR  = (130, 130, 140)
GRASS_COLOR = ( 80, 120,  60)


# ──────────────────────────────────────────────────────────────────────────────
# Projection helpers
# ──────────────────────────────────────────────────────────────────────────────

def world_to_screen(x_world: float, y_world: float):
    """
    Project a world point (x = lateral metres, y = depth ahead of ego)
    to image-pixel coordinates (px, py).
    Returns None if the point is behind the camera or degenerate.
    """
    if y_world <= 0.5:
        return None
    scale = FOCAL_LENGTH / y_world
    px = int(IMG_W / 2 + x_world * scale)
    py = int(HORIZON_Y + CAM_HEIGHT * scale)
    return (px, py)


def vehicle_screen_size(y_world: float):
    """Return (width_px, height_px) for an object at depth y_world."""
    if y_world <= 0.5:
        return (0, 0)
    scale = FOCAL_LENGTH / y_world
    w = max(4, int(2.0 * scale))   # 2 m wide
    h = max(4, int(1.5 * scale))   # 1.5 m tall
    return (w, h)


def in_fov(x_world: float, y_world: float) -> bool:
    """True when the world point falls inside the camera's horizontal FOV."""
    if y_world <= 0:
        return False
    angle = math.degrees(math.atan2(abs(x_world), y_world))
    return angle <= FOV_HALF_DEG


# ──────────────────────────────────────────────────────────────────────────────
# Scene layers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_sky(img: np.ndarray) -> None:
    """Fill the sky region with a vertical gradient."""
    for row in range(HORIZON_Y):
        t = row / HORIZON_Y
        r = int(SKY_TOP[0] * (1 - t) + SKY_BOT[0] * t)
        g = int(SKY_TOP[1] * (1 - t) + SKY_BOT[1] * t)
        b = int(SKY_TOP[2] * (1 - t) + SKY_BOT[2] * t)
        img[row, :] = (b, g, r)   # BGR


def _draw_road(img: np.ndarray) -> None:
    """Render road polygon, grass shoulders, curb lines and lane markings."""
    # Grass baseline
    img[HORIZON_Y:, :] = GRASS_COLOR[::-1]   # RGB→BGR

    # Road polygon (perspective-projected)
    depths = [2, 5, 10, 20, 40, 80, 150]
    road_half = ROAD_WIDTH / 2
    left_pts, right_pts = [], []

    for d in depths:
        lp = world_to_screen(-road_half, d)
        rp = world_to_screen( road_half, d)
        if lp and rp:
            left_pts.append(lp)
            right_pts.append(rp)

    if len(left_pts) >= 2:
        poly = left_pts + list(reversed(right_pts))
        cv2.fillPoly(img, [np.array(poly, dtype=np.int32)], ROAD_COLOR[::-1])

    # Curb lines
    for side in (-road_half, road_half):
        prev = None
        for d in depths:
            pt = world_to_screen(side, d)
            if pt and prev:
                cv2.line(img, prev, pt, CURB_COLOR[::-1], 2)
            prev = pt

    # Dashed lane markings
    for lane_x in (-LANE_WIDTH / 2, LANE_WIDTH / 2):
        for d in np.arange(3, 120, 6):
            p1 = world_to_screen(lane_x, d)
            p2 = world_to_screen(lane_x, d + 2.5)
            if p1 and p2:
                cv2.line(img, p1, p2, (200, 200, 200), 2)


def _draw_vehicle(img: np.ndarray, veh: dict) -> None:
    """
    Render a single vehicle or pedestrian.
    Vehicles in-FOV get a body rect, windshield, shadow and detection box.
    Out-of-FOV objects get an arrow indicator at the image edge.
    """
    x, y   = veh["x"], veh["y"]
    vid    = veh["id"]
    bgr    = veh["color"][::-1]   # RGB → BGR
    is_ped = veh.get("lane", 0) == -1

    pt = world_to_screen(x, y)
    if pt is None:
        return

    px, py = pt
    vw, vh = vehicle_screen_size(y)
    if vw < 2:
        return

    if in_fov(x, y):
        # ── bounding box coords ──────────────────────────────────────────────
        if is_ped:
            head_r = max(3, vh // 3)
            x1, y1 = px - head_r,      py - vh - head_r
            x2, y2 = px + head_r,      py + vh // 2
        else:
            x1, y1 = px - vw // 2, py - vh
            x2, y2 = px + vw // 2, py

        # ── draw shape ───────────────────────────────────────────────────────
        if is_ped:
            cv2.circle(img, (px, py - vh), head_r, bgr, -1)
            cv2.line(img, (px, py - vh + head_r),
                     (px, py + vh // 2), bgr, max(2, vw // 4))
        else:
            # Shadow
            shadow = tuple((np.array(bgr, np.float32) * 0.4).astype(int).tolist())
            cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), shadow, -1)
            # Body
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, -1)
            # Windshield
            ws_y2 = y1 + (y2 - y1) // 3
            cv2.rectangle(img,
                          (x1 + vw // 6, y1), (x2 - vw // 6, ws_y2),
                          (200, 220, 255), -1)
            # Outline
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # ── distance label ───────────────────────────────────────────────────
        label_y = y1 - 4 if not is_ped else py - vh - 10
        font_scale = max(0.3, min(0.7, 15 / y))
        cv2.putText(img, f"{vid}  {y:.1f}m",
                    (px - vw // 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # ── green detection box ──────────────────────────────────────────────
        pad = 4
        cv2.rectangle(img,
                      (x1 - pad, y1 - pad),
                      (x2 + pad, y2 + pad),
                      (0, 255, 0), 1)

    else:
        # ── off-screen arrow indicator ───────────────────────────────────────
        edge_x    = 10 if x < 0 else IMG_W - 10
        edge_y    = max(HORIZON_Y + 20, min(IMG_H - 20, py))
        arrow_dir = 1 if x > 0 else -1
        pts = np.array([
            [edge_x,                    edge_y     ],
            [edge_x - arrow_dir * 18,   edge_y - 10],
            [edge_x - arrow_dir * 18,   edge_y + 10],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], (0, 50, 255))
        cv2.putText(img, vid,
                    (edge_x - arrow_dir * 40, edge_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 50, 255), 1)


def _draw_hud(img: np.ndarray,
              frame_data: dict,
              detected_ids: list,
              v2v_msgs: list) -> None:
    """Overlay HUD: speed bar, timestamp, detected objects, V2V log, FOV arc."""
    ts  = frame_data["timestamp"]
    ego = next(v for v in frame_data["vehicles"] if v["id"] == "ego")

    # Semi-transparent bottom bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, IMG_H - 110), (IMG_W, IMG_H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Speed
    speed_kmh = int(ego["speed"] * 3.6)
    cv2.putText(img, str(speed_kmh),
                (30, IMG_H - 40), cv2.FONT_HERSHEY_DUPLEX,
                2.0, (255, 255, 255), 2)
    cv2.putText(img, "km/h",
                (30, IMG_H - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1)

    # Timestamp
    cv2.putText(img, f"T: {ts:.1f}s",
                (IMG_W - 140, IMG_H - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Detected objects list
    det_str = "DETECTED: " + (", ".join(detected_ids) if detected_ids else "None")
    det_color = (0, 255, 100) if detected_ids else (120, 120, 120)
    cv2.putText(img, det_str,
                (150, IMG_H - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 1, cv2.LINE_AA)

    # V2V message strip
    cv2.putText(img, "V2V:",
                (150, IMG_H - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    for i, msg in enumerate(v2v_msgs[-3:]):
        cv2.putText(img, msg,
                    (200, IMG_H - 55 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

    # FOV arc indicator (top-right corner)
    cx, cy = IMG_W - 60, HORIZON_Y - 30
    cv2.ellipse(img, (cx, cy), (40, 20), 0,
                -FOV_HALF_DEG, FOV_HALF_DEG, (0, 255, 200), 1)
    rad = math.radians(FOV_HALF_DEG)
    cv2.line(img, (cx, cy),
             (cx + int(40 * math.sin(rad)), cy - int(20 * math.cos(rad))),
             (0, 255, 200), 1)
    cv2.line(img, (cx, cy),
             (cx - int(40 * math.sin(rad)), cy - int(20 * math.cos(rad))),
             (0, 255, 200), 1)
    cv2.putText(img, "CAM",
                (cx - 14, cy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 200), 1)

    # Ego label (top-left)
    cv2.putText(img, "EGO VEHICLE — FRONT DASHCAM",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


# ──────────────────────────────────────────────────────────────────────────────
# V2V broadcast log  (Phase 2 produces broadcast messages consumed by Phase 3)
# ──────────────────────────────────────────────────────────────────────────────

_v2v_log: list[str] = []


def simulate_v2v_broadcast(frame_data: dict,
                            detected_ids: list) -> list[str]:
    """
    Simulate the ego vehicle broadcasting its perceived objects over V2V.

    Returns a list of short display strings (last 3 messages) and
    appends full log entries to the module-level _v2v_log.
    Pass the full log to Phase 3 via get_v2v_log().
    """
    ts       = frame_data["timestamp"]
    new_msgs = []

    for vid in detected_ids:
        veh = next((v for v in frame_data["vehicles"] if v["id"] == vid), None)
        if veh:
            full_msg = (
                f"[{ts:.1f}s] EGO→BCAST | "
                f"id={vid} x={veh['x']:.1f} y={veh['y']:.1f} "
                f"spd={veh['speed']:.1f}m/s"
            )
            _v2v_log.append(full_msg)
            new_msgs.append(f"BCAST: {vid} @ ({veh['x']:.1f}, {veh['y']:.1f})")

    return new_msgs[-3:]


def get_v2v_log() -> list[str]:
    """Return all V2V broadcast messages accumulated so far."""
    return list(_v2v_log)


def clear_v2v_log() -> None:
    """Reset the V2V log (call between simulation runs)."""
    _v2v_log.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Public render API
# ──────────────────────────────────────────────────────────────────────────────

def render_frame(frame_data: dict) -> np.ndarray:
    """
    Render one simulation frame to a BGR NumPy array (shape H×W×3).

    Parameters
    ----------
    frame_data : dict
        Single frame dict as produced by Phase 1 (SUMO or dummy generator).

    Returns
    -------
    np.ndarray  (dtype uint8, BGR channel order, shape IMG_H × IMG_W × 3)
    """
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    _draw_sky(img)
    _draw_road(img)

    # Sort vehicles far→near (painter's algorithm)
    others = sorted(
        [v for v in frame_data["vehicles"] if v["id"] != "ego"],
        key=lambda v: v["y"],
        reverse=True,
    )

    detected_ids = []
    for veh in others:
        _draw_vehicle(img, veh)
        if in_fov(veh["x"], veh["y"]) and veh["y"] > 0:
            detected_ids.append(veh["id"])

    v2v_msgs = simulate_v2v_broadcast(frame_data, detected_ids)
    _draw_hud(img, frame_data, detected_ids, v2v_msgs)

    return img


def render_frame_rgb(frame_data: dict) -> np.ndarray:
    """Same as render_frame but returns RGB (suitable for matplotlib / PIL)."""
    return cv2.cvtColor(render_frame(frame_data), cv2.COLOR_BGR2RGB)


def frames_to_pil(frame_data_list: list,
                  verbose: bool = True) -> list:
    """
    Render a sequence of frame dicts into PIL Image objects.

    Parameters
    ----------
    frame_data_list : list[dict]
        Ordered list of frame dicts (Phase 1 output).
    verbose : bool
        Print progress every 10 frames.

    Returns
    -------
    list[PIL.Image.Image]  (RGB mode)
    """
    n = len(frame_data_list)
    pil_frames = []
    if verbose:
        print(f"[Phase 2] Rendering {n} frames …")

    for i, fd in enumerate(frame_data_list):
        rgb = render_frame_rgb(fd)
        pil_frames.append(Image.fromarray(rgb))
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n} frames rendered")

    if verbose:
        print("[Phase 2] Render complete.")

    return pil_frames


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_gif(pil_frames: list,
             path: str = "dashcam_sim.gif",
             fps: float = 10.0) -> str:
    """
    Save a list of PIL frames as an animated GIF.

    Parameters
    ----------
    pil_frames : list[PIL.Image.Image]
    path       : output file path
    fps        : frames per second

    Returns
    -------
    str  — absolute path of the saved file
    """
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    abs_path = os.path.abspath(path)
    print(f"[Phase 2] GIF saved → {abs_path}")
    return abs_path


def save_frames(pil_frames: list,
                out_dir: str = "frames",
                prefix: str = "frame",
                fmt: str = "PNG") -> list[str]:
    """
    Save each PIL frame as an individual image file.

    Parameters
    ----------
    pil_frames : list[PIL.Image.Image]
    out_dir    : directory to write into (created if absent)
    prefix     : filename prefix
    fmt        : image format ("PNG", "JPEG", …)

    Returns
    -------
    list[str]  — paths of saved files
    """
    os.makedirs(out_dir, exist_ok=True)
    ext   = fmt.lower().replace("jpeg", "jpg")
    paths = []
    for i, frame in enumerate(pil_frames):
        p = os.path.join(out_dir, f"{prefix}_{i:04d}.{ext}")
        frame.save(p, format=fmt)
        paths.append(p)
    print(f"[Phase 2] {len(paths)} frames saved to '{out_dir}/'")
    return paths


def show_sample_frames(pil_frames: list,
                       indices: list | None = None,
                       title: str = "Synthetic Dashcam Frames") -> None:
    """
    Display a horizontal strip of sample frames via matplotlib.

    Parameters
    ----------
    pil_frames : list[PIL.Image.Image]
    indices    : which frame indices to display (defaults to 5 spread evenly)
    title      : figure title
    """
    n = len(pil_frames)
    if indices is None:
        indices = [int(n * k / 4) for k in range(5)]
        indices[-1] = min(indices[-1], n - 1)

    indices = [i for i in indices if 0 <= i < n]
    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.imshow(pil_frames[idx])
        ax.set_title(f"Frame {idx}  t={idx * 0.1:.1f}s")
        ax.axis("off")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def print_v2v_log(max_lines: int = 20) -> None:
    """Pretty-print the accumulated V2V broadcast log."""
    log = get_v2v_log()
    print("\n===== V2V BROADCAST LOG (Phase 2) =====")
    for msg in log[:max_lines]:
        print(msg)
    if len(log) > max_lines:
        print(f"… +{len(log) - max_lines} more messages")


def _vehicle_color(veh_id: str) -> tuple[int, int, int]:
    """Stable pseudo-color assignment for vehicle IDs (RGB)."""
    if veh_id.startswith("ego"):
        return (0, 200, 0)
    if veh_id.startswith("coop"):
        return (0, 180, 255)
    seed = abs(hash(veh_id))
    r = 70 + (seed % 150)
    g = 70 + ((seed // 7) % 150)
    b = 70 + ((seed // 13) % 150)
    return (r, g, b)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", value)
    return safe or "veh"


def _to_local_frame_from_observer(raw_frame: dict[str, Any], observer: dict[str, Any]) -> dict[str, Any]:
    """Convert run_sumo world frame to local coordinates for any observer vehicle."""
    obs_x = float(observer["x"])
    obs_y = float(observer["y"])
    h = float(observer.get("heading_rad", 0.0))

    vehicles = [
        {
            "id": "ego",
            "x": 0.0,
            "y": 0.0,
            "speed": float(observer.get("speed_ms", 0.0)),
            "angle": float(observer.get("heading_deg", 0.0)),
            "lane": 1,
            "color": (0, 200, 0),
        }
    ]

    for veh in raw_frame.get("all_vehicles", []):
        veh_id = str(veh.get("id", ""))
        if veh_id == observer.get("id"):
            continue

        dx = float(veh["x"]) - obs_x
        dy = float(veh["y"]) - obs_y
        forward = dx * math.cos(h) + dy * math.sin(h)
        lateral = -dx * math.sin(h) + dy * math.cos(h)

        vehicles.append(
            {
                "id": veh_id,
                "x": lateral,
                "y": forward,
                "speed": float(veh.get("speed_ms", 0.0)),
                "angle": float(veh.get("heading_deg", 0.0)),
                "lane": 1,
                "color": _vehicle_color(veh_id),
            }
        )

    return {
        "timestamp": float(raw_frame.get("sim_time", 0.0)),
        "vehicles": vehicles,
    }


def _to_local_frame(raw_frame: dict[str, Any], observer_key: str) -> dict[str, Any] | None:
    """Convert run_sumo world frame to the local schema expected by renderer."""
    observer = raw_frame.get(observer_key)
    if observer is None:
        return None
    return _to_local_frame_from_observer(raw_frame, observer)


def _visible_ids(frame_data: dict[str, Any]) -> list[str]:
    visible = []
    for veh in frame_data.get("vehicles", []):
        if veh.get("id") == "ego":
            continue
        if in_fov(float(veh.get("x", 0.0)), float(veh.get("y", 0.0))) and float(veh.get("y", 0.0)) > 0.5:
            visible.append(str(veh.get("id")))
    return visible


def _collect_frame_files(frames_dir: Path) -> list[Path]:
    pattern = re.compile(r"^frame_\d{6}\.json$")
    return [p for p in sorted(frames_dir.glob("frame_*.json")) if pattern.match(p.name)]


def render_single_frame(
    raw_frame: dict[str, Any],
    out_dir: Path,
    render_coop: bool = True,
    observer_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Render one run_sumo frame dict to ego/coop images and return a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    step = int(raw_frame.get("step", 0))

    ego_frame = _to_local_frame(raw_frame, "ego")
    if ego_frame is None:
        return None

    ego_rgb = render_frame_rgb(ego_frame)
    ego_img_path = out_dir / f"frame_{step:06d}_ego.png"
    Image.fromarray(ego_rgb).save(ego_img_path)

    coop_img_path = None
    coop_visible: list[str] = []
    if render_coop and raw_frame.get("coop") is not None:
        coop_frame = _to_local_frame(raw_frame, "coop")
        if coop_frame is not None:
            coop_rgb = render_frame_rgb(coop_frame)
            coop_img_path = out_dir / f"frame_{step:06d}_coop.png"
            Image.fromarray(coop_rgb).save(coop_img_path)
            coop_visible = _visible_ids(coop_frame)

    dashboards: dict[str, str] = {}
    ego_id = str(raw_frame.get("ego", {}).get("id", ""))
    coop_data = raw_frame.get("coop")
    coop_id = str(coop_data.get("id", "")) if isinstance(coop_data, dict) else ""
    if ego_id:
        dashboards[ego_id] = str(ego_img_path)
    if coop_id and coop_img_path is not None:
        dashboards[coop_id] = str(coop_img_path)

    observers = observer_ids or []
    vehicle_by_id = {
        str(veh.get("id", "")): veh
        for veh in raw_frame.get("all_vehicles", [])
        if veh.get("id") is not None
    }
    for observer_id in observers:
        observer = vehicle_by_id.get(observer_id)
        if observer is None or observer_id in dashboards:
            continue
        local_frame = _to_local_frame_from_observer(raw_frame, observer)
        dashboard_rgb = render_frame_rgb(local_frame)
        dash_path = out_dir / f"frame_{step:06d}_obs_{_safe_id(observer_id)}.png"
        Image.fromarray(dashboard_rgb).save(dash_path)
        dashboards[observer_id] = str(dash_path)

    manifest = {
        "frame_index": step,
        "sim_time": float(raw_frame.get("sim_time", 0.0)),
        "ego_image": str(ego_img_path),
        "coop_image": str(coop_img_path) if coop_img_path else None,
        "dashboards": dashboards,
        "ego_visible_ids": _visible_ids(ego_frame),
        "coop_visible_ids": coop_visible,
    }
    (out_dir / f"frame_{step:06d}.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def run_batch(frames_dir: str = "data/frames", out_dir: str = "data/images", render_coop: bool = True) -> None:
    """Render all run_sumo frame JSON files into ego/coop synthetic images."""
    frames_path = Path(frames_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_files = _collect_frame_files(frames_path)
    if not frame_files:
        print(f"[generate_images] No frame JSONs found in {frames_path}")
        return

    clear_v2v_log()
    print(f"[generate_images] Rendering {len(frame_files)} frame files...")
    for i, frame_file in enumerate(frame_files, start=1):
        raw = json.loads(frame_file.read_text(encoding="utf-8"))
        step = int(raw.get("step", i - 1))

        manifest = render_single_frame(raw, out_path, render_coop=render_coop)
        if manifest is None:
            continue

        if i % 25 == 0 or i == len(frame_files):
            print(f"  [generate_images] {i}/{len(frame_files)} done")

    print(f"[generate_images] Images and manifests written to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Self-contained demo  (uses the Phase 1 dummy generator when run directly)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 synthetic image generation")
    parser.add_argument("--frames", default="data/frames", help="Input frame JSON directory")
    parser.add_argument("--out", default="data/images", help="Output image directory")
    parser.add_argument(
        "--no-coop",
        action="store_true",
        help="Disable cooperative camera rendering",
    )
    args = parser.parse_args()

    run_batch(frames_dir=args.frames, out_dir=args.out, render_coop=not args.no_coop)