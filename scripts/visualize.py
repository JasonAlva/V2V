"""Phase 4: Render fused outputs as a BEV playback with optional video export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize fused BEV outputs.")
    parser.add_argument("--fused", default="data/fused", help="Fused JSON directory")
    parser.add_argument("--frames", default="data/frames", help="Frame state JSON directory")
    parser.add_argument("--images", default="data/images", help="Reference image directory")
    parser.add_argument("--out", default="data/fused/bev", help="Directory for rendered BEV PNGs")
    parser.add_argument("--video", default="", help="Optional output video path (mp4)")
    parser.add_argument("--scale", type=float, default=5.0, help="Pixels per meter in BEV")
    parser.add_argument("--range-m", type=float, default=70.0, help="Forward range shown in BEV")
    parser.add_argument(
        "--blind-spot",
        action="store_true",
        help="Highlight coop-only objects as V2V blind-spot detections",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _world_to_ego_local(world_x: float, world_y: float, ego: dict[str, Any]) -> tuple[float, float]:
    dx = world_x - float(ego["x"])
    dy = world_y - float(ego["y"])
    h = float(ego.get("heading_rad", 0.0))
    forward = dx * np.cos(h) + dy * np.sin(h)
    lateral = -dx * np.sin(h) + dy * np.cos(h)
    return float(lateral), float(forward)


def _local_to_canvas(x_local: float, y_local: float, width: int, height: int, scale: float) -> tuple[int, int]:
    cx = int(width * 0.45)
    cy = int(height * 0.88)
    px = int(cx + x_local * scale)
    py = int(cy - y_local * scale)
    return px, py


def _draw_background(canvas: np.ndarray, width: int, height: int, range_m: float, scale: float) -> None:
    canvas[:] = (18, 18, 24)
    for r in (10, 20, 30, 40, 50, 60):
        radius = int(r * scale)
        cv2.circle(canvas, (int(width * 0.45), int(height * 0.88)), radius, (45, 45, 60), 1)
        cv2.putText(canvas, f"{r}m", (int(width * 0.45) + 4, int(height * 0.88) - radius - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 150), 1)

    cv2.line(canvas, (0, int(height * 0.88)), (width, int(height * 0.88)), (50, 50, 70), 1)
    cv2.line(canvas, (int(width * 0.45), 0), (int(width * 0.45), height), (50, 50, 70), 1)
    cv2.putText(canvas, f"BEV Range: {range_m:.0f}m", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)


def _draw_ego(canvas: np.ndarray, width: int, height: int, label: str = "EGO") -> None:
    cx = int(width * 0.45)
    cy = int(height * 0.88)
    ego_poly = np.array(
        [[cx, cy - 16], [cx - 10, cy + 12], [cx + 10, cy + 12]],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [ego_poly], (0, 210, 120))
    cv2.putText(canvas, label, (cx - 16, cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 220), 1)


def _draw_objects(
    canvas: np.ndarray,
    fused_payload: dict[str, Any],
    observer: dict[str, Any] | None,
    width: int,
    height: int,
    scale: float,
    range_m: float,
    show_blind_spot: bool,
) -> None:
    if observer is None:
        return

    for obj in fused_payload.get("objects", []):
        pos = obj.get("position_world", {})
        if "x" not in pos or "y" not in pos:
            continue
        x_local, y_local = _world_to_ego_local(float(pos["x"]), float(pos["y"]), observer)
        if y_local < 0.0 or y_local > range_m:
            continue
        if abs(x_local) > range_m * 0.6:
            continue

        px, py = _local_to_canvas(x_local, y_local, width, height, scale)
        source = obj.get("source", "unknown")
        if source == "fused":
            color = (0, 230, 255)
        elif source == "ego_only":
            color = (0, 190, 0)
        elif source == "coop_only" and show_blind_spot:
            color = (255, 60, 220)
        else:
            color = (255, 180, 0)

        cv2.circle(canvas, (px, py), 6, color, -1)
        if source == "coop_only" and show_blind_spot:
            cv2.circle(canvas, (px, py), 10, color, 2)
        track_id = str(obj.get("track_id", "obj"))
        cv2.putText(canvas, track_id, (px + 7, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1)


def _blind_spot_objects(fused_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [obj for obj in fused_payload.get("objects", []) if obj.get("source") == "coop_only"]


def _project_blind_spot_boxes(
    fused_payload: dict[str, Any],
    frame_payload: dict[str, Any],
    observer: dict[str, Any] | None,
) -> list[list[float]]:
    if observer is None:
        return []

    intr = frame_payload.get("camera_intrinsics", {})
    mount = frame_payload.get("camera_mount", {})
    fx = float(intr.get("fx", 600.0))
    cx = float(intr.get("cx", 400.0))
    cy = float(intr.get("cy", 300.0))
    cam_h = float(mount.get("z_offset", 1.5))

    boxes: list[list[float]] = []
    for obj in _blind_spot_objects(fused_payload):
        pos = obj.get("position_world", {})
        if "x" not in pos or "y" not in pos:
            continue

        x_local, y_local = _world_to_ego_local(float(pos["x"]), float(pos["y"]), observer)
        if y_local <= 0.5:
            continue

        scale = fx / y_local
        size = obj.get("coop", {}).get("size_m", {})
        width_px = max(4.0, float(size.get("width", 2.0)) * scale)
        height_px = max(4.0, float(size.get("height", 1.6)) * scale)
        px = cx + x_local * scale
        py = cy + cam_h * scale
        boxes.append([px - width_px / 2, py - height_px, px + width_px / 2, py])

    return boxes


def _compose_frame(
    fused_payload: dict[str, Any],
    frame_payload: dict[str, Any],
    image_path: Path | None,
    observer: dict[str, Any] | None,
    observer_label: str,
    scale: float,
    range_m: float,
    show_blind_spot: bool,
) -> np.ndarray:
    width, height = 1200, 700
    bev_w = 760
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    bev = canvas[:, :bev_w]
    _draw_background(bev, bev_w, height, range_m, scale)
    _draw_ego(bev, bev_w, height, observer_label)
    _draw_objects(bev, fused_payload, observer, bev_w, height, scale, range_m, show_blind_spot)

    sim_time = float(fused_payload.get("sim_time", 0.0))
    counts = fused_payload.get("counts", {})
    info_lines = [
        f"t = {sim_time:.2f}s",
        f"ego: {counts.get('ego', 0)}",
        f"coop: {counts.get('coop', 0)}",
        f"fused: {counts.get('fused', 0)}",
        f"matches: {counts.get('matched_pairs', 0)}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line, (790, 40 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1)

    if image_path is not None and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            target_h = 440
            scale_img = target_h / max(1, img.shape[0])
            target_w = int(img.shape[1] * scale_img)
            if show_blind_spot:
                for box in _project_blind_spot_boxes(fused_payload, frame_payload, observer):
                    x1, y1, x2, y2 = [int(v * scale_img) for v in box]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 255), 2)
                    cv2.putText(img, "V2V", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 255), 1)
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            x0, y0 = 780, 200
            x1, y1 = min(width, x0 + target_w), min(height, y0 + target_h)
            canvas[y0:y1, x0:x1] = img[: y1 - y0, : x1 - x0]
            cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (120, 120, 120), 1)
            cv2.putText(canvas, "Ego Camera", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)

    return canvas


def main() -> None:
    args = parse_args()
    fused_dir = Path(args.fused)
    frames_dir = Path(args.frames)
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fused_files = sorted(fused_dir.glob("frame_*_fused.json"))
    if not fused_files:
        print(f"[visualize] No fused files found in {fused_dir}")
        return

    video_writer = None
    if args.video:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (1200, 700),
        )

    print(f"[visualize] Rendering {len(fused_files)} frames...")
    written = 0
    for i, fused_path in enumerate(fused_files, start=1):
        fused = _load_json(fused_path)
        frame_index = int(fused.get("frame_index", 0))

        frame_path = frames_dir / f"frame_{frame_index:06d}.json"
        if not frame_path.exists():
            continue
        frame = _load_json(frame_path)

        ego_img = images_dir / f"frame_{frame_index:06d}_ego.png"
        observer = frame.get("ego")
        canvas = _compose_frame(
            fused,
            frame,
            ego_img if ego_img.exists() else None,
            observer,
            "EGO",
            args.scale,
            args.range_m,
            args.blind_spot,
        )

        out_file = out_dir / f"frame_{frame_index:06d}_bev.png"
        cv2.imwrite(str(out_file), canvas)
        written += 1

        if video_writer is not None:
            video_writer.write(canvas)

        if i % 25 == 0 or i == len(fused_files):
            print(f"  [visualize] {i}/{len(fused_files)} done")

    if video_writer is not None:
        video_writer.release()
        print(f"[visualize] Video written to {args.video}")

    print(f"[visualize] Wrote {written} BEV frames to {out_dir}")


if __name__ == "__main__":
    main()
