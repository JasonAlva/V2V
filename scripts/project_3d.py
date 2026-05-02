"""Minimal project_3d shim for pipeline tests.
Provides small implementations of the functions `run_live_pipeline` expects so
we can run the end-to-end test. This is intentionally lightweight and not a
full camera projection — it's sufficient for simulation/testing purposes.
"""

import math
from typing import Any, Dict

# Simple pinhole-like intrinsics used for crude estimates
INTRINSICS = {"fx": 600.0, "fy": 600.0, "cx": 400.0, "cy": 300.0, "width": 800, "height": 600}


def _bbox_from_obj(obj: Dict[str, Any], frame: Dict[str, Any], observer: Dict[str, Any]):
    # Project object center to image plane using a very simple model.
    try:
        ox = float(observer.get("x", 0.0))
        oy = float(observer.get("y", 0.0))
        oh = float(observer.get("heading_rad", 0.0))
        tx = float(obj.get("x", 0.0))
        ty = float(obj.get("y", 0.0))
    except Exception:
        return None

    dx = tx - ox
    dy = ty - oy
    forward = dx * math.cos(oh) + dy * math.sin(oh)
    lateral = -dx * math.sin(oh) + dy * math.cos(oh)

    if forward <= 0.1:
        return None

    # crude image coords
    fx = INTRINSICS["fx"]
    fy = INTRINSICS["fy"]
    cx = INTRINSICS["cx"]
    cy = INTRINSICS["cy"]

    x_img = cx + (lateral * fx) / max(forward, 0.1)
    y_img = cy - (0.0 * fy) / max(forward, 0.1)  # ignore height for simplicity

    w = max(8.0, fx * 2.0 / max(forward, 1.0))
    h = max(8.0, fy * 1.0 / max(forward, 1.0))

    x1 = x_img - w / 2.0
    y1 = y_img - h / 2.0
    x2 = x_img + w / 2.0
    y2 = y_img + h / 2.0

    return [x1, y1, x2, y2]


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _world_to_local(x: float, y: float, observer: Dict[str, Any]):
    ox = float(observer.get("x", 0.0))
    oy = float(observer.get("y", 0.0))
    oh = float(observer.get("heading_rad", 0.0))
    dx = x - ox
    dy = y - oy
    forward = dx * math.cos(oh) + dy * math.sin(oh)
    lateral = -dx * math.sin(oh) + dy * math.cos(oh)
    return {"forward": forward, "lateral": lateral, "dist": math.hypot(dx, dy)}


def _estimate_from_bbox(bbox, frame, observer):
    # Very rough depth estimate: larger bbox height -> closer object.
    # Use intrinsics to back-calc an approximate forward distance.
    _, _, _, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_h = max(4.0, h - bbox[1])
    # crude mapping
    forward = max(1.0, INTRINSICS["fy"] * 1.2 / bbox_h)
    # assume centered laterally
    lateral = 0.0
    ox = float(observer.get("x", 0.0))
    oy = float(observer.get("y", 0.0))
    oh = float(observer.get("heading_rad", 0.0))

    wx = ox + forward * math.cos(oh) - lateral * math.sin(oh)
    wy = oy + forward * math.sin(oh) + lateral * math.cos(oh)

    return {
        "position_world": {"x": wx, "y": wy},
        "position_observer": {"forward": forward, "lateral": lateral, "dist": math.hypot(forward, lateral)},
    }
