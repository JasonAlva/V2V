"""Minimal project_3d shim for pipeline tests.

Provides small implementations of the functions `run_live_pipeline` expects so
we can run the end-to-end test. This is intentionally lightweight and not a
full camera projection — it's sufficient for simulation/testing purposes.

Fixes applied vs original:
  1. y_img now uses actual vertical projection (was multiplied by 0.0).
  2. _estimate_from_bbox now recovers lateral offset from bbox x-centre.
  3. bbox_h variable renamed/clarified (h was misleadingly reused for bbox[3]).
  4. _iou and _estimate_from_bbox promoted to public API (underscore removed).
  5. frame parameter annotated clearly; unused but kept for API compatibility.
"""

import math
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Simple pinhole-camera intrinsics used for crude projection estimates.
# Override by calling configure_intrinsics() before using the module.
# ---------------------------------------------------------------------------
INTRINSICS: Dict[str, float] = {
    "fx": 600.0,
    "fy": 600.0,
    "cx": 400.0,
    "cy": 300.0,
    "width": 800.0,
    "height": 600.0,
    # Assumed object half-height in world metres (used for y projection).
    "obj_half_height_m": 0.75,
}


def configure_intrinsics(**kwargs: float) -> None:
    """Override any intrinsic parameter at runtime."""
    INTRINSICS.update(kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _world_to_observer(
    x: float,
    y: float,
    observer: Dict[str, Any],
) -> Dict[str, float]:
    """Convert a world (x, y) point into the observer's local frame.

    Returns a dict with keys: forward, lateral, dist.
    """
    ox = float(observer.get("x", 0.0))
    oy = float(observer.get("y", 0.0))
    oh = float(observer.get("heading_rad", 0.0))

    dx = x - ox
    dy = y - oy

    forward = dx * math.cos(oh) + dy * math.sin(oh)
    lateral = -dx * math.sin(oh) + dy * math.cos(oh)

    return {
        "forward": forward,
        "lateral": lateral,
        "dist": math.hypot(dx, dy),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bbox_from_obj(
    obj: Dict[str, Any],
    frame: Dict[str, Any],  # kept for API compatibility; unused in shim
    observer: Dict[str, Any],
) -> Optional[List[float]]:
    """Project a world object into an image bounding box [x1, y1, x2, y2].

    Returns None when the object is behind the observer (forward <= 0).
    """
    try:
        local = _world_to_observer(
            float(obj.get("x", 0.0)),
            float(obj.get("y", 0.0)),
            observer,
        )
    except Exception:
        return None

    forward = local["forward"]
    lateral = local["lateral"]

    if forward <= 0.1:
        return None

    fx = INTRINSICS["fx"]
    fy = INTRINSICS["fy"]
    cx = INTRINSICS["cx"]
    cy = INTRINSICS["cy"]
    half_h = INTRINSICS["obj_half_height_m"]

    # FIX 1: x_img uses lateral offset correctly.
    x_img = cx + (lateral * fx) / forward

    # FIX 2: y_img now uses a non-zero vertical offset (object top/bottom).
    # We assume the camera is mounted at roughly the same height as the object
    # centre, so the vertical angle is atan2(half_h, forward).
    y_img = cy - (half_h * fy) / forward  # top edge of object in image

    # Box size: larger when closer.
    w = max(8.0, fx * 2.0 / max(forward, 1.0))
    h = max(8.0, fy * (2.0 * half_h) / max(forward, 1.0))

    x1 = x_img - w / 2.0
    y1 = y_img - h / 2.0
    x2 = x_img + w / 2.0
    y2 = y_img + h / 2.0

    return [x1, y1, x2, y2]


def iou(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """Compute Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
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

    return inter / union if union > 0.0 else 0.0


def world_to_local(
    x: float,
    y: float,
    observer: Dict[str, Any],
) -> Dict[str, float]:
    """Public alias for _world_to_observer (used by run_live_pipeline)."""
    return _world_to_observer(x, y, observer)


def estimate_from_bbox(
    bbox: List[float],
    frame: Dict[str, Any],  # kept for API compatibility; unused in shim
    observer: Dict[str, Any],
) -> Dict[str, Any]:
    """Back-project an image bbox to an approximate world position.

    Fixes vs original:
      - Recovers lateral offset from bbox x-centre (was always 0.0).
      - Uses correct bbox height (y2 - y1), not a raw coordinate.
      - Reconstructs world position properly from forward + lateral.
    """
    x1, y1, x2, y2 = bbox

    # FIX 3: bbox height is (y2 - y1), not a raw y coordinate.
    bbox_h = max(4.0, y2 - y1)
    half_h = INTRINSICS["obj_half_height_m"]
    fy = INTRINSICS["fy"]
    fx = INTRINSICS["fx"]
    cx = INTRINSICS["cx"]

    # Recover forward distance from vertical size of box.
    forward = max(1.0, fy * (2.0 * half_h) / bbox_h)

    # FIX 4: Recover lateral offset from horizontal centre of bbox.
    x_centre = (x1 + x2) / 2.0
    lateral = (x_centre - cx) * forward / fx

    ox = float(observer.get("x", 0.0))
    oy = float(observer.get("y", 0.0))
    oh = float(observer.get("heading_rad", 0.0))

    # Reconstruct world position from observer frame.
    wx = ox + forward * math.cos(oh) - lateral * math.sin(oh)
    wy = oy + forward * math.sin(oh) + lateral * math.cos(oh)

    dist = math.hypot(forward, lateral)

    return {
        "position_world": {"x": wx, "y": wy},
        "position_observer": {
            "forward": forward,
            "lateral": lateral,
            "dist": dist,
        },
    }


# ---------------------------------------------------------------------------
# Back-compat aliases (in case existing code uses the private names)
# ---------------------------------------------------------------------------
_bbox_from_obj = bbox_from_obj
_iou = iou
_world_to_local = world_to_local
_estimate_from_bbox = estimate_from_bbox