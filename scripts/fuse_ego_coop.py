"""Phase 3: Hungarian matching + ego/coop 3D fusion."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


EGO_RE = re.compile(r"^frame_(\d{6})_ego_3d\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse ego and cooperative 3D detections.")
    parser.add_argument("--lifted", default="data/fused/lifted", help="Input lifted 3D directory")
    parser.add_argument("--out", default="data/fused", help="Output fused JSON directory")
    parser.add_argument("--max-match-dist", type=float, default=6.0, help="Max world-distance for ego/coop match")
    parser.add_argument("--ego-weight", type=float, default=0.55, help="Position blending weight for ego")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax = float(a["position_world"]["x"])
    ay = float(a["position_world"]["y"])
    bx = float(b["position_world"]["x"])
    by = float(b["position_world"]["y"])
    return math.hypot(ax - bx, ay - by)


def _blend_position(ego_obj: dict[str, Any], coop_obj: dict[str, Any], ego_weight: float) -> dict[str, float]:
    ex = float(ego_obj["position_world"]["x"])
    ey = float(ego_obj["position_world"]["y"])
    cx = float(coop_obj["position_world"]["x"])
    cy = float(coop_obj["position_world"]["y"])
    return {
        "x": ego_weight * ex + (1.0 - ego_weight) * cx,
        "y": ego_weight * ey + (1.0 - ego_weight) * cy,
    }


def _build_cost(ego_objs: list[dict[str, Any]], coop_objs: list[dict[str, Any]]) -> np.ndarray:
    if not ego_objs or not coop_objs:
        return np.zeros((len(ego_objs), len(coop_objs)), dtype=np.float64)

    cost = np.zeros((len(ego_objs), len(coop_objs)), dtype=np.float64)
    for i, e in enumerate(ego_objs):
        for j, c in enumerate(coop_objs):
            cost[i, j] = _distance(e, c)
    return cost


def _fuse_frame(
    frame_index: int,
    sim_time: float,
    ego_objs: list[dict[str, Any]],
    coop_objs: list[dict[str, Any]],
    max_match_dist: float,
    ego_weight: float,
) -> dict[str, Any]:
    fused_objects: list[dict[str, Any]] = []

    matched_ego: set[int] = set()
    matched_coop: set[int] = set()

    cost = _build_cost(ego_objs, coop_objs)
    if cost.size > 0:
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            dist = float(cost[r, c])
            if dist > max_match_dist:
                continue

            eobj = ego_objs[r]
            cobj = coop_objs[c]
            fused_pos = _blend_position(eobj, cobj, ego_weight)

            fused_objects.append(
                {
                    "track_id": eobj.get("track_id") or cobj.get("track_id") or f"fused_{r}_{c}",
                    "source": "fused",
                    "confidence": max(float(eobj.get("confidence", 0.0)), float(cobj.get("confidence", 0.0))),
                    "position_world": fused_pos,
                    "contributors": ["ego", "coop"],
                    "match_distance": dist,
                    "ego": eobj,
                    "coop": cobj,
                }
            )
            matched_ego.add(r)
            matched_coop.add(c)

    for i, obj in enumerate(ego_objs):
        if i in matched_ego:
            continue
        fused_objects.append(
            {
                "track_id": obj.get("track_id") or f"ego_{i}",
                "source": "ego_only",
                "confidence": float(obj.get("confidence", 0.0)),
                "position_world": obj.get("position_world", {}),
                "contributors": ["ego"],
                "match_distance": None,
                "ego": obj,
            }
        )

    for i, obj in enumerate(coop_objs):
        if i in matched_coop:
            continue
        fused_objects.append(
            {
                "track_id": obj.get("track_id") or f"coop_{i}",
                "source": "coop_only",
                "confidence": float(obj.get("confidence", 0.0)),
                "position_world": obj.get("position_world", {}),
                "contributors": ["coop"],
                "match_distance": None,
                "coop": obj,
            }
        )

    fused_objects.sort(
        key=lambda o: (
            float(o.get("position_world", {}).get("x", 0.0)),
            float(o.get("position_world", {}).get("y", 0.0)),
        )
    )

    return {
        "frame_index": frame_index,
        "sim_time": sim_time,
        "counts": {
            "ego": len(ego_objs),
            "coop": len(coop_objs),
            "fused": len(fused_objects),
            "matched_pairs": len(matched_ego),
        },
        "objects": fused_objects,
    }


def main() -> None:
    args = parse_args()
    lifted_dir = Path(args.lifted)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ego_files = sorted(p for p in lifted_dir.glob("frame_*_ego_3d.json") if EGO_RE.match(p.name))
    if not ego_files:
        print(f"[fuse_ego_coop] No ego lifted files found in {lifted_dir}")
        return

    print(f"[fuse_ego_coop] Fusing {len(ego_files)} frames...")
    written = 0
    for idx, ego_file in enumerate(ego_files, start=1):
        m = EGO_RE.match(ego_file.name)
        if m is None:
            continue
        frame_index = int(m.group(1))

        coop_file = lifted_dir / f"frame_{frame_index:06d}_coop_3d.json"
        ego_payload = _load_json(ego_file)
        coop_payload = _load_json(coop_file) if coop_file.exists() else {"objects_3d": []}

        fused = _fuse_frame(
            frame_index=frame_index,
            sim_time=float(ego_payload.get("sim_time", 0.0)),
            ego_objs=list(ego_payload.get("objects_3d", [])),
            coop_objs=list(coop_payload.get("objects_3d", [])),
            max_match_dist=args.max_match_dist,
            ego_weight=args.ego_weight,
        )

        out_file = out_dir / f"frame_{frame_index:06d}_fused.json"
        out_file.write_text(json.dumps(fused, indent=2), encoding="utf-8")
        written += 1

        if idx % 25 == 0 or idx == len(ego_files):
            print(f"  [fuse_ego_coop] {idx}/{len(ego_files)} done")

    print(f"[fuse_ego_coop] Wrote {written} fused frames to {out_dir}")


if __name__ == "__main__":
    main()
