"""Phase 2.5: Run YOLOv8 inference on synthetic ego/coop frames."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FRAME_RE = re.compile(r"^frame_(\d{6})_(ego|coop)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 on generated synthetic frames.")
    parser.add_argument("--images", default="data/images", help="Input image directory")
    parser.add_argument("--out", default="data/images/detections", help="Output detection JSON directory")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path or name")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO predictions",
    )
    return parser.parse_args()


def _collect_images(images_dir: Path) -> list[tuple[int, str, Path]]:
    items: list[tuple[int, str, Path]] = []
    for path in sorted(images_dir.glob("frame_*_*.png")):
        m = FRAME_RE.match(path.name)
        if m is None:
            continue
        items.append((int(m.group(1)), m.group(2), path))
    return items


def _try_load_model(model_name: str):
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency optional at runtime
        print(f"[run_detection] ultralytics import failed: {exc}")
        return None

    try:
        return YOLO(model_name)
    except Exception as exc:  # pragma: no cover
        print(f"[run_detection] model load failed ({model_name}): {exc}")
        return None


def _infer_one(model, image_path: Path, conf: float) -> list[dict[str, Any]]:
    if model is None:
        return []

    results = model.predict(str(image_path), conf=conf, verbose=False)
    if not results:
        return []

    result = results[0]
    names = result.names
    detections: list[dict[str, Any]] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        if isinstance(names, dict):
            class_name = str(names.get(cls_id, str(cls_id)))
        elif isinstance(names, list) and 0 <= cls_id < len(names):
            class_name = str(names[cls_id])
        else:
            class_name = str(cls_id)
        detections.append(
            {
                "bbox": [float(v) for v in xyxy],
                "confidence": confidence,
                "class_id": cls_id,
                "class_name": class_name,
            }
        )

    return detections


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_items = _collect_images(images_dir)
    if not image_items:
        print(f"[run_detection] No matching images found in {images_dir}")
        return

    model = _try_load_model(args.model)
    if model is None:
        print("[run_detection] Continuing with empty detections (model unavailable).")

    print(f"[run_detection] Processing {len(image_items)} images...")
    for idx, (frame_index, view, image_path) in enumerate(image_items, start=1):
        detections = _infer_one(model, image_path, args.conf)
        payload = {
            "frame_index": frame_index,
            "view": view,
            "image_file": str(image_path),
            "model": args.model,
            "conf_threshold": args.conf,
            "detections": detections,
        }

        out_file = out_dir / f"frame_{frame_index:06d}_{view}.json"
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if idx % 25 == 0 or idx == len(image_items):
            print(f"  [run_detection] {idx}/{len(image_items)} done")

    print(f"[run_detection] Detection JSONs written to {out_dir}")


if __name__ == "__main__":
    main()
