"""Run the full SUMO V2V cooperative perception pipeline end-to-end."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all pipeline stages in order.")

    parser.add_argument("--cfg", default="sumo_scenario/scenario.sumocfg", help="SUMO config path")
    parser.add_argument("--ego", default="ego_vehicle", help="Ego vehicle ID")
    parser.add_argument("--coop", default="coop_vehicle", help="Cooperative vehicle ID")

    parser.add_argument("--frames", default="data/frames", help="Phase 1 frame output dir")
    parser.add_argument("--images", default="data/images", help="Phase 2 image output dir")
    parser.add_argument("--detections", default="data/images/detections", help="Phase 2.5 detection output dir")
    parser.add_argument("--lifted", default="data/fused/lifted", help="Phase 2.75 lifted output dir")
    parser.add_argument("--fused", default="data/fused", help="Phase 3 fused output dir")
    parser.add_argument("--bev", default="data/fused/bev", help="Phase 4 BEV output dir")
    parser.add_argument("--video", default="data/fused/fused_bev.mp4", help="Phase 4 video output path")

    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name/path")
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
    parser.add_argument("--step-length", type=float, default=0.05, help="SUMO step length")
    parser.add_argument("--radius", type=float, default=80.0, help="Neighborhood radius in meters")
    parser.add_argument("--no-coop-images", action="store_true", help="Render only ego images")

    parser.add_argument("--skip-detection", action="store_true", help="Skip YOLO stage")
    parser.add_argument("--skip-video", action="store_true", help="Skip MP4 export")
    parser.add_argument("--clean", action="store_true", help="Delete prior outputs before running")

    return parser.parse_args()


def run_stage(name: str, cmd: list[str], cwd: Path) -> float:
    print("\n" + "=" * 80)
    print(f"[PIPELINE] {name}")
    print("=" * 80)
    print("[CMD] " + " ".join(cmd))

    start = time.perf_counter()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    duration = time.perf_counter() - start

    print(f"[PIPELINE] {name} completed in {duration:.2f}s")
    return duration


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    python_exe = sys.executable

    frames = repo_root / args.frames
    images = repo_root / args.images
    detections = repo_root / args.detections
    lifted = repo_root / args.lifted
    fused = repo_root / args.fused
    bev = repo_root / args.bev
    video = repo_root / args.video

    if args.clean:
        print("[PIPELINE] Cleaning previous outputs...")
        remove_path(frames)
        remove_path(images)
        remove_path(detections)
        remove_path(lifted)
        remove_path(bev)
        remove_path(video)

    frames.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    detections.mkdir(parents=True, exist_ok=True)
    lifted.mkdir(parents=True, exist_ok=True)
    fused.mkdir(parents=True, exist_ok=True)
    bev.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)

    timings: list[tuple[str, float]] = []

    sumo_cmd = [
        python_exe,
        "scripts/run_sumo.py",
        "--cfg",
        args.cfg,
        "--ego",
        args.ego,
        "--coop",
        args.coop,
        "--out",
        args.frames,
        "--step-length",
        str(args.step_length),
        "--radius",
        str(args.radius),
    ]
    if args.gui:
        sumo_cmd.append("--gui")
    timings.append(("Phase 1: SUMO", run_stage("Phase 1: SUMO", sumo_cmd, repo_root)))

    image_cmd = [
        python_exe,
        "scripts/generate_images.py",
        "--frames",
        args.frames,
        "--out",
        args.images,
    ]
    if args.no_coop_images:
        image_cmd.append("--no-coop")
    timings.append(("Phase 2: Synthetic Images", run_stage("Phase 2: Synthetic Images", image_cmd, repo_root)))

    if not args.skip_detection:
        detect_cmd = [
            python_exe,
            "scripts/run_detection.py",
            "--images",
            args.images,
            "--out",
            args.detections,
            "--model",
            args.model,
        ]
        timings.append(("Phase 2.5: Detection", run_stage("Phase 2.5: Detection", detect_cmd, repo_root)))
    else:
        print("[PIPELINE] Skipping detection stage (--skip-detection)")

    lift_cmd = [
        python_exe,
        "scripts/project_3d.py",
        "--frames",
        args.frames,
        "--detections",
        args.detections,
        "--out",
        args.lifted,
    ]
    timings.append(("Phase 2.75: 3D Lift", run_stage("Phase 2.75: 3D Lift", lift_cmd, repo_root)))

    fuse_cmd = [
        python_exe,
        "scripts/fuse_ego_coop.py",
        "--lifted",
        args.lifted,
        "--out",
        args.fused,
    ]
    timings.append(("Phase 3: Fusion", run_stage("Phase 3: Fusion", fuse_cmd, repo_root)))

    vis_cmd = [
        python_exe,
        "scripts/visualize.py",
        "--fused",
        args.fused,
        "--frames",
        args.frames,
        "--images",
        args.images,
        "--out",
        args.bev,
    ]
    if not args.skip_video:
        vis_cmd.extend(["--video", args.video])
    timings.append(("Phase 4: Visualization", run_stage("Phase 4: Visualization", vis_cmd, repo_root)))

    total = sum(sec for _, sec in timings)
    print("\n" + "#" * 80)
    print("[PIPELINE] COMPLETE")
    for stage, seconds in timings:
        print(f"  - {stage:<28} {seconds:8.2f}s")
    print(f"  - {'Total':<28} {total:8.2f}s")
    print("#" * 80)

    print("\nOutputs:")
    print(f"  Frames      : {frames}")
    print(f"  Images      : {images}")
    print(f"  Detections  : {detections}")
    print(f"  Lifted 3D   : {lifted}")
    print(f"  Fused       : {fused}")
    print(f"  BEV frames  : {bev}")
    if not args.skip_video:
        print(f"  BEV video   : {video}")


if __name__ == "__main__":
    main()
