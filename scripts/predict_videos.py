#!/usr/bin/env python3
"""
Run YOLO segmentation inference on one or more local videos.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def collect_videos(source: Path, recursive: bool) -> List[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        return []

    if recursive:
        files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    else:
        files = [p for p in source.glob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(files)


def count_detections(result) -> int:
    if getattr(result, "boxes", None) is not None:
        return len(result.boxes)
    if getattr(result, "masks", None) is not None and getattr(result.masks, "data", None) is not None:
        return int(result.masks.data.shape[0])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO inference on local videos")
    parser.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--source", required=True, help="Video file or directory")
    parser.add_argument("--recursive", action="store_true", help="Recurse into source directory")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="onc_local")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--save-txt", action="store_true", help="Save per-frame labels")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence in saved labels")
    parser.add_argument("--classes", nargs="+", type=int, default=None, help="Optional class IDs to keep")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Install with: pip install ultralytics", file=sys.stderr)
        return 1

    model_path = Path(args.model).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    videos = collect_videos(source_path, recursive=args.recursive)
    if not videos:
        print(f"No videos found in source: {source_path}", file=sys.stderr)
        return 1

    model = YOLO(str(model_path))

    output_dir = Path(args.project).expanduser().resolve() / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for idx, video in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] {video}")
        frame_count = 0
        frames_with_detections = 0
        total_detections = 0

        results = model.predict(
            source=str(video),
            stream=True,
            save=True,
            project=str(Path(args.project).expanduser().resolve()),
            name=args.name,
            exist_ok=True,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            vid_stride=args.vid_stride,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            classes=args.classes,
            verbose=False,
        )

        for result in results:
            frame_count += 1
            n = count_detections(result)
            if n > 0:
                frames_with_detections += 1
            total_detections += n

        summary_rows.append(
            {
                "video": str(video),
                "frames": frame_count,
                "frames_with_detections": frames_with_detections,
                "total_detections": total_detections,
                "detection_rate": (frames_with_detections / frame_count) if frame_count else 0.0,
            }
        )

    summary_path = output_dir / "inference_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "frames",
                "frames_with_detections",
                "total_detections",
                "detection_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("")
    print(f"Annotated outputs: {output_dir}")
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
