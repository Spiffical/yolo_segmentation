#!/usr/bin/env python3
"""
Convert COCO annotations to YOLO detection or segmentation format.

Detection label format:
    <class_id> <x_center> <y_center> <width> <height>

Segmentation label format:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    apply_label_plan,
    create_category_mapping,
    filter_by_categories,
    get_top_categories,
    load_coco_json,
    remap_categories_to_binary,
    split_dataset,
)

try:
    import cv2
    from pycocotools import mask as mask_utils
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install pycocotools opencv-python")
    sys.exit(1)


DEFAULT_LABEL_PLANS = Path(__file__).resolve().parent.parent / "configs" / "label_plans.yaml"


def decode_rle_to_mask(segmentation: Dict, height: int, width: int) -> np.ndarray:
    """Decode COCO RLE or polygon segmentation into a binary mask."""
    if isinstance(segmentation, dict) and "counts" in segmentation:
        if isinstance(segmentation["counts"], str):
            rle = segmentation
        else:
            rle = mask_utils.frPyObjects(segmentation, height, width)
        return mask_utils.decode(rle)

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle)

    return np.zeros((height, width), dtype=np.uint8)


def mask_to_polygons(
    mask: np.ndarray,
    min_area: int = 10,
    epsilon_factor: float = 0.001,
) -> List[np.ndarray]:
    """Convert a binary mask to simplified polygons."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) >= 3:
            polygons.append(simplified.squeeze())

    return polygons


def polygon_to_yolo_format(polygon: np.ndarray, img_width: int, img_height: int) -> List[float]:
    """Convert polygon coordinates to normalized YOLO segmentation format."""
    normalized = []
    for point in polygon:
        x = max(0.0, min(1.0, point[0] / img_width))
        y = max(0.0, min(1.0, point[1] / img_height))
        normalized.extend([x, y])
    return normalized


def bbox_to_yolo_format(bbox: List[float], img_width: int, img_height: int) -> Optional[List[float]]:
    """Convert COCO XYWH bbox to normalized YOLO detection format."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    x, y, w, h = bbox
    if w <= 0 or h <= 0 or img_width <= 0 or img_height <= 0:
        return None

    x1 = max(0.0, min(float(img_width), float(x)))
    y1 = max(0.0, min(float(img_height), float(y)))
    x2 = max(0.0, min(float(img_width), float(x + w)))
    y2 = max(0.0, min(float(img_height), float(y + h)))

    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = clipped_w / img_width
    height = clipped_h / img_height
    return [x_center, y_center, width, height]


def process_image(
    *,
    task: str,
    image_info: Dict,
    annotations: List[Dict],
    category_id_to_idx: Dict[int, int],
    output_dir: Path,
    min_polygon_area: int = 10,
) -> Tuple[str, int, int]:
    """Write one YOLO label file for a single image."""
    file_name = image_info["file_name"]
    img_width = image_info["width"]
    img_height = image_info["height"]
    label_path = output_dir / f"{Path(file_name).stem}.txt"

    lines: List[str] = []
    seen_items = set()
    items_written = 0

    for ann in annotations:
        cat_id = ann.get("category_id")
        if cat_id not in category_id_to_idx:
            continue

        class_idx = category_id_to_idx[cat_id]

        try:
            if task == "detect":
                coords = bbox_to_yolo_format(ann.get("bbox"), img_width, img_height)
                if coords is None:
                    continue
                signature = (class_idx, tuple(round(c, 6) for c in coords))
                if signature in seen_items:
                    continue
                seen_items.add(signature)
                coord_str = " ".join(f"{c:.6f}" for c in coords)
                lines.append(f"{class_idx} {coord_str}")
                items_written += 1
                continue

            segmentation = ann.get("segmentation")
            if segmentation is None:
                continue

            mask = decode_rle_to_mask(segmentation, img_height, img_width)
            polygons = mask_to_polygons(mask, min_area=min_polygon_area)
            for polygon in polygons:
                if len(polygon) < 3:
                    continue
                coords = polygon_to_yolo_format(polygon, img_width, img_height)
                signature = (class_idx, tuple(round(c, 4) for c in coords))
                if signature in seen_items:
                    continue
                seen_items.add(signature)
                coord_str = " ".join(f"{c:.6f}" for c in coords)
                lines.append(f"{class_idx} {coord_str}")
                items_written += 1
        except Exception:
            continue

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return file_name, len(annotations), items_written


def write_dataset_yaml(yaml_path: Path, dataset_root: Path, class_names: List[str], task: str) -> None:
    """Write Ultralytics dataset YAML."""
    content = (
        f"# MBARI FathomNet {task} dataset\n"
        "# Auto-generated by convert_coco_to_yolo.py\n\n"
        f"path: {dataset_root.absolute()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(class_names)}\n\n"
        "names:\n"
    )
    for i, name in enumerate(class_names):
        escaped = name.replace("'", "''")
        content += f"  {i}: '{escaped}'\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


def write_conversion_summary(summary_path: Path, payload: Dict) -> None:
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def convert_coco_to_yolo(
    *,
    coco_json_path: str,
    output_dir: str,
    image_dir: Optional[str] = None,
    task: str = "segment",
    val_ratio: float = 0.2,
    mode: str = "all",
    top_n: int = 50,
    min_annotations: int = 10,
    workers: int = 8,
    seed: int = 42,
    label_plan: Optional[str] = None,
    label_plans_file: Optional[str] = None,
) -> Dict:
    """Convert COCO annotations into a YOLO dataset."""
    print(f"Loading COCO JSON from {coco_json_path}...")
    data = load_coco_json(coco_json_path)

    source_stats = {
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "categories": len(data["categories"]),
    }
    print(f"  - Images: {source_stats['images']}")
    print(f"  - Annotations: {source_stats['annotations']}")
    print(f"  - Categories: {source_stats['categories']}")

    label_plan_summary = None
    if mode == "binary":
        print("\nConverting to binary labels (object vs background)...")
        data = remap_categories_to_binary(data)
    elif mode == "top_n":
        print(f"\nFiltering to top {top_n} categories by annotation count...")
        top_cats = set(get_top_categories(data, top_n))
        data = filter_by_categories(data, top_cats, min_annotations=min_annotations)
    elif mode == "coarse":
        if not label_plan:
            raise ValueError("--label_plan is required when --mode coarse")
        plans_path = label_plans_file or str(DEFAULT_LABEL_PLANS)
        print(f"\nApplying coarse label plan '{label_plan}' from {plans_path}...")
        data, label_plan_summary = apply_label_plan(data, plans_path, label_plan)
    elif min_annotations > 0:
        print(f"\nFiltering categories with at least {min_annotations} annotations...")
        data = filter_by_categories(data, min_annotations=min_annotations)

    filtered_stats = {
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "categories": len(data["categories"]),
    }
    print("After filtering/remapping:")
    print(f"  - Images: {filtered_stats['images']}")
    print(f"  - Annotations: {filtered_stats['annotations']}")
    print(f"  - Categories: {filtered_stats['categories']}")
    if label_plan_summary is not None:
        print(
            f"  - Label plan coverage: {label_plan_summary['mapped_annotation_count']} / "
            f"{source_stats['annotations']} annotations "
            f"({label_plan_summary['coverage_fraction'] * 100:.2f}%)"
        )

    print(f"\nSplitting dataset ({1 - val_ratio:.0%} train, {val_ratio:.0%} val)...")
    train_data, val_data = split_dataset(data, val_ratio=val_ratio, seed=seed)
    print(f"  - Train: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"  - Val: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")

    category_id_to_idx, class_names = create_category_mapping(data)

    output_path = Path(output_dir)
    train_labels_dir = output_path / "labels" / "train"
    val_labels_dir = output_path / "labels" / "val"
    train_images_dir = output_path / "images" / "train"
    val_images_dir = output_path / "images" / "val"
    for directory in (train_labels_dir, val_labels_dir, train_images_dir, val_images_dir):
        directory.mkdir(parents=True, exist_ok=True)

    instance_label = "boxes" if task == "detect" else "polygons"
    stats = {"task": task, "train": {}, "val": {}}

    for split_name, split_data, labels_dir, images_dir in (
        ("train", train_data, train_labels_dir, train_images_dir),
        ("val", val_data, val_labels_dir, val_images_dir),
    ):
        print(f"\nProcessing {split_name} split...")
        img_to_anns: Dict[int, List[Dict]] = {}
        for ann in split_data["annotations"]:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

        total_images = len(split_data["images"])
        total_anns = 0
        total_items = 0

        with tqdm(total=total_images, desc=f"Converting {split_name}") as pbar:
            for img_info in split_data["images"]:
                anns = img_to_anns.get(img_info["id"], [])
                file_name, n_anns, n_items = process_image(
                    task=task,
                    image_info=img_info,
                    annotations=anns,
                    category_id_to_idx=category_id_to_idx,
                    output_dir=labels_dir,
                    min_polygon_area=10,
                )
                total_anns += n_anns
                total_items += n_items

                if image_dir:
                    src_path = Path(image_dir).expanduser().resolve() / file_name
                    dst_path = images_dir / file_name
                    if src_path.exists() and not dst_path.exists():
                        try:
                            dst_path.symlink_to(src_path)
                        except OSError:
                            shutil.copy2(src_path, dst_path)

                pbar.update(1)

        stats[split_name] = {
            "images": total_images,
            "annotations": total_anns,
            instance_label: total_items,
        }
        print(f"  - Wrote {total_items} {instance_label} from {total_anns} annotations")

    yaml_path = output_path / "dataset.yaml"
    write_dataset_yaml(yaml_path, output_path, class_names, task)
    names_path = output_path / "classes.txt"
    with open(names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    summary_payload = {
        "task": task,
        "mode": mode,
        "top_n": top_n,
        "min_annotations": min_annotations,
        "seed": seed,
        "val_ratio": val_ratio,
        "workers_requested": workers,
        "label_plan": label_plan,
        "label_plans_file": str(Path(label_plans_file).expanduser().resolve()) if label_plans_file else "",
        "source": source_stats,
        "filtered": filtered_stats,
        "splits": stats,
        "class_names": class_names,
        "label_plan_summary": label_plan_summary,
    }
    write_conversion_summary(output_path / "conversion_summary.json", summary_payload)

    print(f"\nWrote dataset config to {yaml_path}")
    print(f"Wrote conversion summary to {output_path / 'conversion_summary.json'}")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO detection or segmentation format")
    parser.add_argument(
        "--coco_json",
        "-j",
        type=str,
        default="data/seg_masks/train.json",
        help="Path to COCO JSON file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/yolo_dataset",
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--image_dir",
        "-i",
        type=str,
        default="/mnt/z/yolo/data/images/train",
        help="Directory containing source images (for symlinking)",
    )
    parser.add_argument(
        "--task",
        choices=["detect", "segment"],
        default="segment",
        help="YOLO task to emit labels for",
    )
    parser.add_argument(
        "--val_ratio",
        "-v",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["all", "top_n", "binary", "coarse"],
        default="all",
        help="Label mode: all, top_n, binary, or coarse",
    )
    parser.add_argument(
        "--top_n",
        "-n",
        type=int,
        default=50,
        help="Number of top categories to keep (for mode=top_n)",
    )
    parser.add_argument(
        "--min_annotations",
        type=int,
        default=10,
        help="Minimum annotations per category to include",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="Requested conversion worker count (currently informational)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--label_plan",
        "--label-plan",
        type=str,
        default=None,
        help="Named coarse label plan to apply when mode=coarse",
    )
    parser.add_argument(
        "--label_plans_file",
        "--label-plans-file",
        type=str,
        default=str(DEFAULT_LABEL_PLANS),
        help="Path to YAML file containing coarse label plans",
    )

    args = parser.parse_args()
    convert_coco_to_yolo(
        coco_json_path=args.coco_json,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        task=args.task,
        val_ratio=args.val_ratio,
        mode=args.mode,
        top_n=args.top_n,
        min_annotations=args.min_annotations,
        workers=args.workers,
        seed=args.seed,
        label_plan=args.label_plan,
        label_plans_file=args.label_plans_file,
    )


if __name__ == "__main__":
    main()
