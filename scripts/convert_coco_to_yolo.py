#!/usr/bin/env python3
"""
Convert COCO JSON with RLE segmentation masks to YOLO polygon format.

YOLO segmentation format (per line in .txt file):
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    
Where all coordinates are normalized to [0, 1] relative to image dimensions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    load_coco_json,
    split_dataset,
    filter_by_categories,
    remap_categories_to_binary,
    create_category_mapping,
    get_top_categories,
)

try:
    from pycocotools import mask as mask_utils
    import cv2
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install pycocotools opencv-python")
    sys.exit(1)


def decode_rle_to_mask(segmentation: Dict, height: int, width: int) -> np.ndarray:
    """
    Decode RLE or polygon segmentation to binary mask.
    
    Args:
        segmentation: COCO segmentation dict (RLE or polygon format)
        height: Image height
        width: Image width
        
    Returns:
        Binary mask as numpy array
    """
    if isinstance(segmentation, dict):
        # RLE format
        if 'counts' in segmentation:
            if isinstance(segmentation['counts'], str):
                # Compressed RLE
                rle = segmentation
            else:
                # Uncompressed RLE
                rle = mask_utils.frPyObjects(segmentation, height, width)
            return mask_utils.decode(rle)
    elif isinstance(segmentation, list):
        # Polygon format - convert to RLE first
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle)
    
    return np.zeros((height, width), dtype=np.uint8)


def mask_to_polygons(
    mask: np.ndarray,
    min_area: int = 10,
    epsilon_factor: float = 0.001
) -> List[np.ndarray]:
    """
    Convert binary mask to polygon contours.
    
    Args:
        mask: Binary mask (H, W)
        min_area: Minimum contour area to keep
        epsilon_factor: Simplification factor (multiplied by arc length)
        
    Returns:
        List of polygon contours as numpy arrays
    """
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,  # Only external contours
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplify contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        
        # Need at least 3 points for a valid polygon
        if len(simplified) >= 3:
            polygons.append(simplified.squeeze())
    
    return polygons


def polygon_to_yolo_format(
    polygon: np.ndarray,
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Convert polygon coordinates to YOLO normalized format.
    
    Args:
        polygon: Polygon as (N, 2) array of (x, y) coordinates
        img_width: Image width for normalization
        img_height: Image height for normalization
        
    Returns:
        Flattened list of normalized coordinates [x1, y1, x2, y2, ...]
    """
    normalized = []
    for point in polygon:
        x = point[0] / img_width
        y = point[1] / img_height
        # Clamp to [0, 1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        normalized.extend([x, y])
    return normalized


def process_image(
    image_info: Dict,
    annotations: List[Dict],
    category_id_to_idx: Dict[int, int],
    output_dir: Path,
    min_polygon_area: int = 10
) -> Tuple[str, int, int]:
    """
    Process a single image's annotations and write YOLO format label file.
    
    Args:
        image_info: COCO image dict
        annotations: List of annotations for this image
        category_id_to_idx: Mapping from category ID to YOLO class index
        output_dir: Directory to write label files
        min_polygon_area: Minimum polygon area to keep
        
    Returns:
        Tuple of (filename, num_annotations, num_polygons_written)
    """
    file_name = image_info['file_name']
    img_width = image_info['width']
    img_height = image_info['height']
    
    # Label file has same name but .txt extension
    label_name = Path(file_name).stem + '.txt'
    label_path = output_dir / label_name
    
    lines = []
    total_polygons = 0
    
    for ann in annotations:
        cat_id = ann['category_id']
        if cat_id not in category_id_to_idx:
            continue
        
        class_idx = category_id_to_idx[cat_id]
        segmentation = ann.get('segmentation')
        
        if segmentation is None:
            continue
        
        try:
            # Decode RLE to mask
            mask = decode_rle_to_mask(segmentation, img_height, img_width)
            
            # Convert mask to polygons
            polygons = mask_to_polygons(mask, min_area=min_polygon_area)
            
            # Write each polygon as a separate line
            for polygon in polygons:
                if len(polygon) >= 3:
                    coords = polygon_to_yolo_format(polygon, img_width, img_height)
                    coord_str = ' '.join(f'{c:.6f}' for c in coords)
                    lines.append(f'{class_idx} {coord_str}')
                    total_polygons += 1
                    
        except Exception as e:
            # Skip problematic annotations
            continue
    
    # Write label file (even if empty, for completeness)
    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return file_name, len(annotations), total_polygons


def convert_coco_to_yolo(
    coco_json_path: str,
    output_dir: str,
    image_dir: Optional[str] = None,
    val_ratio: float = 0.2,
    mode: str = 'all',  # 'all', 'top_n', 'binary'
    top_n: int = 50,
    min_annotations: int = 10,
    workers: int = 8,
    seed: int = 42
) -> Dict:
    """
    Convert COCO JSON to YOLO segmentation format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Output directory for YOLO dataset
        image_dir: Directory containing images (for symlinking)
        val_ratio: Validation split ratio
        mode: Category mode - 'all', 'top_n', or 'binary'
        top_n: Number of top categories to keep (if mode='top_n')
        min_annotations: Minimum annotations per category
        workers: Number of parallel workers
        seed: Random seed
        
    Returns:
        Statistics dict
    """
    print(f"Loading COCO JSON from {coco_json_path}...")
    data = load_coco_json(coco_json_path)
    
    print(f"  - Images: {len(data['images'])}")
    print(f"  - Annotations: {len(data['annotations'])}")
    print(f"  - Categories: {len(data['categories'])}")
    
    # Apply category filtering based on mode
    if mode == 'binary':
        print("\nConverting to binary segmentation (object vs background)...")
        data = remap_categories_to_binary(data)
    elif mode == 'top_n':
        print(f"\nFiltering to top {top_n} categories by annotation count...")
        top_cats = set(get_top_categories(data, top_n))
        data = filter_by_categories(data, top_cats, min_annotations=min_annotations)
    elif min_annotations > 0:
        print(f"\nFiltering categories with at least {min_annotations} annotations...")
        data = filter_by_categories(data, min_annotations=min_annotations)
    
    print(f"After filtering:")
    print(f"  - Images: {len(data['images'])}")
    print(f"  - Annotations: {len(data['annotations'])}")
    print(f"  - Categories: {len(data['categories'])}")
    
    # Split into train/val
    print(f"\nSplitting dataset ({1-val_ratio:.0%} train, {val_ratio:.0%} val)...")
    train_data, val_data = split_dataset(data, val_ratio=val_ratio, seed=seed)
    
    print(f"  - Train: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"  - Val: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    
    # Create category mapping
    category_id_to_idx, class_names = create_category_mapping(data)
    
    # Setup output directories
    output_path = Path(output_dir)
    train_labels_dir = output_path / 'labels' / 'train'
    val_labels_dir = output_path / 'labels' / 'val'
    train_images_dir = output_path / 'images' / 'train'
    val_images_dir = output_path / 'images' / 'val'
    
    for d in [train_labels_dir, val_labels_dir, train_images_dir, val_images_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Process train and val sets
    stats = {'train': {}, 'val': {}}
    
    for split_name, split_data, labels_dir, images_dir in [
        ('train', train_data, train_labels_dir, train_images_dir),
        ('val', val_data, val_labels_dir, val_images_dir)
    ]:
        print(f"\nProcessing {split_name} split...")
        
        # Create image_id to annotations mapping
        img_to_anns = {}
        for ann in split_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Create image_id to image_info mapping
        id_to_img = {img['id']: img for img in split_data['images']}
        
        total_images = len(split_data['images'])
        total_anns = 0
        total_polygons = 0
        
        # Process images
        with tqdm(total=total_images, desc=f"Converting {split_name}") as pbar:
            for img_info in split_data['images']:
                img_id = img_info['id']
                anns = img_to_anns.get(img_id, [])
                
                file_name, n_anns, n_polys = process_image(
                    img_info,
                    anns,
                    category_id_to_idx,
                    labels_dir,
                    min_polygon_area=10
                )
                
                total_anns += n_anns
                total_polygons += n_polys
                
                # Create symlink for image if image_dir provided
                if image_dir:
                    src_path = Path(image_dir).absolute() / file_name
                    dst_path = images_dir / file_name
                    if src_path.exists() and not dst_path.exists():
                        try:
                            dst_path.symlink_to(src_path.absolute())
                        except OSError:
                            # Fallback to copy if symlink fails
                            import shutil
                            shutil.copy2(src_path, dst_path)
                
                pbar.update(1)
        
        stats[split_name] = {
            'images': total_images,
            'annotations': total_anns,
            'polygons': total_polygons
        }
        print(f"  - Wrote {total_polygons} polygons from {total_anns} annotations")
    
    # Write dataset YAML
    yaml_path = output_path / 'dataset.yaml'
    write_dataset_yaml(yaml_path, output_path, class_names)
    print(f"\nWrote dataset config to {yaml_path}")
    
    # Write class names file
    names_path = output_path / 'classes.txt'
    with open(names_path, 'w') as f:
        f.write('\n'.join(class_names))
    
    return stats


def write_dataset_yaml(yaml_path: Path, dataset_root: Path, class_names: List[str]):
    """Write YOLO dataset configuration YAML."""
    content = f"""# MBARI FathomNet Segmentation Dataset
# Auto-generated by convert_coco_to_yolo.py

path: {dataset_root.absolute()}
train: images/train
val: images/val

# Number of classes
nc: {len(class_names)}

# Class names
names:
"""
    for i, name in enumerate(class_names):
        # Escape special characters in YAML
        escaped = name.replace("'", "''")
        content += f"  {i}: '{escaped}'\n"
    
    with open(yaml_path, 'w') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO RLE segmentation to YOLO polygon format"
    )
    parser.add_argument(
        '--coco_json', '-j',
        type=str,
        default='data/seg_masks/train.json',
        help='Path to COCO JSON file'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='data/yolo_dataset',
        help='Output directory for YOLO dataset'
    )
    parser.add_argument(
        '--image_dir', '-i',
        type=str,
        default='/mnt/z/yolo/data/images/train',
        help='Directory containing source images (for symlinking)'
    )
    parser.add_argument(
        '--val_ratio', '-v',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['all', 'top_n', 'binary'],
        default='all',
        help='Category mode: all (all categories), top_n (top N categories), binary (object vs bg)'
    )
    parser.add_argument(
        '--top_n', '-n',
        type=int,
        default=50,
        help='Number of top categories to keep (when mode=top_n)'
    )
    parser.add_argument(
        '--min_annotations',
        type=int,
        default=10,
        help='Minimum annotations per category to include'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=8,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    stats = convert_coco_to_yolo(
        coco_json_path=args.coco_json,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        val_ratio=args.val_ratio,
        mode=args.mode,
        top_n=args.top_n,
        min_annotations=args.min_annotations,
        workers=args.workers,
        seed=args.seed
    )
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print("="*50)
    for split, s in stats.items():
        print(f"{split}: {s['images']} images, {s['polygons']} polygons")


if __name__ == '__main__':
    main()
