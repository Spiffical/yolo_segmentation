#!/usr/bin/env python3
"""
Validate converted YOLO segmentation dataset.

Visualizes samples to verify the RLE-to-polygon conversion was successful.
"""

import argparse
import os
import sys
import random
from pathlib import Path
from typing import List, Tuple
import yaml

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install opencv-python numpy")
    sys.exit(1)


def parse_yolo_segmentation_line(line: str) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Parse a YOLO segmentation label line.
    
    Returns:
        Tuple of (class_id, list of (x, y) normalized coordinates)
    """
    parts = line.strip().split()
    if len(parts) < 7:  # class_id + at least 3 points (6 coords)
        return None, []
    
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    
    # Group into (x, y) pairs
    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    return class_id, points


def draw_segmentation(
    image: np.ndarray,
    points: List[Tuple[float, float]],
    class_id: int,
    class_name: str = None,
    color: Tuple[int, int, int] = None
) -> np.ndarray:
    """Draw a segmentation polygon on an image."""
    h, w = image.shape[:2]
    
    # Generate color from class_id if not provided
    if color is None:
        random.seed(class_id)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    # Convert normalized coords to pixel coords
    pts = np.array([(int(x * w), int(y * h)) for x, y in points], dtype=np.int32)
    
    # Draw filled polygon with transparency
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
    
    # Draw polygon outline
    cv2.polylines(image, [pts], True, color, 2)
    
    # Draw label
    if class_name:
        label_pos = (pts[0][0], max(pts[0][1] - 10, 20))
        cv2.putText(image, f"{class_id}: {class_name}", label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


def load_class_names(dataset_yaml: str) -> dict:
    """Load class names from dataset YAML."""
    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    names = data.get('names', {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return names


def visualize_sample(
    image_path: str,
    label_path: str,
    class_names: dict,
    output_path: str = None,
    show: bool = True
) -> np.ndarray:
    """Visualize a single sample with its segmentation masks."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Parse label file
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            class_id, points = parse_yolo_segmentation_line(line)
            if class_id is not None and len(points) >= 3:
                class_name = class_names.get(class_id, f"class_{class_id}")
                image = draw_segmentation(image, points, class_id, class_name)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")
    
    # Show if requested
    if show:
        # Resize for display if too large
        max_dim = 1200
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
        cv2.imshow('Segmentation Preview', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Validate YOLO segmentation dataset conversion"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='data/yolo_dataset',
        help='Path to converted YOLO dataset directory'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='Dataset split to visualize'
    )
    parser.add_argument(
        '--n_samples', '-n',
        type=int,
        default=5,
        help='Number of random samples to visualize'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Directory to save visualization images (optional)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display images (only save)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for sample selection'
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    
    # Check dataset exists
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        print("Run the conversion script first:")
        print("  python scripts/convert_coco_to_yolo.py")
        sys.exit(1)
    
    # Load configuration
    yaml_path = dataset_path / 'dataset.yaml'
    if not yaml_path.exists():
        print(f"Dataset YAML not found: {yaml_path}")
        sys.exit(1)
    
    class_names = load_class_names(str(yaml_path))
    print(f"Loaded {len(class_names)} class names")
    
    # Find image and label directories
    images_dir = dataset_path / 'images' / args.split
    labels_dir = dataset_path / 'labels' / args.split
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Get list of label files (as they have the annotations)
    label_files = list(labels_dir.glob('*.txt'))
    if not label_files:
        print(f"No label files found in: {labels_dir}")
        sys.exit(1)
    
    print(f"Found {len(label_files)} label files in {args.split} split")
    
    # Filter to files with content (non-empty)
    non_empty = [f for f in label_files if f.stat().st_size > 0]
    print(f"  - {len(non_empty)} with annotations")
    
    # Select random samples
    if args.seed is not None:
        random.seed(args.seed)
    
    samples = random.sample(non_empty, min(args.n_samples, len(non_empty)))
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize samples
    for i, label_path in enumerate(samples):
        stem = label_path.stem
        
        # Find corresponding image (could be .jpg or .png)
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            print(f"Image not found for: {stem}")
            continue
        
        print(f"\n[{i+1}/{len(samples)}] {stem}")
        
        output_path = None
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"{stem}_viz.jpg")
        
        visualize_sample(
            str(image_path),
            str(label_path),
            class_names,
            output_path=output_path,
            show=not args.no_show
        )
    
    print("\nValidation complete!")


if __name__ == '__main__':
    main()
