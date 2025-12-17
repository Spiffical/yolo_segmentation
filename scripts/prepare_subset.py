#!/usr/bin/env python3
"""
Prepare a small subset of the dataset for quick testing.

This creates a minimal dataset for validating the training pipeline
before running on the full dataset.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Set


def create_subset(
    coco_json_path: str,
    output_dir: str,
    n_images: int = 100,
    image_dir: str = None,
    seed: int = 42,
    copy_images: bool = False
) -> Dict:
    """
    Create a small subset of the COCO dataset.
    
    Args:
        coco_json_path: Path to full COCO JSON
        output_dir: Output directory for subset
        n_images: Number of images to include
        image_dir: Source image directory
        seed: Random seed
        copy_images: If True, copy images; else symlink
        
    Returns:
        Statistics dict
    """
    random.seed(seed)
    
    print(f"Loading COCO JSON from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    print(f"Full dataset: {len(images)} images, {len(annotations)} annotations")
    
    # Build image_id -> annotations mapping
    img_to_anns: Dict[int, List] = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Filter to images that have annotations
    images_with_anns = [img for img in images if img['id'] in img_to_anns]
    print(f"Images with annotations: {len(images_with_anns)}")
    
    # Further filter to images that exist in image_dir (if provided)
    if image_dir:
        available_images = []
        for img in images_with_anns:
            img_path = Path(image_dir) / img['file_name']
            if img_path.exists():
                available_images.append(img)
        images_with_anns = available_images
        print(f"Images available in {image_dir}: {len(images_with_anns)}")
    
    # Sample N images
    n_images = min(n_images, len(images_with_anns))
    selected_images = random.sample(images_with_anns, n_images)
    selected_img_ids = {img['id'] for img in selected_images}
    
    print(f"Selected {n_images} images for subset")
    
    # Get annotations for selected images
    selected_anns = [ann for ann in annotations if ann['image_id'] in selected_img_ids]
    
    # Get categories used in selected annotations
    used_cat_ids = {ann['category_id'] for ann in selected_anns}
    selected_cats = [cat for cat in categories if cat['id'] in used_cat_ids]
    
    print(f"Subset: {len(selected_images)} images, {len(selected_anns)} annotations, {len(selected_cats)} categories")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save subset JSON
    subset_data = {
        'images': selected_images,
        'annotations': selected_anns,
        'categories': selected_cats
    }
    
    subset_json_path = output_path / 'subset.json'
    with open(subset_json_path, 'w') as f:
        json.dump(subset_data, f, indent=2)
    print(f"Saved subset JSON to {subset_json_path}")
    
    # Copy/symlink images if image_dir provided
    if image_dir:
        images_output_dir = output_path / 'images'
        images_output_dir.mkdir(exist_ok=True)
        
        copied = 0
        for img in selected_images:
            src = Path(image_dir) / img['file_name']
            dst = images_output_dir / img['file_name']
            
            if src.exists() and not dst.exists():
                if copy_images:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.symlink_to(src.absolute())
                    except OSError:
                        shutil.copy2(src, dst)
                copied += 1
        
        print(f"{'Copied' if copy_images else 'Linked'} {copied} images to {images_output_dir}")
    
    return {
        'n_images': len(selected_images),
        'n_annotations': len(selected_anns),
        'n_categories': len(selected_cats),
        'subset_json': str(subset_json_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create a small subset of the dataset for testing"
    )
    parser.add_argument(
        '--coco_json', '-j',
        type=str,
        default='data/seg_masks/train.json',
        help='Path to full COCO JSON file'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='data/subset',
        help='Output directory for subset'
    )
    parser.add_argument(
        '--n_images', '-n',
        type=int,
        default=100,
        help='Number of images to include in subset'
    )
    parser.add_argument(
        '--image_dir', '-i',
        type=str,
        default='/mnt/z/yolo/data/images/train',
        help='Source image directory'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy images instead of symlinking'
    )
    
    args = parser.parse_args()
    
    stats = create_subset(
        coco_json_path=args.coco_json,
        output_dir=args.output_dir,
        n_images=args.n_images,
        image_dir=args.image_dir,
        seed=args.seed,
        copy_images=args.copy
    )
    
    print("\n" + "="*50)
    print("Subset created successfully!")
    print("="*50)
    print(f"  Images: {stats['n_images']}")
    print(f"  Annotations: {stats['n_annotations']}")
    print(f"  Categories: {stats['n_categories']}")
    print(f"\nNext step: Convert to YOLO format:")
    print(f"  python scripts/convert_coco_to_yolo.py \\")
    print(f"    --coco_json {stats['subset_json']} \\")
    print(f"    --output_dir data/yolo_subset \\")
    print(f"    --image_dir {args.output_dir}/images")


if __name__ == '__main__':
    main()
