"""
Data utilities shared across YOLO detection and segmentation workflows.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import random
from collections import defaultdict

import yaml


def load_coco_json(json_path: str) -> Dict:
    """Load and return COCO format JSON data."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_category_stats(data: Dict) -> Dict[int, int]:
    """Get annotation count per category."""
    stats = defaultdict(int)
    for ann in data.get('annotations', []):
        stats[ann['category_id']] += 1
    return dict(stats)


def get_top_categories(data: Dict, n: int = 50) -> List[int]:
    """Get the top N categories by annotation count."""
    stats = get_category_stats(data)
    sorted_cats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    return [cat_id for cat_id, count in sorted_cats[:n]]


def split_dataset(
    data: Dict,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify: bool = True
) -> Tuple[Dict, Dict]:
    """
    Split COCO dataset into train and validation sets.
    
    Args:
        data: COCO format data dict
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility
        stratify: If True, try to maintain category distribution
        
    Returns:
        Tuple of (train_data, val_data) in COCO format
    """
    random.seed(seed)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    # Create image_id to annotations mapping
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    # Get all image IDs that have annotations
    image_ids_with_anns = list(img_to_anns.keys())
    
    if stratify:
        # Group images by their primary category
        img_to_primary_cat = {}
        for img_id in image_ids_with_anns:
            # Use the category with most annotations in this image
            cat_counts = defaultdict(int)
            for ann in img_to_anns[img_id]:
                cat_counts[ann['category_id']] += 1
            img_to_primary_cat[img_id] = max(cat_counts.items(), key=lambda x: x[1])[0]
        
        # Group images by category
        cat_to_imgs = defaultdict(list)
        for img_id, cat_id in img_to_primary_cat.items():
            cat_to_imgs[cat_id].append(img_id)
        
        # Split each category
        val_img_ids = set()
        for cat_id, img_ids in cat_to_imgs.items():
            random.shuffle(img_ids)
            if len(img_ids) == 1:
                # For single-image categories, randomly assign to train or val
                if random.random() < val_ratio:
                    val_img_ids.add(img_ids[0])
            else:
                # For multi-image categories, take proportional split
                n_val = max(1, int(len(img_ids) * val_ratio))
                val_img_ids.update(img_ids[:n_val])
    else:
        # Simple random split
        random.shuffle(image_ids_with_anns)
        n_val = int(len(image_ids_with_anns) * val_ratio)
        val_img_ids = set(image_ids_with_anns[:n_val])
    
    train_img_ids = set(image_ids_with_anns) - val_img_ids
    
    # Create image lookup
    id_to_img = {img['id']: img for img in images}
    
    # Build split datasets
    train_images = [id_to_img[img_id] for img_id in train_img_ids if img_id in id_to_img]
    val_images = [id_to_img[img_id] for img_id in val_img_ids if img_id in id_to_img]
    
    train_anns = [ann for ann in annotations if ann['image_id'] in train_img_ids]
    val_anns = [ann for ann in annotations if ann['image_id'] in val_img_ids]
    
    train_data = {
        'images': train_images,
        'annotations': train_anns,
        'categories': categories
    }
    
    val_data = {
        'images': val_images,
        'annotations': val_anns,
        'categories': categories
    }
    
    return train_data, val_data


def filter_by_categories(
    data: Dict,
    category_ids: Optional[Set[int]] = None,
    min_annotations: int = 0
) -> Dict:
    """
    Filter COCO dataset to only include specified categories.
    
    Args:
        data: COCO format data dict
        category_ids: Set of category IDs to keep. If None, keep all.
        min_annotations: Minimum annotations a category must have
        
    Returns:
        Filtered COCO data dict
    """
    if category_ids is None and min_annotations == 0:
        return data
    
    # If filtering by min_annotations, find eligible categories
    if min_annotations > 0:
        stats = get_category_stats(data)
        eligible = {cat_id for cat_id, count in stats.items() if count >= min_annotations}
        if category_ids is not None:
            category_ids = category_ids & eligible
        else:
            category_ids = eligible
    
    # Filter annotations
    filtered_anns = [
        ann for ann in data['annotations']
        if ann['category_id'] in category_ids
    ]
    
    # Get image IDs that still have annotations
    valid_img_ids = {ann['image_id'] for ann in filtered_anns}
    
    # Filter images
    filtered_images = [
        img for img in data['images']
        if img['id'] in valid_img_ids
    ]
    
    # Filter categories
    filtered_cats = [
        cat for cat in data['categories']
        if cat['id'] in category_ids
    ]
    
    return {
        'images': filtered_images,
        'annotations': filtered_anns,
        'categories': filtered_cats
    }


def remap_categories_to_binary(data: Dict) -> Dict:
    """
    Convert multi-class dataset to binary (object vs background).
    All objects become class 0.
    
    Args:
        data: COCO format data dict
        
    Returns:
        Modified COCO data dict with single category
    """
    # Remap all annotations to category 0
    remapped_anns = []
    for ann in data['annotations']:
        new_ann = ann.copy()
        new_ann['category_id'] = 0
        remapped_anns.append(new_ann)
    
    return {
        'images': data['images'],
        'annotations': remapped_anns,
        'categories': [{'id': 0, 'name': 'object', 'supercategory': 'none'}]
    }


def create_category_mapping(data: Dict) -> Tuple[Dict[int, int], List[str]]:
    """
    Create a mapping from original category IDs to contiguous YOLO class indices.
    
    Args:
        data: COCO format data dict
        
    Returns:
        Tuple of (category_id_to_class_idx, class_names)
    """
    categories = sorted(data['categories'], key=lambda x: x['id'])
    category_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    class_names = [cat['name'] for cat in categories]
    return category_id_to_idx, class_names


def load_label_plan(label_plans_path: str, plan_name: str) -> Dict[str, Any]:
    """
    Load one named label plan from a YAML file.

    The YAML format is expected to look like:

    plans:
      coarse_v1:
        drop_unmapped: true
        classes:
          fish:
            - Sebastolobus
          sponge:
            - Porifera
    """
    path = Path(label_plans_path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid label plan file: {path}")

    plans = payload.get("plans")
    if not isinstance(plans, dict):
        raise ValueError(f"Label plan file missing top-level 'plans' mapping: {path}")

    def _resolve(name: str, stack: Optional[List[str]] = None) -> Dict[str, Any]:
        if stack is None:
            stack = []
        if name in stack:
            raise ValueError(f"Cyclic label plan inheritance detected: {' -> '.join(stack + [name])}")
        raw = plans.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"Unknown label plan '{name}' in {path}")

        merged: Dict[str, Any] = {
            "name": name,
            "description": raw.get("description", ""),
            "drop_unmapped": bool(raw.get("drop_unmapped", True)),
            "classes": {},
        }

        parent = raw.get("extends")
        if parent:
            base = _resolve(str(parent), stack + [name])
            merged["description"] = merged["description"] or base.get("description", "")
            merged["drop_unmapped"] = raw.get("drop_unmapped", base.get("drop_unmapped", True))
            merged["classes"].update(base.get("classes", {}))

        classes = raw.get("classes", {})
        if not isinstance(classes, dict):
            raise ValueError(f"Label plan '{name}' has non-mapping 'classes'")

        for class_name, category_names in classes.items():
            if not isinstance(category_names, list) or not all(isinstance(x, str) for x in category_names):
                raise ValueError(
                    f"Label plan '{name}' class '{class_name}' must map to a list of category-name strings"
                )
            merged["classes"][str(class_name)] = list(category_names)

        return merged

    resolved = _resolve(plan_name)

    seen_names: Dict[str, str] = {}
    for class_name, category_names in resolved["classes"].items():
        for category_name in category_names:
            prev = seen_names.get(category_name)
            if prev is not None and prev != class_name:
                raise ValueError(
                    f"Category name '{category_name}' is assigned to multiple coarse classes: "
                    f"'{prev}' and '{class_name}'"
                )
            seen_names[category_name] = class_name

    return resolved


def apply_label_plan(data: Dict, label_plans_path: str, plan_name: str) -> Tuple[Dict, Dict[str, Any]]:
    """
    Remap native category labels onto a named coarse label plan.

    Unmapped annotations are dropped when the plan sets `drop_unmapped: true`.
    """
    plan = load_label_plan(label_plans_path, plan_name)
    categories = data.get("categories", [])
    annotations = data.get("annotations", [])
    images = data.get("images", [])

    category_name_to_new_class: Dict[str, Tuple[int, str]] = {}
    class_names = list(plan["classes"].keys())
    for idx, class_name in enumerate(class_names):
        for category_name in plan["classes"][class_name]:
            category_name_to_new_class[category_name] = (idx, class_name)

    id_to_cat = {cat["id"]: cat for cat in categories if "id" in cat}
    remapped_anns = []
    dropped_annotations = 0
    dropped_category_ids = set()
    mapped_category_ids = set()

    for ann in annotations:
        cat = id_to_cat.get(ann.get("category_id"))
        if cat is None:
            dropped_annotations += 1
            continue

        mapping = category_name_to_new_class.get(cat.get("name"))
        if mapping is None:
            dropped_annotations += 1
            dropped_category_ids.add(cat["id"])
            continue

        new_class_id, _ = mapping
        new_ann = ann.copy()
        new_ann["category_id"] = new_class_id
        remapped_anns.append(new_ann)
        mapped_category_ids.add(cat["id"])

    valid_img_ids = {ann["image_id"] for ann in remapped_anns}
    remapped_images = [img for img in images if img.get("id") in valid_img_ids]
    remapped_categories = [
        {"id": idx, "name": class_name, "supercategory": "coarse"}
        for idx, class_name in enumerate(class_names)
    ]

    summary = {
        "label_plan": plan_name,
        "label_plan_path": str(Path(label_plans_path).expanduser().resolve()),
        "class_names": class_names,
        "mapped_category_count": len(mapped_category_ids),
        "dropped_category_count": len(dropped_category_ids),
        "mapped_annotation_count": len(remapped_anns),
        "dropped_annotation_count": dropped_annotations,
        "coverage_fraction": (len(remapped_anns) / len(annotations)) if annotations else 0.0,
        "drop_unmapped": bool(plan.get("drop_unmapped", True)),
        "unmapped_category_names_sample": sorted(
            id_to_cat[cid]["name"] for cid in list(dropped_category_ids)[:25] if cid in id_to_cat
        ),
    }

    return {
        "images": remapped_images,
        "annotations": remapped_anns,
        "categories": remapped_categories,
    }, summary


def get_image_path(file_name: str, image_dirs: List[str]) -> Optional[str]:
    """
    Find the full path to an image given possible directories.
    
    Args:
        file_name: Image filename
        image_dirs: List of directories to search
        
    Returns:
        Full path to image if found, None otherwise
    """
    for img_dir in image_dirs:
        path = os.path.join(img_dir, file_name)
        if os.path.exists(path):
            return path
    return None
