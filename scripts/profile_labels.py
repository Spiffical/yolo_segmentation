#!/usr/bin/env python3
"""
Profile label distribution from a COCO annotation JSON.

Useful for deciding multiclass experiment settings (e.g., top_n threshold).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def _quantile(sorted_values: List[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile class labels from COCO JSON")
    parser.add_argument(
        "--coco-json",
        default="data/seg_masks/train.json",
        help="Path to COCO annotation JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/label_profile",
        help="Directory to write summary files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top classes to print",
    )
    args = parser.parse_args()

    json_path = Path(args.coco_json).expanduser().resolve()
    if not json_path.exists():
        raise SystemExit(f"COCO JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    annotations = data.get("annotations", [])
    images = data.get("images", [])

    id_to_cat: Dict[int, Dict] = {int(c["id"]): c for c in categories if "id" in c}

    ann_counts = Counter()
    image_ids_per_cat = defaultdict(set)
    for ann in annotations:
        cid = ann.get("category_id")
        iid = ann.get("image_id")
        if cid is None:
            continue
        ann_counts[cid] += 1
        if iid is not None:
            image_ids_per_cat[cid].add(iid)

    counts_sorted = sorted(ann_counts.values())
    non_empty_supercats = {
        c.get("supercategory")
        for c in categories
        if c.get("supercategory") not in (None, "")
    }

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "category_counts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category_id",
                "name",
                "supercategory",
                "annotation_count",
                "image_count",
                "name_token_count",
            ],
        )
        writer.writeheader()
        for cid, count in sorted(ann_counts.items(), key=lambda kv: kv[1], reverse=True):
            cat = id_to_cat.get(cid, {})
            name = str(cat.get("name", f"id_{cid}"))
            writer.writerow(
                {
                    "category_id": cid,
                    "name": name,
                    "supercategory": cat.get("supercategory", ""),
                    "annotation_count": count,
                    "image_count": len(image_ids_per_cat.get(cid, set())),
                    "name_token_count": len(name.split()),
                }
            )

    md_path = out_dir / "label_profile.md"
    lines: List[str] = []
    lines.append("# Label Profile")
    lines.append("")
    lines.append(f"- Source: `{json_path}`")
    lines.append(f"- Images: {len(images)}")
    lines.append(f"- Annotations: {len(annotations)}")
    lines.append(f"- Categories declared: {len(categories)}")
    lines.append(f"- Categories with annotations: {len(ann_counts)}")
    lines.append(f"- Distinct non-empty supercategories: {len(non_empty_supercats)}")
    if non_empty_supercats:
        preview = ", ".join(sorted(str(x) for x in list(non_empty_supercats)[:20]))
        lines.append(f"- Supercategory preview: {preview}")
    lines.append("")
    lines.append("## Annotation Count Quantiles Per Class")
    for p in (0.5, 0.75, 0.9, 0.95, 0.99):
        lines.append(f"- q{int(p*100)}: {_quantile(counts_sorted, p):.1f}")
    lines.append("")
    lines.append("## Classes Above Count Threshold")
    for t in (1, 5, 10, 20, 50, 100, 200, 500, 1000):
        n = sum(v >= t for v in ann_counts.values())
        lines.append(f"- >= {t}: {n}")
    lines.append("")
    lines.append(f"## Top {args.top_k} Classes By Annotation Count")
    lines.append("| rank | category_id | name | supercategory | ann_count | image_count |")
    lines.append("|---|---:|---|---|---:|---:|")
    for rank, (cid, count) in enumerate(ann_counts.most_common(args.top_k), start=1):
        cat = id_to_cat.get(cid, {})
        name = str(cat.get("name", f"id_{cid}")).replace("|", "\\|")
        sup = str(cat.get("supercategory", "")).replace("|", "\\|")
        lines.append(
            f"| {rank} | {cid} | {name} | {sup} | {count} | {len(image_ids_per_cat.get(cid, set()))} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
