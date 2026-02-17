#!/usr/bin/env python3
"""
Benchmark multiple YOLO checkpoints on one dataset and produce a single report.
"""

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PRIMARY_METRIC_DEFAULT = "metrics/mAP50-95(M)"
FALLBACK_METRICS = [
    "metrics/mAP50-95(M)",
    "metrics/mAP50-95(B)",
    "metrics/mAP50(M)",
    "metrics/mAP50(B)",
    "fitness",
]


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _available_splits(dataset_yaml: Path) -> List[str]:
    data = _load_yaml(dataset_yaml)
    out = []
    for split in ("train", "val", "test"):
        value = data.get(split)
        if isinstance(value, str) and value.strip():
            out.append(split)
    return out


def _collect_checkpoints(
    explicit: List[str],
    roots: List[str],
    pattern: str,
) -> List[Path]:
    found: List[Path] = []

    for item in explicit:
        p = Path(item).expanduser()
        if p.is_file():
            found.append(p.resolve())
        elif p.is_dir():
            found.extend(sorted(x.resolve() for x in p.rglob(pattern)))

    for root in roots:
        p = Path(root).expanduser()
        if p.is_dir():
            found.extend(sorted(x.resolve() for x in p.rglob(pattern)))

    unique = sorted(set(found))
    return unique


def _extract_metrics(metrics_obj: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}

    results_dict = getattr(metrics_obj, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            v = _to_float(value)
            if v is not None:
                out[str(key)] = v

    for prefix, attr in (("B", "box"), ("M", "seg")):
        block = getattr(metrics_obj, attr, None)
        if block is None:
            continue

        pairs = {
            f"metrics/precision({prefix})": getattr(block, "mp", None),
            f"metrics/recall({prefix})": getattr(block, "mr", None),
            f"metrics/mAP50({prefix})": getattr(block, "map50", None),
            f"metrics/mAP50-95({prefix})": getattr(block, "map", None),
        }
        for key, value in pairs.items():
            v = _to_float(value)
            if v is not None:
                out[key] = v

    fitness = _to_float(getattr(metrics_obj, "fitness", None))
    if fitness is not None:
        out["fitness"] = fitness

    return out


def _choose_score(row: Dict[str, Any], primary_metric: str) -> Tuple[float, str]:
    ordered = [primary_metric] + [m for m in FALLBACK_METRICS if m != primary_metric]
    for metric in ordered:
        value = _to_float(row.get(metric))
        if value is not None and not math.isnan(value):
            return value, metric
    return float("-inf"), ""


def _extract_run_metadata(checkpoint: Path) -> Dict[str, Any]:
    # Expected layout:
    #   .../<run_dir>/train/weights/best.pt
    train_dir = checkpoint.parent.parent
    run_dir = train_dir.parent
    run_config_path = run_dir / "run_config.yaml"
    args_path = train_dir / "args.yaml"

    run_cfg = _load_yaml(run_config_path) if run_config_path.exists() else {}
    train_args = _load_yaml(args_path) if args_path.exists() else {}

    metadata: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "train_dir": str(train_dir),
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "model": run_cfg.get("model") or train_args.get("model"),
        "mode": run_cfg.get("mode"),
        "top_n": run_cfg.get("top_n"),
        "val_ratio": run_cfg.get("val_ratio"),
        "split_seed": run_cfg.get("split_seed"),
        "epochs": run_cfg.get("epochs") or train_args.get("epochs"),
        "batch": run_cfg.get("batch") or train_args.get("batch"),
        "seed": train_args.get("seed"),
        "train_results_plot": str(train_dir / "results.png") if (train_dir / "results.png").exists() else "",
    }
    return metadata


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fixed_cols = [
        "rank",
        "status",
        "split",
        "run_name",
        "checkpoint",
        "score",
        "score_metric",
        "model",
        "mode",
        "top_n",
        "val_ratio",
        "split_seed",
        "epochs",
        "batch",
        "seed",
        "train_dir",
        "run_dir",
        "eval_dir",
        "train_results_plot",
        "confusion_matrix",
        "confusion_matrix_normalized",
        "error",
    ]

    metric_cols = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.startswith("metrics/") or key == "fitness"
        }
    )

    all_cols = fixed_cols + [c for c in metric_cols if c not in fixed_cols]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    primary_metric: str,
) -> None:
    best = summary.get("best", {})
    lines = []
    lines.append("# YOLO Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append(f"- Dataset: `{summary['dataset']}`")
    lines.append(f"- Requested splits: `{', '.join(summary['requested_splits'])}`")
    lines.append(f"- Available splits in dataset: `{', '.join(summary['available_splits'])}`")
    lines.append(f"- Ranking split: `{summary['rank_split']}`")
    lines.append(f"- Primary metric: `{primary_metric}`")
    lines.append(f"- Checkpoints discovered: {summary['total_checkpoints']}")
    lines.append(f"- Evaluations executed: {summary['total_evaluations']}")
    lines.append(f"- Rankable evaluations succeeded: {summary['successful_ranked_evaluations']}")
    lines.append("")
    if best:
        lines.append(f"- Best checkpoint: `{best.get('checkpoint')}`")
        lines.append(f"- Best score: {best.get('score'):.6f} ({best.get('score_metric')})")
        lines.append("")

    table_cols = [
        "rank",
        "split",
        "run_name",
        "score",
        "score_metric",
        "metrics/mAP50-95(M)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(M)",
        "metrics/mAP50(B)",
        "fitness",
        "status",
    ]
    lines.append("| " + " | ".join(table_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(table_cols)) + "|")
    for row in rows:
        cells = []
        for col in table_cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple YOLO checkpoints and write a ranked report"
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset YAML")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint file or directory (repeatable)",
    )
    parser.add_argument(
        "--checkpoints-root",
        action="append",
        default=[],
        help="Root directory to recursively search for checkpoints (repeatable)",
    )
    parser.add_argument(
        "--pattern",
        default="weights/best.pt",
        help="Recursive file pattern used under checkpoint roots",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="One or more dataset splits to evaluate (e.g. --splits val test)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Deprecated alias for a single split; overrides --splits when provided.",
    )
    parser.add_argument(
        "--rank-split",
        default="val",
        choices=["train", "val", "test"],
        help="Split used for ranking checkpoints",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--primary-metric",
        default=PRIMARY_METRIC_DEFAULT,
        help="Metric key used for ranking",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Optional cap on number of checkpoints evaluated (0 = all)",
    )
    parser.add_argument("--project", default="runs/benchmark")
    parser.add_argument("--name", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Install with: pip install ultralytics", file=sys.stderr)
        return 1

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset YAML not found: {dataset_path}", file=sys.stderr)
        return 1

    eval_splits = list(args.splits)
    if args.split is not None:
        eval_splits = [args.split]
    # Preserve order but remove duplicates.
    seen = set()
    eval_splits = [s for s in eval_splits if not (s in seen or seen.add(s))]
    available_splits = _available_splits(dataset_path)

    checkpoints = _collect_checkpoints(
        explicit=args.checkpoint,
        roots=args.checkpoints_root,
        pattern=args.pattern,
    )
    if not checkpoints:
        print("No checkpoints found. Provide --checkpoint or --checkpoints-root.", file=sys.stderr)
        return 1

    if args.max_models > 0:
        checkpoints = checkpoints[: args.max_models]

    run_name = args.name or dt.datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
    out_dir = Path(args.project).expanduser().resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_project = out_dir / "eval_runs"
    eval_project.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, checkpoint in enumerate(checkpoints, start=1):
        print(f"[{idx}/{len(checkpoints)}] Evaluating {checkpoint}")
        model = YOLO(str(checkpoint))

        for split in eval_splits:
            row = _extract_run_metadata(checkpoint)
            row["split"] = split
            row["status"] = "ok"
            row["error"] = ""
            row["eval_dir"] = ""
            row["confusion_matrix"] = ""
            row["confusion_matrix_normalized"] = ""

            if split not in available_splits:
                row["status"] = "missing_split"
                row["error"] = f"Dataset YAML has no '{split}' entry"
                row["score"] = ""
                row["score_metric"] = ""
                rows.append(row)
                continue

            eval_name = f"{idx:03d}_{checkpoint.parent.parent.parent.name}_{split}"
            try:
                metrics_obj = model.val(
                    data=str(dataset_path),
                    split=split,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    workers=args.workers,
                    device=args.device,
                    project=str(eval_project),
                    name=eval_name,
                    exist_ok=True,
                    verbose=args.verbose,
                    plots=True,
                )
                row.update(_extract_metrics(metrics_obj))

                save_dir = Path(getattr(metrics_obj, "save_dir", eval_project / eval_name))
                row["eval_dir"] = str(save_dir)
                cm = save_dir / "confusion_matrix.png"
                cmn = save_dir / "confusion_matrix_normalized.png"
                if cm.exists():
                    row["confusion_matrix"] = str(cm)
                if cmn.exists():
                    row["confusion_matrix_normalized"] = str(cmn)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "failed"
                row["error"] = str(exc)

            if split == args.rank_split:
                score, score_metric = _choose_score(row, args.primary_metric)
                row["score"] = score if score != float("-inf") else ""
                row["score_metric"] = score_metric
            else:
                row["score"] = ""
                row["score_metric"] = ""
            rows.append(row)

    successful = [
        r
        for r in rows
        if r.get("split") == args.rank_split and r.get("status") == "ok" and r.get("score") != ""
    ]
    successful_sorted = sorted(successful, key=lambda x: float(x["score"]), reverse=True)

    for rank, row in enumerate(successful_sorted, start=1):
        row["rank"] = rank
    for row in rows:
        row.setdefault("rank", "")

    ordered_rows = successful_sorted + [r for r in rows if r not in successful_sorted]

    best = successful_sorted[0] if successful_sorted else {}
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "requested_splits": eval_splits,
        "available_splits": available_splits,
        "rank_split": args.rank_split,
        "primary_metric": args.primary_metric,
        "total_checkpoints": len(checkpoints),
        "total_evaluations": len(rows),
        "successful_ranked_evaluations": len(successful_sorted),
        "best": {
            "checkpoint": best.get("checkpoint"),
            "score": best.get("score"),
            "score_metric": best.get("score_metric"),
            "run_name": best.get("run_name"),
        }
        if best
        else {},
    }

    csv_path = out_dir / "benchmark_report.csv"
    json_path = out_dir / "benchmark_report.json"
    md_path = out_dir / "benchmark_report.md"
    _write_csv(csv_path, ordered_rows)
    _write_markdown(md_path, summary, ordered_rows, args.primary_metric)

    report_payload = {"summary": summary, "results": ordered_rows}
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print("")
    print("Benchmark complete")
    print(f"Report (Markdown): {md_path}")
    print(f"Report (CSV): {csv_path}")
    print(f"Report (JSON): {json_path}")
    if best:
        print(f"Best checkpoint: {best.get('checkpoint')}")
        print(f"Best score: {best.get('score')} ({best.get('score_metric')})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
