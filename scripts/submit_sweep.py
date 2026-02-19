#!/usr/bin/env python3
"""
Submit a set of YOLO training jobs to SLURM from a YAML config.

This script creates a reproducible manifest (commands + job IDs) so you can
later benchmark checkpoints and trace back exactly how each model was trained.
"""

import argparse
import csv
import datetime as dt
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


SBATCH_KEYS = [
    "repo",
    "venv",
    "data",
    "mode",
    "top_n",
    "val_ratio",
    "split_seed",
    "model",
    "epochs",
    "batch",
    "wandb_project",
    "wandb_name",
    "wandb_group",
    "wandb_tags",
]

SBATCH_FLAG_MAP = {
    "repo": "--repo",
    "venv": "--venv",
    "data": "--data",
    "mode": "--mode",
    "top_n": "--top_n",
    "val_ratio": "--val_ratio",
    "split_seed": "--split-seed",
    "model": "--model",
    "epochs": "--epochs",
    "batch": "--batch",
    "wandb_project": "--wandb-project",
    "wandb_name": "--wandb-name",
    "wandb_group": "--wandb-group",
    "wandb_tags": "--wandb-tags",
}

SBATCH_RESOURCE_KEYS = {
    "cpus_per_task": "--cpus-per-task",
    "mem": "--mem",
    "time": "--time",
}


def _flag(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _append_opt(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, _stringify(value)])


def _append_train_args(cmd: List[str], train_cfg: Dict[str, Any]) -> None:
    for key, value in train_cfg.items():
        flag = _flag(key)
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                cmd.extend([flag, _stringify(item)])
            continue
        cmd.extend([flag, _stringify(value)])


def _merge_dict(parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(parent)
    merged.update(child)
    return merged


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must be a YAML mapping")
    if "experiments" not in cfg:
        raise ValueError("Config must contain an 'experiments' list")
    if not isinstance(cfg["experiments"], list) or not cfg["experiments"]:
        raise ValueError("'experiments' must be a non-empty list")
    return cfg


def _build_commands(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    global_cfg = cfg.get("global", {})
    if global_cfg is None:
        global_cfg = {}
    if not isinstance(global_cfg, dict):
        raise ValueError("'global' must be a mapping when present")

    global_train = global_cfg.get("train", {})
    if global_train is None:
        global_train = {}
    if not isinstance(global_train, dict):
        raise ValueError("'global.train' must be a mapping when present")

    commands: List[Dict[str, Any]] = []
    for idx, exp in enumerate(cfg["experiments"], start=1):
        if not isinstance(exp, dict):
            raise ValueError(f"Experiment #{idx} must be a mapping")

        merged = _merge_dict(global_cfg, exp)
        train_cfg = _merge_dict(global_train, exp.get("train", {}))

        if "model" not in merged:
            raise ValueError(f"Experiment #{idx} is missing required field 'model'")
        if "batch" not in merged:
            raise ValueError(f"Experiment #{idx} is missing required field 'batch'")

        exp_name = merged.get("name", f"exp_{idx:03d}")
        wandb_name = merged.get("wandb_name", exp_name)
        gpus_per_node = merged.get("gpus_per_node", 1)

        cmd: List[str] = [
            "sbatch",
            f"--gpus-per-node={gpus_per_node}",
        ]

        for key, flag in SBATCH_RESOURCE_KEYS.items():
            _append_opt(cmd, flag, merged.get(key))

        cmd.append("slurm/train.sh")

        resolved = dict(merged)
        resolved["wandb_name"] = wandb_name
        resolved["name"] = exp_name

        for key in SBATCH_KEYS:
            value = resolved.get(key)
            if key == "wandb_name":
                value = wandb_name
            _append_opt(cmd, SBATCH_FLAG_MAP[key], value)

        _append_train_args(cmd, train_cfg)

        train_args_list = merged.get("train_args", [])
        if train_args_list is None:
            train_args_list = []
        if not isinstance(train_args_list, list):
            raise ValueError(f"Experiment '{exp_name}' has non-list 'train_args'")
        cmd.extend([_stringify(x) for x in train_args_list])

        commands.append(
            {
                "index": idx,
                "name": exp_name,
                "command": cmd,
                "resolved": resolved,
                "train": train_cfg,
            }
        )

    return commands


def _parse_job_id(stdout: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", stdout)
    return match.group(1) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit a SLURM sweep for YOLO training"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to sweep YAML config",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="runs/sweeps",
        help="Directory for sweep manifests",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit jobs (default: dry-run only)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    try:
        cfg = _load_config(config_path)
        commands = _build_commands(cfg)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = f"{timestamp}_{config_path.stem}"
    out_dir = Path(args.output_dir) / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg_path = out_dir / "resolved_config.yaml"
    with resolved_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    commands_txt = out_dir / "commands.txt"
    with commands_txt.open("w", encoding="utf-8") as f:
        for item in commands:
            line = " ".join(shlex.quote(part) for part in item["command"])
            f.write(line + "\n")

    manifest_rows = []
    for item in commands:
        cmd = item["command"]
        cmd_str = " ".join(shlex.quote(part) for part in cmd)
        status = "planned"
        stdout = ""
        stderr = ""
        job_id = ""

        if args.submit:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
            )
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            status = "submitted" if proc.returncode == 0 else "failed"
            job_id = _parse_job_id(stdout)

        manifest_rows.append(
            {
                "index": item["index"],
                "name": item["name"],
                "status": status,
                "job_id": job_id,
                "command": cmd_str,
                "stdout": stdout,
                "stderr": stderr,
            }
        )

    manifest_path = out_dir / "submission_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "name", "status", "job_id", "command", "stdout", "stderr"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    mode = "submitted" if args.submit else "dry-run"
    print(f"Sweep {mode}: {len(commands)} experiments")
    print(f"Commands: {commands_txt}")
    print(f"Manifest: {manifest_path}")
    if args.submit:
        ok = sum(1 for row in manifest_rows if row["status"] == "submitted")
        print(f"Submitted successfully: {ok}/{len(commands)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
