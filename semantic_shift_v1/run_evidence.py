"""Drive-authoritative training-run evidence persistence for frozen G4.

The helpers create the exact frozen leaf names, refuse pre-existing run folders,
append only within an active run, and finalize immutable completion/checksum records.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from .artifact_protocol import (
    create_training_run_dirs,
    write_new_json,
    write_new_text,
    write_checksum_manifest,
    sha256_file,
)


EPOCH_FIELDS = (
    "epoch",
    "train_loss_batch_mean",
    "validation_metric",
    "learning_rate",
    "best_metric",
    "best_epoch",
)


def environment_snapshot():
    out = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    for package in ("albumentations", "rasterio", "pandas"):
        try:
            mod = __import__(package)
            out[package] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[package] = None
    return out


def initialize_run_evidence(
    project_root,
    cfg,
    *,
    implementation_sha,
    immutable_config_sha256,
    input_identities,
):
    """Create exact run/checkpoint dirs and the initial required run evidence."""
    run_id = str(cfg["run_id"])
    if not immutable_config_sha256 or len(str(immutable_config_sha256)) != 64:
        raise ValueError("immutable_config_sha256 must be a 64-character SHA-256")
    if not isinstance(input_identities, dict) or not input_identities:
        raise ValueError("input_identities must be a non-empty dictionary")

    run_dir, ckpt_dir = create_training_run_dirs(project_root, run_id)
    write_new_json(run_dir / "run_config.json", cfg)
    write_new_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "condition_id": cfg["condition_id"],
            "role": cfg["role"],
            "seed": int(cfg["seed"]),
            "implementation_sha": str(implementation_sha),
            "immutable_config_sha256": str(immutable_config_sha256),
            "input_identities": input_identities,
            "g4_status_required": "CLOSED/FROZEN",
        },
    )
    write_new_json(run_dir / "environment.json", environment_snapshot())
    write_new_text(
        run_dir / "training_log.txt",
        "Semantic Shift frozen run\n"
        f"run_id={run_id}\n"
        f"implementation_sha={implementation_sha}\n"
        "status=STARTED\n",
    )
    with (run_dir / "epoch_metrics.csv").open("x", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=EPOCH_FIELDS).writeheader()
    return run_dir, ckpt_dir


def append_training_log(run_dir, text):
    p = Path(run_dir) / "training_log.txt"
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open("a", encoding="utf-8") as f:
        f.write(str(text))
        if not str(text).endswith("\n"):
            f.write("\n")
        f.flush()


def append_epoch_metrics(run_dir, row):
    p = Path(run_dir) / "epoch_metrics.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    missing = [k for k in EPOCH_FIELDS if k not in row]
    if missing:
        raise KeyError(f"Missing epoch-metric field(s): {missing}")
    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPOCH_FIELDS)
        w.writerow({k: row[k] for k in EPOCH_FIELDS})
        f.flush()


def finalize_run_evidence(run_dir, ckpt_dir, *, status, best_epoch, best_metric, error=None):
    """Create completion status and checkpoint checksum manifest exactly once."""
    run_dir = Path(run_dir)
    ckpt_dir = Path(ckpt_dir)
    status = str(status)
    if status not in {"COMPLETED", "FAILED"}:
        raise ValueError("status must be COMPLETED or FAILED")

    checkpoint_files = sorted(
        p for p in ckpt_dir.iterdir()
        if p.is_file() and p.name != "CHECKPOINT_SHA256SUMS.txt"
    )
    if checkpoint_files:
        write_checksum_manifest(
            ckpt_dir / "CHECKPOINT_SHA256SUMS.txt",
            checkpoint_files,
            relative_to=ckpt_dir,
        )
    else:
        write_new_text(ckpt_dir / "CHECKPOINT_SHA256SUMS.txt", "")

    completion = {
        "status": status,
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "error": None if error is None else str(error),
        "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "epoch_metrics_sha256": sha256_file(run_dir / "epoch_metrics.csv"),
        "training_log_sha256": sha256_file(run_dir / "training_log.txt"),
        "checkpoint_sha256_manifest": "07_Checkpoints/{run_id}/CHECKPOINT_SHA256SUMS.txt",
    }
    write_new_json(run_dir / "completion_status.json", completion)
    return completion


def verify_required_run_leaves(run_dir, ckpt_dir):
    run_dir = Path(run_dir)
    ckpt_dir = Path(ckpt_dir)
    run_names = {
        "run_config.json",
        "run_manifest.json",
        "environment.json",
        "training_log.txt",
        "epoch_metrics.csv",
        "completion_status.json",
    }
    checkpoint_names = {"best_validation.pt", "CHECKPOINT_SHA256SUMS.txt"}
    missing_run = sorted(name for name in run_names if not (run_dir / name).is_file())
    missing_ckpt = sorted(name for name in checkpoint_names if not (ckpt_dir / name).is_file())
    if missing_run or missing_ckpt:
        raise RuntimeError(f"Missing frozen run evidence: run={missing_run}, checkpoint={missing_ckpt}")
    return True
