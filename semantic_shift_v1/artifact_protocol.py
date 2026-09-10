"""Frozen Semantic Shift artifact naming and no-overwrite primitives.

This module centralizes the Drive-authoritative artifact contract without executing
training or evaluation. All creation helpers default to create-new semantics and
refuse silent overwrite.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


TRAIN_RUN_LEAVES = (
    "run_config.json",
    "run_manifest.json",
    "environment.json",
    "training_log.txt",
    "epoch_metrics.csv",
    "completion_status.json",
)

CHECKPOINT_FIXED_LEAVES = (
    "best_validation.pt",
    "CHECKPOINT_SHA256SUMS.txt",
)

EVAL_RUN_LEAVES = (
    "event_sufficient_statistics.csv",
    "domain_summary.json",
    "evaluation_log.txt",
    "ARTIFACT_SHA256SUMS.txt",
)

STAGE_ROOT_MANIFEST = {
    "E1": "E1_STAGE_MANIFEST.json",
    "E2": "E2_STAGE_MANIFEST.json",
    "E3": "E3_STAGE_MANIFEST.json",
}

STAGE_ROOT_DIR = {
    "E1": "08_Evaluation/E1_GEOID_SOURCE",
    "E2": "08_Evaluation/E2_GEOID_HELDOUT_2026",
    "E3": "08_Evaluation/E3_KURO_TEST_GRD",
}

RESULT_LEAVES = (
    "condition_summary_event_macro_iou.csv",
    "primary_contrasts_event_macro_iou.csv",
    "bootstrap_primary_event_macro_iou.npz",
    "permanent_water_fpr_summary.csv",
    "permanent_water_fpr_contrasts.csv",
    "bootstrap_permanent_water_fpr.npz",
    "no_water_fpr_negative_control.csv",
    "supporting_mechanism_contrasts.csv",
    "confirmatory_statistical_summary.json",
    "RESULTS_SHA256SUMS.txt",
)


def training_run_dir(project_root, run_id):
    return Path(project_root) / "06_Training_Runs" / str(run_id)


def checkpoint_run_dir(project_root, run_id):
    return Path(project_root) / "07_Checkpoints" / str(run_id)


def stage_root_dir(project_root, stage):
    stage = str(stage)
    if stage not in STAGE_ROOT_DIR:
        raise ValueError(f"Unknown evaluation stage: {stage}")
    return Path(project_root) / STAGE_ROOT_DIR[stage]


def stage_manifest_path(project_root, stage):
    stage = str(stage)
    if stage not in STAGE_ROOT_MANIFEST:
        raise ValueError(f"Unknown evaluation stage: {stage}")
    return stage_root_dir(project_root, stage) / STAGE_ROOT_MANIFEST[stage]


def assert_new_training_run(project_root, run_id):
    """Refuse a run when either authoritative output directory already exists."""
    run_dir = training_run_dir(project_root, run_id)
    ckpt_dir = checkpoint_run_dir(project_root, run_id)
    existing = [str(p) for p in (run_dir, ckpt_dir) if p.exists()]
    if existing:
        raise FileExistsError(
            "Frozen no-overwrite policy: run/checkpoint directory already exists: "
            + ", ".join(existing)
        )
    return run_dir, ckpt_dir


def create_training_run_dirs(project_root, run_id):
    """Create the two authoritative run directories exactly once."""
    run_dir, ckpt_dir = assert_new_training_run(project_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=False)
    except Exception:
        # Avoid leaving a half-created pair if checkpoint creation fails.
        try:
            run_dir.rmdir()
        except OSError:
            pass
        raise
    return run_dir, ckpt_dir


def write_new_bytes(path, data):
    """Create a new file and fail if the destination already exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = bytes(data)
    with p.open("xb") as f:
        f.write(payload)
        f.flush()
    return p


def write_new_text(path, text, encoding="utf-8"):
    return write_new_bytes(path, str(text).encode(encoding))


def write_new_json(path, obj):
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return write_new_text(path, text)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(int(chunk_size))
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def checksum_lines(paths, relative_to=None):
    """Return deterministic sha256sum-style lines sorted by displayed path."""
    base = None if relative_to is None else Path(relative_to)
    entries = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        shown = p.name if base is None else p.relative_to(base).as_posix()
        entries.append((shown, sha256_file(p)))
    entries.sort(key=lambda x: x[0])
    return [f"{digest}  {shown}" for shown, digest in entries]


def write_checksum_manifest(path, files, relative_to=None):
    lines = checksum_lines(files, relative_to=relative_to)
    return write_new_text(path, "\n".join(lines) + "\n")
