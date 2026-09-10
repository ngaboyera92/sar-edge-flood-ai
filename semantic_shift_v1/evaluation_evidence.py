"""Frozen G4-06 evaluation evidence persistence and stage barriers."""
from __future__ import annotations

import csv
from pathlib import Path

from .artifact_protocol import (
    stage_root_dir,
    stage_manifest_path,
    write_new_json,
    write_new_text,
    write_checksum_manifest,
    sha256_file,
)
from .evaluation_protocol import assert_stage_preconditions


EVENT_FIELDS = (
    "event_id",
    "valid_pixels",
    "no_water_pixels",
    "permanent_water_pixels",
    "reference_flood_pixels",
    "predicted_flood_pixels",
    "tp",
    "fp",
    "fp_no_water",
    "fp_permanent_water",
    "fn",
    "tn",
    "flood_iou",
    "permanent_water_fpr",
    "no_water_fpr",
    "average_precision",
    "brier_sum",
    "brier_count",
)


def evaluation_run_dir(project_root, stage, run_id):
    return stage_root_dir(project_root, stage) / str(run_id)


def completed_stage_manifests(project_root):
    out = []
    for stage in ("E1", "E2", "E3"):
        if stage_manifest_path(project_root, stage).is_file():
            out.append(stage)
    return out


def assert_evaluation_stage_authorized(project_root, stage):
    return assert_stage_preconditions(stage, completed_stage_manifests(project_root))


def write_evaluation_evidence(
    project_root,
    stage,
    run_id,
    *,
    event_rows,
    domain_summary,
    provenance,
):
    """Persist one condition/seed/domain package with create-new semantics."""
    assert_evaluation_stage_authorized(project_root, stage)
    out = evaluation_run_dir(project_root, stage, run_id)
    out.mkdir(parents=True, exist_ok=False)

    event_path = out / "event_sufficient_statistics.csv"
    with event_path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in event_rows:
            writer.writerow({key: row.get(key) for key in EVENT_FIELDS})

    write_new_json(
        out / "domain_summary.json",
        {**domain_summary, "provenance": provenance},
    )
    write_new_text(
        out / "evaluation_log.txt",
        "Semantic Shift frozen evaluation\n"
        f"stage={stage}\n"
        f"run_id={run_id}\n"
        "status=COMPLETED\n",
    )
    payload_files = [
        event_path,
        out / "domain_summary.json",
        out / "evaluation_log.txt",
    ]
    write_checksum_manifest(
        out / "ARTIFACT_SHA256SUMS.txt",
        payload_files,
        relative_to=out,
    )
    return out


def evaluation_evidence_identity(path):
    p = Path(path)
    required = (
        "event_sufficient_statistics.csv",
        "domain_summary.json",
        "evaluation_log.txt",
        "ARTIFACT_SHA256SUMS.txt",
    )
    missing = [x for x in required if not (p / x).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete evaluation evidence: {missing}")
    return {
        name: sha256_file(p / name)
        for name in required
    }


def finalize_stage_manifest(
    project_root,
    stage,
    expected_run_ids,
    *,
    input_identities,
):
    """Create the immutable stage barrier only after every expected run package exists."""
    assert_evaluation_stage_authorized(project_root, stage)
    expected = [str(x) for x in expected_run_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected_run_ids contains duplicates")
    entries = []
    for run_id in expected:
        d = evaluation_run_dir(project_root, stage, run_id)
        if not d.is_dir():
            raise RuntimeError(f"Missing evaluation package for {stage}/{run_id}")
        entries.append({
            "run_id": run_id,
            "status": "COMPLETED",
            "artifact_hashes": evaluation_evidence_identity(d),
        })
    manifest = {
        "stage": str(stage),
        "status": "COMPLETED/FROZEN",
        "expected_run_count": len(expected),
        "runs": entries,
        "input_identities": input_identities,
    }
    path = stage_manifest_path(project_root, stage)
    write_new_json(path, manifest)
    return path
