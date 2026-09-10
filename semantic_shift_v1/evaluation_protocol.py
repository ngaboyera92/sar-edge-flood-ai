"""Frozen G4-06 evaluation primitives for event-level sufficient statistics.

This module contains no dataset traversal and no outcome-dependent adaptation.
It only converts fixed-threshold predictions and reference labels into additive
sufficient statistics and enforces the frozen E1 -> E2 -> E3 stage order.
"""
from collections import defaultdict
import math
import numpy as np
import torch


_STAGE_PREDECESSORS = {
    "E1": (),
    "E2": ("E1",),
    "E3": ("E1", "E2"),
}


def assert_stage_preconditions(stage, completed_stages):
    stage = str(stage)
    if stage not in _STAGE_PREDECESSORS:
        raise ValueError(f"Unknown evaluation stage: {stage}")
    completed = set(str(x) for x in completed_stages)
    missing = [x for x in _STAGE_PREDECESSORS[stage] if x not in completed]
    if missing:
        raise RuntimeError(
            f"Evaluation stage {stage} blocked; missing completed predecessor(s): {missing}"
        )
    return True


def binary_prediction_from_probability(probability, threshold=0.5):
    if float(threshold) != 0.5:
        raise ValueError("Frozen Semantic Shift threshold is exactly 0.5")
    p = torch.as_tensor(probability)
    return p > 0.5


def chip_sufficient_statistics(probability, label, valid, threshold=0.5):
    """Return additive pixel counts for one chip.

    Label semantics are frozen as:
      0 = no water, 1 = permanent water (non-flood), 2 = flood.
    Only valid pixels participate. Prediction uses probability > 0.5.
    """
    p = torch.as_tensor(probability)
    y = torch.as_tensor(label)
    v = torch.as_tensor(valid, dtype=torch.bool)

    if p.ndim == y.ndim + 1 and p.shape[0] == 1:
        p = p[0]
    if p.shape != y.shape or v.shape != y.shape:
        raise ValueError(
            f"Shape mismatch: probability={tuple(p.shape)}, label={tuple(y.shape)}, valid={tuple(v.shape)}"
        )
    if bool((v & ~((y == 0) | (y == 1) | (y == 2))).any()):
        raise RuntimeError("Valid mask includes label outside frozen set {0,1,2}")

    pred = binary_prediction_from_probability(p, threshold=threshold) & v
    flood = (y == 2) & v
    perm = (y == 1) & v
    no_water = (y == 0) & v
    nonflood = (~flood) & v

    tp = int((pred & flood).sum().item())
    fp = int((pred & nonflood).sum().item())
    fn = int(((~pred) & flood & v).sum().item())
    tn = int(((~pred) & nonflood & v).sum().item())

    return {
        "valid_pixels": int(v.sum().item()),
        "no_water_pixels": int(no_water.sum().item()),
        "permanent_water_pixels": int(perm.sum().item()),
        "reference_flood_pixels": int(flood.sum().item()),
        "predicted_flood_pixels": int(pred.sum().item()),
        "tp": tp,
        "fp": fp,
        "fp_no_water": int((pred & no_water).sum().item()),
        "fp_permanent_water": int((pred & perm).sum().item()),
        "fn": fn,
        "tn": tn,
    }


_COUNT_FIELDS = (
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
)


def aggregate_event_statistics(records):
    """Aggregate chip counts by event_id without averaging chip metrics."""
    totals = defaultdict(lambda: {k: 0 for k in _COUNT_FIELDS})
    for rec in records:
        if "event_id" not in rec:
            raise KeyError("Each record must contain event_id")
        eid = str(rec["event_id"])
        for key in _COUNT_FIELDS:
            totals[eid][key] += int(rec[key])

    out = []
    for eid in sorted(totals):
        row = {"event_id": eid, **totals[eid]}
        union = row["tp"] + row["fp"] + row["fn"]
        row["flood_iou"] = (row["tp"] / union) if union > 0 else math.nan
        row["permanent_water_fpr"] = (
            row["fp_permanent_water"] / row["permanent_water_pixels"]
            if row["permanent_water_pixels"] > 0 else math.nan
        )
        row["no_water_fpr"] = (
            row["fp_no_water"] / row["no_water_pixels"]
            if row["no_water_pixels"] > 0 else math.nan
        )
        out.append(row)
    return out


def primary_positive_event_rows(event_rows):
    """Frozen primary endpoint uses only events with GT flood-positive support."""
    return [r for r in event_rows if int(r["reference_flood_pixels"]) > 0]


def event_macro_flood_iou(event_rows):
    rows = primary_positive_event_rows(event_rows)
    if not rows:
        raise RuntimeError("Primary event-macro IoU undefined: no flood-positive events")
    vals = np.asarray([float(r["flood_iou"]) for r in rows], dtype=np.float64)
    if not np.isfinite(vals).all():
        raise RuntimeError("Primary event-macro IoU encountered non-finite event IoU")
    return float(vals.mean())
