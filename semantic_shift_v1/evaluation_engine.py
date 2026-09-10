"""Frozen G4-06/G4-07 evaluation engine.

Consumes an already-frozen model and loader, uses no TTA, and writes no model state.
Threshold-free scores are spooled to DuckDB so exact pooled/event AP does not require
retaining the full domain in RAM.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import torch

from .core import flood_probability
from .evaluation_protocol import chip_sufficient_statistics, aggregate_event_statistics
from .secondary_metrics import brier_sum_count, reliability_bin_sufficient_statistics
from .score_store import DuckDBScoreStore


_COUNT_FIELDS = (
    "valid_pixels", "no_water_pixels", "permanent_water_pixels",
    "reference_flood_pixels", "predicted_flood_pixels",
    "tp", "fp", "fp_no_water", "fp_permanent_water", "fn", "tn",
)


def _safe_ratio(num, den):
    return float(num) / float(den) if int(den) > 0 else math.nan


def evaluate_frozen_model(model, loader, condition, device, *, score_db_path):
    """Evaluate one frozen checkpoint on one frozen domain without TTA."""
    condition = str(condition)
    model.eval()
    chip_records = []
    event_brier = defaultdict(lambda: [0.0, 0])
    pooled_brier = [0.0, 0]
    pooled_bins = {
        "bin_count": np.zeros(10, dtype=np.int64),
        "bin_probability_sum": np.zeros(10, dtype=np.float64),
        "bin_positive_count": np.zeros(10, dtype=np.float64),
    }

    score_path = Path(score_db_path)
    if score_path.exists():
        raise FileExistsError(score_path)

    with DuckDBScoreStore(score_path) as scores:
        with torch.no_grad():
            for batch in loader:
                image = batch["image"].to(device)
                label = batch["label"].to(device)
                valid = batch["valid"].to(device).bool()
                logits = model(image)
                prob = flood_probability(logits, condition)
                if prob.ndim == 4 and prob.shape[1] == 1:
                    prob = prob[:, 0]
                if prob.shape != label.shape:
                    raise RuntimeError(
                        f"Evaluation probability/label shape mismatch: {tuple(prob.shape)} != {tuple(label.shape)}"
                    )
                event_ids = batch["event_id"]
                if len(event_ids) != int(label.shape[0]):
                    raise RuntimeError("event_id batch length mismatch")

                for i, event_id in enumerate(event_ids):
                    eid = str(event_id)
                    p = prob[i].detach().cpu().float().numpy()
                    y_class = label[i].detach().cpu().numpy()
                    v = valid[i].detach().cpu().numpy().astype(bool)
                    y = (y_class == 2).astype(np.uint8)

                    counts = chip_sufficient_statistics(p, y_class, v, threshold=0.5)
                    chip_records.append({"event_id": eid, **counts})

                    b = brier_sum_count(p, y, v)
                    event_brier[eid][0] += b["brier_sum"]
                    event_brier[eid][1] += b["brier_count"]
                    pooled_brier[0] += b["brier_sum"]
                    pooled_brier[1] += b["brier_count"]

                    bins = reliability_bin_sufficient_statistics(p, y, v)
                    for key in pooled_bins:
                        pooled_bins[key] += bins[key]

                    scores.append(eid, p, y, v)

        event_rows = aggregate_event_statistics(chip_records)
        event_ap = scores.event_average_precision()
        pooled_ap = scores.pooled_average_precision()

        for row in event_rows:
            eid = row["event_id"]
            row["average_precision"] = event_ap.get(eid, math.nan)
            row["brier_sum"] = float(event_brier[eid][0])
            row["brier_count"] = int(event_brier[eid][1])
            row["brier_score"] = _safe_ratio(row["brier_sum"], row["brier_count"])

        total = {k: sum(int(r[k]) for r in event_rows) for k in _COUNT_FIELDS}
        flood_positive = [r for r in event_rows if int(r["reference_flood_pixels"]) > 0]
        if len(flood_positive) < 2:
            raise RuntimeError("G4-07 inferential domain requires at least two flood-positive event units")
        perm_positive = [r for r in event_rows if int(r["permanent_water_pixels"]) > 0]
        no_water_positive = [r for r in event_rows if int(r["no_water_pixels"]) > 0]

        micro_iou = _safe_ratio(total["tp"], total["tp"] + total["fp"] + total["fn"])
        precision = _safe_ratio(total["tp"], total["tp"] + total["fp"])
        recall = _safe_ratio(total["tp"], total["tp"] + total["fn"])
        f1 = _safe_ratio(2 * total["tp"], 2 * total["tp"] + total["fp"] + total["fn"])

        summary = {
            "condition_id": condition,
            "threshold": 0.5,
            "event_count": len(event_rows),
            "flood_positive_event_count": len(flood_positive),
            "pooled_counts": total,
            "pooled_micro_iou": micro_iou,
            "pooled_precision": precision,
            "pooled_recall": recall,
            "pooled_f1": f1,
            "event_macro_flood_iou": float(np.mean([r["flood_iou"] for r in flood_positive])),
            "pooled_permanent_water_fpr": _safe_ratio(total["fp_permanent_water"], total["permanent_water_pixels"]),
            "event_macro_permanent_water_fpr": (
                float(np.mean([r["permanent_water_fpr"] for r in perm_positive]))
                if perm_positive else math.nan
            ),
            "pooled_no_water_fpr": _safe_ratio(total["fp_no_water"], total["no_water_pixels"]),
            "event_macro_no_water_fpr": (
                float(np.mean([r["no_water_fpr"] for r in no_water_positive]))
                if no_water_positive else math.nan
            ),
            "pooled_average_precision": pooled_ap,
            "event_macro_average_precision": float(np.mean([event_ap[r["event_id"]] for r in flood_positive])),
            "pooled_brier_score": _safe_ratio(pooled_brier[0], pooled_brier[1]),
            "event_macro_brier_score": float(np.mean([r["brier_score"] for r in event_rows])),
            "brier_sum": float(pooled_brier[0]),
            "brier_count": int(pooled_brier[1]),
            "reliability_bins": {
                "edges": [float(x) for x in np.linspace(0.0, 1.0, 11)],
                "count": pooled_bins["bin_count"].astype(int).tolist(),
                "probability_sum": pooled_bins["bin_probability_sum"].astype(float).tolist(),
                "positive_count": pooled_bins["bin_positive_count"].astype(float).tolist(),
                "empty_bins_are_NA": True,
                "final_bin_includes_1": True,
            },
            "score_store_backend": "DuckDB FLOAT/UTINYINT disk-backed exact AP",
        }
        return event_rows, summary
