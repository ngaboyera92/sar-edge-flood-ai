"""Frozen G4-07 secondary metric primitives.

Pure numerical functions only. They do not read datasets, train models, or alter
frozen thresholds. Average precision is the step-weighted area under the empirical
precision-recall curve, grouping tied probabilities at one observed threshold.
"""
from __future__ import annotations

import math
import numpy as np


FIXED_THRESHOLD = 0.5
RELIABILITY_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)


def _valid_binary_arrays(probability, target, valid=None):
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(target).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("probability/target shape mismatch")
    if valid is None:
        v = np.ones(p.shape, dtype=bool)
    else:
        v = np.asarray(valid, dtype=bool).reshape(-1)
        if v.shape != p.shape:
            raise ValueError("valid shape mismatch")
    p = p[v]
    y = y[v]
    if p.size == 0:
        raise RuntimeError("No valid pixels")
    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite flood probability")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("Flood probabilities must lie in [0,1]")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Binary target must contain only 0/1 on valid pixels")
    return p, y.astype(np.uint8, copy=False)


def average_precision_step(probability, target, valid=None):
    """Exact step-weighted AP on supplied valid pixels, with tied scores grouped."""
    p, y = _valid_binary_arrays(probability, target, valid)
    positives = int(y.sum())
    if positives <= 0:
        raise RuntimeError("Average precision undefined with zero positive support")

    # Stable descending sort. Grouping ties makes within-tie order irrelevant.
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ys = y[order].astype(np.int64)

    # End index of each equal-score group.
    ends = np.r_[np.flatnonzero(ps[1:] != ps[:-1]), ps.size - 1]
    cum_tp = np.cumsum(ys, dtype=np.int64)
    ranks = ends + 1
    tp = cum_tp[ends].astype(np.float64)
    precision = tp / ranks.astype(np.float64)
    recall = tp / float(positives)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision, dtype=np.float64))


def threshold_counts(probability, target, valid=None, threshold=FIXED_THRESHOLD):
    if float(threshold) != FIXED_THRESHOLD:
        raise ValueError("Frozen threshold is exactly 0.5")
    p, y = _valid_binary_arrays(probability, target, valid)
    pred = p > FIXED_THRESHOLD
    pos = y == 1
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & ~pos).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_metrics(probability, target, valid=None, threshold=FIXED_THRESHOLD):
    c = threshold_counts(probability, target, valid, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn
    iou = tp / iou_den if iou_den else math.nan
    precision = tp / precision_den if precision_den else math.nan
    recall = tp / recall_den if recall_den else math.nan
    f1_den = 2 * tp + fp + fn
    f1 = (2 * tp) / f1_den if f1_den else math.nan
    return {**c, "iou": iou, "precision": precision, "recall": recall, "f1": f1}


def brier_sum_count(probability, target, valid=None):
    p, y = _valid_binary_arrays(probability, target, valid)
    err = (p - y.astype(np.float64)) ** 2
    return {"brier_sum": float(err.sum(dtype=np.float64)), "brier_count": int(err.size)}


def reliability_bin_sufficient_statistics(probability, target, valid=None):
    """Ten fixed bins [0,.1),...,[.9,1.0], final bin includes 1.0."""
    p, y = _valid_binary_arrays(probability, target, valid)
    # floor(10*p), with p==1 clipped into final bin.
    idx = np.minimum((p * 10.0).astype(np.int64), 9)
    count = np.bincount(idx, minlength=10).astype(np.int64)
    prob_sum = np.bincount(idx, weights=p, minlength=10).astype(np.float64)
    pos_count = np.bincount(idx, weights=y.astype(np.float64), minlength=10).astype(np.float64)
    return {
        "bin_count": count,
        "bin_probability_sum": prob_sum,
        "bin_positive_count": pos_count,
    }


def empirical_pr_curve(probability, target, valid=None):
    """Return observed-score PR points using prediction p>=threshold for diagnostics."""
    p, y = _valid_binary_arrays(probability, target, valid)
    positives = int(y.sum())
    if positives <= 0:
        raise RuntimeError("PR curve undefined with zero positive support")
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ys = y[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(ps[1:] != ps[:-1]), ps.size - 1]
    cum_tp = np.cumsum(ys, dtype=np.int64)
    tp = cum_tp[ends].astype(np.float64)
    n_pred = (ends + 1).astype(np.float64)
    thresholds = ps[ends]
    precision = tp / n_pred
    recall = tp / float(positives)
    return {
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
    }


def operating_point_matches(a3_probability, target, comparator_precision, comparator_recall, valid=None):
    """Frozen A3 diagnostic: closest precision and closest recall; ties -> higher threshold."""
    curve = empirical_pr_curve(a3_probability, target, valid)
    t = curve["threshold"]
    p = curve["precision"]
    r = curve["recall"]

    def choose(values, reference):
        if not np.isfinite(reference):
            return {"threshold": math.nan, "precision": math.nan, "recall": math.nan}
        dist = np.abs(values - float(reference))
        best = np.nanmin(dist)
        candidates = np.flatnonzero(dist == best)
        # Exact tie: higher observed threshold.
        j = candidates[np.argmax(t[candidates])]
        return {"threshold": float(t[j]), "precision": float(p[j]), "recall": float(r[j])}

    return {
        "precision_matched": choose(p, comparator_precision),
        "recall_matched": choose(r, comparator_recall),
    }
