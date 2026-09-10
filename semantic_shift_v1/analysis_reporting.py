"""Frozen G4-07 seed summaries, contrast reporting and interpretation helpers."""
from __future__ import annotations

import numpy as np

from .analysis_protocol import paired_crossed_bootstrap_difference


def five_seed_summary(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.shape != (5,) or not np.isfinite(x).all():
        raise ValueError("Frozen seed summary requires exactly five finite values")
    return {
        "seed_values": x.tolist(),
        "mean": float(x.mean()),
        "sample_std_ddof1": float(x.std(ddof=1)),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
    }


def paired_seed_delta_summary(treatment, comparator):
    a = np.asarray(treatment, dtype=np.float64).reshape(-1)
    b = np.asarray(comparator, dtype=np.float64).reshape(-1)
    if a.shape != (5,) or b.shape != (5,):
        raise ValueError("Frozen paired seed summary requires five paired seeds")
    d = a - b
    return {
        "paired_seed_deltas": d.tolist(),
        "seed_wins": int((d > 0).sum()),
        "seed_ties": int((d == 0).sum()),
        "seed_losses": int((d < 0).sum()),
        "mean_delta": float(d.mean()),
    }


def crossed_contrast_row(treatment_panel, comparator_panel, *, favorable_direction="positive"):
    r = paired_crossed_bootstrap_difference(treatment_panel, comparator_panel)
    delta = float(r["point_estimate"])
    if favorable_direction not in {"positive", "negative"}:
        raise ValueError("favorable_direction must be positive or negative")
    favorable = delta > 0 if favorable_direction == "positive" else delta < 0
    return {
        "point_estimate": delta,
        "pointwise_ci_low": r["pointwise_ci"][0],
        "pointwise_ci_high": r["pointwise_ci"][1],
        "familywise_ci_low": r["familywise_ci"][0],
        "familywise_ci_high": r["familywise_ci"][1],
        "direction_favors_A3": bool(favorable),
    }


def primary_familywise_interpretation(ci_low, ci_high):
    lo, hi = float(ci_low), float(ci_high)
    if lo > 0:
        return "family-wise resolved improvement"
    if hi < 0:
        return "family-wise resolved harm"
    return "direction/uncertainty only — simultaneous interval crosses zero"


def shift_claim_scope(rows):
    """Apply the frozen temporal-OOD + Kuro claim rule for one comparator."""
    by_domain = {str(r["domain"]): r for r in rows}
    if "E2" not in by_domain or "E3" not in by_domain:
        raise ValueError("E2 and E3 rows are required")
    e2, e3 = by_domain["E2"], by_domain["E3"]
    positive_both = float(e2["point_estimate"]) > 0 and float(e3["point_estimate"]) > 0
    resolved_both = float(e2["familywise_ci_low"]) > 0 and float(e3["familywise_ci_low"]) > 0
    if resolved_both:
        return "resolved improvement across temporal-OOD and Kuro"
    if positive_both:
        return "positive point estimates across temporal-OOD and Kuro; unresolved family-wise uncertainty"
    return "domain-specific only"
