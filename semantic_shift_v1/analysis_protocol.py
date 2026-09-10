"""Frozen G4-07 confirmatory crossed-bootstrap analysis primitives.

This module contains analysis-only functions. It does not train models, read imagery,
or alter frozen protocol choices.
"""
from __future__ import annotations

import numpy as np


PRIMARY_CONTRASTS = ("A3-A0", "A3-B0")
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20260910
POINTWISE_QUANTILES = (0.025, 0.975)
FAMILYWISE_QUANTILES = (0.0041666667, 0.9958333333)


def _as_paired_panel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired panels must have identical shape: {a.shape} != {b.shape}")
    if a.ndim != 2:
        raise ValueError("Paired panels must be two-dimensional [seed, event]")
    if a.shape[0] != 5:
        raise ValueError("Frozen G4-07 requires exactly five seeds")
    if a.shape[1] < 1:
        raise ValueError("At least one event is required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Bootstrap panels must contain only finite values")
    return a, b


def paired_crossed_bootstrap_difference(
    treatment,
    comparator,
    *,
    replicates=BOOTSTRAP_REPLICATES,
    seed=BOOTSTRAP_SEED,
):
    """Bootstrap the paired mean difference by independently resampling seeds and events.

    Pairing is preserved because both conditions are indexed by the same sampled seed
    and event indices before the difference is averaged.
    """
    treatment, comparator = _as_paired_panel(treatment, comparator)
    replicates = int(replicates)
    seed = int(seed)
    if replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("Frozen G4-07 requires exactly 50,000 bootstrap replicates")
    if seed != BOOTSTRAP_SEED:
        raise ValueError("Frozen G4-07 requires bootstrap seed 20260910")

    diff = treatment - comparator
    n_seed, n_event = diff.shape
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)

    # Sequential replicate generation fixes an unambiguous RNG consumption order.
    for i in range(replicates):
        seed_idx = rng.integers(0, n_seed, size=n_seed)
        event_idx = rng.integers(0, n_event, size=n_event)
        samples[i] = diff[np.ix_(seed_idx, event_idx)].mean(dtype=np.float64)

    return {
        "point_estimate": float(diff.mean(dtype=np.float64)),
        "bootstrap_samples": samples,
        "pointwise_ci": tuple(float(x) for x in np.quantile(samples, POINTWISE_QUANTILES)),
        "familywise_ci": tuple(float(x) for x in np.quantile(samples, FAMILYWISE_QUANTILES)),
        "replicates": replicates,
        "seed": seed,
        "rng": "numpy.default_rng_PCG64",
    }


def summarize_primary_contrasts(panels_by_domain):
    """Run the six frozen primary estimands: A3-A0 and A3-B0 in E1/E2/E3."""
    expected_domains = ("E1", "E2", "E3")
    if tuple(panels_by_domain.keys()) != expected_domains:
        raise ValueError("Domains must be provided in frozen order E1, E2, E3")
    rows = []
    for domain in expected_domains:
        p = panels_by_domain[domain]
        for contrast, comparator in (("A3-A0", "A0"), ("A3-B0", "B0")):
            if "A3" not in p or comparator not in p:
                raise ValueError(f"Missing {domain} panel for {contrast}")
            r = paired_crossed_bootstrap_difference(p["A3"], p[comparator])
            rows.append({
                "domain": domain,
                "contrast": contrast,
                "point_estimate": r["point_estimate"],
                "pointwise_ci_low": r["pointwise_ci"][0],
                "pointwise_ci_high": r["pointwise_ci"][1],
                "familywise_ci_low": r["familywise_ci"][0],
                "familywise_ci_high": r["familywise_ci"][1],
            })
    return rows


def permanent_water_direction_is_favorable(delta):
    """For A3-comparator permanent-water FPR differences, negative favors A3."""
    return float(delta) < 0.0
