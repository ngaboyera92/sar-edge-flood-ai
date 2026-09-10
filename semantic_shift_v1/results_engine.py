"""Frozen G4-07 confirmatory analysis and exact result-artifact writer."""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_protocol import paired_crossed_bootstrap_difference
from .analysis_reporting import five_seed_summary, paired_seed_delta_summary, primary_familywise_interpretation
from .artifact_protocol import RESULT_LEAVES, write_new_bytes, write_new_json, write_new_text, write_checksum_manifest
from .score_store import DuckDBScoreStore


DOMAINS = ("E1_GEOID_SOURCE", "E2_GEOID_HELDOUT_2026", "E3_KURO_TEST_GRD")
CONDITIONS = ("B0", "A0", "B3", "A1", "A2", "A3", "A4", "A5")
SEEDS = (0, 1, 2, 3, 4)
PRIMARY_COMPARATORS = ("A0", "B0")
SUPPORT_COMPARATORS = ("B3", "A1", "A2", "A4", "A5")


def _frame(table):
    return table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame(table)


def _table_for(event_tables, domain, condition, seed):
    try:
        df = _frame(event_tables[domain][condition][seed])
    except Exception as exc:
        raise RuntimeError(f"Missing event table for {domain}/{condition}/s{seed:02d}") from exc
    required = {
        "event_id", "reference_flood_pixels", "permanent_water_pixels", "no_water_pixels",
        "flood_iou", "permanent_water_fpr", "no_water_fpr",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Event table missing columns {missing}")
    if df["event_id"].astype(str).duplicated().any():
        raise RuntimeError(f"Duplicate event IDs in {domain}/{condition}/s{seed:02d}")
    return df.assign(event_id=df["event_id"].astype(str)).set_index("event_id", drop=False)


def _panel(event_tables, domain, condition, metric, denominator_column):
    tables = [_table_for(event_tables, domain, condition, seed) for seed in SEEDS]
    reference = tables[0]
    eligible = tuple(sorted(reference.index[reference[denominator_column].astype(float) > 0]))
    if len(eligible) < 2:
        raise RuntimeError(f"{domain}/{metric} has fewer than two eligible event units")
    reference_ground = reference.loc[list(eligible), denominator_column].astype(float).to_numpy()
    values = []
    for seed, df in zip(SEEDS, tables):
        if tuple(sorted(df.index)) != tuple(sorted(reference.index)):
            raise RuntimeError(f"Event set changed across seeds for {domain}/{condition}")
        got_ground = df.loc[list(eligible), denominator_column].astype(float).to_numpy()
        if not np.array_equal(got_ground, reference_ground):
            raise RuntimeError(f"Ground-truth denominator changed across seeds for {domain}/{condition}")
        row = df.loc[list(eligible), metric].astype(float).to_numpy()
        if not np.isfinite(row).all():
            raise RuntimeError(f"Non-finite eligible metric in {domain}/{condition}/{metric}")
        values.append(row)
    return np.asarray(values, dtype=np.float64), eligible, reference_ground


def _paired_panels(event_tables, domain, treatment, comparator, metric, denominator_column):
    a, events_a, ground_a = _panel(event_tables, domain, treatment, metric, denominator_column)
    b, events_b, ground_b = _panel(event_tables, domain, comparator, metric, denominator_column)
    if events_a != events_b or not np.array_equal(ground_a, ground_b):
        raise RuntimeError(
            f"Paired event eligibility differs for {domain}/{treatment}-{comparator}/{metric}"
        )
    return a, b, events_a


def _condition_summary_rows(event_tables, metric, denominator_column):
    rows = []
    for domain in DOMAINS:
        for condition in CONDITIONS:
            panel, events, _ = _panel(event_tables, domain, condition, metric, denominator_column)
            seed_values = panel.mean(axis=1)
            s = five_seed_summary(seed_values)
            rows.append({
                "domain": domain,
                "condition": condition,
                "eligible_event_count": len(events),
                **{f"seed_{i}": float(seed_values[i]) for i in SEEDS},
                "mean": s["mean"],
                "sample_std_ddof1": s["sample_std_ddof1"],
                "median": s["median"],
                "min": s["min"],
                "max": s["max"],
                "range": s["range"],
            })
    return rows


def _contrast(event_tables, domain, comparator, metric, denominator_column, *, favorable_direction):
    a, b, events = _paired_panels(
        event_tables, domain, "A3", comparator, metric, denominator_column
    )
    boot = paired_crossed_bootstrap_difference(a, b)
    seed_a = a.mean(axis=1)
    seed_b = b.mean(axis=1)
    robust = paired_seed_delta_summary(seed_a, seed_b)
    row = {
        "domain": domain,
        "contrast": f"A3-{comparator}",
        "eligible_event_count": len(events),
        "point_estimate": boot["point_estimate"],
        "pointwise_ci_low": boot["pointwise_ci"][0],
        "pointwise_ci_high": boot["pointwise_ci"][1],
        "familywise_ci_low": boot["familywise_ci"][0],
        "familywise_ci_high": boot["familywise_ci"][1],
        "seed_wins": robust["seed_wins"],
        "seed_ties": robust["seed_ties"],
        "seed_losses": robust["seed_losses"],
        **{f"paired_seed_delta_{i}": robust["paired_seed_deltas"][i] for i in SEEDS},
    }
    if favorable_direction == "positive":
        row["familywise_interpretation"] = primary_familywise_interpretation(
            row["familywise_ci_low"], row["familywise_ci_high"]
        )
    elif favorable_direction == "negative":
        lo, hi = row["familywise_ci_low"], row["familywise_ci_high"]
        row["familywise_interpretation"] = (
            "family-wise resolved reduction" if hi < 0 else
            "family-wise resolved increase" if lo > 0 else
            "direction/uncertainty only — simultaneous interval crosses zero"
        )
    else:
        raise ValueError("Unknown favorable direction")
    return row, boot["bootstrap_samples"]


def _write_csv_new(path, rows):
    text = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n")
    write_new_text(path, text)


def _write_npz_new(path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    write_new_bytes(path, buffer.getvalue())


def compute_and_write_results(
    project_root,
    event_tables,
    *,
    domain_summaries=None,
    a3_score_db_paths=None,
    analysis_identity=None,
):
    """Compute frozen confirmatory tables and create the ten exact 09_Results leaves."""
    root = Path(project_root) / "09_Results"
    root.mkdir(parents=True, exist_ok=True)
    existing = [name for name in RESULT_LEAVES if (root / name).exists()]
    if existing:
        raise FileExistsError(f"Frozen results artifacts already exist: {existing}")

    primary_summary = _condition_summary_rows(
        event_tables, "flood_iou", "reference_flood_pixels"
    )
    permanent_summary = _condition_summary_rows(
        event_tables, "permanent_water_fpr", "permanent_water_pixels"
    )

    primary_rows, primary_draws = [], {}
    permanent_rows, permanent_draws = [], {}
    no_water_rows = []
    supporting_rows = []

    for domain in DOMAINS:
        for comparator in PRIMARY_COMPARATORS:
            p_row, p_draw = _contrast(
                event_tables, domain, comparator, "flood_iou", "reference_flood_pixels",
                favorable_direction="positive",
            )
            primary_rows.append(p_row)
            primary_draws[f"{domain}__A3_minus_{comparator}"] = p_draw

            w_row, w_draw = _contrast(
                event_tables, domain, comparator, "permanent_water_fpr", "permanent_water_pixels",
                favorable_direction="negative",
            )
            permanent_rows.append(w_row)
            permanent_draws[f"{domain}__A3_minus_{comparator}"] = w_draw

            n_row, _ = _contrast(
                event_tables, domain, comparator, "no_water_fpr", "no_water_pixels",
                favorable_direction="negative",
            )
            no_water_rows.append({
                **{k: v for k, v in n_row.items() if not k.startswith("familywise_")},
                "interpretation": "prespecified negative control; pointwise interval only",
            })

        for comparator in SUPPORT_COMPARATORS:
            for endpoint, metric, denominator, favorable in (
                ("event_macro_flood_iou", "flood_iou", "reference_flood_pixels", "positive"),
                ("event_macro_permanent_water_fpr", "permanent_water_fpr", "permanent_water_pixels", "negative"),
            ):
                row, _ = _contrast(
                    event_tables, domain, comparator, metric, denominator,
                    favorable_direction=favorable,
                )
                supporting_rows.append({
                    "endpoint": endpoint,
                    **{k: v for k, v in row.items() if not k.startswith("familywise_")},
                    "interpretation": "supporting mechanism analysis; pointwise interval only",
                })

    operating_points = []
    if a3_score_db_paths is not None:
        if domain_summaries is None:
            raise ValueError("domain_summaries required for operating-point matching")
        for domain in DOMAINS:
            for seed in SEEDS:
                db_path = a3_score_db_paths[domain][seed]
                with DuckDBScoreStore(db_path) as store:
                    for comparator in PRIMARY_COMPARATORS:
                        ref = domain_summaries[domain][comparator][seed]
                        match = store.operating_point_match(
                            ref["pooled_precision"], ref["pooled_recall"]
                        )
                        operating_points.append({
                            "domain": domain,
                            "seed": seed,
                            "comparator": comparator,
                            "comparator_threshold": 0.5,
                            "comparator_precision": ref["pooled_precision"],
                            "comparator_recall": ref["pooled_recall"],
                            **match,
                        })

    statistical_summary = {
        "status": "FROZEN_ANALYSIS_OUTPUT",
        "primary_family_size": 6,
        "key_secondary_permanent_water_family_size": 6,
        "bootstrap_replicates": 50000,
        "bootstrap_seed": 20260910,
        "rng": "numpy.default_rng_PCG64",
        "pointwise_quantiles": [0.025, 0.975],
        "familywise_quantiles": [0.0041666667, 0.9958333333],
        "primary_rows": primary_rows,
        "permanent_water_rows": permanent_rows,
        "operating_point_matches": operating_points,
        "analysis_identity": analysis_identity,
    }

    _write_csv_new(root / "condition_summary_event_macro_iou.csv", primary_summary)
    _write_csv_new(root / "primary_contrasts_event_macro_iou.csv", primary_rows)
    _write_npz_new(root / "bootstrap_primary_event_macro_iou.npz", primary_draws)
    _write_csv_new(root / "permanent_water_fpr_summary.csv", permanent_summary)
    _write_csv_new(root / "permanent_water_fpr_contrasts.csv", permanent_rows)
    _write_npz_new(root / "bootstrap_permanent_water_fpr.npz", permanent_draws)
    _write_csv_new(root / "no_water_fpr_negative_control.csv", no_water_rows)
    _write_csv_new(root / "supporting_mechanism_contrasts.csv", supporting_rows)
    write_new_json(root / "confirmatory_statistical_summary.json", statistical_summary)

    payload = [root / name for name in RESULT_LEAVES if name != "RESULTS_SHA256SUMS.txt"]
    write_checksum_manifest(
        root / "RESULTS_SHA256SUMS.txt", payload, relative_to=root
    )
    return root
