"""Frozen Kuro/Siwo single-frame adapter for Semantic Shift E3.

This module contains only pure sample adaptation and locked-event validation.
It does not download datasets, open network connections, or run evaluation.
"""
from pathlib import Path
import csv

import numpy as np
import torch

from .core import preprocess_linear_sigma


KURO_EXPECTED_EVENTS = 43
KURO_EXPECTED_SAMPLES = 67490
KURO_NATIVE_SHAPE = (224, 224)
KURO_IGNORED_LABELS = (3,)


def canonical_kuro_event_id(actid, flood_date):
    """Build the frozen Kuro event identifier '<actid>|<flood_date>'."""
    return f"{str(actid)}|{str(flood_date)}"


def load_locked_event_ids(path, expected_count=KURO_EXPECTED_EVENTS):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["event_id"]:
            raise RuntimeError(f"Locked Kuro event CSV must contain exactly one 'event_id' column; got {reader.fieldnames}")
        ids = [str(r["event_id"]) for r in reader]
    if len(ids) != int(expected_count):
        raise RuntimeError(f"Locked Kuro event count mismatch: {len(ids)} != {expected_count}")
    if len(set(ids)) != len(ids):
        raise RuntimeError("Locked Kuro event IDs are not unique")
    return tuple(ids)


def adapt_kuro_arrays(vv, vh, mask, valid_mask, actid, flood_date,
                      locked_event_ids=None):
    """Adapt one Kuro test_GRD sample to the frozen E3 student interface.

    Parameters are arrays already supplied by the dataset reader. VV/VH must be
    linear sigma values. Label semantics are 0=no-water, 1=permanent-water,
    2=flood, 3=ignored. The returned image uses the exact frozen G4-02
    preprocessing and channel order [VV, VH].
    """
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)
    mask = np.asarray(mask)
    valid_mask = np.asarray(valid_mask)

    if vv.shape != KURO_NATIVE_SHAPE or vh.shape != KURO_NATIVE_SHAPE:
        raise RuntimeError(f"Kuro VV/VH shape must be {KURO_NATIVE_SHAPE}; got {vv.shape} and {vh.shape}")
    if mask.shape != KURO_NATIVE_SHAPE or valid_mask.shape != KURO_NATIVE_SHAPE:
        raise RuntimeError(f"Kuro mask/valid_mask shape must be {KURO_NATIVE_SHAPE}")

    event_id = canonical_kuro_event_id(actid, flood_date)
    if locked_event_ids is not None and event_id not in set(locked_event_ids):
        raise RuntimeError(f"Kuro sample event is outside the frozen 43-event set: {event_id}")

    raw = np.stack([vv, vh], axis=0)
    image, invalid = preprocess_linear_sigma(raw)

    label = mask.astype(np.int64, copy=False)
    allowed = np.isin(label, (0, 1, 2))
    valid = (valid_mask == 1) & allowed & (~invalid.any(axis=0))

    # A label outside {0,1,2,3} is not part of the frozen Kuro contract.
    if bool((~np.isin(label, (0, 1, 2, 3))).any()):
        raise RuntimeError("Kuro mask contains a label outside frozen values {0,1,2,3}")

    return {
        "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
        "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
        "valid": torch.from_numpy(np.ascontiguousarray(valid)).bool(),
        "event_id": event_id,
    }
