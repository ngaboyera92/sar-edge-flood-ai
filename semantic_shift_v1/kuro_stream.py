"""Execution adapter for the frozen Kuro E3 Hugging Face WebDataset.

The immutable E3 config intentionally leaves dataset_revision=null and requires the
concrete data identity used at execution to be recorded. This module resolves the
repository HEAD exactly once at E3 start, then loads that immutable SHA and exposes
only the post-flood VV/VH + label/valid fields required by the frozen student path.
"""
from __future__ import annotations

import json
from torch.utils.data import IterableDataset

from .kuro_data import adapt_kuro_arrays


KURO_REPOSITORY = "orion-ai-lab/Kuro-Siwo-Webdataset"
KURO_HF_CONFIG = "labelled_GRD"
KURO_HF_SPLIT = "test"


def resolve_kuro_execution_revision():
    """Resolve the concrete Hub SHA to record before E3 traversal."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for Kuro E3") from exc
    info = HfApi().dataset_info(KURO_REPOSITORY)
    sha = getattr(info, "sha", None)
    if not sha or len(str(sha)) != 40:
        raise RuntimeError("Could not resolve a concrete 40-character Kuro repository SHA")
    return str(sha)


def load_kuro_hf_stream(revision):
    """Load labelled_GRD/test at an already-resolved concrete revision."""
    if not revision or len(str(revision)) != 40:
        raise ValueError("Kuro execution revision must be a concrete 40-character SHA")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for Kuro E3 WebDataset streaming") from exc
    return load_dataset(
        KURO_REPOSITORY,
        KURO_HF_CONFIG,
        split=KURO_HF_SPLIT,
        streaming=True,
        revision=str(revision),
    )


def _get(sample, exact_name):
    """Accept the exact WebDataset field name or HF's extension-stripped variant."""
    if exact_name in sample:
        return sample[exact_name]
    stem = exact_name.rsplit(".", 1)[0]
    if stem in sample:
        return sample[stem]
    raise KeyError(f"Kuro sample missing required field {exact_name!r}")


def _decode_info(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        return json.loads(bytes(value).decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Unsupported Kuro info.json representation: {type(value).__name__}")


def adapt_kuro_webdataset_sample(sample, locked_event_ids):
    info = _decode_info(_get(sample, "info.json"))
    if "actid" not in info or "flood_date" not in info:
        raise RuntimeError("Kuro info.json must contain actid and flood_date")
    return adapt_kuro_arrays(
        vv=_get(sample, "flood_vv.npy"),
        vh=_get(sample, "flood_vh.npy"),
        mask=_get(sample, "mask.npy"),
        valid_mask=_get(sample, "valid_mask.npy"),
        actid=info["actid"],
        flood_date=info["flood_date"],
        locked_event_ids=locked_event_ids,
    )


class KuroHFIterableDataset(IterableDataset):
    """Adapt a fixed-revision HF iterable to the frozen E3 student interface."""
    def __init__(self, stream, locked_event_ids):
        super().__init__()
        self.stream = stream
        self.locked_event_ids = tuple(locked_event_ids)

    def __iter__(self):
        for sample in self.stream:
            yield adapt_kuro_webdataset_sample(sample, self.locked_event_ids)
