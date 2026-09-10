"""Manifest-driven GEOID S1-GRD loader for the frozen Semantic Shift protocol."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .core import preprocess_linear_sigma
from .augmentation import SynchronizedD4


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def event_folder_from_label_id(label_id):
    stem = str(label_id).replace("_label", "")
    if "-" not in stem:
        raise ValueError(f"Cannot derive GEOID event folder from label_id={label_id!r}")
    return stem.rsplit("-", 1)[0]


def geoid_paths(data_root, row):
    root = Path(data_root)
    event_folder = event_folder_from_label_id(row["label_id"])
    base = root / event_folder
    return {
        "pre": base / "s1grd" / f"{row['pre_raster_id']}.tif",
        "post": base / "s1grd" / f"{row['post_raster_id']}.tif",
        "label": base / "label" / f"{row['label_id']}.tif",
    }


def _read_window(path, x, y, size, indexes=None):
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError("rasterio is required for GEOID loading") from exc
    with rasterio.open(path) as src:
        window = Window(int(x), int(y), int(size), int(size))
        arr = src.read(indexes=indexes, window=window)
    return arr


class GEOIDManifestDataset(Dataset):
    """Read exactly the rows of one frozen chip manifest; no fallback substitution.

    ``include_teacher_context`` is used only for KD student training.  It keeps the
    student input strictly post-event [VV,VH] while also returning the aligned
    temporal teacher input [VV_pre,VH_pre,VV_post,VH_post] and a KD-valid mask.
    """

    def __init__(self, manifest_path, data_root, role, train=False,
                 expected_sha256=None, expected_chips=None, seed=0,
                 include_teacher_context=False):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.role = str(role)
        self.train = bool(train)
        self.seed = int(seed)
        self.include_teacher_context = bool(include_teacher_context)
        if self.role not in {"teacher", "student"}:
            raise ValueError("role must be teacher or student")
        if self.role == "teacher" and self.include_teacher_context:
            raise ValueError("include_teacher_context is only for student datasets")
        if expected_sha256 is not None:
            got = sha256_file(self.manifest_path)
            if got != expected_sha256:
                raise RuntimeError(f"Manifest SHA-256 mismatch: {got} != {expected_sha256}")
        self.rows = pd.read_csv(self.manifest_path)
        required = {
            "chip_id", "event_aoi_id", "label_id", "x", "y", "size",
            "sar_product", "sar_bands", "pre_raster_id", "post_raster_id",
        }
        missing = sorted(required - set(self.rows.columns))
        if missing:
            raise RuntimeError(f"Manifest missing required columns: {missing}")
        if expected_chips is not None and len(self.rows) != int(expected_chips):
            raise RuntimeError(f"Manifest row count mismatch: {len(self.rows)} != {expected_chips}")
        if not (self.rows["sar_product"].astype(str) == "s1grd").all():
            raise RuntimeError("Frozen GEOID manifest contains non-s1grd row")
        if not (self.rows["sar_bands"].astype(str).str.replace(" ", "") == "VV,VH").all():
            raise RuntimeError("Frozen GEOID band order must be VV,VH")
        self.d4 = SynchronizedD4(self.seed) if self.train else None

    def set_worker_seed(self, seed):
        if self.d4 is not None:
            self.d4.set_seed(int(seed))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[int(idx)]
        paths = geoid_paths(self.data_root, row)
        for key, p in paths.items():
            if not p.is_file():
                raise FileNotFoundError(f"Missing frozen GEOID {key} file: {p}")

        x, y, size = int(row["x"]), int(row["y"]), int(row["size"])
        post_raw = _read_window(paths["post"], x, y, size, indexes=[1, 2]).astype(np.float32)
        if post_raw.shape != (2, size, size):
            raise RuntimeError(f"Unexpected post window shape {post_raw.shape}")
        post, post_invalid = preprocess_linear_sigma(post_raw)

        label = _read_window(paths["label"], x, y, size, indexes=1)
        if label.ndim == 3:
            label = label[0]
        label = np.asarray(label, dtype=np.int64)
        if label.shape != (size, size):
            raise RuntimeError(f"Unexpected label window shape {label.shape}")

        valid = np.isin(label, (0, 1, 2)) & (~post_invalid.any(axis=0))
        teacher_image = None
        kd_valid = None

        need_pre = self.role == "teacher" or self.include_teacher_context
        if need_pre:
            pre_raw = _read_window(paths["pre"], x, y, size, indexes=[1, 2]).astype(np.float32)
            if pre_raw.shape != (2, size, size):
                raise RuntimeError(f"Unexpected pre window shape {pre_raw.shape}")
            pre, pre_invalid = preprocess_linear_sigma(pre_raw)
            temporal = np.concatenate([pre, post], axis=0)

            if self.role == "teacher":
                valid &= ~pre_invalid.any(axis=0)
                image = temporal
            else:
                image = post
                teacher_image = temporal
                kd_valid = valid & (~pre_invalid.any(axis=0))
        else:
            image = post

        if self.d4 is not None:
            if self.include_teacher_context:
                temporal_t, label, valid, extras = self.d4(
                    teacher_image, label, valid, extra_masks=[kd_valid]
                )
                teacher_image = temporal_t
                image = temporal_t[2:4]
                kd_valid = extras[0]
            else:
                image, label, valid = self.d4(image, label, valid)

        out = {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)).bool(),
            "chip_id": str(row["chip_id"]),
            "event_id": str(row["event_aoi_id"]),
        }
        if self.include_teacher_context:
            out["teacher_image"] = torch.from_numpy(np.ascontiguousarray(teacher_image)).float()
            out["kd_valid"] = torch.from_numpy(np.ascontiguousarray(kd_valid)).bool()
        return out
