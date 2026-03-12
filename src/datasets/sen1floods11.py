
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tifffile as tiff


@dataclass
class Sen1Item:
    base_id: str
    image_path: str
    mask_path: str


def _extract_base_id(s: str) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    m = re.search(r'([A-Za-z\-]+_\d+)', s)
    return m.group(1) if m else None


class Sen1Floods11Dataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None, target_transform=None, return_meta: bool = False):
        self.root_dir = root_dir
        self.split = split.lower()
        self.transform = transform
        self.target_transform = target_transform
        self.return_meta = return_meta
        self.items = self._build_index()

    def _split_csv_path(self) -> str:
        split_dir = os.path.join(self.root_dir, "splits", "flood_handlabeled")
        mapping = {
            "train": "flood_train_data.csv",
            "val": "flood_val_data.csv",
            "validation": "flood_val_data.csv",
            "test": "flood_test_data.csv",
        }
        if self.split not in mapping:
            raise ValueError(f"Unknown split '{self.split}'. Use train/val/test.")
        return os.path.join(split_dir, mapping[self.split])

    def _build_maps(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        s1_dir = os.path.join(self.root_dir, "data", "flood_events", "HandLabeled", "S1Hand")
        lb_dir = os.path.join(self.root_dir, "data", "flood_events", "HandLabeled", "LabelHand")

        if not os.path.isdir(s1_dir):
            raise FileNotFoundError(f"Missing S1Hand directory: {s1_dir}")
        if not os.path.isdir(lb_dir):
            raise FileNotFoundError(f"Missing LabelHand directory: {lb_dir}")

        s1_map = {}
        for fn in os.listdir(s1_dir):
            if fn.endswith("_S1Hand.tif"):
                s1_map[fn.replace("_S1Hand.tif", "")] = os.path.join(s1_dir, fn)

        lb_map = {}
        for fn in os.listdir(lb_dir):
            if fn.endswith("_LabelHand.tif"):
                lb_map[fn.replace("_LabelHand.tif", "")] = os.path.join(lb_dir, fn)

        return s1_map, lb_map

    def _read_split_base_ids(self) -> List[str]:
        csv_path = self._split_csv_path()
        df = pd.read_csv(csv_path)

        best_col, best_hits = None, -1
        for col in df.columns:
            hits = df[col].astype(str).head(200).apply(lambda x: _extract_base_id(x) is not None).sum()
            if hits > best_hits:
                best_hits, best_col = hits, col

        if best_col is None or best_hits <= 0:
            raise RuntimeError(f"Could not detect ID column in {csv_path}. Columns={df.columns.tolist()}")

        base_ids = []
        for v in df[best_col].astype(str).tolist():
            b = _extract_base_id(v)
            if b:
                base_ids.append(b)

        seen = set()
        out = []
        for b in base_ids:
            if b not in seen:
                out.append(b)
                seen.add(b)
        return out

    def _build_index(self) -> List[Sen1Item]:
        s1_map, lb_map = self._build_maps()
        base_ids = self._read_split_base_ids()

        items = []
        missing = 0
        for base in base_ids:
            img = s1_map.get(base)
            msk = lb_map.get(base)
            if img is None or msk is None:
                missing += 1
                continue
            items.append(Sen1Item(base, img, msk))

        if len(items) == 0:
            raise RuntimeError("No matched (image, mask) pairs found.")
        if missing > 0:
            print(f"Warning: {missing} IDs from split CSV were not found in folders.")
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = tiff.imread(it.image_path).astype(np.float32)
        msk = tiff.imread(it.mask_path).astype(np.int64)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        msk = np.nan_to_num(msk, nan=0, posinf=0, neginf=0)

        if img.ndim == 2:
            img = img[None, ...]
        elif img.ndim == 3 and (img.shape[0] not in (1, 2, 3, 4)) and (img.shape[-1] in (1, 2, 3, 4)):
            img = np.transpose(img, (2, 0, 1))

        if msk.ndim == 2:
            msk = msk[None, ...]
        elif msk.ndim == 3:
            msk = msk[:1, ...]

        flood = (msk > 0).astype(np.int64)
        img_t = torch.from_numpy(img)
        mask_t = torch.from_numpy(flood)

        if self.transform:
            img_t = self.transform(img_t)
        if self.target_transform:
            mask_t = self.target_transform(mask_t)

        if self.return_meta:
            return img_t, mask_t, {
                "base_id": it.base_id,
                "image_path": it.image_path,
                "mask_path": it.mask_path,
            }

        return img_t, mask_t
