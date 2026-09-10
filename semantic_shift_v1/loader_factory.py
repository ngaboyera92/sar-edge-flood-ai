"""Exact DataLoader construction for frozen GEOID training/validation/evaluation."""
from __future__ import annotations

import torch

from .geoid_data import GEOIDManifestDataset
from .determinism import make_train_generator, seed_worker


def build_geoid_train_val_loaders(cfg, *, data_root, train_manifest_path, validation_manifest_path):
    """Construct loaders exactly as frozen in G4-04 and the immutable run config."""
    if int(cfg["batch_size"]) != 8:
        raise RuntimeError("Frozen batch size must be 8")
    d = cfg["determinism"]
    if not (d["train_shuffle"] is True and d["validation_shuffle"] is False):
        raise RuntimeError("Frozen train/validation shuffle policy mismatch")
    if d["drop_last"] is not False or int(d["num_workers"]) != 2:
        raise RuntimeError("Frozen DataLoader drop_last/num_workers mismatch")

    role = str(cfg["role"])
    condition = str(cfg["condition_id"])
    kd_run = role == "student" and condition not in {"B0", "A5"}

    train_id = cfg["data"]["train_manifest"]
    val_id = cfg["data"]["validation_manifest"]
    seed = int(cfg["seed"])

    train_ds = GEOIDManifestDataset(
        manifest_path=train_manifest_path,
        data_root=data_root,
        role=role,
        train=True,
        expected_sha256=train_id["sha256"],
        expected_chips=train_id["expected_chips"],
        seed=seed,
        include_teacher_context=kd_run,
    )
    val_ds = GEOIDManifestDataset(
        manifest_path=validation_manifest_path,
        data_root=data_root,
        role=role,
        train=False,
        expected_sha256=val_id["sha256"],
        expected_chips=val_id["expected_chips"],
        seed=seed,
        include_teacher_context=False,
    )

    generator = make_train_generator(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
    return train_loader, val_loader, generator


def build_geoid_eval_loader(eval_cfg, *, data_root, manifest_path, role="student"):
    """Construct the no-augmentation GEOID E1/E2 evaluation loader."""
    if str(eval_cfg["stage"]) not in {"E1", "E2"}:
        raise ValueError("GEOID evaluation loader is only valid for E1/E2")
    if eval_cfg["test_time_augmentation"] != "none":
        raise RuntimeError("Frozen evaluation forbids TTA")
    d = eval_cfg["dataset"]
    dataset = GEOIDManifestDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        role=str(role),
        train=False,
        expected_sha256=d["sha256"],
        expected_chips=d["expected_chips"],
        seed=0,
        include_teacher_context=False,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
