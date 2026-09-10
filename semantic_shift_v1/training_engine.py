"""Frozen Semantic Shift training engine.

The public entry point performs the overall-G4 authorization check before any model,
optimizer, artifact directory, or training loop is created. The module is therefore
safe to import while G4 is OPEN; execution is blocked until the caller supplies the
frozen implementation SHA and CLOSED/FROZEN status.
"""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

import torch

from .config import validate_run_config
from .core import seed_everything
from .losses import teacher_loss, student_loss
from .models import build_model
from .determinism import make_train_generator
from .training_protocol import (
    assert_training_authorized,
    build_optimizer,
    build_scheduler,
    strict_improvement,
    binary_micro_iou,
    teacher_three_class_macro_iou,
    make_recovery_payload,
    recovery_filename,
)
from .artifact_protocol import create_training_run_dirs


def _atomic_torch_save(obj, path):
    """Atomically replace a within-run mutable checkpoint file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".tmp-", dir=str(p.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, p)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _student_validation_metric(model, loader, condition, device):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            valid = batch["valid"].to(device).bool()
            logits = model(image)
            if condition == "A5":
                prob = torch.softmax(logits, dim=1)[:, 2]
            else:
                prob = torch.sigmoid(logits[:, 0])
            pred = prob > 0.5
            gt = label == 2
            tp += int((pred & gt & valid).sum().item())
            fp += int((pred & (~gt) & valid).sum().item())
            fn += int(((~pred) & gt & valid).sum().item())
    return binary_micro_iou(tp, fp, fn)


def _teacher_validation_metric(model, loader, device):
    model.eval()
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            valid = batch["valid"].to(device).bool()
            pred = torch.softmax(model(image), dim=1).argmax(dim=1)
            for g in range(3):
                for p in range(3):
                    confusion[g, p] += int(((label == g) & (pred == p) & valid).sum().item())
    return teacher_three_class_macro_iou(confusion)


def train_frozen_run(
    cfg,
    *,
    project_root,
    train_loader,
    val_loader,
    implementation_sha,
    frozen_implementation_sha,
    overall_g4_status,
    device,
    teacher_model=None,
    train_generator=None,
):
    """Execute exactly one frozen 40-epoch teacher or student run.

    This function must not be called while G4 is OPEN. Authorization is checked
    before model construction, optimizer creation, output-directory creation, or
    any batch iteration.
    """
    assert_training_authorized(
        overall_g4_status=overall_g4_status,
        implementation_sha=implementation_sha,
        frozen_implementation_sha=frozen_implementation_sha,
    )
    validate_run_config(cfg)

    seed = int(cfg["seed"])
    seed_everything(seed)
    if train_generator is None:
        train_generator = make_train_generator(seed)

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    kd_run = cfg["role"] == "student" and cfg["condition_id"] not in {"B0", "A5"}
    if kd_run:
        if teacher_model is None:
            raise RuntimeError("Frozen KD student run requires teacher_model")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    run_dir, ckpt_dir = create_training_run_dirs(project_root, cfg["run_id"])

    best_metric = None
    best_epoch = None

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            valid = batch["valid"].to(device).bool()

            optimizer.zero_grad(set_to_none=True)
            logits = model(image)

            if cfg["role"] == "teacher":
                loss = teacher_loss(logits, label, valid)
            else:
                condition = cfg["condition_id"]
                teacher_logits = None
                kd_valid = None
                if kd_run:
                    if "teacher_image" not in batch or "kd_valid" not in batch:
                        raise RuntimeError(
                            "KD student batch must contain aligned teacher_image and kd_valid"
                        )
                    teacher_image = batch["teacher_image"].to(device)
                    kd_valid = batch["kd_valid"].to(device).bool()
                    with torch.no_grad():
                        teacher_logits = teacher_model(teacher_image)
                loss = student_loss(
                    condition,
                    logits,
                    label,
                    valid,
                    teacher_logits=teacher_logits,
                    beta=0.3,
                    kd_valid=kd_valid,
                )

            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()

        if cfg["role"] == "teacher":
            val_metric = _teacher_validation_metric(model, val_loader, device)
        else:
            val_metric = _student_validation_metric(
                model, val_loader, cfg["condition_id"], device
            )

        scheduler.step(val_metric)

        if strict_improvement(val_metric, best_metric):
            best_metric = float(val_metric)
            best_epoch = int(epoch)
            _atomic_torch_save(
                {
                    "epoch": best_epoch,
                    "metric": best_metric,
                    "model_state_dict": model.state_dict(),
                    "implementation_sha": str(implementation_sha),
                    "run_id": cfg["run_id"],
                    "run_seed": seed,
                },
                ckpt_dir / "best_validation.pt",
            )

        recovery = make_recovery_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            train_generator=train_generator,
            extra={
                "best_epoch": best_epoch,
                "run_id": cfg["run_id"],
                "run_seed": seed,
                "implementation_sha": str(implementation_sha),
            },
        )
        recovery_path = ckpt_dir / recovery_filename(epoch)
        if recovery_path.exists():
            raise FileExistsError(f"Recovery checkpoint already exists: {recovery_path}")
        torch.save(recovery, recovery_path)

    return {
        "model": model,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "run_dir": run_dir,
        "checkpoint_dir": ckpt_dir,
    }
