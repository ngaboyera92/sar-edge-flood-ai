"""Frozen Semantic Shift training engine with Drive-authoritative evidence.

Authorization is checked before model/optimizer/artifact/loader side effects. The
engine implements the frozen 40-epoch training/checkpoint policy, exact required run
leaves, versioned epoch recovery, and epoch-boundary continuation after interruption.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

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
    restore_recovery_payload,
    recovery_filename,
)
from .artifact_protocol import training_run_dir, checkpoint_run_dir
from .run_evidence import (
    initialize_run_evidence,
    append_training_log,
    append_epoch_metrics,
    finalize_run_evidence,
    verify_required_run_leaves,
)


def _atomic_torch_save(obj, path):
    """Atomically replace the within-run mutable best-validation checkpoint."""
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


def _validate_resume_identity(cfg, implementation_sha, run_dir, payload):
    frozen_cfg = json.loads((Path(run_dir) / "run_config.json").read_text(encoding="utf-8"))
    if frozen_cfg != cfg:
        raise RuntimeError("Resume config differs from frozen run_config.json")
    extra = payload.get("extra", {})
    if str(extra.get("run_id")) != str(cfg["run_id"]):
        raise RuntimeError("Recovery run_id mismatch")
    if str(extra.get("implementation_sha")) != str(implementation_sha):
        raise RuntimeError("Recovery implementation SHA mismatch")
    if int(extra.get("run_seed")) != int(cfg["seed"]):
        raise RuntimeError("Recovery run seed mismatch")


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
    immutable_config_sha256,
    input_identities,
    teacher_model=None,
    train_generator=None,
    resume_checkpoint=None,
):
    """Execute or epoch-boundary-resume exactly one frozen teacher/student run."""
    # MUST remain first: no side effect may occur while G4 is open.
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

    best_metric = None
    best_epoch = None
    start_epoch = 1

    if resume_checkpoint is None:
        run_dir, ckpt_dir = initialize_run_evidence(
            project_root,
            cfg,
            implementation_sha=implementation_sha,
            immutable_config_sha256=immutable_config_sha256,
            input_identities=input_identities,
        )
    else:
        run_dir = training_run_dir(project_root, cfg["run_id"])
        ckpt_dir = checkpoint_run_dir(project_root, cfg["run_id"])
        if not run_dir.is_dir() or not ckpt_dir.is_dir():
            raise RuntimeError("Cannot resume: frozen run/checkpoint directories are missing")
        if (run_dir / "completion_status.json").exists():
            raise RuntimeError("Cannot resume a run that already has completion_status.json")
        recovery_path = Path(resume_checkpoint)
        if recovery_path.parent.resolve() != ckpt_dir.resolve():
            raise RuntimeError("Recovery checkpoint is outside the frozen checkpoint directory")
        payload = torch.load(recovery_path, map_location=device)
        _validate_resume_identity(cfg, implementation_sha, run_dir, payload)
        completed_epoch, best_metric, extra = restore_recovery_payload(
            payload, model, optimizer, scheduler, train_generator=train_generator
        )
        start_epoch = int(completed_epoch) + 1
        best_epoch = extra.get("best_epoch")
        append_training_log(run_dir, f"status=RESUMED_FROM_EPOCH_{completed_epoch:02d}")

    try:
        for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
            model.train()
            loss_sum = 0.0
            batch_count = 0
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
                loss_sum += float(loss.detach().cpu().item())
                batch_count += 1

            if batch_count <= 0:
                raise RuntimeError("Training loader produced zero batches")

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

            append_epoch_metrics(
                run_dir,
                {
                    "epoch": epoch,
                    "train_loss_batch_mean": loss_sum / batch_count,
                    "validation_metric": float(val_metric),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                },
            )
            append_training_log(
                run_dir,
                f"epoch={epoch:02d} val_metric={float(val_metric):.12g} best_epoch={best_epoch}",
            )

        append_training_log(run_dir, "status=COMPLETED")
        finalize_run_evidence(
            run_dir,
            ckpt_dir,
            status="COMPLETED",
            best_epoch=best_epoch,
            best_metric=best_metric,
        )
        verify_required_run_leaves(run_dir, ckpt_dir)
    except Exception as exc:
        # Preserve evidence only if initialization succeeded and the completion file does not exist.
        if Path(run_dir).is_dir() and not (Path(run_dir) / "completion_status.json").exists():
            append_training_log(run_dir, f"status=FAILED error={type(exc).__name__}: {exc}")
            finalize_run_evidence(
                run_dir,
                ckpt_dir,
                status="FAILED",
                best_epoch=best_epoch,
                best_metric=best_metric,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise

    return {
        "model": model,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "run_dir": run_dir,
        "checkpoint_dir": ckpt_dir,
    }
