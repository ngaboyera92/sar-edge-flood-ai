"""Frozen G4-04 optimizer, scheduler, checkpoint, and resume primitives.

This module intentionally does not contain a training loop. Training remains gated
until overall G4 is explicitly CLOSED/FROZEN and the frozen implementation SHA is
bound into the run manifest.
"""
from copy import deepcopy
import torch

from .determinism import capture_recovery_rng, restore_recovery_rng


def assert_training_authorized(overall_g4_status, implementation_sha, frozen_implementation_sha):
    if str(overall_g4_status) != "CLOSED/FROZEN":
        raise RuntimeError("Training prohibited: overall G4 is not CLOSED/FROZEN")
    if not implementation_sha or implementation_sha != frozen_implementation_sha:
        raise RuntimeError("Training prohibited: implementation SHA is not the frozen SHA")
    return True


def build_optimizer(model, cfg):
    o = cfg["optimizer"]
    if o["name"] != "Adam":
        raise ValueError("Frozen optimizer must be Adam")
    return torch.optim.Adam(
        model.parameters(),
        lr=float(o["lr"]),
        betas=tuple(float(x) for x in o["betas"]),
        eps=float(o["eps"]),
        weight_decay=float(o["weight_decay"]),
        amsgrad=bool(o["amsgrad"]),
    )


def build_scheduler(optimizer, cfg):
    s = cfg["scheduler"]
    if s["name"] != "ReduceLROnPlateau":
        raise ValueError("Frozen scheduler must be ReduceLROnPlateau")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=s["mode"],
        factor=float(s["factor"]),
        patience=int(s["patience"]),
        threshold=float(s["threshold"]),
        threshold_mode=s["threshold_mode"],
        cooldown=int(s["cooldown"]),
        min_lr=float(s["min_lr"]),
        eps=float(s["eps"]),
    )


def strict_improvement(metric, best_metric):
    if best_metric is None:
        return True
    return float(metric) > float(best_metric)


def binary_micro_iou(tp, fp, fn):
    denom = int(tp) + int(fp) + int(fn)
    if denom <= 0:
        raise RuntimeError("Undefined binary micro IoU: zero union")
    return float(tp) / float(denom)


def teacher_three_class_macro_iou(confusion):
    c = torch.as_tensor(confusion, dtype=torch.int64)
    if tuple(c.shape) != (3, 3):
        raise ValueError("Teacher confusion matrix must be 3x3")
    intersections = torch.diag(c)
    gt = c.sum(dim=1)
    pred = c.sum(dim=0)
    unions = gt + pred - intersections
    if bool((unions <= 0).any()):
        raise RuntimeError("Teacher validation STOP: at least one class union is zero")
    iou = intersections.to(torch.float64) / unions.to(torch.float64)
    return float(iou.mean().item())


def make_recovery_payload(model, optimizer, scheduler, epoch, best_metric, train_generator=None, extra=None):
    return {
        "epoch": int(epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "model_state_dict": deepcopy(model.state_dict()),
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": deepcopy(scheduler.state_dict()),
        "rng_state": capture_recovery_rng(train_generator),
        "extra": {} if extra is None else deepcopy(extra),
    }


def restore_recovery_payload(payload, model, optimizer, scheduler, train_generator=None):
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    restore_recovery_rng(payload["rng_state"], train_generator)
    return int(payload["epoch"]), payload["best_metric"], deepcopy(payload.get("extra", {}))


def recovery_filename(epoch):
    e = int(epoch)
    if e < 0:
        raise ValueError("epoch must be non-negative")
    return f"recovery_epoch_{e:02d}.pt"
