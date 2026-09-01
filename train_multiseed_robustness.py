import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.datasets.sen1floods11 import Sen1Floods11Dataset
from src.models.unet import UNet


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def batch_iou(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    if union.item() == 0:
        return torch.tensor(1.0, device=logits.device)
    return intersection / union


def task_plus_kd_loss(student_logits, teacher_logits, targets, beta):
    bce = F.binary_cross_entropy_with_logits(student_logits, targets)
    dice = dice_loss_from_logits(student_logits, targets)
    task_loss = bce + dice
    distill = F.mse_loss(torch.sigmoid(student_logits), torch.sigmoid(teacher_logits))
    return task_loss + beta * distill


def load_matrix(path: Path):
    with path.open() as f:
        return yaml.safe_load(f)


def split_paths(data_root: Path):
    d = data_root / "splits" / "flood_handlabeled"
    return {
        "train": d / "flood_train_data.csv",
        "val": d / "flood_val_data.csv",
        "test": d / "flood_test_data.csv",
    }


def audit_repo(expected_commit=None):
    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    if status:
        raise RuntimeError(f"Working tree is not clean:\n{status}")
    if expected_commit and head != expected_commit:
        raise RuntimeError(f"Commit mismatch: expected {expected_commit}, found {head}")
    return {"commit": head, "status": "clean", "branch": git_output("rev-parse", "--abbrev-ref", "HEAD")}


def audit_data(data_root: Path, protocol):
    paths = split_paths(data_root)
    out = {}
    for split, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing split CSV: {p}")
        rows = len(pd.read_csv(p))
        digest = sha256_file(p)
        expected_rows = int(protocol["expected_split_counts"][split])
        expected_hash = protocol["expected_split_sha256"][split]
        if rows != expected_rows:
            raise RuntimeError(f"{split} row count mismatch: expected {expected_rows}, found {rows}")
        if digest != expected_hash:
            raise RuntimeError(f"{split} SHA-256 mismatch: expected {expected_hash}, found {digest}")
        out[split] = {"path": str(p), "rows": rows, "sha256": digest}

    datasets = {s: Sen1Floods11Dataset(str(data_root), split=s) for s in ("train", "val", "test")}
    for split, ds in datasets.items():
        expected_rows = int(protocol["expected_split_counts"][split])
        if len(ds) != expected_rows:
            raise RuntimeError(f"{split} matched dataset length mismatch: expected {expected_rows}, found {len(ds)}")
    return out, datasets


def audit_teacher(project_root: Path, protocol, device):
    teacher_path = project_root / protocol["teacher_relpath"]
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_path}")
    digest = sha256_file(teacher_path)
    if digest != protocol["teacher_sha256"]:
        raise RuntimeError(f"Teacher SHA-256 mismatch: expected {protocol['teacher_sha256']}, found {digest}")
    checkpoint = torch.load(teacher_path, map_location=device)
    teacher = UNet(in_channels=2, out_channels=1, base=64).to(device)
    teacher.load_state_dict(checkpoint["model_state"], strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher_path, digest, checkpoint, teacher


def build_manifest(project_root, matrix_path, matrix, run, repo, split_info, teacher_path, teacher_hash, teacher_ckpt, device):
    return {
        "experiment_id": matrix["experiment_id"],
        "run_name": run["run_name"],
        "order": int(run["order"]),
        "seed": int(run["seed"]),
        "beta": float(run["beta"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "matrix_path": str(matrix_path),
        "repository": repo,
        "protocol": matrix["protocol"],
        "splits": split_info,
        "teacher": {
            "path": str(teacher_path),
            "sha256": teacher_hash,
            "epoch": int(teacher_ckpt.get("epoch", -1)),
            "val_iou": float(teacher_ckpt.get("val_iou", float("nan"))),
            "strict_load": True,
            "frozen": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def evaluate(model, loader, device, threshold):
    model.eval()
    batch_scores = []
    tp = fp = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).float()
            logits = model(x)
            batch_scores.append(batch_iou(logits, y, threshold).item())
            pred = torch.sigmoid(logits) > threshold
            target = y > 0.5
            tp += int((pred & target).sum().item())
            fp += int((pred & ~target).sum().item())
            fn += int((~pred & target).sum().item())
    micro_union = tp + fp + fn
    micro_iou = 1.0 if micro_union == 0 else tp / micro_union
    return float(np.mean(batch_scores)), float(micro_iou), {"tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="experiments/fw02a_multiseed_matrix.yaml")
    ap.add_argument("--project-root", required=True, help="Authoritative Drive project root containing datasets/ and outputs/")
    ap.add_argument("--run-name", help="Exact run_name from the predeclared matrix")
    ap.add_argument("--expected-commit", default=None, help="Freeze all runs to one audited experiment commit")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = load_matrix(matrix_path)
    protocol = matrix["protocol"]
    project_root = Path(args.project_root).resolve()
    data_root = project_root / "datasets" / "Sen1Floods11" / "v1.2"
    outputs_root = project_root / "outputs"

    repo = audit_repo(args.expected_commit)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this controlled Stage A experiment.")
    device = torch.device("cuda")
    split_info, datasets = audit_data(data_root, protocol)
    teacher_path, teacher_hash, teacher_ckpt, teacher = audit_teacher(project_root, protocol, device)

    parent = outputs_root / matrix["parent_output"]
    existing_runs = [r["run_name"] for r in matrix["runs"] if (parent / r["run_name"]).exists()]

    print("STATUS: READY TO START")
    print(json.dumps({
        "repository": repo,
        "split_counts": {k: v["rows"] for k, v in split_info.items()},
        "split_sha256": {k: v["sha256"] for k, v in split_info.items()},
        "teacher_sha256": teacher_hash,
        "teacher_epoch": teacher_ckpt.get("epoch"),
        "teacher_val_iou": teacher_ckpt.get("val_iou"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "parent_output": str(parent),
        "predeclared_runs": len(matrix["runs"]),
        "existing_predeclared_run_folders": existing_runs,
    }, indent=2))

    if args.preflight_only:
        return
    if not args.run_name:
        raise RuntimeError("--run-name is required unless --preflight-only is used")

    matches = [r for r in matrix["runs"] if r["run_name"] == args.run_name]
    if len(matches) != 1:
        raise RuntimeError(f"run_name must match exactly one predeclared run; got {args.run_name}")
    run = matches[0]

    epochs = int(protocol["epochs"])
    batch_size = int(protocol["batch_size"])
    lr = float(protocol["lr"])
    base = int(protocol["base"])
    threshold = float(protocol["threshold"])
    beta = float(run["beta"])
    seed = int(run["seed"])
    if beta not in (0.0, 0.3):
        raise RuntimeError(f"Unexpected beta {beta}")

    run_dir = parent / run["run_name"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to repeat or overwrite existing run directory: {run_dir}")
    parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=False, exist_ok=False)

    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=2,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=2,
                            worker_init_fn=seed_worker)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=2,
                             worker_init_fn=seed_worker)

    manifest = build_manifest(project_root, matrix_path, matrix, run, repo, split_info,
                              teacher_path, teacher_hash, teacher_ckpt, device)
    with (run_dir / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    with (run_dir / "run_config.yaml").open("w") as f:
        yaml.safe_dump({"run": run, "protocol": protocol}, f, sort_keys=False)

    student = UNet(in_channels=2, out_channels=1, base=base).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_iou = -1.0
    best_epoch = None
    best_path = run_dir / "best_validation_checkpoint.pt"
    metrics = []
    log_path = run_dir / "training.log"

    with log_path.open("w", buffering=1) as logf:
        def log(msg):
            print(msg, flush=True)
            logf.write(msg + "\n")

        log(f"RUN START {datetime.now(timezone.utc).isoformat()}")
        log(f"run_name={run['run_name']} order={run['order']} seed={seed} beta={beta}")
        log(f"repo_commit={repo['commit']} device={torch.cuda.get_device_name(0)}")

        for epoch in range(1, epochs + 1):
            student.train()
            train_loss = 0.0
            train_iou = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device).float()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_logits = teacher(x)
                student_logits = student(x)
                loss = task_plus_kd_loss(student_logits, teacher_logits, y, beta)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_iou += batch_iou(student_logits, y, threshold).item()
            train_loss /= len(train_loader)
            train_iou /= len(train_loader)

            student.eval()
            val_loss = 0.0
            val_iou = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device).float()
                    teacher_logits = teacher(x)
                    student_logits = student(x)
                    loss = task_plus_kd_loss(student_logits, teacher_logits, y, beta)
                    val_loss += loss.item()
                    val_iou += batch_iou(student_logits, y, threshold).item()
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            scheduler.step(val_iou)
            lr_now = optimizer.param_groups[0]["lr"]

            metrics.append({
                "epoch": epoch, "lr": lr_now, "train_loss": train_loss, "train_iou": train_iou,
                "val_loss": val_loss, "val_iou": val_iou,
            })
            log(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.6f} | train_iou={train_iou:.6f} | val_loss={val_loss:.6f} | val_iou={val_iou:.6f} | lr={lr_now:.8g}")

            if val_iou > best_iou:
                best_iou = val_iou
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state": student.state_dict(),
                    "val_iou": val_iou,
                    "seed": seed,
                    "beta": beta,
                    "run_name": run["run_name"],
                    "repo_commit": repo["commit"],
                }, best_path)

        pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)

        best_ckpt = torch.load(best_path, map_location=device)
        final_model = UNet(in_channels=2, out_channels=1, base=base).to(device)
        final_model.load_state_dict(best_ckpt["model_state"], strict=True)
        test_batch_mean_iou, test_micro_iou, counts = evaluate(final_model, test_loader, device, threshold)
        ckpt_hash = sha256_file(best_path)

        summary = {
            "experiment_id": matrix["experiment_id"],
            "run_name": run["run_name"],
            "order": int(run["order"]),
            "seed": seed,
            "beta": beta,
            "best_epoch": int(best_epoch),
            "best_validation_iou_batch_mean": float(best_iou),
            "test_iou_batch_mean": test_batch_mean_iou,
            "test_iou_micro": test_micro_iou,
            "test_counts": counts,
            "threshold": threshold,
            "checkpoint": str(best_path),
            "checkpoint_sha256": ckpt_hash,
            "repository_commit": repo["commit"],
            "teacher_sha256": teacher_hash,
        }
        with (run_dir / "run_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        log("RUN COMPLETE")
        log(json.dumps(summary, indent=2))

    with (run_dir / "SHA256SUMS.txt").open("w") as f:
        for name in ("best_validation_checkpoint.pt", "metrics.csv", "run_manifest.json", "run_summary.json", "run_config.yaml", "training.log"):
            p = run_dir / name
            f.write(f"{sha256_file(p)}  {name}\n")


if __name__ == "__main__":
    main()
