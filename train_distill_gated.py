import os
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.sen1floods11 import Sen1Floods11Dataset
from src.models.unet import UNet


def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def compute_iou(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    if union == 0:
        return torch.tensor(1.0, device=logits.device)
    return intersection / union


def distillation_loss(student_logits, teacher_logits, targets, beta=0.5, gate_threshold=1.0):
    bce = F.binary_cross_entropy_with_logits(student_logits, targets)
    dice = dice_loss_from_logits(student_logits, targets)
    task_loss = bce + dice

    teacher_probs = torch.sigmoid(teacher_logits)
    student_probs = torch.sigmoid(student_logits)

    if gate_threshold < 1.0:
        # Confidence gate: only apply MSE where teacher confidence > threshold
        mask = (teacher_probs > gate_threshold) | (teacher_probs < (1 - gate_threshold))
        mask = mask.float()
        distill = F.mse_loss(student_probs * mask, teacher_probs * mask, reduction='sum') / (mask.sum() + 1e-8)
    else:
        distill = F.mse_loss(student_probs, teacher_probs)

    return task_loss + beta * distill


def main():
    config_path = "configs/distill_config_gated.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    base = cfg.get("base", 16)
    beta = cfg.get("beta", 0.5)
    gate_threshold = cfg.get("gate_threshold", 1.0)

    data_root = "/content/drive/MyDrive/sar-edge-flood-ai/datasets/Sen1Floods11/v1.2"
    outputs_root = "/content/drive/MyDrive/sar-edge-flood-ai/outputs"
    out_dir = os.path.join(outputs_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    teacher_path = "/content/drive/MyDrive/sar-edge-flood-ai/outputs/teacher_unet_base64/teacher_unet_best_0.5532_epoch30.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Run name:", run_name)
    print("Beta:", beta, " Gate threshold:", gate_threshold)
    print("Device:", device)
    print("Data root:", data_root)
    print("Output dir:", out_dir)
    print("Teacher checkpoint:", teacher_path)
    print("=" * 60)

    train_ds = Sen1Floods11Dataset(data_root, split="train")
    val_ds = Sen1Floods11Dataset(data_root, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    checkpoint = torch.load(teacher_path, map_location=device)
    teacher = UNet(in_channels=2, out_channels=1, base=64).to(device)
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    student = UNet(in_channels=2, out_channels=1, base=base).to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_iou = 0.0
    metrics = []

    for epoch in range(1, epochs + 1):
        student.train()
        train_loss = 0.0
        train_iou = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = student(x)

            loss = distillation_loss(student_logits, teacher_logits, y,
                                     beta=beta, gate_threshold=gate_threshold)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += compute_iou(student_logits, y).item()

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        student.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).float()

                teacher_logits = teacher(x)
                student_logits = student(x)

                loss = distillation_loss(student_logits, teacher_logits, y,
                                         beta=beta, gate_threshold=gate_threshold)

                val_loss += loss.item()
                val_iou += compute_iou(student_logits, y).item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_iou)

        metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} | val_iou={val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt = {
                "epoch": epoch,
                "model_state": student.state_dict(),
                "val_iou": val_iou,
            }
            torch.save(ckpt, os.path.join(out_dir, "student_distilled_best.pt"))
            torch.save(
                ckpt,
                os.path.join(out_dir, f"student_distilled_best_epoch{epoch}_iou{val_iou:.4f}.pt")
            )

    df = pd.DataFrame(metrics)
    csv_path = os.path.join(out_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    print("Saved metrics CSV ->", csv_path)
    print("Best val IoU:", best_iou)


if __name__ == "__main__":
    main()
