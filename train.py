
import os
import yaml
import torch
import numpy as np
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
        return torch.tensor(1.0)

    return intersection / union


def main():

    config_path = "configs/teacher_base32.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    base = cfg["base"]

    data_root = "/content/drive/MyDrive/sar-edge-flood-ai/datasets/Sen1Floods11/v1.2"
    outputs_root = "/content/drive/MyDrive/sar-edge-flood-ai/outputs"
    out_dir = os.path.join(outputs_root, run_name)

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("Run name:", run_name)
    print("Device:", device)
    print("Data root:", data_root)
    print("Output dir:", out_dir)
    print("="*60)

    train_ds = Sen1Floods11Dataset(data_root, split="train")
    val_ds = Sen1Floods11Dataset(data_root, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=2, out_channels=1, base=base).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_iou = 0.0
    metrics = []

    for epoch in range(1, epochs+1):

        model.train()
        train_loss = 0
        train_iou = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()

            logits = model(x)

            bce = F.binary_cross_entropy_with_logits(logits, y)
            dice = dice_loss_from_logits(logits, y)

            loss = bce + dice
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_iou += compute_iou(logits, y).item()

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        model.eval()

        val_loss = 0
        val_iou = 0

        with torch.no_grad():

            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).float()

                logits = model(x)

                bce = F.binary_cross_entropy_with_logits(logits, y)
                dice = dice_loss_from_logits(logits, y)

                loss = bce + dice

                val_loss += loss.item()
                val_iou += compute_iou(logits, y).item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_iou)

        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f}"
        )

        metrics.append({
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
        })

        if val_iou > best_iou:

            best_iou = val_iou

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_iou": val_iou,
            }

            torch.save(
                ckpt,
                os.path.join(out_dir, "teacher_unet_best.pt")
            )

            torch.save(
                ckpt,
                os.path.join(out_dir, f"teacher_unet_best_epoch{epoch}_iou{val_iou:.4f}.pt")
            )

    df = pd.DataFrame(metrics)
    csv_path = os.path.join(out_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\nSaved metrics CSV ->", csv_path)
    print("Best val IoU:", best_iou)


if __name__ == "__main__":
    main()
