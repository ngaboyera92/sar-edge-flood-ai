"""Frozen Semantic Shift model construction."""
from src.models.unet import UNet


def build_model(cfg):
    """Construct the architecture declared by a validated immutable run config."""
    a = cfg["architecture"]
    if a["family"] != "U-Net":
        raise ValueError(f"Unsupported architecture family: {a['family']}")
    out_channels = 3 if a["output"] == "3_logits_N_P_F" else 1
    return UNet(
        in_channels=len(a["input_channels"]),
        out_channels=out_channels,
        base=int(a["base"]),
    )
