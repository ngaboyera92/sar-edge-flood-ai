# SAR Edge Flood AI

Lightweight SAR flood segmentation for edge deployment using the Sen1Floods11 dataset.

## Overview

This repository implements **knowledge distillation** to train a compact UNet student model (base=16) from a larger teacher UNet (base=64). The distilled student achieves **0.5671 IoU** – slightly outperforming both the teacher and the baseline student – while being **16× smaller** (3.25 MB vs 51 MB) and **11× faster** on CPU (150 ms vs 1673 ms per image).

## Key Results

| Model | Params | Size (MB) | Val IoU | Test IoU | CPU ms/img |
|-------|--------|-----------|---------|----------|-------------|
| Teacher UNet (base64) | 13.4M | 51.2 | 0.553 | 0.564 | 1673 |
| Student UNet (base16) | 0.84M | 3.25 | 0.558 | 0.564 | 150 |
| **Distilled Student (β=0.3)** | **0.84M** | **3.25** | **0.567** | **0.567** | **150** |
| Distilled Student (gated, β=0.3) | 0.84M | 3.25 | 0.552 | 0.544 | 150 |

## Repository Structure & File Descriptions
sar-edge-flood-ai/
├── train.py # Teacher training script (UNet base=64)
├── train_distill.py # Student distillation (ungated, β=0.3, no gating)
├── train_distill_gated.py # Confidence‑gated distillation (β=0.3, gate=0.95)
├── requirements.txt # Python dependencies
├── configs/
│ ├── teacher_base32.yaml # Teacher config (base=32)
│ ├── teacher_base64.yaml # Teacher config (base=64)
│ ├── student_base16.yaml # Baseline student config (no distillation)
│ └── distill_config_gated.yaml # Gated distillation config (β=0.3, gate=0.95)
└── src/
├── datasets/
│ └── sen1floods11.py # Dataset loader for Sen1Floods11
└── models/
└── unet.py # UNet model (configurable base_filters)


## Colab Setup & Training

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/ngaboyera92/sar-edge-flood-ai.git
%cd sar-edge-flood-ai
!pip install -r requirements.txt

# Train teacher
!python train.py

# Train ungated student
!python train_distill.py

# Train gated student (ablation)
!python train_distill_gated.py

Citation
If you use this code, please cite this repository.
