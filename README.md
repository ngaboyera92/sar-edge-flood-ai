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
