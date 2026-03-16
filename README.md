# SAR Edge Flood AI

Lightweight SAR flood segmentation for edge deployment using the Sen1Floods11 dataset.

## Structure
- train.py : teacher training
- train_distill.py : student distillation training
- configs/ : experiment configs
- src/ : dataset and model code
- requirements.txt : dependencies

## Colab Setup
1. Mount Drive
2. Clone repo
3. Install requirements
4. Run training

Example:

from google.colab import drive
drive.mount('/content/drive')

git clone https://github.com/ngaboyera92/sar-edge-flood-ai.git
cd sar-edge-flood-ai
pip install -r requirements.txt
