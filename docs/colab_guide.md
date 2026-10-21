# Google Colab Execution Guide

All experiments are fully supported on Google Colab (Free T4 or Pro A100 instances).

---

## 1. Mount Google Drive & Clone Code

In a fresh Colab notebook cell:

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/baohuy2209/deep-learning-for-computer-vision.git
%cd deep-learning-for-computer-vision

# Install editable package and dependencies
!pip install -q --upgrade albumentations timm torchmetrics pyyaml
!pip install -e .
```

---

## 2. Using Pre-generated LODO Splits

To avoid downloading and parsing 55,000 raw images, upload `lodo_folds.zip` to Google Drive at:
`/content/drive/MyDrive/lodo_folds.zip`

Unzip directly in Colab:
```python
!unzip -q /content/drive/MyDrive/lodo_folds.zip -d /content/deep-learning-for-computer-vision/data/processed/lodo_folds/
```

---

## 3. Run Experiments via CLI in Colab

```python
# Run LODO fold 0 (APTOS held out)
!python -m dr_multiscale_mil.train_lodo --config configs/train/lodo_benchmark.yaml --fold 0 --device cuda:0

# Generate publication figures
!python -m dr_multiscale_mil.visualize --output-dir results/figures --dpi 300
```
