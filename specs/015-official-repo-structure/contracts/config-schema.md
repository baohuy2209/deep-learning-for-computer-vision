# Configuration Schema Contract: Official ICCPR Repository

**Feature**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/015-official-repo-structure/spec.md) | **Date**: 2026-09-19  
**Paper**: *Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression* (ICCPR Conference)

---

## 1. Experiment Configuration Specification (`configs/train/in_domain.yaml`)

```yaml
experiment:
  name: "multiscale_mil_in_domain_champion"
  seed: 42
  device: "cuda:0"
  output_dir: "results/in_domain"

data:
  raw_dir: "data/raw"
  native_cropped_dir: "data/raw/hybrid/native_cropped"
  train_csv: "data/raw/hybrid/train_balanced_metadata.csv"
  val_csv: "data/raw/hybrid/val_raw_metadata.csv"
  test_csv: "data/raw/hybrid/test_raw_metadata.csv"
  num_workers: 4
  pin_memory: true

model:
  architecture: "MultiScaleMILNet"
  global_branch:
    backbone: "efficientnet_b0"
    pretrained: true
    img_size: 512
    preprocessing: "ben_graham_green"  # options: 'ben_graham_green', 'clahe_green', 'raw_green'
  local_branch:
    backbone: "efficientnet_b0"
    pretrained: true
    patch_size: 224
    num_patches: 4
    patch_selection: "saliency_topk"   # options: 'saliency_topk', 'grid'
  attention:
    type: "cbam"                       # options: 'cbam', 'msag', 'vanilla'
    reduction_ratio: 16
  head:
    type: "consistent_ordinal"         # options: 'consistent_ordinal', 'coral', 'softmax', 'regression'
    num_classes: 5

training:
  epochs: 35
  batch_size: 16
  mixed_precision: true                # PyTorch AMP (torch.cuda.amp)
  optimizer:
    name: "AdamW"
    lr: 1.0e-4
    weight_decay: 1.0e-2
  scheduler:
    name: "CosineAnnealingLR"
    T_max: 35
    eta_min: 1.0e-6
  early_stopping:
    monitor: "val_qwk"
    mode: "max"
    patience: 7
    min_delta: 0.002
  checkpointing:
    save_top_k: 1
    save_last: true
    monitor: "val_qwk"
```

---

## 2. LODO Benchmark Configuration Specification (`configs/train/lodo_benchmark.yaml`)

```yaml
experiment:
  name: "lodo_6fold_benchmark"
  seed: 42
  device: "cuda:0"

lodo:
  folds_dir: "data/processed/lodo_folds"
  summary_csv: "data/processed/lodo_folds/lodo_splits_summary.csv"
  folds:
    0: { key: "APTOS", slug: "aptos" }
    1: { key: "IDRiD", slug: "idrid" }
    2: { key: "Messidor-2", slug: "messidor2" }
    3: { key: "DDR", slug: "ddr" }
    4: { key: "EyePACS", slug: "eyepacs" }
    5: { key: "DeepDRiD", slug: "deepdrid" }

model:
  # Inherits champion model architecture
  global_branch:
    backbone: "efficientnet_b0"
    img_size: 512
    preprocessing: "ben_graham_green"
  local_branch:
    patch_size: 224
    num_patches: 4
  attention:
    type: "cbam"
  head:
    type: "consistent_ordinal"

training:
  epochs: 30
  batch_size: 16
  mixed_precision: true
  early_stopping:
    monitor: "val_qwk"
    patience: 6
```
