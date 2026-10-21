# CLI Interface Contracts: Parity Execution Suite

**Feature**: `018-restore-notebook-parity` | **Date**: 2026-09-24

This document defines the Command-Line Interface (CLI) contracts for the reproduction scripts under `scripts/`. These scripts mirror the canonical notebook execution workflow.

---

## 1. `scripts/run_lodo.py`

Mirrors `notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb`.

### Usage
```bash
python scripts/run_lodo.py --variant {v0,v1,v2,v3} [--held-out {aptos,idrid,messidor2,ddr,eyepacs,deepdrid} | --all-folds] [options]
```

### Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `--variant` | `str` | *Required* | Model variant: `v0`, `v1`, `v2`, `v3` |
| `--held-out` | `str` | `None` | Single target cohort to evaluate out-of-distribution |
| `--all-folds` | `flag` | `False` | Sequentially execute all 6 LODO evaluation folds |
| `--splits-dir` | `Path` | `data/processed/lodo_folds` | Directory containing the 18 pre-generated fold CSVs |
| `--results-dir`| `Path` | `results/lodo` | Directory for serialized result JSON/CSV outputs |
| `--checkpoints-dir` | `Path` | `checkpoints` | Directory to save/load model checkpoints |
| `--batch-size` | `int` | `4` | Mini-batch size |
| `--epochs` | `int` | `30` | Maximum training epochs |
| `--lr` | `float` | `1e-4` | Base learning rate |
| `--seed` | `int` | `42` | Random seed |
| `--resume` | `flag` | `False` | Resume from existing fold checkpoints or JSON results |
| `--dry-run` | `flag` | `False` | Execute 1 fast epoch on synthetic data for verification |
| `--device` | `str` | `auto` | Execution device: `auto`, `cuda`, `cpu` |
| `--amp` | `str` | `auto` | Automatic mixed precision: `auto`, `fp16`, `bf16`, `none` |

### Outputs
- Checkpoint: `checkpoints/lodo_{variant}_fold_{fold_idx}_{held_out}_best.pth`
- Fold Metric JSON: `results/lodo/{variant}_fold{fold_idx}_{held_out}.json`
- Aggregate Summary: `results/lodo/{variant}_lodo_summary.csv`

---

## 2. `scripts/run_head_ablation.py`

Mirrors `notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb`.

### Usage
```bash
python scripts/run_head_ablation.py [--head {softmax_ce,softmax_qwk,coral,corn,clm_qwk} | --all-heads] [options]
```

### Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `--head` | `str` | `None` | Specific output head formulation to train |
| `--all-heads` | `flag` | `False` | Sequentially train and evaluate all 5 output heads |
| `--data-dir` | `Path` | `data/raw/hybrid` | Directory containing standard train/val/test metadata |
| `--results-dir`| `Path` | `results/head_ablation` | Output directory for ablation metrics |
| `--checkpoints-dir` | `Path` | `checkpoints_head_ablation` | Directory for head ablation checkpoints |
| `--dry-run` | `flag` | `False` | Fast synthetic verification pass |

---

## 3. `scripts/generate_lodo_splits.py`

Mirrors `notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb`.

### Usage
```bash
python scripts/generate_lodo_splits.py [options]
```

### Arguments
| Argument | Type | Default | Description |
|---|---|---|---|
| `--data-dir` | `Path` | `data/raw` | Base directory containing raw/total cohort CSVs |
| `--native-dir`| `Path` | `data/raw/hybrid/native_cropped` | Directory containing cropped native images |
| `--out-dir` | `Path` | `data/processed/lodo_folds` | Destination directory for the 18 fold CSVs |
| `--seed` | `int` | `42` | Base random seed |

### Output
- Exactly 18 CSV files: `fold_{0-5}_{cohort}_{train,val,test}.csv`
- Summary file: `lodo_splits_summary.csv`
