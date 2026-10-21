# CLI Interface Contracts: Public Academic Repository Refactor

**Feature Branch**: `017-public-repo-refactor`
**Date**: 2026-09-23
**Status**: Completed

All entry-point scripts in `scripts/` MUST support standard command-line flags and provide self-documenting `--help` text.

---

## 1. `scripts/download_dataset.py`

**Purpose**: Download official public subsets into `data/raw/<dataset_key>`.

```bash
python scripts/download_dataset.py --dataset <dataset_key> [--data-root <path>]
```

| Argument | Type | Default | Choices / Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | *Required* | `['ddr', 'deepdrid', 'idrid', 'idrid-segmentation', 'messidor2']` |
| `--data-root` | `Path` | `./data/raw` | Destination root directory for raw dataset archives |

*Note*: If `--dataset eyepacs` is requested, the script cleanly exits with an informative error instructing the user to refer to `docs/DATA.md` for Kaggle competition terms and manual acquisition steps.

---

## 2. `scripts/prepare_dataset.py`

**Purpose**: Standardize raw cohorts into native circular cropped ROIs and produce cohort CSVs.

```bash
python scripts/prepare_dataset.py --dataset <dataset_key> [--raw-root <path>] [--output-root <path>]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | *Required* | Cohort key (`aptos`, `ddr`, `deepdrid`, `idrid`, `idrid-segmentation`, `messidor2`, `all`) |
| `--raw-root` | `Path` | `./data/raw` | Input directory containing raw data |
| `--output-root` | `Path` | `./data/processed` | Destination for cropped images and intermediate CSVs |

---

## 3. `scripts/build_master_metadata.py`

**Purpose**: Harmonize all cohort CSVs into the unified master metadata table.

```bash
python scripts/build_master_metadata.py [--processed-root <path>] [--output-csv <path>]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--processed-root` | `Path` | `./data/processed` | Directory containing individual prepared cohort CSVs |
| `--output-csv` | `Path` | `./data/processed/master_fundus_metadata.csv` | Output canonical metadata path |

---

## 4. `scripts/generate_lodo_splits.py`

**Purpose**: Generate 6 deterministic LODO fold CSVs and summary audit statistics.

```bash
python scripts/generate_lodo_splits.py [--metadata-csv <path>] [--splits-dir <path>] [--seed <int>]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--metadata-csv` | `Path` | `./data/processed/master_fundus_metadata.csv` | Master harmonized metadata table |
| `--splits-dir` | `Path` | `./data/processed/lodo_folds` | Output directory for 18 fold CSVs + summary |
| `--seed` | `int` | `42` | Random seed for source-domain validation partitioning |

---

## 5. `scripts/train_variant.py`

**Purpose**: Train a specific architectural variant (V0..V3) on standard in-domain or custom splits.

```bash
python scripts/train_variant.py --variant <v0|v1|v2|v3> [options]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--variant` | `str` | *Required* | Model variant: `v0`, `v1`, `v2`, `v3` |
| `--head-type` | `str` | `softmax_ce` | Output head (`softmax_ce`, `clm_qwk`, etc.) |
| `--epochs` | `int` | `30` | Maximum training epochs |
| `--batch-size` | `int` | `8` | Training batch size |
| `--lr` | `float` | `1e-4` | Backbone learning rate |
| `--output-dir` | `Path` | `./checkpoints` | Checkpoint and log storage directory |
| `--resume` | `Path` | `None` | Path to checkpoint file to resume from |
| `--amp` | `str` | `auto` | Mixed precision mode (`auto`, `fp16`, `bf16`, `none`) |

---

## 6. `scripts/run_lodo.py`

**Purpose**: Execute 6-fold Leave-One-Dataset-Out cross-domain evaluation.

```bash
python scripts/run_lodo.py --variant <v0|v1|v2|v3> [--held-out <target> | --all-folds] [options]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--variant` | `str` | *Required* | Model variant: `v0`, `v1`, `v2`, `v3` |
| `--held-out` | `str` | `None` | Single target: `aptos`, `ddr`, `deepdrid`, `idrid`, `messidor2`, `eyepacs` |
| `--all-folds` | `flag` | `False` | Sequentially execute all 6 LODO folds |
| `--splits-dir` | `Path` | `./data/processed/lodo_folds` | Directory containing pre-generated LODO fold CSVs |
| `--results-dir` | `Path` | `./results/lodo` | Directory for serialized result JSON/CSV outputs |
| `--resume` | `flag` | `False` | Automatically resume from existing fold checkpoints |

---

## 7. `scripts/run_head_ablation.py`

**Purpose**: Execute controlled 5-head benchmark on standardized ResNet-50 features.

```bash
python scripts/run_head_ablation.py [--head <head_name> | --all-heads] [options]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--head` | `str` | `None` | Single head: `softmax_ce`, `softmax_qwk`, `coral`, `corn`, `clm_qwk` |
| `--all-heads` | `flag` | `False` | Sequentially train and evaluate all 5 heads |
| `--results-dir` | `Path` | `./results/head_ablation` | Output directory for ablation metrics |

---

## 8. `scripts/run_xai.py`

**Purpose**: Generate Grad-CAM and MIL attention saliency and evaluate against IDRiD lesion masks.

```bash
python scripts/run_xai.py --checkpoint <path> [--dataset-dir <path>] [--results-dir <path>]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--checkpoint` | `Path` | *Required* | Path to trained model weights (`.pt`) |
| `--dataset-dir` | `Path` | `./data/raw/idrid_segmentation` | Directory containing fundus images and lesion masks |
| `--results-dir` | `Path` | `./results/xai` | Output directory for quantitative metrics and visual overlays |
