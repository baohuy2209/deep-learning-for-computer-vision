# Quickstart & Verification Guide: Public Academic Repository Refactor

**Feature Branch**: `017-public-repo-refactor`
**Date**: 2026-09-23
**Status**: Completed

This guide provides fast, runnable commands to verify the refactored package, data integrity, unit tests, and CLI entry points on both CPU workstations and GPU environments.

---

## 1. Environment Setup & Verification

```bash
# 1. Clone & create virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1

# 2. Install package in editable mode with development dependencies
pip install -e ".[dev]"

# 3. Verify clean package import
python -c "import drdg; print('drdg version:', drdg.__version__)"
```

**Expected Outcome**: Imports succeed with zero warnings or errors.

---

## 2. Fast CPU Unit Testing Suite

Run the full CPU test suite verifying patient parsing, zero-leakage partitions, model tensor shapes, ordinal cutpoints, and RDR threshold isolation:

```bash
pytest tests/ -v --durations=10
```

**Expected Outcome**: All tests pass in $< 180$ seconds.

---

## 3. Data Download & Preparation Pipeline

```bash
# 1. Download supported public cohorts (DDR, DeepDRiD, IDRiD, Messidor-2)
python scripts/download_dataset.py --dataset idrid
python scripts/download_dataset.py --dataset idrid-segmentation

# 2. Prepare cohort into native cropped ROI format
python scripts/prepare_dataset.py --dataset idrid

# 3. Build unified master metadata
python scripts/build_master_metadata.py

# 4. Generate deterministic 6-Fold LODO splits with leakage audit
python scripts/generate_lodo_splits.py
```

**Expected Outcome**:
- `data/processed/master_fundus_metadata.csv` is populated with verified columns.
- `data/processed/lodo_folds/` contains 18 fold CSVs + `lodo_splits_summary.csv`.
- Zero patient overlap across partitions is logged.

---

## 4. Model Variant Smoke Tests (Dry-Run on CPU)

Verify that all four architectural variants instantiate and execute forward passes:

```bash
python scripts/train_variant.py --variant v0 --epochs 1 --batch-size 2 --dry-run
python scripts/train_variant.py --variant v1 --epochs 1 --batch-size 2 --dry-run
python scripts/train_variant.py --variant v2 --epochs 1 --batch-size 2 --dry-run
python scripts/train_variant.py --variant v3 --epochs 1 --batch-size 2 --dry-run
```

**Expected Outcome**: Models initialize weights, perform single-batch forward/backward passes, compute QWK losses, and output status logs without crashing.

---

## 5. Controlled Head Ablation Smoke Test

Verify independent training of the 5 classification/ordinal heads on standardized representations:

```bash
python scripts/run_head_ablation.py --head softmax_ce --dry-run
python scripts/run_head_ablation.py --head clm_qwk --dry-run
```

**Expected Outcome**: Outputs structured summary dictionaries conforming to [`result_schemas.md`](contracts/result_schemas.md).

---

## 6. LODO Execution & Evaluation (Single Fold)

Run Fold 0 (APTOS held-out) with automatic checkpointing:

```bash
# Run Fold 0 evaluation for Variant V3
python scripts/run_lodo.py --variant v3 --held-out aptos --batch-size 8 --epochs 30
```

**Expected Outcome**:
- Logs training progress, source-validation QWK, and early stopping.
- Evaluates on 100% natural prevalence of held-out APTOS cohort.
- Serializes `results/lodo/v3_fold0_aptos.json` with partitioned sections `source_validation`, `source_test`, and `target_test`.

---

## 7. Explainability (XAI) Evaluation

Evaluate spatial localization against IDRiD pixel lesion masks:

```bash
python scripts/run_xai.py \
    --checkpoint checkpoints/v3_best.pt \
    --dataset-dir data/raw/idrid_segmentation \
    --results-dir results/xai
```

**Expected Outcome**: Produces `results/xai/metrics_summary.json` containing Pointing Game accuracy, Area-Matched Saliency Recall, and pixel AUROC for both Grad-CAM and MIL attention maps.
