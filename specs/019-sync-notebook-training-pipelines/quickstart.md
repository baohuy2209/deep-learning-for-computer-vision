# Quickstart & Verification Guide: Unified Notebook Training Pipelines

**Feature**: `019-sync-notebook-training-pipelines`  
**Date**: 2026-09-24  
**Status**: Ready for Implementation  

---

## 1. Overview & Verification Goal

This guide provides researchers with end-to-end verification steps to confirm that:
1. `notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb`
2. `notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb`
3. `notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb`

all share the exact same training engine architecture, parameter defaults, decoupled dual monitoring behavior, and evaluation schemas.

---

## 2. Automated Static Verification & Code Parity Audit

Run the automated inspection script to verify AST syntax and structural parity across the three notebooks:

```bash
python scratch/verify_notebook_training_parity.py
```

### Expected Output:
- `main_model (6).ipynb`:
  - `DRTrainer` implements `self.lr_scheduler.step(val_qwk)` for plateau schedulers.
  - Cell 70 passes all parameters: `epochs`, `model_name`, `patience_early_stopping`, `patience_reduce_lr`, `reduce_lr_factor`, `min_lr`, `resume_checkpoint_path`.
- `main_model_head_ablation (3).ipynb`:
  - `DRHeadTrainer` uses epoch-level QWK dynamic warmup (`qwk_warmup_epochs=15`).
  - Decoupled counters `epochs_no_improve_es` (on `val_qwk`) and `epochs_no_improve_lr` (on `val_loss`).
  - Checkpoint schema matches `contracts/checkpoint_schema.json`.
- `documents/`:
  - `README.md` maps active champion notebooks.
  - `07_technical_report_and_experimental_appendix.md` reflects Table 2.1 hyperparameters.

---

## 3. Smoke Test / Dry-Run Execution (Colab or Local GPU)

To execute a 1-epoch dry run in any of the notebooks without consuming GPU compute hours:
1. In the configuration cell, set `SMOKE_TEST = True` (or `EPOCHS = 1`, `LODO_EPOCHS = 1`, `ABLATION_EPOCHS = 1`).
2. Run all cells sequentially.
3. Verify that:
   - Checkpoint files `_best.pth` and `_last.pth` are created.
   - The checkpoint payload matches `contracts/checkpoint_schema.json`.
   - Logging outputs the standard epoch line:
     `[Epoch 01/01 - XX.Xs] Train Loss: X.XXXX | Train QWK: X.XXXX || Val Loss: X.XXXX | Val QWK: X.XXXX | Val Within-1: XX.X% | Val Ref-Sens: XX.X% | Val Ref-Sens@Spec95: XX.X% | ... | LR: X.XXe-XX`
   - Evaluation restores `_best.pth` and computes held-out test metrics and screening calibration.
