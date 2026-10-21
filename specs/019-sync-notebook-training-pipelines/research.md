# Research & Decision Record: Synchronizing Notebook Training Pipelines

**Feature**: `019-sync-notebook-training-pipelines`  
**Date**: 2026-09-24  
**Status**: Completed  

---

## 1. Context & Motivation

The research codebase relies on three core Jupyter notebooks under `notebooks/diabetic_retinopathy/pipeline/model/`:
1. `main_model_lodo.ipynb`: The Leave-One-Dataset-Out (LODO) benchmark orchestration notebook. It contains the most mature, hardened, and battle-tested execution of `DRTrainer`, implementing decoupled dual monitoring, dynamic linear QWK loss warmup over early epochs, automatic bfloat16/float16 mixed precision, robust dual-checkpointing (`_best.pth` and `_last.pth`), completion flag protection (`is_completed`), and screening calibration at fixed specificity cutoffs (95%, 90%, 85%, 80%).
2. `main_model (6).ipynb`: The primary full-architecture notebook (Dual-Branch ResNet-50 + NonLocal + CBAM + QuadrantTokens + EfficientNet-B0 Dual-Pool MIL + LayerNorm MLP Fusion + CLM_QWK). While based on `DRTrainer`, it contains minor divergences (e.g. `self.lr_scheduler.step(val_loss)` vs `val_qwk`, missing arguments in `trainer.fit(...)`, unstandardized checkpoint completion handling).
3. `main_model_head_ablation (3).ipynb`: The 5-round output head ablation benchmark notebook (Softmax CE, Softmax QWK, CORAL, CORN, CLM_QWK on ResNet-50). It utilizes `DRHeadTrainer` which has internal discrepancies with `DRTrainer` (step-level warmup vs epoch-level warmup, non-standard metric return signature in `evaluate`, distinct checkpoint dictionary structures).
4. `documents/`: Master documentation suite (`README.md`, `02_pipeline_architecture.md`, `04_factorial_baseline_and_domain_shift_framework.md`, `07_technical_report_and_experimental_appendix.md`), which requires synchronization with the latest notebook names, hyperparameter configurations, and proposed framework architecture.

---

## 2. Research Decisions & Technical Rationale

### Decision 1: Harmonization of `DRTrainer` in `main_model (6).ipynb` with `main_model_lodo.ipynb`

- **Decision**: Update `DRTrainer` and training execution Cell 70 in `main_model (6).ipynb` to achieve exact 1-to-1 operational parity with `main_model_lodo.ipynb`.
- **Key Deltas**:
  1. `self.lr_scheduler.step(val_qwk)`: When an external plateau scheduler is supplied, step it on `val_qwk` instead of `val_loss`, aligning with the project constitution principle that QWK is the primary optimization metric.
  2. Full argument pass in `Cell 70`: Explicitly pass `epochs=EPOCHS`, `model_name=model_tag`, `patience_early_stopping=PATIENCE_EARLY_STOPPING`, `patience_reduce_lr=PATIENCE_REDUCE_LR`, `reduce_lr_factor=REDUCE_LR_FACTOR`, `min_lr=MIN_LR`, and `resume_checkpoint_path=resume_ckpt_path`.
  3. Standardized Checkpoint Payload: Ensure `checkpoint_state` contains `epoch`, `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `scheduler_state_dict`, `val_qwk`, `best_val_qwk`, `best_val_loss`, `history`, `val_metrics`, `epochs_no_improve_es`, and `epochs_no_improve_lr`.
  4. Best checkpoint completion flag: Add `ckpt_meta["is_completed"] = True` and persist final test metrics upon training completion.
- **Rationale**: Eliminates behavioral divergences between full-model single runs and multi-center LODO evaluations, ensuring that identical hyperparameter schedules and early stopping triggers govern both pipelines.
- **Alternatives Considered**: Keeping `main_model (6).ipynb` as-is was rejected because discrepancies in learning rate decay and early stopping counters cause inconsistent performance metrics across the same architecture.

---

### Decision 2: Harmonization of `DRHeadTrainer` in `main_model_head_ablation (3).ipynb` with `DRTrainer`

- **Decision**: Refactor `DRHeadTrainer` in `main_model_head_ablation (3).ipynb` to inherit the exact operational architecture of `DRTrainer` from `main_model_lodo.ipynb`, adapting only the tensor ingestion to single-branch images (`batch["image"]`).
- **Key Deltas**:
  1. Epoch-Level QWK Loss Warmup: Replace step-level warmup with epoch-level linear warmup (`progress = min(1.0, epoch / max(1, self.qwk_warmup_epochs))`), matching `DRTrainer`. This eliminates high intra-epoch variance caused by noisy mini-batch soft confusion matrices.
  2. Decoupled Dual Monitoring: Enforce the exact same decoupling where `_best.pth` and early stopping (`epochs_no_improve_es`) are strictly driven by `val_qwk`, while `ReduceLROnPlateau` (`epochs_no_improve_lr`) tracks `val_loss`.
  3. Metric Function Standardization: Use `calculate_dr_metrics` for training epoch summary rather than raw `cohen_kappa_score` alone, ensuring uniform metric logging.
  4. Unified Evaluation Tuple: Align `evaluate()` return signature with `DRTrainer` (`metrics, y_true, y_pred`), embedding `probs` and `Monotonicity_Violations` directly within the `metrics` dictionary.
  5. Checkpoint Keys Parity: Save `_best(2).pth` and `_last(2).pth` with identical dictionary keys, including `scaler_state_dict` and `scheduler_state_dict`.
- **Rationale**: Isolates Hypothesis H3 without confounding effects introduced by different warmup schedules or metric extraction procedures.
- **Alternatives Considered**: Rewriting `main_model_head_ablation (3).ipynb` to take dual-branch inputs was rejected because head ablation is designed on ResNet-50 512x512 to evaluate classification heads independently of local patch MIL.

---

### Decision 3: Preserving Self-Containment and Colab Execution

- **Decision**: All three notebooks must remain 100% self-contained and independently executable in Google Colab / cloud GPU environments.
- **Rationale**: Per project constitution and user directives, notebooks must not become thin wrappers importing from `drdg` or require external local installations that fail in fresh Colab runtimes.
- **Alternatives Considered**: Having notebooks import `DRTrainer` from `drdg.training.trainer` was rejected because it violates user instructions and breaks standalone Colab execution.

---

### Decision 4: Master Technical Documentation Suite Synchronization (`documents/`)

- **Decision**: Update all relevant markdown monographs in `documents/` to reflect:
  1. Current active notebook filenames: `main_model (6).ipynb`, `main_model_head_ablation (3).ipynb`, `main_model_lodo.ipynb`.
  2. Exact hyperparameter configuration table (Table 2.1 in `07_technical_report_and_experimental_appendix.md`): `EPOCHS`, `T_MAX`, `PATIENCE_EARLY_STOPPING`, `PATIENCE_REDUCE_LR`, `REDUCE_LR_FACTOR`, `TARGET_QWK_WEIGHT=0.4`, `QWK_WARMUP_EPOCHS=15`, and `MIN_LR=1e-6`.
  3. Decoupled Dual Monitoring mathematical description and rationale in `02_pipeline_architecture.md` and `07_technical_report_and_experimental_appendix.md`.
  4. Head ablation and LODO evaluation protocol updates in `04_factorial_baseline_and_domain_shift_framework.md`.
  5. Repository map update in `documents/README.md`.
- **Rationale**: Ensures complete provenance and zero discrepancy between academic reporting and the executable codebase.

---

### Decision 5: Modular Library (`src/drdg/`) Parity Verification

- **Decision**: Verify and ensure that `src/drdg/training/trainer.py` and `src/drdg/training/head_trainer.py` maintain exact logic parity with the synchronized notebook implementations.
- **Rationale**: Ensures researchers running the pure-Python modular pipeline via CLI/scripts experience the identical training behavior as those executing the Colab notebooks.
