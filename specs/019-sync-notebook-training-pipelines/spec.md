# Feature Specification: 019-sync-notebook-training-pipelines

**Feature Branch**: `019-sync-notebook-training-pipelines`

**Created**: 2026-09-24

**Status**: Draft

**Input**: User description: "/speckit-specify @[notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb] @[notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb] @[notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb] giúp thực hiện đồng bộ code cho @[notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb] và @[notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb] theo cơ chế train trong @[notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb] để cho thống nhất nhé, và cập nhật lại phần @directory:documents cho chuẩn theo luôn nhé"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Full Architecture Notebook Training Synchronization (`main_model (6).ipynb`) (Priority: P1)

As a machine learning researcher conducting Diabetic Retinopathy (DR) grading experiments, I need `main_model (6).ipynb` to use the exact training orchestration mechanism established in `main_model_lodo.ipynb`, so that training dynamics, learning rate scheduling, mixed-precision configurations, decoupled dual monitoring, and checkpoint state representations are completely unified across single-run and multi-center LODO benchmarks.

**Why this priority**: `main_model (6).ipynb` is the flagship research notebook representing the full proposed Dual-Branch Multi-Scale MIL architecture with the CLM_QWK champion head. Divergences in training logic between this notebook and the LODO benchmark undermine research reproducibility and introduce confounding factors into empirical evaluations.

**Independent Test**: Can be validated by executing a dry-run / smoke-test epoch of `main_model (6).ipynb` and verifying that the `DRTrainer` initialization, training step, metric logging, checkpoint payload, and evaluation procedures match the exact behavior and schema of `main_model_lodo.ipynb`.

**Acceptance Scenarios**:

1. **Given** `main_model (6).ipynb`, **When** initializing `DRTrainer`, **Then** it uses identical bfloat16/float16 automatic mixed precision detection, GPU-side normalization constants, gradient clipping (`GRAD_CLIP=1.0`), and checkpoint directory handling as `main_model_lodo.ipynb`.
2. **Given** a training session in `main_model (6).ipynb`, **When** `fit()` runs across epochs, **Then** it applies decoupled dual monitoring (early stopping strictly on `Validation QWK` via `epochs_no_improve_es`, and learning rate reduction on `Validation Loss` via `epochs_no_improve_lr`), dynamic linear QWK loss warmup over `QWK_WARMUP_EPOCHS`, and steps `lr_scheduler` on `val_qwk` when using an external plateau scheduler.
3. **Given** a completed or interrupted epoch in `main_model (6).ipynb`, **When** saving `_last.pth` and `_best.pth`, **Then** the saved state dictionary contains identical keys (`epoch`, `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `scheduler_state_dict`, `val_qwk`, `best_val_qwk`, `best_val_loss`, `history`, `val_metrics`, `epochs_no_improve_es`, `epochs_no_improve_lr`) enabling seamless resumption.
4. **Given** training execution Cell 70 in `main_model (6).ipynb`, **When** calling `trainer.fit(...)`, **Then** all parameters (`epochs`, `model_name`, `patience_early_stopping`, `patience_reduce_lr`, `reduce_lr_factor`, `min_lr`, `resume_checkpoint_path`) are explicitly passed with unified variable defaults.

---

### User Story 2 - Head Ablation Benchmark Training Alignment (`main_model_head_ablation (3).ipynb`) (Priority: P2)

As a research engineer evaluating output head formulations, I need `main_model_head_ablation (3).ipynb` to adopt the unified training mechanism of `main_model_lodo.ipynb`, so that all 5 output heads (Softmax CE, Softmax QWK, CORAL, CORN, CLM_QWK) are trained under an identical optimization regime, warmup schedule, and evaluation protocol.

**Why this priority**: The 5-round head ablation benchmark isolates Hypothesis H3 (output head superiority). Inconsistencies in warmup formulation (step-level vs epoch-level), metric reporting, or checkpoint schemas between head ablation and full model training compromise the validity of comparative ablation tables.

**Independent Test**: Can be validated by executing a dry-run round of `main_model_head_ablation (3).ipynb` and checking that `DRHeadTrainer` adheres to the exact `DRTrainer` pattern (epoch-level QWK warmup, decoupled monitoring, unified checkpoint schema, standardized epoch logging, and clean VRAM offloading).

**Acceptance Scenarios**:

1. **Given** `main_model_head_ablation (3).ipynb`, **When** defining `DRHeadTrainer`, **Then** its initialization, AMP scaler setup, decoupled early stopping / learning rate reduction, checkpoint state dictionary keys, and logging string format match `main_model_lodo.ipynb`.
2. **Given** training of any of the 5 heads, **When** executing each epoch, **Then** QWK loss weight ($\lambda$) linearly warms up at the epoch level (`progress = min(1.0, epoch / max(1, self.qwk_warmup_epochs))`), matching `main_model_lodo.ipynb`.
3. **Given** the 5-round execution loop in Cell 29, **When** each round completes, **Then** best checkpoints are restored, validation and held-out test evaluations are computed using standardized clinical metrics, post-hoc screening calibration (at 95%, 90%, 85%, 80% specificity) is performed on validation and applied to test, and model memory is cleanly garbage-collected.

---

### User Story 3 - Technical Documentation & Architectural Specifications Alignment (`documents/`) (Priority: P3)

As an academic author and research auditor, I need the complete documentation suite under `documents/` to accurately reflect the synchronized training mechanism, hyperparameter settings, notebook filenames, and architectural diagrams.

**Why this priority**: Discrepancies between technical reports and executable code mislead reviewers, violate research provenance, and undermine academic publication integrity.

**Independent Test**: Can be validated by inspecting `documents/README.md`, `documents/02_pipeline_architecture.md`, `documents/04_factorial_baseline_and_domain_shift_framework.md`, and `documents/07_technical_report_and_experimental_appendix.md` to ensure all notebook filenames, hyperparameter tables, training equations, and architecture references match the synchronized codebase.

**Acceptance Scenarios**:

1. **Given** `documents/README.md`, **When** reviewing the Source Code Directory Map, **Then** it references active champion notebooks (`main_model (6).ipynb`, `main_model_head_ablation (3).ipynb`, `main_model_lodo.ipynb`) with accurate descriptions.
2. **Given** `documents/07_technical_report_and_experimental_appendix.md`, **When** reviewing Section 2 (Hyperparameters & Training Dynamics), **Then** all hyperparameter tables (`EPOCHS`, `T_MAX`, `PATIENCE_EARLY_STOPPING`, `PATIENCE_REDUCE_LR`, `REDUCE_LR_FACTOR`, `TARGET_QWK_WEIGHT`, `QWK_WARMUP_EPOCHS`) and decoupled dual monitoring formulations precisely match the unified implementation.
3. **Given** `documents/02_pipeline_architecture.md`, **When** reviewing Section 4 and Section 5, **Then** training procedures, mixed precision protocols, early stopping rules, and screening calibration methodologies match the notebook implementation.
4. **Given** `documents/04_factorial_baseline_and_domain_shift_framework.md`, **When** reading the Head Ablation and LODO protocols, **Then** descriptions of the 5-round benchmark and 6-fold LODO evaluation faithfully mirror the code.

---

### Edge Cases

- **Mixed-Precision Hardware Diversity**: What happens when running on an environment without `bfloat16` support (e.g. NVIDIA T4 with compute capability 7.5)?
  - The trainer automatically checks `torch.cuda.is_bf16_supported()`; if unsupported, it falls back to `float16` and enables `torch.amp.GradScaler`, preventing numerical overflow while maintaining accelerated execution.
- **Mid-Session Colab Timeout / Disconnection**: How does the system resume when a training run is interrupted?
  - `_last.pth` is updated at the conclusion of every single epoch containing full model, optimizer, scheduler, scaler, metrics history, and decoupled counters (`epochs_no_improve_es`, `epochs_no_improve_lr`), ensuring exact state restoration from epoch $E+1$ without loss of momentum or counter divergence.
- **Checkpoint Completed Flag**: What happens when re-running a notebook where folds or rounds were already completed?
  - The execution loop checks `ckpt_meta.get("is_completed", False)` or existing summary results; if completed, it skips redundant re-training, preserves compute units, and loads saved artifacts for downstream visualization.
- **Single-Branch vs Dual-Branch Input Structure**: How does the unified training logic accommodate single-image inputs in `main_model_head_ablation (3).ipynb` versus dual-branch (`global_img` + `local_patches`) in `main_model (6).ipynb` and `main_model_lodo.ipynb`?
  - `DRTrainer` handles dual-branch inputs while `DRHeadTrainer` handles single-branch inputs with an identical operational skeleton, identical method names, identical checkpoint structures, and identical metric outputs.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST synchronize `DRTrainer` in `main_model (6).ipynb` to match the exact method signatures, AMP configuration, decoupled monitoring, scheduler stepping, and checkpoint keys found in `main_model_lodo.ipynb`.
- **FR-002**: Training execution Cell 70 in `main_model (6).ipynb` MUST pass all configured hyperparameters (`epochs=EPOCHS`, `model_name=model_tag`, `patience_early_stopping=PATIENCE_EARLY_STOPPING`, `patience_reduce_lr=PATIENCE_REDUCE_LR`, `reduce_lr_factor=REDUCE_LR_FACTOR`, `min_lr=MIN_LR`, `resume_checkpoint_path=resume_ckpt_path`) to `fit()`.
- **FR-003**: The evaluation and post-hoc screening calibration in `main_model (6).ipynb` MUST compute and log all clinical screening metrics (QWK, within-1 accuracy, referable sensitivity at 95%, 90%, 85%, 80% specificity, referable AUC, and rank monotonicity violations).
- **FR-004**: `DRHeadTrainer` in `main_model_head_ablation (3).ipynb` MUST be harmonized to follow the identical training mechanism as `DRTrainer` in `main_model_lodo.ipynb`, including epoch-level dynamic QWK loss warmup, decoupled dual monitoring, unified checkpoint schema (`_best.pth` and `_last.pth`), and standardized metric logging.
- **FR-005**: The head ablation orchestration loop in `main_model_head_ablation (3).ipynb` MUST preserve full VRAM cleanup, checkpoint completion protection, and per-head metric export while maintaining exact parity with `DRTrainer`.
- **FR-006**: Both notebooks MUST remain fully self-contained and directly executable on Google Colab / cloud GPU environments without requiring external package installations beyond standard requirements.
- **FR-007**: The documentation suite under `documents/` (`README.md`, `02_pipeline_architecture.md`, `04_factorial_baseline_and_domain_shift_framework.md`, `07_technical_report_and_experimental_appendix.md`) MUST be updated to reflect the active notebooks (`main_model (6).ipynb`, `main_model_head_ablation (3).ipynb`, `main_model_lodo.ipynb`), exact hyperparameter tables, decoupled dual monitoring rationale, and unified screening calibration metrics.
- **FR-008**: The modular library implementation in `src/drdg/` MUST maintain exact logic parity with the synchronized notebook training mechanism.

### Key Entities

- **DRTrainer**: High-performance PyTorch training and evaluation engine managing AMP precision, gradient clipping, decoupled dual monitoring, epoch-level QWK loss warmup, dual checkpointing (`_best.pth` and `_last.pth`), and full resumption capability.
- **DRHeadTrainer**: Specialized variant of `DRTrainer` tailored for single-branch ResNet-50 head ablation, sharing identical method lifecycle, decoupled monitoring, checkpoint schemas, and clinical evaluation reporting.
- **Decoupled Dual Monitoring System**: Optimization control mechanism separating model selection (`_best.pth` and early stopping driven strictly by `Validation QWK`) from learning rate decay (`ReduceLROnPlateau` driven by `Validation Loss`).
- **Checkpoint State Payload**: Standardized Python dictionary saved to disk containing `epoch`, `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `scheduler_state_dict`, `val_qwk`, `best_val_qwk`, `best_val_loss`, `history`, `val_metrics`, `epochs_no_improve_es`, and `epochs_no_improve_lr`.
- **Screening Calibration Suite**: Clinical utility calibration evaluating referable DR sensitivity at fixed clinical specificity thresholds ($95\%, 90\%, 85\%, 80\%$) calibrated on in-domain validation and evaluated on held-out test distributions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% structural and behavioral parity in training orchestration (optimizer setup, learning rate scheduling, decoupled dual monitoring, AMP handling, checkpoint dictionary keys, and metric logging) between `main_model (6).ipynb`, `main_model_head_ablation (3).ipynb`, and `main_model_lodo.ipynb`.
- **SC-002**: All three notebooks pass pre-flight dry-run syntax and cell structure verification without syntax errors, missing variables, or broken dependency chains.
- **SC-003**: 100% of existing unit and regression tests in the test suite pass with zero regressions.
- **SC-004**: All technical documentation monographs in `documents/` (`README.md`, `02_pipeline_architecture.md`, `04_factorial_baseline_and_domain_shift_framework.md`, `07_technical_report_and_experimental_appendix.md`) are verified to be fully synchronized with actual codebase implementations and hyperparameter configurations.

## Assumptions

- The existing training mechanism in `main_model_lodo.ipynb` serves as the authoritative gold standard reference for training orchestration across the research project.
- Notebooks must remain strictly self-contained and independently executable on Google Colab with local cell execution order.
- Changes to notebooks must follow minimal-scope delta-only principles, preserving all existing markdown explanations, cell IDs, and visualization blocks.
- The `documents/` directory contains academic documentation and technical reports that must remain synchronized with the research codebase to ensure publication integrity.
