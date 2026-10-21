# Feature Specification: Full LODO Benchmark Strictly Aligned with `main_model (6).ipynb`

**Feature Branch**: `012-lodo-architecture-ablation`  
**Feature Directory**: `specs/012-lodo-architecture-ablation`  
**Created**: 2026-09-15  
**Status**: Ready for Implementation  
**Input**: User request: "áp dụng logic từ main_model (6).ipynb cho thí nghiệm main_model_lodo.ipynb đi, đối với lodo thì đọc thêm dữ liệu csv cho fold lodo nữa thôi, thì cái này tui sẽ tải lên cố định với path /content/lodo_folds/{ten file}, bạn cập nhật lại, hoàn thiện lại main_model_lodo.ipynb cho đúng chuẩn thí nghiệm đi"

---

## Executive Summary & Core Principle

The benchmark notebook [`notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb) must strictly adopt the proven, battle-tested pipeline logic of [`main_model (6).ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model%20(6).ipynb) without introducing divergent abstractions, redundant split functions, or synthetic fallbacks.

The **ONLY** difference between `main_model (6).ipynb` and `main_model_lodo.ipynb` is:
Instead of training once on a single train/val/test split from `train_balanced_metadata.csv`, `main_model_lodo.ipynb` iterates over the 6 Leave-One-Dataset-Out (LODO) folds, where for each fold it ingests pre-generated CSV splits from the fixed path:
`/content/lodo_folds/{filename}`  
(e.g., `fold_{fold_idx}_{held_out_key}_train.csv`, `fold_{fold_idx}_{held_out_key}_val.csv`, `fold_{fold_idx}_{held_out_key}_test.csv`).

All other components:
1. Automated image archive unpacking (`native_cropped.zip` $\to$ `/content/dataset_local`).
2. Directory paths (`DATA_ROOT = '/content/dataset_local'`, `image_root_dir = f"{DATA_ROOT}/native_cropped"`).
3. Preprocessing (`apply_green_channel_ablation`, `get_global_augmentations`).
4. Dual-Branch Dataset (`FundusDualBranchDataset` resolving `img_path = f"{self.img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"` and caching `.npz`).
5. FOV mask & sliding-window anchor extraction with `trainer_compatible_collate_fn`.
6. Deep Neural Network Architecture (`GlobalBranch`, `DualBranchMILModel`, `DRFullModel`, `build_model`).
7. Training Engine (`DRTrainer` with mixed precision, CosineAnnealing, EarlyStopping, ReduceLROnPlateau).
8. Evaluation metrics (`calculate_dr_metrics`, `calculate_referable_sensitivity_at_specificity`, `extract_rank_probs`).

remain 100% identical and faithful to `main_model (6).ipynb`.

---

## User Scenarios & Prioritized User Stories

### User Story 1 - Exact Architectural & Pipeline Parity with `main_model (6).ipynb` (Priority: P1) 🎯 MVP
As an ML researcher, I need `main_model_lodo.ipynb` to share the exact same classes, functions, and logic as `main_model (6).ipynb` (including `FundusDualBranchDataset`, `trainer_compatible_collate_fn`, `DualBranchMILModel`, `DRFullModel`, `DRTrainer`), with parametric controls in `DualBranchMILModel` (`use_local_branch`, `use_nonlocal`, `use_quadrant_tokens`) so that any ablation variant ($V_0 \dots V_3$) runs with identical numerical behavior.

**Acceptance Scenarios**:
1. `FundusDualBranchDataset` initializes with `csv_path` or `metadata`, loads images from `image_root_dir = f"{DATA_ROOT}/native_cropped"`, and resolves `f"{self.img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"`.
2. `trainer_compatible_collate_fn` handles stochastic sub-sampling for train mode and dense sweeps for eval/test mode.
3. `build_model()` instantiates `DRFullModel` with exact weights, dimensions ($512\text{-dim}$ fusion), and dispatchable heads (`CLM_QWK` or `Softmax_CE`).

---

### User Story 2 - Fixed LODO Fold CSV Ingestion at `/content/lodo_folds` (Priority: P1) 🎯 MVP
As a researcher running on Google Colab, I will upload the pre-generated fold CSV files to `/content/lodo_folds/{filename}`. I need `main_model_lodo.ipynb` to load these CSVs directly for each fold without on-the-fly balancing, candidate fallbacks, or redundant splitting helper functions, while providing a local fallback to `data/processed/lodo_folds` if run locally.

**Acceptance Scenarios**:
1. Configures `LODO_FOLDS_DIR = '/content/lodo_folds'` if exists, else fallback to `data/processed/lodo_folds`.
2. For fold `fold_idx` holding out `held_out_key`, directly loads:
   - `train_csv = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_train.csv")`
   - `val_csv   = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_val.csv")`
   - `test_csv  = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_test.csv")`
3. Instantiates `train_dataset = FundusDualBranchDataset(csv_path=train_csv, img_root=image_root_dir, split="train", ...)` and corresponding `val_dataset` and `test_dataset`.

---

### User Story 3 - Robust Multi-Fold Orchestration Loop & Checkpoint Resumption (Priority: P1) 🎯 MVP
As a researcher training across 6 cohorts $\times$ 100 epochs, I need the LODO benchmark loop to:
1. Automatically resume from `_last.pth` if a run was interrupted mid-epoch.
2. Skip completed folds if `is_completed=True` in checkpoint metadata.
3. Ingest existing champion results ($V_3$) from `lodo_summary.json` without re-running if `"run": False`.
4. Compute and log `in_domain_val_qwk`, `ood_test_qwk`, and $\Delta_{\text{LODO}} = \text{val\_qwk} - \text{ood\_qwk}$ per fold.
5. Save aggregated results to `lodo_summary.json` and `lodo_summary.csv`.

---

### User Story 4 - Multi-Variant Publication Visualizations with `(2)` Suffix (Priority: P2)
As lead author, I need the publication visualization suite (Figs 1 to 8) to generate publication plots at 300 DPI, saving multi-variant plots (Fig 3 Slope Chart, Fig 4 Waterfall Decomposition, Fig 5 LODO Generalization Heatmap, Fig 7 ROC Comparison) with `(2)` appended to filenames so existing Drive outputs are never overwritten.

---

## Requirements

### Functional Requirements

- **FR-001**: `main_model_lodo.ipynb` MUST reuse the exact preprocessing, dataset, collate, and model definitions from `main_model (6).ipynb`.
- **FR-002**: Automated dataset unzipping MUST extract `/content/drive/MyDrive/native_cropped.zip` to `/content/dataset_local`.
- **FR-003**: `image_root_dir` MUST point to `f"{DATA_ROOT}/native_cropped"`.
- **FR-004**: LODO fold CSVs MUST be read from `LODO_FOLDS_DIR = '/content/lodo_folds'` (fallback: `data/processed/lodo_folds`).
- **FR-005**: All redundant functions (e.g. `load_lodo_fold_splits`, `load_and_harmonize_master_cohorts`, candidate roots, synthetic fallbacks) MUST NOT be present.
- **FR-006**: For each fold, `train_csv`, `val_csv`, and `test_csv` MUST be passed directly to `FundusDualBranchDataset(csv_path=..., img_root=image_root_dir, ...)`.
- **FR-007**: `FundusDualBranchDataset` MUST resolve `img_path = f"{self.img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"`.
- **FR-008**: `DualBranchMILModel` MUST support parametric flags `use_local_branch`, `use_nonlocal`, and `use_quadrant_tokens`.
- **FR-009**: Checkpoints MUST be namespaced by variant and fold: `lodo_{variant_name}_fold_{fold_idx}_{held_out_key}_best.pth` and `_last.pth`.
- **FR-010**: Checkpoint resumption MUST restore optimizer, scheduler, epoch, and best metric from `_last.pth` if present.
- **FR-011**: $V_3$ champion results MUST be automatically merged from `lodo_summary.json` when `"run": False`.
- **FR-012**: Figures 3, 4, 5, 7 MUST append `(2)` before file extensions (`.png` and `.pdf`) and be saved at 300 DPI.
- **FR-013**: All code cells in `main_model_lodo.ipynb` MUST pass `ast.parse()` with zero syntax errors or warnings.
- **FR-014**: Figure 4 (Waterfall Decomposition) MUST strictly decompose Out-of-Domain QWK (QWK ngoại miền) computed directly from the empirical 6-fold LODO test evaluations of each architecture ablation variant ($V_0 \to V_1 \to V_2 \to V_3$) without any synthetic formulas, fabricated multiplier fallbacks, or hardcoded numbers.
- **FR-015**: Figure 5 (LODO Generalization Heatmap) MUST seamlessly support multi-variant results dictionaries by targeting the Champion model ($V_3$) or primary variant, safely mapping all evaluated clinical metrics (`in_domain_val_qwk`, `ood_test_qwk`, `delta_lodo`, `ood_within_1_grade_acc`, `ood_referable_sensitivity_at_95spec`).

---

## Success Criteria

- **SC-001**: `main_model_lodo.ipynb` runs end-to-end on Colab using `/content/lodo_folds/` CSVs and `/content/dataset_local/native_cropped/` images with 0 missing files.
- **SC-002**: Data loading and training mechanics are byte-for-byte identical to `main_model (6).ipynb`.
- **SC-003**: In-domain validation and OOD test evaluation produce accurate QWK, per-grade recall, and $\Delta_{\text{LODO}}$ metrics.
- **SC-004**: All generated artifacts are persisted to Google Drive under `RESULTS_DIR_LODO`.
