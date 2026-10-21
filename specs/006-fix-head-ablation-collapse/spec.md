# Feature Specification: Head Ablation Pipeline Collapse Fix & Advisor Directives (HD_1-4) Alignment

**Feature Branch / Directory**: `specs/006-fix-head-ablation-collapse`  
**Created**: 2026-09-03  
**Status**: Draft  
**Target System**: 5-Round End-to-End Head Ablation Benchmark on ResNet-50 (`main_model_head_ablation (3).ipynb`)  
**Reference Documents**: Advisor Guidelines (`documents/HD_1.pdf`, `documents/HD_2.pdf`, `documents/HD_3.pdf`, `documents/HD_4.pdf`)  
**Input**: Comprehensive review of HD_1-4 and diagnostic analysis of total model collapse (QWK=0.0000, Ref-AUC=0.5000, Within-1 fixed at majority class) observed in `main_model_head_ablation (2).ipynb`.

---

## 1. Context & Problem Statement

In `main_model_head_ablation (2).ipynb`, all 5 output heads (Softmax_CE, Softmax_QWK, CORAL, CORN, CLM_QWK) exhibited complete model collapse:
- **Cohen's Quadratic Weighted Kappa (QWK)** was exactly `0.0000` for all heads on both training and held-out validation/test splits.
- **Referable AUC (Ref-AUC)** was locked at `0.5000` (exact random chance level).
- **Within-1-Grade Accuracy** froze at fixed majority-class distribution proportions (`32.1%` when predicting Grade 2, `92.8%` when predicting Grade 1 in validation; `71.6%` / `92.7%` / `31.7%` in test).
- **Post-Hoc Threshold Calibration** yielded $\tau = \infty$ and Sensitivity = `0.0%`.

### Root Cause Analysis:
1. **Cross-Platform Path Resolution & Silent Gray Fallback**:
   The metadata CSVs (`train_balanced_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`) store Windows absolute paths in `native_path` (e.g. `d:\Github\...\native_cropped\IDRiD\IDRiD_118.jpg`). When executed on Linux (Google Colab), `os.path.exists(native_path)` evaluated to `False` for 100% of samples.
   In `ResNetFundusDataset.__getitem__`, when path check failed, the code silently fell back to:
   `img_rgb = np.full((self.img_size, self.img_size, 3), 128, dtype=np.uint8)`
   It then saved this blank gray array into the local `.npy` cache! Consequently, all subsequent batches and all 5 ablation rounds read constant gray 128 images (<1ms loading), causing complete feature blindness and mathematical collapse to prior class predictions ($P_o = P_e \implies \text{QWK} = 0.0000$, $\text{AUC} = 0.5000$).
2. **CORAL Bias Ordering Inversion**:
   `CORALHead.__init__` initialized rank biases as `torch.linspace(-2.0, 2.0, 4)` (increasing order). Since $P(y > k) = \sigma(W^Tx + b_k)$, monotonic ranking $P(y>0) \ge P(y>1) \ge P(y>2) \ge P(y>3)$ requires strictly decreasing biases ($b_0 > b_1 > b_2 > b_3$). Increasing initialization caused 100% monotonicity violations at epoch 1.
3. **Stale Checkpoint Resumption**:
   The execution loop in Cell 20 unconditionally resumed from any existing `_last.pth` checkpoint, propagating collapsed weights from prior runs instead of training fresh from epoch 1.

---

## 2. User Scenarios & Testing *(mandatory)*

### User Story 1 - Robust Cross-Platform Data Ingestion & Zero Silent Failures (Priority: P1)
As a machine learning researcher, I need the dataset loader in `main_model_head_ablation (3).ipynb` to reliably locate and decode fundus images across both Windows and Linux/Colab environments, and to fail loudly if an image cannot be found rather than silently poisoning the cache with blank arrays.

**Why this priority**: Without valid retinal image signals reaching the backbone, no training or ablation can occur.

**Independent Test**: Load 50 random samples via `verify_dataset_and_cache`. All samples must resolve to actual image files on disk, decode to valid RGB arrays, and possess image standard deviation $\sigma > 5.0$.

**Acceptance Scenarios**:
1. **Given** a metadata CSV containing Windows absolute paths, **When** executed on Linux with `DATASET_ROOT = '/content/dataset_local'`, **Then** `resolve_image_path` correctly resolves the image to `/content/dataset_local/native_cropped/{dataset_key}/{filename}` or `/content/dataset_local/{dataset_key}/{filename}`.
2. **Given** an unresolvable or missing image path, **When** `__getitem__` is called, **Then** the loader raises an informative `FileNotFoundError` detailing all checked candidate paths instead of returning an all-gray array.
3. **Given** a pre-existing cache directory, **When** cached `.npy` arrays have $\sigma < 1.0$ (blank array), **Then** the system detects corruption, recomputes from source, or prompts cache purge.

---

### User Story 2 - Elimination of Stale Checkpoint Pollution (Priority: P1)
As an experimenter running 5 ablation rounds, I need an explicit `FORCE_FRESH_TRAINING` toggle so that fresh ResNet-50 backbones and heads are initialized and trained cleanly from Epoch 1 without contamination from prior collapsed checkpoints.

**Why this priority**: Unconditional resumption of broken checkpoints prevents models from learning even after code fixes are applied.

**Independent Test**: Execute Round 1 with `FORCE_FRESH_TRAINING = True`. Training starts at Epoch 1/25 with newly initialized weights.

**Acceptance Scenarios**:
1. **Given** existing `_last.pth` checkpoints from a prior collapsed run, **When** `FORCE_FRESH_TRAINING = True`, **Then** `DRHeadTrainer.fit` bypasses old checkpoints and starts from Epoch 1.

---

### User Story 3 - Mathematical Rigor in Ordinal Output Heads (Priority: P1)
As an algorithm designer, I need the CORAL head to be initialized with monotonic decreasing biases ($b_0 > b_1 > b_2 > b_3$), the CLM head to maintain monotonic softplus cutpoints, and the continuous QWK loss to stabilize smoothly under $\lambda=0.4$ with 15-epoch warmup.

**Why this priority**: Directly enforces hypotheses H1-H3 and satisfies HD_1, HD_2, and HD_4 guidance.

**Independent Test**: Verify that CORAL initial rank probabilities satisfy $P(y>0) \ge P(y>1) \ge P(y>2) \ge P(y>3)$, and initial violation rate is $0\%$.

**Acceptance Scenarios**:
1. **Given** `CORALHead`, **When** initialized, **Then** `self.biases` are set to `torch.linspace(2.0, -2.0, 4)`.
2. **Given** `CLM_QWK` and `Softmax_QWK`, **When** training begins, **Then** QWK penalty weight warms up from $0.0 \to 0.4$ over 15 epochs.

---

### User Story 4 - Standardized Clinical Evaluation & Post-Hoc Calibration (Priority: P2)
As a clinical evaluator, I need post-hoc screening threshold calibration evaluated on held-out test data at specificities (95%, 90%, 85%, 80%), and full classification metrics (QWK, Per-grade Recall, Within-1, Monotonicity Violations, Generalization Gap) exported to summary CSV/JSON.

**Why this priority**: Delivers the exact publication table required by the advisor (HD_1, HD_4 Part 1 & 3).

**Independent Test**: Complete execution of all 5 heads produces `head_ablation_summary.csv` and individual calibration curves.

**Acceptance Scenarios**:
1. **Given** validation and test probabilities, **When** `calibrate_and_evaluate_screening` runs, **Then** thresholds $\tau_s$ are calibrated on Val and evaluated on Test, reporting sensitivity against UK NSC benchmark ($\ge 80\%$ Sens at $\ge 95\%$ Spec).

---

## 3. Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Implement `resolve_image_path(row, dataset_root)` with multi-tier fallback: (1) native path if exists, (2) `dataset_root/native_cropped/{dataset_key}/{filename}`, (3) `dataset_root/{dataset_key}/{filename}`, (4) relative workspace path `data/raw/hybrid/...`.
- **FR-002**: Remove silent gray fallback (`np.full(..., 128)`) from `ResNetFundusDataset.__getitem__`. Raise `FileNotFoundError` on missing images.
- **FR-003**: Implement pre-flight integrity check `verify_dataset_and_cache(dataset)` asserting that 100% of tested samples exist and have non-trivial pixel variance ($\sigma > 5.0$).
- **FR-004**: Correct `CORALHead` bias initialization from increasing (`-2.0 -> 2.0`) to decreasing (`2.0 -> -2.0`).
- **FR-005**: Add `FORCE_FRESH_TRAINING = True` configuration flag in Section 1.1, and wire it into `DRHeadTrainer.fit` to prevent accidental resumption from corrupted checkpoints.
- **FR-006**: Calibrate QWK loss target to $\lambda = 0.4$ and linear warmup to 15 epochs for both `Softmax_QWK` and `CLM_QWK`.
- **FR-007**: Synchronize EarlyStopping and ModelCheckpoint on `Val QWK` with patience 15.
- **FR-008**: Maintain post-hoc threshold calibration for screening ($y \ge 2$) at target specificities `(0.80, 0.85, 0.90, 0.95)`.
- **FR-009**: Ensure `main_model_head_ablation (3).ipynb` compiles cleanly with zero syntax errors via `ast.parse`.

---

## 4. Success Criteria *(mandatory)*

### Measurable Outcomes
- **SC-001**: Data loader sanity check verifies 100% valid image arrays with $\sigma > 5.0$, completely eliminating all-gray inputs.
- **SC-002**: Training QWK and Validation QWK ascend significantly above 0.0000 starting from early epochs (targeting Val QWK $> 0.60$ for CLM/CORAL).
- **SC-003**: Validation and Test Referable AUC exceed $0.8500$ (well above the collapsed $0.5000$ chance line).
- **SC-004**: CORAL initial monotonicity violations rate drops from $100.0\%$ to $\le 5.0\%$.
- **SC-005**: All 5 heads successfully generate calibrated screening metrics and export `head_ablation_summary.csv`.

---

## 5. Assumptions

- Image data on Google Colab will be unzipped to `/content/dataset_local` or accessible via Google Drive.
- The 6 constituent datasets in the hybrid pool maintain their established directory keys: `APTOS`, `DDR`, `DeepDRiD`, `EyePACS`, `IDRiD`, `Messidor-2`.
- `main_model_head_ablation (3).ipynb` remains the authoritative head ablation notebook.
