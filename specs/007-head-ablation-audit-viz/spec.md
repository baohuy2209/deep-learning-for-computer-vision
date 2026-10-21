# Feature Specification: Head Ablation Benchmark Technical Audit & Data Integrity Visualization Suite

**Feature Branch / Directory**: `specs/007-head-ablation-audit-viz`  
**Created**: 2026-09-04  
**Status**: Draft  
**Target System**: 5-Round End-to-End Output Head Ablation Benchmark on ResNet-50 (`notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb`)  
**Reference Documents**: Advisor Review Directives (`documents/HD_1.pdf`, `documents/HD_2.pdf`, `documents/HD_3.pdf`, `documents/HD_4.pdf`)  
**Reference Model & Data Pipeline**: Dual-Branch Baseline Notebook (`notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb`)  
**Input**: Comprehensive technical audit of `main_model_head_ablation (3).ipynb` against HD_1-4 guidelines and integration of data visualization & tensor sanity inspection to guarantee data integrity.

---

## 1. Executive Summary & Audit Overview

The purpose of this specification is to:
1. **Audit `main_model_head_ablation (3).ipynb`** against the 4 advisor guideline documents (`HD_1.pdf` to `HD_4.pdf`) to determine technical validity, theoretical soundness, and benchmark fairness for comparing the 5 output heads (`Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, `CLM_QWK`).
2. **Identify Critical Vulnerabilities & Bugs** that threaten experiment validity (such as silent gray fallback, inverted CORAL bias initialization, stale checkpoint contamination, and syntax clutter).
3. **Specify the Integration of Data Integrity & Visual Inspection Tools** into `main_model_head_ablation (3).ipynb`, adapting the high-resolution batch inspection, sample diagnostics, pixel variance assertions, and class distribution charts from `main_model (4).ipynb` to ensure 100% data integrity before training starts.

---

## 2. Technical Audit Summary: Compliance with HD_1, HD_2, HD_3, HD_4

| Advisor Document | Key Directives & Requirements | Current Implementation Status in Notebook (3) | Evaluation & Action Required |
|---|---|---|---|
| **HD_1: Output Head Architecture & H3** | 1. Compare 5 heads on fixed backbone (`Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, `CLM_QWK`).<br>2. Evaluate Hypothesis H3: Ordinal heads eliminate ordering loss and preserve performance under domain shift.<br>3. Mandatory metric suite: QWK, Per-grade recall (G1 & G3), Within-1-Grade accuracy, Ref-Sens @ fixed specificities (80, 85, 90, 95), Monotonicity Violations, Generalization Gap ($\Delta_{\text{val-test}}$). | **ALIGNED**: All 5 heads are defined with ResNet-50 backbone. Publication figures (Figs 1, 3, 4, 6, 7) directly target H3. | **Pass with Bugfix**: Core logic conforms to HD_1, but CORAL rank monotonicity calculation is distorted by inverted bias initialization (see below). |
| **HD_2: Split Hygiene & End-to-End Training** | 1. Split before balance: Natural distribution for Val and Test; train-only balancing.<br>2. True End-to-End training: Train backbone separately 5 times (no shared latent cache bias).<br>3. Fix `rank_monotonicity_violations` measurement on cumulative rank probabilities.<br>4. Fix cross-platform path separators. | **MOSTLY ALIGNED**: Loads `train_balanced_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`. Section 5 loops 5 rounds with fresh ResNet-50. | **Action Required**: `ResNetFundusDataset` still uses Windows-centric path resolution and lacks fallback logic. |
| **HD_3: Performance & IPC Optimization** | 1. `cv2.setNumThreads(0)` to prevent worker thread contention.<br>2. Transmit `uint8` tensors across DataLoader workers to slash shared memory IPC from ~755 MB to ~189 MB.<br>3. Local SSD uncompressed `.npy` caching.<br>4. Val loader must use evaluation mode (no random augmentations). | **ALIGNED**: `cv2.setNumThreads(0)` present in cell 3; `tensor_u8` returned in `__getitem__`; `normalize_fundus_tensor` operates on GPU; `self.is_train` controls augmentation. | **Pass**: Highly optimized DataLoader pipeline. |
| **HD_4: Threshold Calibration & Loss Warmup** | 1. Part 1 & 3: Ref-AUC = 0.9063 proves feature separation is strong; problem is operating threshold $\tau$. Do NOT use class weights to shift threshold; calibrate $\tau$ post-hoc on Val, evaluate on Test at Spec 95/90/85/80.<br>2. Part 4: Stabilize continuous QWK loss: extend warmup to 15 epochs; set target $\lambda = 0.4$.<br>3. Synchronize `EarlyStopping` and `ModelCheckpoint` on `Val QWK` with patience $\ge 15$. | **ALIGNED**: `USE_CLASS_WEIGHTS = False`, `TARGET_QWK_WEIGHT = 0.4`, `QWK_WARMUP_EPOCHS = 15`, `PATIENCE_EARLY_STOP = 15`. `calibrate_and_evaluate_screening` is implemented. | **Pass**: Fully matches advisor's updated optimization and calibration instructions. |

---

## 3. Discovered Vulnerabilities & Required Technical Corrections

### Finding 1: Silent Gray Fallback & Missing Cross-Platform Path Resolution (CRITICAL)
- **Problem**: In Cell 12 (`ResNetFundusDataset.__getitem__`), lines 437-442:
  ```python
  if os.path.exists(native_path_str):
      img_bgr = cv2.imread(native_path_str)
      img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else np.full((self.img_size, self.img_size, 3), 128, dtype=np.uint8)
  else:
      img_rgb = np.full((self.img_size, self.img_size, 3), 128, dtype=np.uint8)
  ```
  The metadata CSVs store Windows absolute paths (e.g., `d:\Github\...\IDRiD_118.jpg`). When executed on Linux / Google Colab (or when dataset is extracted to `/content/dataset_local`), `os.path.exists(native_path_str)` is `False` for 100% of samples. The dataset silently generates constant gray `128` arrays and saves them into the `.npy` cache! This caused the total model collapse (QWK=0.0000, AUC=0.5000) diagnosed in `specs/006-fix-head-ablation-collapse`.
- **Correction**:
  1. Implement `resolve_image_path(row, dataset_root)` supporting multi-tier candidate paths.
  2. Remove `np.full(..., 128)` fallback. If an image cannot be found or decoded, immediately raise a descriptive `FileNotFoundError`.

### Finding 2: Inverted Bias Initialization in `CORALHead` (CRITICAL)
- **Problem**: In Cell 14 (`CORALHead.__init__`), line 693:
  ```python
  self.biases.copy_(torch.linspace(-2.0, 2.0, self.num_thresholds))
  ```
  This initializes thresholds in **increasing** order ($[-2.0, -0.67, +0.67, +2.0]$). Because $P(y > k) = \sigma(W^T x + b_k)$, increasing biases guarantee that $b_0 < b_3 \implies P(y > 0) < P(y > 3)$, generating an initial 100% rank monotonicity violation rate and forcing the optimizer to invert all bias signs.
- **Correction**: Change to strictly decreasing order:
  ```python
  self.biases.copy_(torch.linspace(2.0, -2.0, self.num_thresholds))
  ```

### Finding 3: Unconditional Checkpoint Resumption (Risk of Stale Checkpoint Poisoning)
- **Problem**: In Cell 20, the training loop automatically checks `if os.path.exists(last_ckpt_path): resume`. If a previous run failed or trained on all-gray data, re-running the notebook automatically resumes the corrupted checkpoint rather than re-training from Epoch 1.
- **Correction**: Add explicit toggle `FORCE_FRESH_TRAINING = True` in Section 1.1, allowing users to train from scratch cleanly.

### Finding 4: Absence of Data Integrity Verification & Batch Inspection
- **Problem**: Unlike `main_model (4).ipynb` (which contains extensive sample diagnostics, visual batch inspection, and dataset distribution charts in Cells 26-31), `main_model_head_ablation (3).ipynb` directly jumps from dataset initialization to training without any visual verification. The user cannot visually verify that retina images are loaded correctly, that colors are natural after Ben Graham enhancement, or that pixel arrays are non-trivial.
- **Correction**: Add dedicated Data Visualization and Tensor Sanity Inspection cells modeled after `main_model (4).ipynb`.

### Finding 5: Stray Code Artifact
- **Problem**: Cell 8 contains only the naked expression `NUM_WORKERS`.
- **Correction**: Clean up or merge into Section 1.1 configuration.

---

## 4. User Scenarios & Testing *(mandatory)*

### User Story 1 - Visual Fundus Batch Inspection & Tensor Sanity Check (Priority: P1)
As a researcher evaluating 5 output heads, I need interactive visualization cells in `main_model_head_ablation (3).ipynb` that display raw/preprocessed retinal images, inspect sample tensor diagnostics, and plot a multi-sample batch grid across different DR grades, so that I can visually verify data integrity and contrast enhancement before running 125 epochs of training.

**Why this priority**: Visual inspection is the first line of defense against corrupted data paths, incorrect color channels, inverted normalization, or blank gray caches.

**Independent Test**: Execute the new visualization cells immediately after DataLoader initialization. A Matplotlib window must display a multi-image panel of retinal images with clearly visible vascular trees and optic discs across Grades 0 to 4, accompanied by tensor value ranges, mean intensity, and pixel standard deviation $\sigma > 5.0$.

**Acceptance Scenarios**:
1. **Given** initialized `train_dataset` and `train_loader`, **When** the inspection cell runs, **Then** it prints sample diagnostics: native path, ICDRS grade, tensor shape `[3, 512, 512]`, dtype `torch.uint8`, min/max values `[0, 255]`, and mean pixel intensity.
2. **Given** a batch from `train_loader`, **When** `visualize_batch_inspection` runs, **Then** it uses `safe_convert_for_plot` to render a clean grid of fundus images with diagnosis labels and verified pixel intensity ranges $[0.0, 1.0]$.
3. **Given** an invalid or missing image file, **When** accessed, **Then** the pipeline raises `FileNotFoundError` with clear diagnostics instead of showing a blank gray image.

---

### User Story 2 - Pre-Flight Data Integrity Assertion (Priority: P1)
As an experimenter running long GPU jobs, I need an automated pre-flight integrity check that validates 50 random samples from the dataset to confirm that 100% of samples resolve to valid retinal images with pixel standard deviation $\sigma > 5.0$, completely preventing silent all-gray cache poisoning.

**Why this priority**: Prevents wasting hours of GPU compute on corrupted or gray inputs.

**Independent Test**: Run `verify_dataset_and_cache(train_dataset, num_samples=50)`. It must report `100% Passed (0 corrupted/gray samples detected)`.

**Acceptance Scenarios**:
1. **Given** any sample with pixel variance $\sigma < 1.0$ (indicating an all-gray or blank array), **When** checked, **Then** the assertion fails immediately and alerts the user to purge the cache.

---

### User Story 3 - Class Distribution & Natural Prevalence Breakdown (Priority: P2)
As an author preparing publication tables, I need a visual distribution chart comparing the balanced training cohort against the natural prevalence of the Validation and Test sets (HD_2 Lỗi 3, HD_4 Part 1), confirming that Test prevalence remains untampered (~64% Grade 0).

**Why this priority**: Satisfies the thesis advisor's explicit directive to document test set prevalence and prove split integrity.

**Independent Test**: Run `visualize_dataset_splits_distribution()` to generate a structured console table and publication-grade bar chart showing exact sample counts and percentages for Grades 0-4 across Train, Val, and Test splits.

**Acceptance Scenarios**:
1. **Given** `train_metadata`, `val_metadata`, and `test_metadata`, **When** plotted, **Then** the chart displays Train (balanced ~1300-2300 per grade) alongside Val and Test (natural distribution with Grade 0 ~64%).

---

### User Story 4 - Output Head Mathematical Correction & Fresh Training Guard (Priority: P1)
As an ordinal regression researcher, I need `CORALHead` biases to initialize monotonically decreasing, and `DRHeadTrainer` to support `FORCE_FRESH_TRAINING`, so that all 5 rounds train cleanly and produce mathematically valid monotonicity metrics.

**Why this priority**: Enforces theoretical correctness of the benchmark for Hypothesis H3.

**Independent Test**: Instantiate `CORALHead` and verify `biases = [2.0, 0.67, -0.67, -2.0]`. Run Round 1 with `FORCE_FRESH_TRAINING = True` to verify fresh initialization.

**Acceptance Scenarios**:
1. **Given** `CORALHead`, **When** forward pass runs at initialization, **Then** initial rank monotonicity violation rate is $\le 5.0\%$ (instead of 100%).
2. **Given** `FORCE_FRESH_TRAINING = True`, **When** training starts, **Then** old checkpoints are ignored and training starts at Epoch 1.

---

## 5. Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Implement `resolve_image_path(row, dataset_root)` in `main_model_head_ablation (3).ipynb` with hierarchical resolution:
  1. Native path if exists on current filesystem.
  2. `os.path.join(dataset_root, "native_cropped", dataset_key, filename)`
  3. `os.path.join(dataset_root, dataset_key, filename)`
  4. `os.path.join("data", "raw", "hybrid", "native_cropped", dataset_key, filename)`
- **FR-002**: Eliminate silent gray fallback (`np.full(..., 128)`) from `ResNetFundusDataset.__getitem__`. Raise `FileNotFoundError` on unresolvable files.
- **FR-003**: Implement `safe_convert_for_plot(img_data)` handling uint8, float32, and ImageNet Z-score denormalization into $[0.0, 1.0]$ NumPy array.
- **FR-004**: Add **Section 2.3: Sample Diagnostics & Tensor Inspection** displaying tensor dimensions, min/max, mean, and 5x5 center pixel matrix for arbitrary dataset indices.
- **FR-005**: Add **Section 2.4: Visual Mini-Batch Inspection** rendering an 8-sample grid from `train_loader` across available clinical grades with image titles showing diagnosis grade and pixel range.
- **FR-006**: Add **Section 2.5: Pre-Flight Dataset & Cache Integrity Verification** asserting non-trivial variance ($\sigma > 5.0$) across 50 random samples.
- **FR-007**: Add **Section 2.6: Split Distribution Breakdown** generating console summary and 3-split comparative bar chart (Train balanced vs Val natural vs Test natural).
- **FR-008**: Fix `CORALHead` bias initialization to decreasing order: `torch.linspace(2.0, -2.0, self.num_thresholds)`.
- **FR-009**: Add `FORCE_FRESH_TRAINING: bool = True` configuration in Section 1.1 and wire it into trainer checkpoint resumption logic.
- **FR-010**: Remove stray Cell 8 (`NUM_WORKERS`).
- **FR-011**: Maintain strict compliance with Constitution v1.1.0 (zero unintended refactoring, delta-only edits, syntax validation via `ast.parse`).

---

## 6. Success Criteria *(mandatory)*

### Measurable Outcomes
- **SC-001**: 100% of tested dataset samples resolve to actual image files on disk with pixel standard deviation $\sigma > 5.0$, with 0 occurrences of blank gray 128 images.
- **SC-002**: Interactive visualization cells render publication-grade Matplotlib figures of retinal fundus images and class distributions directly inside the notebook.
- **SC-003**: `CORALHead` initial rank monotonicity violation rate drops from $100.0\%$ to $\le 5.0\%$.
- **SC-004**: All 5 ablation rounds execute smoothly end-to-end on ResNet-50 with 15-epoch QWK warmup ($\lambda=0.4$), EarlyStopping on Val QWK, and post-hoc threshold calibration.
- **SC-005**: Notebook AST syntax parses with 0 errors (`ast.parse`).

---

## 7. Assumptions & Environment Dependencies

- The notebook is executed in Google Colab (with Google Drive mounted or dataset extracted to `/content/dataset_local`) or on Windows local development environment.
- The 6 hybrid constituent datasets (`APTOS`, `DDR`, `DeepDRiD`, `EyePACS`, `IDRiD`, `Messidor-2`) are present in `data/raw/hybrid/` or `/content/dataset_local`.
- Target image resolution is $512 \times 512$ with Green-channel Ben Graham contrast enhancement.
