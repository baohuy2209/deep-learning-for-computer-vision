# Tasks: Robust Image Preprocessing & Harmonization Pipeline for 6-Dataset DR Hybrid Pool

**Input**: Design documents from `/specs/002-dr-preprocessing-harmonization/`

**Prerequisites**: [plan.md](plan.md) (required), [spec.md](spec.md) (required), [research.md](research.md), [data-model.md](data-model.md), [contracts/preprocessing_pipeline_contract.md](contracts/preprocessing_pipeline_contract.md), [quickstart.md](quickstart.md)

**Tests**: Not explicitly requested in feature spec; validation handled via quickstart execution steps.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different functions/blocks, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- File paths are explicitly specified in all descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, import configuration, and directory preparation

- [x] T001 Verify output directory structure under `data/processed/hybrid/300x300/` and `data/processed/hybrid/512x512/` (and reference `data/raw/hybrid/`)
- [x] T002 [P] Initialize Python imports (`cv2`, `albumentations`, `sklearn`, `pandas`, `numpy`, `PIL`, `tqdm`) and set random seeds in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`
- [x] T003 [P] Implement dynamic path resolver for raw data and processed destination folders in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core dataset configuration mapping and ingestion helpers

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Define dataset configuration dictionary mapping the 6 raw DR datasets in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`
- [x] T005 [P] Implement data ingestion helper to load raw label tables in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 - Multi-Center Dataset Harmonization & Leak-Free Stratified Splitting (Priority: P1) 🎯 MVP

**Goal**: Concatenate 6 datasets into a single master DataFrame and perform a 70/15/15 Stratified Train/Val/Test split (`stratify=y`) before augmentation.

**Independent Test**: Train (70%), Val (15%), and Test (15%) partitions generated with identical ICDRS 0-4 class distributions.

### Implementation for User Story 1

- [x] T006 [P] [US1] Implement master harmonization function (`merge_and_harmonize_datasets()`) in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`
- [x] T007 [P] [US1] Implement stratified 70/15/15 train/val/test splitting (`create_stratified_splits()`) using `sklearn.model_selection.train_test_split` with `stratify=y` in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`
- [x] T008 [US1] Save baseline split metadata CSVs (`train_split_raw.csv`, `val_split_raw.csv`, `test_split_raw.csv`) in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`

**Checkpoint**: At this point, User Story 1 (Leak-free Harmonization & Splitting) is fully functional and independently testable.

---

## Phase 4: User Story 2 - Automated ROI Auto-Cropping & Dual-Resolution Image Enhancement (Priority: P2)

**Goal**: Crop non-diagnostic black borders, extract Green Channel CLAHE contrast enhancement, apply Gaussian blur, and resize to both 300x300 and 512x512 pixels.

**Independent Test**: Processed image subplots display tight fundus boundaries without black margins at dual spatial resolutions.

### Implementation for User Story 2

- [x] T009 [P] [US2] Implement Ben Graham / contour-based ROI auto-cropping function (`crop_fundus_roi()`) in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`
- [x] T010 [P] [US2] Implement Green channel extraction, CLAHE contrast enhancement, and Gaussian blur filter (`enhance_fundus_image()`) in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`
- [x] T011 [US2] Implement batch image preprocessing loop exporting images to `data/processed/hybrid/300x300/` and `data/processed/hybrid/512x512/` in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`

**Checkpoint**: User Stories 1 AND 2 work independently, producing cropped, CLAHE-enhanced dual-resolution images.

---

## Phase 5: User Story 3 - Targeted Training Set Balancing & Albumentations Augmentation (Priority: P3)

**Goal**: Undersample dominant classes (0, 1, 2) and oversample/augment minority classes (3, 4) using Albumentations strictly on the 70% Training set to yield 2,000 images per class (10,000 total).

**Independent Test**: Training set metadata confirms exactly 2,000 images per grade, while Val and Test set counts remain unaugmented.

### Implementation for User Story 3

- [x] T012 [P] [US3] Define Albumentations transformation pipeline (Random Flips, Rotation up to 90°, ShiftScaleRotate, Random Brightness/Contrast) in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`
- [x] T013 [P] [US3] Implement targeted training set undersampling and oversampling balancing function (`balance_and_augment_train_set()`) in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`
- [x] T014 [US3] Generate and save balanced 10,000 training image set (2,000 per class) at dual resolutions (300x300 and 512x512) in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`

**Checkpoint**: Training set achieves exactly 2,000 images per grade (10,000 total images) without altering Val/Test distributions.

---

## Phase 6: User Story 4 - Pixel Normalization & Pipeline Notebook Deliverable (Priority: P4)

**Goal**: Normalize pixel values (0-1 / ImageNet Z-score) and package the complete pipeline in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb` with detailed clinical/technical markdown annotations.

**Independent Test**: Preprocessing notebook runs end-to-end generating valid normalized tensors and complete documentation.

### Implementation for User Story 4

- [x] T015 [P] [US4] Implement pixel intensity normalization function (`normalize_image_tensor()`) scaling to [0, 1] or ImageNet Z-score in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`
- [x] T016 [US4] Add clinical and technical markdown annotations detailing data leakage prevention, microaneurysm visualization via CLAHE, and QWK optimization rationale in `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing_script.py`

**Checkpoint**: All user stories are independently functional and fully implemented.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Conversion to Jupyter Notebook and end-to-end quickstart validation

- [x] T017 [P] Convert standalone script `preprocessing_script.py` to Jupyter Notebook `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb`
- [x] T018 Execute end-to-end quickstart validation scenarios defined in `specs/002-dr-preprocessing-harmonization/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phases 3–6)**: Depend on Foundational phase completion
  - Sequential priority order: US1 (P1) $\rightarrow$ US2 (P2) $\rightarrow$ US3 (P3) $\rightarrow$ US4 (P4)
- **Polish (Phase 7)**: Depends on completion of all User Stories

### Parallel Opportunities

- T002 and T003 can run in parallel during Setup
- T005 can run in parallel during Foundational
- T006 [US1], T007 [US1], T009 [US2], T010 [US2], T012 [US3], T013 [US3], and T015 [US4] are independent functions that can be implemented in parallel within their respective phases.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T005)
3. Complete Phase 3: User Story 1 (T006–T008)
4. **VALIDATE**: Ensure 53,977 records are merged and split 70/15/15 without data leakage.

### Full Preprocessing Pipeline Incremental Rollout

1. Complete MVP (US1) $\rightarrow$ Harmonized 70/15/15 splits saved
2. Complete US2 (T009–T011) $\rightarrow$ ROI auto-cropping, CLAHE enhancement, dual spatial resolutions (300x300 & 512x512)
3. Complete US3 (T012–T014) $\rightarrow$ 10,000 balanced training images generated (2,000/class)
4. Complete US4 + Polish (T015–T018) $\rightarrow$ Pixel normalization & deliverable notebook at `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb`
