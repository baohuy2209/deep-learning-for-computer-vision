# Tasks: Comprehensive Exploratory Data Analysis (EDA) for 6 DR Datasets

**Input**: Design documents from `/specs/001-dr-datasets-eda/`

**Prerequisites**: [plan.md](plan.md) (required), [spec.md](spec.md) (required), [research.md](research.md), [data-model.md](data-model.md), [contracts/eda_pipeline_contract.md](contracts/eda_pipeline_contract.md), [quickstart.md](quickstart.md)

**Tests**: Not explicitly requested in feature spec; validation handled via quickstart notebook execution steps.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different functions/blocks, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- File paths are explicitly specified in all descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, import configuration, and relative path resolution

- [x] T001 Verify raw data directory structure under `data/raw/` (APTOS, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS)
- [x] T002 [P] Initialize Python imports (`pandas`, `numpy`, `matplotlib`, `seaborn`, `cv2`, `PIL`, `tqdm`) and set random seeds in `notebooks/diabetic_retinopathy/eda_script.py`
- [x] T003 [P] Implement dynamic base directory path resolver (`get_base_path()`) in `notebooks/diabetic_retinopathy/eda_script.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data schema mapping and dataset ingestion infrastructure

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Define `DATASET_CONFIG` configuration dictionary for all 6 raw DR datasets in `notebooks/diabetic_retinopathy/eda_script.py`
- [x] T005 [P] Implement dataset loader and schema normalizer (`load_and_standardize_dataset()`) in `notebooks/diabetic_retinopathy/eda_script.py`

**Checkpoint**: Foundation ready — user story implementation can now begin

---

## Phase 3: User Story 1 - Multi-Dataset Standardization and Integrity Auditing (Priority: P1) 🎯 MVP

**Goal**: Standardize CSV schemas across 6 DR datasets and perform automated data integrity sanity auditing (duplicates, missing labels, out-of-range grades).

**Independent Test**: Standardized DataFrames returned for all 6 datasets with complete audit summary log tables without unhandled exceptions.

### Implementation for User Story 1

- [x] T006 [P] [US1] Implement automated sanity check logic (duplicate IDs, missing NaNs, invalid grades outside [0..4]) in `notebooks/diabetic_retinopathy/eda_script.py`
- [x] T007 [US1] Execute ingestion and sanity auditing loop across all 6 DR datasets in `notebooks/diabetic_retinopathy/eda_script.py`
- [x] T008 [US1] Render summary markdown table of dataset records and integrity check findings in `notebooks/diabetic_retinopathy/eda_script.py`

**Checkpoint**: At this point, User Story 1 (Data Loading & Auditing MVP) is fully functional and independently testable.

---

## Phase 4: User Story 2 - Class Distribution & Eye Metadata Analysis (Priority: P2)

**Goal**: Extract patient IDs and eye lateral orientation (left vs. right), and visualize DR grade class distributions (bar & pie charts) for individual and merged datasets.

**Independent Test**: Render side-by-side Bar and Pie charts for each individual dataset and the combined 6-dataset hybrid pool.

### Implementation for User Story 2

- [x] T009 [P] [US2] Implement patient ID and lateral eye side (`left`/`right`) regex parser functions in `notebooks/diabetic_retinopathy/eda_script.py`
- [x] T010 [P] [US2] Implement class distribution plotting function (`plot_dataset_class_distribution()`) rendering Bar and Pie charts in `notebooks/diabetic_retinopathy/eda_script.py`
- [x] T011 [US2] Calculate and render aggregated class distribution for merged 6-dataset hybrid pool in `notebooks/diabetic_retinopathy/eda_script.py`

**Checkpoint**: User Stories 1 AND 2 work independently and deliver class imbalance insights.

---

## Phase 5: User Story 3 - Visual Sampling, Resolution Distribution & Quality Auditing (Priority: P3)

**Goal**: Sample representative images per DR level (1x5 grid), measure image dimensions and aspect ratios, compute grayscale brightness and OpenCV Laplacian sharpness, and plot quality distributions.

**Independent Test**: 1x5 image subplots displayed per dataset, dimension scatter/KDE plots rendered, and overlapping quality histograms ($\alpha=0.5$) generated.

### Implementation for User Story 3

- [x] T012 [P] [US3] Implement 1x5 visual sample grid generator (`plot_sample_images_per_level()`) with exception handling in `notebooks/diabetic_retinopathy/eda_script.py`
- [x] T013 [P] [US3] Implement resolution and quality metric sampling function (`compute_image_metrics_sample()`) computing width, height, aspect ratio, brightness, and OpenCV Laplacian variance in `notebooks/diabetic_retinopathy/eda_script.py`
- [x] T014 [US3] Render resolution scatter/KDE plots and overlapping quality histograms ($\alpha=0.5$) comparing brightness and sharpness across datasets in `notebooks/diabetic_retinopathy/eda_script.py`
- [x] T015 [US3] Add clinical and biological markdown explanations covering contrast normalization (CLAHE) and motion blur filtering in `notebooks/diabetic_retinopathy/eda_script.py`

**Checkpoint**: All user stories are independently functional and fully implemented.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Conversion to Jupyter Notebook and end-to-end quickstart validation

- [x] T016 [P] Convert standalone script `notebooks/diabetic_retinopathy/eda_script.py` to Jupyter Notebook `notebooks/diabetic_retinopathy/eda_6_datasets.ipynb`
- [x] T017 Execute end-to-end quickstart validation scenarios defined in `specs/001-dr-datasets-eda/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **User Stories (Phases 3–5)**: Depend on Foundational phase completion
  - Sequential priority order: US1 (P1) $\rightarrow$ US2 (P2) $\rightarrow$ US3 (P3)
- **Polish (Phase 6)**: Depends on completion of all User Stories

### Parallel Opportunities

- T002 and T003 can run in parallel during Setup
- T005 can run in parallel during Foundational
- T006 [US1], T009 [US2], T010 [US2], T012 [US3], and T013 [US3] are helper functions that can be implemented in parallel within their respective user story phases.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T005)
3. Complete Phase 3: User Story 1 (T006–T008)
4. **VALIDATE**: Ensure all 6 datasets are loaded, standardized, and audited without errors.

### Full EDA Pipeline Incremental Rollout

1. Complete MVP (US1) $\rightarrow$ Data standardized
2. Complete US2 (T009–T011) $\rightarrow$ Class imbalance & eye lateral distributions analyzed
3. Complete US3 (T012–T015) $\rightarrow$ Sample images, resolution scatter plots, and brightness/sharpness quality histograms rendered
4. Complete Polish (T016–T017) $\rightarrow$ Final `.ipynb` generated and validated.
