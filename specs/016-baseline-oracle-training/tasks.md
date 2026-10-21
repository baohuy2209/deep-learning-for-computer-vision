# Tasks: Baseline Oracle Training for 6 Fundus Cohorts

**Feature**: `016-baseline-oracle-training`  
**Date**: 2026-09-22  
**Input**: Design documents from `specs/016-baseline-oracle-training/` (`spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`)

---

## Format: `- [X] [TaskID] [P?] [Story?] Description with file path`

- **[P]**: Parallelizable task (independent file or distinct cohort)
- **[US1]**: User Story 1 (Priority P1) - Generate and configure 6 in-domain Oracle training notebooks
- **[US2]**: User Story 2 (Priority P2) - Automated in-domain vs. LODO cross-domain evaluation and transfer gap ($\Delta_{\text{transfer}}$) reporting

---

## Phase 1: Setup & Environment Initialization

**Purpose**: Create target directory and initialize shared template configurations.

- [X] T001 Create target directory `notebooks/diabetic_retinopathy/pipeline/oracle/` for all 6 Oracle notebooks
- [X] T002 [P] Verify availability of preprocessed image directories in `data/raw/` and `data/raw/hybrid/native_cropped/`
- [X] T003 [P] Verify availability of pre-generated LODO fold test CSVs and checkpoints in `data/processed/lodo_folds/` and `checkpoints/`

---

## Phase 2: Foundational Components (Shared Pipeline Architecture)

**Purpose**: Verify core architectural modules from `main_model (6).ipynb` and `main_model_lodo.ipynb` before notebook scaffolding.

- [X] T004 Validate `DualBranchMILModel` architecture parity (ResNet-50 Global + EfficientNet-B0 Local Attention MIL)
- [X] T005 [P] Validate `CumulativeLinkModelQWKHead` (Head 5) ordinal cutpoints and continuous QWK loss formula
- [X] T006 [P] Validate patient-level `GroupShuffleSplit` (70/15/15) split logic with seed 42 to guarantee zero cross-eye leakage

---

## Phase 3: User Story 1 - 6 In-Domain Oracle Training Notebooks (Priority: P1) 🎯 MVP

**Goal**: Deliver 6 self-contained, fully executable Jupyter Notebooks for Google Colab with NVIDIA A100 AMP acceleration, each pre-configured for its specific cohort metadata, patient grouping, and dynamic class weights.

- [X] T007 [P] [US1] Create `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_aptos.ipynb` (APTOS 2019, 3,662 images, 1-image-1-patient rule)
- [X] T008 [P] [US1] Create `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_ddr.ipynb` (DDR Dataset, 12,522 images, 2-dash patient grouping rule)
- [X] T009 [P] [US1] Create `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_deepdrid.ipynb` (DeepDRiD, 2,000 images, 500 patients, underscore prefix grouping)
- [X] T010 [P] [US1] Create `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_idrid.ipynb` (IDRiD, 516 images, trailing ID grouping rule)
- [X] T011 [P] [US1] Create `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_messidor2.ipynb` (Messidor-2, 1,744 images, date prefix patient grouping)
- [X] T012 [P] [US1] Create `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_eyepacs.ipynb` (EyePACS, 35,126 images, 17,563 patients, underscore prefix grouping)

---

## Phase 4: User Story 2 - Benchmark Comparison & Transfer Gap ($\Delta_{\text{transfer}}$) (Priority: P2)

**Goal**: Equip each notebook with an automated evaluation block that evaluates the trained in-domain Oracle model on the 15% held-out test split, restores the corresponding LODO champion checkpoint, computes $\Delta_{\text{transfer}} = QWK_{\text{oracle}} - QWK_{\text{OOD}}$, and exports structured JSON results complying with `contracts/oracle_evaluation_schema.json`.

- [X] T013 [P] [US2] Integrate LODO checkpoint restoration and test set evaluation in `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_aptos.ipynb`
- [X] T014 [P] [US2] Integrate LODO checkpoint restoration and test set evaluation in `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_ddr.ipynb`
- [X] T015 [P] [US2] Integrate LODO checkpoint restoration and test set evaluation in `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_deepdrid.ipynb`
- [X] T016 [P] [US2] Integrate LODO checkpoint restoration and test set evaluation in `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_idrid.ipynb`
- [X] T017 [P] [US2] Integrate LODO checkpoint restoration and test set evaluation in `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_messidor2.ipynb`
- [X] T018 [P] [US2] Integrate LODO checkpoint restoration and test set evaluation in `notebooks/diabetic_retinopathy/pipeline/oracle/oracle_eyepacs.ipynb`

---

## Phase 5: Verification & Polish

**Purpose**: Validate code integrity, reproducible seeds, and non-regression across all 6 generated notebooks.

- [X] T019 Implement offline validation script `scripts/validate_oracle_notebooks.py` to test notebook syntax, cell execution graphs, and schema compliance
- [X] T020 Run `python scripts/validate_oracle_notebooks.py` across all 6 notebooks to confirm zero errors
- [X] T021 Update feature status and documentation links in `specs/016-baseline-oracle-training/plan.md`
