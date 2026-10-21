# Tasks: Restore Exact Notebook-to-Python Logic Parity (`drdg`)

**Input**: Design documents from `specs/018-restore-notebook-parity/` (`plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`)

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Organization**: Tasks are grouped by user story (US1 through US5) in priority order to enable independent implementation and testing of each story.

---

## Format: `- [ ] [TaskID] [P?] [Story?] Description with file path`

- **[P]**: Can run in parallel (different files, no blocking dependencies)
- **[Story]**: Which user story this task belongs to (e.g., `[US1]`, `[US2]`, `[US3]`, `[US4]`, `[US5]`)
- Every task includes an explicit, verifiable file path.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Environment verification, dependency validation, and baseline directory preparation.

- [x] T001 Verify Python environment ($\ge 3.10$) and editable package installation via `pip install -e .` in `pyproject.toml`
- [x] T002 [P] Verify presence, accessibility, and unmodified state of the 5 canonical notebooks in `notebooks/diabetic_retinopathy/`
- [x] T003 [P] Verify directory layout for raw, processed, checkpoint, and result assets in `data/raw/`, `data/processed/lodo_folds/`, `checkpoints/`, `results/`
- [x] T004 [P] Configure Pytest test runner options and markers in `pyproject.toml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core extraction harnesses and symbol inventory that all user stories depend on.

**⚠️ CRITICAL**: Must complete before user story implementation to establish golden baselines.

- [x] T005 Extract AST inventory of classes, functions, and constants from the 5 canonical notebooks using `audit_tools/extract_symbols.py` into `audit_artifacts/notebook_symbols.json`
- [x] T006 [P] Extract symbol inventory from the modular Python package into `audit_artifacts/drdg_symbols.json` using `audit_tools/extract_symbols.py`
- [x] T007 [P] Verify centralized path definitions and directory creation in `src/drdg/paths.py`
- [x] T008 [P] Verify typed dataclass configurations (strictly zero YAML) in `src/drdg/config.py`

**Checkpoint**: Foundational symbol inventories and paths verified — user story tasks can begin.

---

## Phase 3: User Story 1 - Canonical Dual-Implementation Model & Notebook Preservation (Priority: P1) 🎯 MVP

**Goal**: Preserve the 5 canonical Jupyter research notebooks 100% self-contained and unmodified while ensuring `drdg` acts as an independent pure-Python equivalent.

**Independent Test**: Execute `python scripts/verify_notebook_integrity.py` to confirm zero git diffs on all 5 notebooks, and confirm that `import drdg` initializes cleanly without file I/O or Colab dependencies.

### Implementation for User Story 1

- [x] T009 [P] [US1] Create automated notebook integrity verification script in `scripts/verify_notebook_integrity.py` asserting zero git modifications on the 5 canonical notebooks
- [x] T010 [P] [US1] Audit and clean top-level package exports in `src/drdg/__init__.py` to expose public models, datasets, losses, and utilities with zero side effects
- [x] T011 [US1] Audit `src/drdg/paths.py` and `src/drdg/config.py` to ensure no Google Drive mounting or Colab-specific path hardcoding executes at import time
- [x] T012 [US1] Add unit test in `tests/test_imports.py` validating that importing `drdg` and all subpackages succeeds on clean Python runtime with zero side-effects

**Checkpoint**: Dual-layer architecture established; original notebooks protected; `drdg` imports cleanly.

---

## Phase 4: User Story 2 - Exact 3-Split LODO Dataset Generation & Bipartite Partition Parity (Priority: P1)

**Goal**: Restore the authentic 3-split LODO partitioning (`train`, `val`, `test` across 18 CSVs) and bipartite evaluation schema, permanently removing all invalid 4-way split logic (`source_test`, `target_test.csv`).

**Independent Test**: Run `pytest tests/parity/test_lodo_split_parity.py` verifying that each of the 6 folds produces/loads exactly 3 CSVs in `data/processed/lodo_folds/` with zero patient leakage.

### Tests for User Story 2

- [x] T013 [P] [US2] Update unit tests in `tests/test_lodo_splits.py` asserting exactly 3 CSVs per fold (`fold_{fold_idx}_{fold_slug}_{split}.csv`), zero patient leakage, and zero references to `source_test`
- [x] T014 [P] [US2] Update result schema tests in `tests/test_result_schema.py` to assert the bipartite schema (`in_domain_val` + `ood_test` + `delta_lodo`)
- [x] T015 [P] [US2] Create behavioral parity test in `tests/parity/test_lodo_split_parity.py` asserting exact fold membership and quota-balanced counts against `merge_dataset_lodo.ipynb`

### Implementation for User Story 2

- [x] T016 [US2] Audit and correct LODO split generator in `src/drdg/data/lodo.py` to strictly produce 3 splits (`train`, `val`, `test`), matching `merge_dataset_lodo.ipynb` Cell 7
- [x] T017 [US2] Audit and correct `src/drdg/experiments/lodo.py` so that `LODOExperimentRunner` loads `fold_{fold_idx}_{cohort}_{split}.csv` and evaluates bipartite partitions (`in_domain_val` and `ood_test`)
- [x] T018 [US2] Audit and correct `src/drdg/evaluation/lodo.py` to evaluate in-domain validation for threshold calibration and held-out target test with frozen threshold
- [x] T019 [US2] Audit and correct `src/drdg/evaluation/result_schema.py` to serialize `LODOFoldResult` matching `main_model_lodo.ipynb` Cell 23 & 27
- [x] T020 [P] [US2] Update LODO split generation CLI script in `scripts/generate_lodo_splits.py` to output the 18 CSV files and `lodo_splits_summary.csv`
- [x] T021 [US2] Search the entire codebase and remove any lingering references to `source_test` or `target_test.csv` in `src/drdg/` and `scripts/`

**Checkpoint**: LODO pipeline strictly produces and consumes 18 fold CSVs with verified zero-leakage bipartite evaluation.

---

## Phase 5: User Story 3 - End-to-End Research Workflow & Execution Dependency Parity (Priority: P1)

**Goal**: Reproduce and document the sequential 6-stage Colab workflow (Raw Acquisition $\to$ Total Prep $\to$ Merge Dataset $\to$ Native Pool $\to$ Standard Experiments & LODO Splits $\to$ LODO Training).

**Independent Test**: Verify that all intermediate artifacts documented in `docs/DATA_FLOW_AUDIT.md` are correctly generated and consumed by downstream scripts.

### Implementation for User Story 3

- [x] T022 [P] [US3] Audit dataset-specific preparation scripts in `scripts/prepare_dataset.py`, `total_aptos.py`, `total_deepdrid.py`, `total_idrid.py`, `total_idrid-segmentation.py` against notebook data ingestion
- [x] T023 [P] [US3] Audit multi-cohort metadata harmonization and non-black circular ROI cropping in `src/drdg/data/harmonization.py` against `merge-dataset.ipynb`
- [x] T024 [P] [US3] Audit master metadata generation CLI in `scripts/build_master_metadata.py` to produce `native_cropped/`, `train_balanced_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`
- [x] T025 [US3] Create comprehensive producer-consumer artifact audit document in `docs/DATA_FLOW_AUDIT.md` mapping all files across the 6 execution stages
- [x] T026 [P] [US3] Verify isolation of IDRiD segmentation dataset in `src/drdg/data/` as an independent XAI evaluation pipeline

**Checkpoint**: End-to-end data flow dependency graph verified and documented with no missing intermediate links.

---

## Phase 6: User Story 4 - Numerical, Architectural & Training-Step Parity (Priority: P2)

**Goal**: Guarantee that all neural network components, custom losses, training engines, and optimization schedules in `drdg` match notebook code cell execution to within numerical tolerance.

**Independent Test**: Run `pytest tests/parity/` confirming FP32 max absolute error $< 1\times 10^{-6}$ for forward passes and losses, $< 1\times 10^{-5}$ for gradients, and matching 1-step optimizer parameter updates.

### Tests for User Story 4

- [x] T027 [P] [US4] Create forward-pass parity test for CBAM, NonLocalBlock2D, and QuadrantTokenAggregator in `tests/parity/test_attention_parity.py`
- [x] T028 [P] [US4] Create forward-pass parity test for ResNet-50 GlobalContextStream in `tests/parity/test_global_branch_parity.py`
- [x] T029 [P] [US4] Create forward-pass parity test for EfficientNet-B0 LocalMILBranch with GatedAttentionPool in `tests/parity/test_local_mil_parity.py`
- [x] T030 [P] [US4] Create forward-pass parity test for FusionNeck in `tests/parity/test_fusion_parity.py`
- [x] T031 [P] [US4] Create forward-pass parity test for all 5 Output Heads (`Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, `CLM_QWK`) in `tests/parity/test_heads_parity.py`
- [x] T032 [P] [US4] Create loss and gradient parity test in `tests/parity/test_losses_parity.py` comparing forward scalars and backward gradients for `ContinuousQWKLoss`, `CORAL`, `CORN`, and `CLM_QWK`
- [x] T033 [P] [US4] Create architectural variant parity test for V0, V1, V2, V3 in `tests/parity/test_variant_parity.py` against `main_model_lodo.ipynb`
- [x] T034 [P] [US4] Create deterministic 1-step training update parity test in `tests/parity/test_training_step_parity.py` comparing model weights after forward-loss-backward-clip-step
- [x] T035 [P] [US4] Create optimizer parameter group and LR scheduler trajectory test in `tests/parity/test_optimizer_scheduler_parity.py`
- [x] T036 [P] [US4] Create Referable DR (RDR) threshold calibration parity test in `tests/parity/test_rdr_parity.py`
- [x] T037 [P] [US4] Create XAI Grad-CAM and patch attention splatting parity test in `tests/parity/test_xai_parity.py`

### Implementation for User Story 4

- [x] T038 [US4] Audit and align `src/drdg/models/attention.py` (CBAM, NonLocalBlock2D, QuadrantTokenAggregator) with `main_model (6).ipynb` Cell 12
- [x] T039 [US4] Audit and align `src/drdg/models/global_stream.py` with `main_model (6).ipynb` Cell 13
- [x] T040 [US4] Audit and align `src/drdg/models/local_mil.py` (patch projection, GatedAttentionPool) with `main_model (6).ipynb` Cell 14
- [x] T041 [US4] Audit and align `src/drdg/models/fusion.py` with `main_model (6).ipynb` Cell 15
- [x] T042 [US4] Audit and align `src/drdg/models/heads.py` (all 5 heads) with `main_model_head_ablation (3).ipynb`
- [x] T043 [US4] Audit and align `src/drdg/training/losses.py` (ContinuousQWKLoss, CumulativeLinkLoss, CORAL, CORN) with notebook loss formulations
- [x] T044 [US4] Audit and align `src/drdg/models/variants.py` (exact V0, V1, V2, V3 component flags) with `main_model_lodo.ipynb` Cell 16
- [x] T045 [US4] Audit and align `src/drdg/training/optimizer.py`, `src/drdg/training/scheduler.py` (two-tier LR, Cosine Annealing, linear QWK warmup), and `src/drdg/training/early_stopping.py` with `main_model_lodo.ipynb`
- [x] T046 [US4] Audit and align `src/drdg/evaluation/rdr.py` (operating threshold $\tau^*$ calibration at $\ge 80\%$ specificity) with notebook logic
- [x] T047 [US4] Audit and align `src/drdg/xai/` (Grad-CAM hooks, local patch attention splatting) with `main_model (6).ipynb` Cell 25

**Checkpoint**: All neural network modules, losses, training steps, and evaluation metrics pass numerical parity tests.

---

## Phase 7: User Story 5 - Code Mapping, Discrepancy Auditing & Documentation Reconstruction (Priority: P3)

**Goal**: Document the canonical symbol map, record all discrepancies and resolutions in audit reports, and rewrite documentation to reflect the dual-layer architecture.

**Independent Test**: Review `docs/NOTEBOOK_CODE_MAP.md`, `docs/REFACTOR_AUDIT.md`, and `README.md` to confirm 100% symbol coverage, transparent status reporting, and complete elimination of invalid wrapper assumptions.

### Implementation for User Story 5

- [x] T048 [US5] Generate comprehensive notebook-to-module mapping document in `docs/NOTEBOOK_CODE_MAP.md` covering 100% of research-critical symbols from the 5 notebooks
- [x] T049 [US5] Generate exhaustive discrepancy and integrity certification report in `docs/REFACTOR_AUDIT.md` with final status `VERIFIED`
- [x] T050 [US5] Search the entire repository and remove false legacy assumptions stating notebooks should be thin wrappers or should not contain model classes
- [x] T051 [US5] Rewrite `README.md` to explicitly describe the dual-layer architecture (canonical complete notebooks + modular Python package) and reproduction workflow
- [x] T052 [P] [US5] Rewrite `docs/ARCHITECTURE.md` to document the verified dual-branch architecture and pure-Python module structure
- [x] T053 [P] [US5] Rewrite `docs/DATA.md` to document raw acquisition, consolidation, native cropping, and the 18-file LODO fold structure
- [x] T054 [P] [US5] Rewrite `docs/EXPERIMENTS.md` and `docs/REPRODUCIBILITY.md` detailing how to run both notebooks and Python CLI scripts
- [x] T055 [P] [US5] Rewrite `docs/RESULTS_SCHEMA.md` and `docs/XAI.md` to describe the bipartite evaluation metrics and explainability modules

**Checkpoint**: Code map, audit report, and public documentation complete and fully aligned with code truth.

---

## Phase 8: Polish & Full Verification

**Purpose**: Execute end-to-end test suites, dry-run CLI scripts, and perform final integrity verification.

- [x] T056 Execute full test suite including all behavioral parity tests (`pytest tests/ tests/parity/ -v`)
- [x] T057 [P] Execute dry-run LODO benchmark across all variants (`python scripts/run_lodo.py --variant v3 --held-out aptos --dry-run`)
- [x] T058 [P] Execute dry-run head ablation across all heads (`python scripts/run_head_ablation.py --head clm_qwk --dry-run`)
- [x] T059 Run final notebook preservation check (`python scripts/verify_notebook_integrity.py` or `git status -- notebooks/`)
- [x] T060 Update final status to `VERIFIED` in `docs/REFACTOR_AUDIT.md`

---

## Dependencies & Execution Order

```mermaid
graph TD
    subgraph Phase 1 & 2: Setup & Foundations
        T001 --> T005
        T002 --> T005
        T003 --> T005
        T004 --> T005
        T005 --> T006
        T006 --> T007
        T007 --> T008
    end

    subgraph Phase 3: User Story 1 (P1 - Notebook Preservation)
        T008 --> T009
        T009 --> T010
        T010 --> T011
        T011 --> T012
    end

    subgraph Phase 4: User Story 2 (P1 - 3-Split LODO Parity)
        T012 --> T013
        T013 --> T014
        T014 --> T015
        T015 --> T016
        T016 --> T017
        T017 --> T018
        T018 --> T019
        T019 --> T020
        T020 --> T021
    end

    subgraph Phase 5: User Story 3 (P1 - End-to-End Workflow Parity)
        T021 --> T022
        T022 --> T023
        T023 --> T024
        T024 --> T025
        T025 --> T026
    end

    subgraph Phase 6: User Story 4 (P2 - Numerical & Step Parity)
        T026 --> T027
        T027 --> T028
        T028 --> T029
        T029 --> T030
        T030 --> T031
        T031 --> T032
        T032 --> T033
        T033 --> T034
        T034 --> T035
        T035 --> T036
        T036 --> T037
        T037 --> T038
        T038 --> T039
        T039 --> T040
        T040 --> T041
        T041 --> T042
        T042 --> T043
        T043 --> T044
        T044 --> T045
        T045 --> T046
        T046 --> T047
    end

    subgraph Phase 7: User Story 5 (P3 - Auditing & Documentation)
        T047 --> T048
        T048 --> T049
        T049 --> T050
        T050 --> T051
        T051 --> T052
        T052 --> T053
        T053 --> T054
        T054 --> T055
    end

    subgraph Phase 8: Final Verification
        T055 --> T056
        T056 --> T057
        T057 --> T058
        T058 --> T059
        T059 --> T060
    end
```

---

## Parallel Execution Opportunities

- **Setup & Foundations**: `T002`, `T003`, `T004` can run in parallel after `T001`.
- **US1**: `T009` and `T010` can run in parallel.
- **US2 Tests**: `T013`, `T014`, `T015` can be developed in parallel.
- **US3 Data Pipeline**: `T022`, `T023`, `T024`, `T026` can be executed in parallel.
- **US4 Parity Tests**: `T027` through `T037` can be implemented and run in parallel as independent test modules.
- **US5 Documentation**: `T052` through `T055` can be updated concurrently once `T048` and `T049` are complete.

---

## Implementation Strategy (MVP & Incremental Delivery)

1. **MVP Scope (Phase 1-4: US1 + US2)**:
   - Establish and guarantee absolute notebook preservation (`T009`-`T012`).
   - Fix the LODO 3-split schema, eradicate `source_test`, and ensure 18 fold CSVs load properly (`T013`-`T021`).
   - At this point, the repository has fixed the primary defect identified by the user.

2. **Incremental Delivery 2 (Phase 5: US3)**:
   - Verify the 6-stage end-to-end data workflow and publish `docs/DATA_FLOW_AUDIT.md` (`T022`-`T026`).

3. **Incremental Delivery 3 (Phase 6: US4)**:
   - Build out comprehensive `tests/parity/` suite, proving FP32 numerical and gradient equivalence across all streams, heads, and losses (`T027`-`T047`).

4. **Incremental Delivery 4 (Phase 7-8: US5 + Polish)**:
   - Complete `docs/NOTEBOOK_CODE_MAP.md`, `docs/REFACTOR_AUDIT.md`, rewrite all user-facing documentation to reflect the dual-layer model, and certify status `VERIFIED` (`T048`-`T060`).
