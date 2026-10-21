# Tasks: Public Academic Repository Refactor (`drdg`)

**Input**: Design documents from `specs/017-public-repo-refactor/` (`spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`)

**Prerequisites**: `plan.md` (required), `spec.md` (required for user stories), `research.md`, `data-model.md`, `contracts/`

**Organization**: Tasks are grouped strictly by user story to enable independent implementation, testing, and incremental verification.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (`[US1]` through `[US6]`)
- Every task includes the exact target file path.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, directory structure, build packaging, and environment templates.

- [ ] T001 Create repository directory scaffolding (`src/drdg/`, `scripts/`, `docs/`, `notebooks/`, `tests/`, `data/raw/`, `data/processed/`, `results/lodo/`, `results/head_ablation/`, `results/xai/`)
- [ ] T002 [P] Configure package build configuration supporting editable installation in `pyproject.toml`
- [ ] T003 [P] Configure minimal runtime package dependencies in `requirements.txt`
- [ ] T004 [P] Configure Git ignore rules and empty credential template in `.gitignore` and `.env.example`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure, configuration dataclasses, and common utilities that all user stories depend on.

- [ ] T005 [P] Implement path resolution helpers in `src/drdg/paths.py`
- [ ] T006 [P] Implement strongly typed configuration dataclasses (`DataConfig`, `ModelConfig`, `TrainingConfig`, `LocalMILConfig`, `LODOConfig`, `XAIConfig`) in `src/drdg/config.py`
- [ ] T007 [P] Implement deterministic seed management (Python, NumPy, PyTorch) in `src/drdg/utils/seed.py`
- [ ] T008 [P] Implement formatted console and file logging utilities in `src/drdg/utils/logging.py`
- [ ] T009 [P] Implement device detection and mixed-precision helpers in `src/drdg/utils/device.py`
- [ ] T010 [P] Implement atomic JSON and CSV file I/O helpers in `src/drdg/utils/io.py`
- [ ] T011 Implement package root version and top-level exports in `src/drdg/__init__.py` and `src/drdg/utils/__init__.py`

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Modular Research Package Architecture (`drdg`) (Priority: P1) 🎯 MVP

**Goal**: Extract and decouple streams, attention modules, fusion, loss functions, output heads, and base trainers into `src/drdg/` so that components can be imported, instantiated, and trained independently without notebook state.

**Independent Test**: Run `pip install -e .` and `pytest tests/test_model_variants.py tests/test_ordinal_heads.py -v` to verify package import, forward passes, tensor shapes, and loss computations on CPU.

### Tests for User Story 1
- [X] T012 [P] [US1] Unit test for model variants and forward tensor shapes in `tests/test_model_variants.py`
- [X] T013 [P] [US1] Unit test for ordinal heads and cutpoint monotonicity in `tests/test_ordinal_heads.py`

### Implementation for User Story 1
- [X] T014 [P] [US1] Implement CBAM (Channel & Spatial attention) in `src/drdg/models/attention/cbam.py`
- [X] T015 [P] [US1] Implement NonLocalBlock2D in `src/drdg/models/attention/non_local.py`
- [X] T016 [P] [US1] Implement QuadrantTokenAggregator in `src/drdg/models/attention/quadrant_tokens.py`
- [X] T017 [P] [US1] Implement attention package exports in `src/drdg/models/attention/__init__.py`
- [X] T018 [US1] Implement GlobalContextStream (ResNet-50 + stage-4 attention + projection) in `src/drdg/models/streams/global_context.py`
- [X] T019 [P] [US1] Implement GatedAttentionPool and LocalMILBranch in `src/drdg/models/streams/local_mil.py`
- [X] T020 [P] [US1] Implement streams package exports in `src/drdg/models/streams/__init__.py`
- [X] T021 [US1] Implement multi-scale dual-branch feature fusion in `src/drdg/models/fusion.py`
- [X] T022 [P] [US1] Implement Continuous Quadratic Weighted Kappa loss (`ContinuousQWKLoss`) in `src/drdg/losses/qwk.py`
- [X] T023 [P] [US1] Implement ordinal binary cross-entropy and rank losses in `src/drdg/losses/ordinal.py`
- [X] T024 [P] [US1] Implement losses package exports in `src/drdg/losses/__init__.py`
- [X] T025 [P] [US1] Implement SoftmaxCEHead and SoftmaxQWKHead in `src/drdg/models/heads/softmax.py`
- [X] T026 [P] [US1] Implement CORALHead in `src/drdg/models/heads/coral.py`
- [X] T027 [P] [US1] Implement CORNHead in `src/drdg/models/heads/corn.py`
- [X] T028 [P] [US1] Implement CumulativeLinkModelQWKHead (CLM) with softplus cutpoints in `src/drdg/models/heads/clm.py`
- [X] T029 [P] [US1] Implement heads package exports in `src/drdg/models/heads/__init__.py`
- [X] T030 [US1] Implement DRFullModel and ResNet50AblationModel in `src/drdg/models/full_model.py`
- [X] T031 [P] [US1] Implement paper architectural variants (V0 baseline, V1 Local MIL, V2 Global Context, V3 Ordinal CLM) in `src/drdg/models/variants/v0.py`, `v1.py`, `v2.py`, `v3.py`, and `src/drdg/models/variants/__init__.py`
- [X] T032 [US1] Implement unified model factory (`build_model`, `build_variant`) in `src/drdg/models/factory.py` and `src/drdg/models/__init__.py`
- [X] T033 [P] [US1] Implement checkpoint state save and resume helpers in `src/drdg/training/checkpoint.py`
- [X] T034 [P] [US1] Implement optimizer and lr scheduler builders (AdamW, CosineAnnealing, QWK warmup) in `src/drdg/training/optimization.py`
- [X] T035 [US1] Implement DRTrainer engine (AMP, gradient clipping, early stopping, best checkpoint tracking) in `src/drdg/training/trainer.py`
- [X] T036 [US1] Implement DRHeadTrainer for controlled representation benchmarks in `src/drdg/training/head_trainer.py` and `src/drdg/training/__init__.py`
- [X] T037 [US1] Implement CLI single-variant training entry point in `scripts/train_variant.py`

**Checkpoint**: Core architecture and training engine complete, verified via unit tests and CLI dry-run.

---

## Phase 4: User Story 2 - Harmonized Multi-Cohort Data Pipeline & Leakage-Free Splitting (Priority: P1)

**Goal**: Automate dataset acquisition, preparation, patient parsing with namespacing, circular FOV extraction, sliding-window patch caching, and deterministic leakage-free split generation across all six cohorts.

**Independent Test**: Run `pytest tests/test_patient_ids.py tests/test_patient_splits.py tests/test_lodo_splits.py tests/test_fov.py tests/test_patch_extraction.py -v`.

### Tests for User Story 2
- [X] T038 [P] [US2] Unit test for patient ID parsing conventions across all 6 cohorts in `tests/test_patient_ids.py`
- [X] T039 [P] [US2] Unit test for disjoint patient split verification in `tests/test_patient_splits.py`
- [X] T040 [P] [US2] Unit test for LODO held-out isolation and zero-leakage assertions in `tests/test_lodo_splits.py`
- [X] T041 [P] [US2] Unit test for morphological FOV mask computation in `tests/test_fov.py`
- [X] T042 [P] [US2] Unit test for sliding-window patch extraction and foreground filtering in `tests/test_patch_extraction.py`

### Implementation for User Story 2
- [X] T043 [P] [US2] Implement cohort metadata registry in `src/drdg/data/registry.py`
- [X] T044 [P] [US2] Implement canonical MasterFundusRecord schema and validation in `src/drdg/data/metadata.py`
- [X] T045 [P] [US2] Implement regex patient ID parser and namespace generator in `src/drdg/data/patient_ids.py`
- [X] T046 [P] [US2] Implement automated dataset downloaders (DDR, DeepDRiD, IDRiD, IDRiD segmentation, Messidor-2) in `src/drdg/data/downloaders/`
- [X] T047 [P] [US2] Implement dataset preparation modules (APTOS, DeepDRiD, IDRiD, IDRiD segmentation, resize) in `src/drdg/data/preparation/`
- [X] T048 [P] [US2] Implement circular non-black retinal ROI cropping and green channel extraction in `src/drdg/data/preprocessing.py`
- [X] T049 [P] [US2] Implement morphological FOV mask computation in `src/drdg/data/fov.py`
- [X] T050 [P] [US2] Implement native-resolution sliding-window patch extractor and foreground cache in `src/drdg/data/patches.py`
- [X] T051 [P] [US2] Implement global and patch image augmentations and normalization transforms in `src/drdg/data/transforms.py`
- [X] T052 [US2] Implement FundusDualBranchDataset in `src/drdg/data/datasets.py`
- [X] T053 [P] [US2] Implement dynamic MIL bag collate function (`trainer_compatible_collate_fn`) in `src/drdg/data/collate.py`
- [X] T054 [P] [US2] Implement training-only quota balancing in `src/drdg/data/balancing.py`
- [X] T055 [P] [US2] Implement patient-level stratified splitting (70/15/15) in `src/drdg/data/splits.py`
- [X] T056 [US2] Implement 6-fold LODO split generation and leakage audit in `src/drdg/data/lodo.py` and `src/drdg/data/__init__.py`
- [X] T057 [US2] Implement CLI dataset downloader script in `scripts/download_dataset.py`
- [X] T058 [US2] Implement CLI dataset preparation script in `scripts/prepare_dataset.py`
- [X] T059 [US2] Implement CLI master metadata builder script in `scripts/build_master_metadata.py`
- [X] T060 [US2] Implement CLI 6-fold LODO split generator script in `scripts/generate_lodo_splits.py`
- [X] T061 [P] [US2] Create data harmonization walkthrough notebook in `notebooks/00_dataset_harmonization.ipynb`
- [X] T062 [P] [US2] Create LODO split generation walkthrough notebook in `notebooks/01_lodo_split_generation.ipynb`
- [X] T063 [P] [US2] Create dataset audit and leakage verification notebook in `notebooks/02_dataset_audit.ipynb`

**Checkpoint**: Multi-cohort data pipeline complete, verified zero-leakage splits generated, and walkthrough notebooks ready.

---

## Phase 5: User Story 3 - Six-Fold Leave-One-Dataset-Out (LODO) Generalization Benchmark (Priority: P1)

**Goal**: Execute 6-fold LODO cross-domain evaluation across variants V0-V3, saving checkpoint-resumable progress and serializing tripartite results (`source_validation`, `source_test`, `target_test`).

**Independent Test**: Run `python scripts/run_lodo.py --help`, run dry-run LODO test, and validate generated JSON schemas against `tests/test_result_schema.py`.

### Tests for User Story 3
- [X] T064 [P] [US3] Unit test for tripartite metric schema validation in `tests/test_result_schema.py`

### Implementation for User Story 3
- [X] T065 [P] [US3] Implement classification metrics (QWK, accuracy, confusion matrix) in `src/drdg/evaluation/metrics.py`
- [X] T066 [P] [US3] Implement grade-specific metrics (per-grade sensitivity, within-1 accuracy) in `src/drdg/evaluation/grade_metrics.py`
- [X] T067 [P] [US3] Implement ordinal severity metrics and consistency checks in `src/drdg/evaluation/ordinal_metrics.py`
- [X] T068 [P] [US3] Implement paired Wilcoxon signed-rank test on matched LODO cohorts in `src/drdg/evaluation/statistics.py`
- [X] T069 [US3] Implement tripartite result serialization and schema validation in `src/drdg/evaluation/result_schema.py`
- [X] T070 [US3] Implement LODO evaluation orchestration engine in `src/drdg/evaluation/lodo.py` and `src/drdg/evaluation/__init__.py`
- [X] T071 [US3] Implement 6-fold LODO experiment runner in `src/drdg/experiments/lodo.py` and `src/drdg/experiments/__init__.py`
- [X] T072 [US3] Implement CLI 6-fold LODO runner script in `scripts/run_lodo.py`
- [X] T073 [P] [US3] Create V0 baseline demonstration notebook in `notebooks/10_v0_global_baseline.ipynb`
- [X] T074 [P] [US3] Create V1 Local MIL demonstration notebook in `notebooks/11_v1_local_mil.ipynb`
- [X] T075 [P] [US3] Create V2 Global Context demonstration notebook in `notebooks/12_v2_global_context.ipynb`
- [X] T076 [P] [US3] Create V3 Ordinal CLM demonstration notebook in `notebooks/13_v3_ordinal_clm.ipynb`
- [X] T077 [P] [US3] Create LODO benchmark result inspector notebook in `notebooks/20_lodo_benchmark.ipynb`
- [X] T078 [P] [US3] Create grade-specific analysis notebook in `notebooks/22_grade_specific_analysis.ipynb`

**Checkpoint**: 6-fold LODO evaluation engine operational with structured result serialization and demonstration notebooks.

---

## Phase 6: User Story 4 - Controlled Output-Head & Model Variant Ablation Suite (Priority: P2)

**Goal**: Train and benchmark the 5 output heads independently on standardized ResNet-50 representations and export machine-readable summaries.

**Independent Test**: Run `python scripts/run_head_ablation.py --help` and execute single head dry-run.

- [X] T079 [US4] Implement controlled head ablation experiment runner in `src/drdg/experiments/head_ablation.py`
- [X] T080 [US4] Implement CLI head ablation runner script in `scripts/run_head_ablation.py`
- [X] T081 [P] [US4] Create head ablation inspection and comparison notebook in `notebooks/21_head_ablation.ipynb`

**Checkpoint**: Controlled 5-head benchmark executable via CLI and inspectable via notebook.

---

## Phase 7: User Story 5 - Referable DR Screening Calibration & Quantitative Explainability (Priority: P2)

**Goal**: Calibrate RDR operating threshold at 95% specificity exclusively on source validation data, and compute quantitative XAI metrics against IDRiD pixel lesion masks.

**Independent Test**: Run `pytest tests/test_rdr.py -v` and `python scripts/run_xai.py --help`.

### Tests for User Story 5
- [X] T082 [P] [US5] Unit test for source-validation RDR threshold isolation in `tests/test_rdr.py`

### Implementation for User Story 5
- [X] T083 [US5] Implement RDR operating threshold calibration ($\ge 95\%$ specificity) and evaluation in `src/drdg/evaluation/rdr.py`
- [X] T084 [P] [US5] Implement Global Branch Grad-CAM in `src/drdg/explainability/gradcam.py`
- [X] T085 [P] [US5] Implement Local MIL Attention Saliency mapping in `src/drdg/explainability/mil_saliency.py`
- [X] T086 [P] [US5] Implement IDRiD lesion mask dataset and evaluator in `src/drdg/explainability/idrid_dataset.py`
- [X] T087 [P] [US5] Implement Pointing Game, Area-Matched Saliency Recall, and pixel AUROC in `src/drdg/explainability/metrics.py` and `src/drdg/explainability/__init__.py`
- [X] T088 [US5] Implement quantitative XAI experiment runner in `src/drdg/experiments/xai.py`
- [X] T089 [US5] Implement CLI XAI runner script in `scripts/run_xai.py`
- [X] T090 [P] [US5] Create RDR screening evaluation notebook in `notebooks/23_rdr_screening.ipynb`
- [X] T091 [P] [US5] Create IDRiD quantitative XAI evaluation notebook in `notebooks/24_xai_idrid.ipynb`

**Checkpoint**: RDR screening calibration and quantitative XAI evaluation operational and independently testable.

---

## Phase 8: User Story 6 - Public Open-Source Packaging, Security Audit, & Reproducibility (Priority: P3)

**Goal**: Perform repository-wide secret scrubbing, provide comprehensive documentation, set up GitHub Actions CI, and create paper figure export utilities.

**Independent Test**: Run repository credential scan, verify all docs render with valid links, and run `python scripts/export_paper_artifacts.py --help`.

- [X] T092 [P] [US6] Scrub all hardcoded Kaggle credentials (including `src/download-dataset/idrid-segmentation.py`) and replace with environment variables
- [X] T093 [US6] Create mandatory refactor audit log in `docs/REFACTOR_AUDIT.md` recording all observed code vs comment discrepancies
- [X] T094 [P] [US6] Create publication figure generation utilities in `src/drdg/visualization/paper_figures.py` and `src/drdg/visualization/__init__.py`
- [X] T095 [US6] Implement CLI paper artifacts export script in `scripts/export_paper_artifacts.py`
- [X] T096 [P] [US6] Create paper figures generation notebook in `notebooks/30_paper_figures.ipynb` and index in `notebooks/README.md`
- [X] T097 [P] [US6] Create data acquisition and licensing guide in `docs/DATA.md`
- [X] T098 [P] [US6] Create architecture and tensor shapes guide in `docs/ARCHITECTURE.md`
- [X] T099 [P] [US6] Create experimental protocols documentation in `docs/EXPERIMENTS.md`
- [X] T100 [P] [US6] Create reproducibility, hardware, and determinism notes in `docs/REPRODUCIBILITY.md`
- [X] T101 [P] [US6] Create results schema documentation in `docs/RESULTS_SCHEMA.md`
- [X] T102 [P] [US6] Create explainability methodology guide in `docs/XAI.md`
- [X] T103 [P] [US6] Create legacy migration code map in `docs/CODE_MAP.md`
- [X] T104 [P] [US6] Create data directory README in `data/README.md` and results README in `results/README.md`
- [X] T105 [P] [US6] Create GitHub Actions automated CPU test workflow in `.github/workflows/tests.yml`
- [X] T106 [P] [US6] Create paper citation metadata in `CITATION.cff` and open-source contribution guidelines in `CONTRIBUTING.md` and `LICENSE`
- [X] T107 [US6] Create comprehensive academic repository `README.md` with neutral scientific framing and paper-to-code mapping

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, output stripping, security audit, and end-to-end testing.

- [X] T108 Strip large cell execution outputs and personal paths from all demonstration notebooks in `notebooks/`
- [X] T109 Run full repository secret scan for `KAGGLE_KEY`, `KAGGLE_USERNAME`, tokens, passwords, and personal paths
- [X] T110 Execute full CPU pytest suite (`pytest tests/ -v`) and verify all test modules pass

---

## Dependencies & Execution Order

```
Phase 1: Setup (T001-T004)
     ↓
Phase 2: Foundational (T005-T011)
     ↓
+--------------------------------------------------------------------------------+
| User Story 1 (T012-T037)  --> Core Package Architecture & Factory (MVP)       |
+--------------------------------------------------------------------------------+
     ↓
+--------------------------------------------------------------------------------+
| User Story 2 (T038-T063)  --> Multi-Cohort Data Pipeline & Zero-Leakage Splits |
+--------------------------------------------------------------------------------+
     ↓
+--------------------------------------------------------------------------------+
| User Story 3 (T064-T078)  --> 6-Fold LODO Cross-Domain Evaluation              |
+--------------------------------------------------------------------------------+
     ↓
+----------------------------------------+---------------------------------------+
| User Story 4 (T079-T081) Head Ablation | User Story 5 (T082-T091) RDR & XAI   |
+----------------------------------------+---------------------------------------+
     ↓                                        ↓
+--------------------------------------------------------------------------------+
| User Story 6 (T092-T107)  --> Documentation, CI/CD, Secrets & Paper Figures    |
+--------------------------------------------------------------------------------+
     ↓
Phase 9: Polish & Final Quality Audit (T108-T110)
```

---

## Implementation Strategy & MVP Scope

1. **MVP Scope (Phase 1 to Phase 3)**:
   - Scaffold the `drdg` package.
   - Decouple all attention layers, feature streams, fusion, losses, heads, variants, and base trainers.
   - Run `pip install -e .` and `pytest tests/test_model_variants.py tests/test_ordinal_heads.py`.
   - Result: A working, importable Python package that builds and evaluates all model variants on dummy tensors.
2. **Increment 2 (Phase 4)**:
   - Implement data downloaders, preparation functions, patient regex parsing, circular FOV cropping, sliding-window patch extractor, and 6-fold LODO split generation.
   - Result: Deterministic, verified zero-leakage splits across the 6 cohorts.
3. **Increment 3 (Phase 5 to Phase 7)**:
   - Wire up LODO evaluation, controlled head ablation, RDR threshold calibration, and quantitative IDRiD XAI.
   - Result: Full experimental orchestration operational with structured JSON/CSV logging.
4. **Increment 4 (Phase 8 to Phase 9)**:
   - Security scrubbing, documentation (`README.md`, `docs/*.md`), CI workflow, clean demonstration notebooks, and final validation.
