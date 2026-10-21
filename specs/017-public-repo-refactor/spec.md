# Feature Specification: Public Academic Repository Refactor for Diabetic Retinopathy Domain Shift

**Feature Branch**: `017-public-repo-refactor`

**Created**: 2026-09-23

**Status**: Draft

**Input**: User description: "Refactor existing notebook-centric diabetic retinopathy research project into a clean, reproducible, well-documented, public GitHub repository ('deconstructing-dr-domain-shift') supporting the research paper 'Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression'."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Modular Research Package Architecture (`drdg`) (Priority: P1)

As a Machine Learning Research Engineer or academic peer reviewer, I want the core algorithmic, model, dataset, loss, training, and evaluation logic extracted from monolithic Jupyter notebooks into a cleanly structured, pip-installable Python package (`drdg`), so that I can import, test, and compose components programmatically without executing interactive notebooks or relying on hidden notebook state.

**Why this priority**: Decoupling the executable research code from Jupyter notebooks is the foundational architectural prerequisite for automated testing, continuous integration, command-line reproducibility, and clean academic open-sourcing.

**Independent Test**: Can be tested independently by running `pip install -e .`, importing `import drdg`, and instantiating core streams, attention layers, losses, and dataset classes in isolated CPU tests.

**Acceptance Scenarios**:
1. **Given** a clean Python environment with dependencies installed, **When** `pip install -e .` is executed from the repository root, **Then** `python -c "import drdg"` succeeds with exit code 0.
2. **Given** the `drdg` package installed, **When** components such as `GlobalContextStream`, `LocalMILBranch`, `ContinuousQWKLoss`, or `CumulativeLinkModelQWKHead` are imported in an external script or test, **Then** they instantiate and execute forward passes without relying on notebook global variables or path hacks.
3. **Given** any module within `src/drdg/`, **When** the module is imported, **Then** no file I/O, heavy training, or dataset downloading executes automatically at import time.

---

### User Story 2 - Harmonized Multi-Cohort Data Pipeline & Leakage-Free Splitting (Priority: P1)

As a researcher evaluating multi-source domain generalization across heterogeneous retinal cohorts, I want automated, deterministic pipelines for data acquisition, dataset-specific preparation, patient ID parsing with namespace collision prevention, native retinal non-black circular ROI cropping, sliding-window patch extraction, and stratified patient-level splitting, so that all six cohorts (APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS) are harmonized under a single verified metadata schema without patient-level data leakage.

**Why this priority**: In multi-cohort medical imaging, data leakage across patient eyes (bilateral correlation) or across training and evaluation splits completely invalidates clinical claims. Automated harmonization guarantees integrity and zero-leakage constraints.

**Independent Test**: Can be tested independently by running unit tests in `tests/test_patient_ids.py` and `tests/test_lodo_splits.py` to verify regex patient parsing, namespace prefixing (`dataset_key::patient_id`), and asserting zero patient intersection between train, validation, and test sets.

**Acceptance Scenarios**:
1. **Given** raw downloaded cohorts in `data/raw/`, **When** `scripts/prepare_dataset.py` is invoked for a cohort, **Then** it standardizes images into native-resolution non-black circular FOVs and produces standardized metadata with verified columns (`image_id`, `diagnosis`, `dataset_name`, `dataset_key`, `raw_full_path`, `native_path`, `patient_id`, `side`).
2. **Given** harmonized metadata across all cohorts, **When** `scripts/generate_lodo_splits.py` is executed, **Then** it generates deterministic 6-fold LODO split files where each held-out cohort is completely excluded from source training and source validation partitions.
3. **Given** any generated split (in-domain or LODO), **When** evaluated by automated split integrity checks, **Then** `Train_Patients ∩ Validation_Patients = ∅` and `Train_Patients ∩ Test_Patients = ∅` are strictly asserted.
4. **Given** cohorts with official download scripts (DDR, DeepDRiD, IDRiD, IDRiD segmentation, Messidor-2), **When** `scripts/download_dataset.py --dataset <key>` is called, **Then** it downloads and stages files into `data/raw/<key>`, while EyePACS is clearly documented as requiring manual third-party acquisition.

---

### User Story 3 - Six-Fold Leave-One-Dataset-Out (LODO) Generalization Benchmark (Priority: P1)

As a clinical AI researcher investigating cross-domain generalization, I want to execute a deterministic six-fold LODO benchmark across model variants V0 (Global Baseline), V1 (Local MIL), V2 (Global Relational Context), and V3 (Ordinal Cumulative Link Model), so that I can rigorously evaluate out-of-distribution performance when each clinical center is held out as an unseen target.

**Why this priority**: LODO validation is the central empirical pillar of the research paper ("Deconstructing Domain Shift in Diabetic Retinopathy Grading").

**Independent Test**: Can be tested independently by running `python scripts/run_lodo.py --variant v0 --held-out aptos --dry-run` or executing a single fold with synthetic data, verifying checkpoint resume, deterministic training-only quota balancing, and structured JSON output serialization.

**Acceptance Scenarios**:
1. **Given** pre-generated LODO fold definitions, **When** `python scripts/run_lodo.py --variant v2 --held-out messidor2` is executed, **Then** training runs on the remaining 5 cohorts with training-only balancing, model selection is performed strictly on source validation data, and zero-shot generalization is evaluated on 100% of the held-out cohort.
2. **Given** training execution across any fold, **When** an interruption occurs, **Then** the training engine successfully resumes from the latest fold checkpoint without corrupting previous epoch metrics or random state determinism.
3. **Given** completed fold evaluations, **When** results are serialized to `results/lodo/`, **Then** the JSON schema strictly separates `source_validation`, `source_test`, and `target_test` metrics, preventing validation metrics from being mislabeled as in-domain test results.

---

### User Story 4 - Controlled Output-Head & Model Variant Ablation Suite (Priority: P2)

As a machine learning researcher, I want to execute controlled, independent benchmarks across five classification and ordinal regression heads (`Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, `CLM_QWK`) on standardized ResNet-50 feature representations, and evaluate the stepwise architectural progression ($V0 \rightarrow V1 \rightarrow V2 \rightarrow V3$), so that I can quantify the exact empirical effect of loss formulations and architectural components under identical representations.

**Why this priority**: Isolating output-head loss behavior from feature backbone variations is critical for establishing whether ordinal formulation advantages are target-dependent or representation-dependent.

**Independent Test**: Can be tested independently by running `tests/test_ordinal_heads.py` to verify mathematical constraints (strictly monotonic cutpoints, finite probability normalization, output tensor shapes $[B, 5]$) and running `python scripts/run_head_ablation.py --head coral --dry-run`.

**Acceptance Scenarios**:
1. **Given** the standardized ResNet-50 ablation representation, **When** `python scripts/run_head_ablation.py --all-heads` is run, **Then** all five heads train independently using their respective loss functions (`ContinuousQWKLoss`, binary cross-entropy thresholds for CORAL/CORN, or ordinal link likelihood for CLM).
2. **Given** completed head ablation runs, **When** metrics are summarized, **Then** machine-readable results are saved to `results/head_ablation/summary.json` and `results/head_ablation/summary.csv` without subjective ranking labels or fabricated winners.
3. **Given** model factory builders, **When** `build_variant("v0")` through `build_variant("v3")` are called, **Then** the instantiated architectures compose shared modular streams (`GlobalContextStream`, `LocalMILBranch`, `DualBranchFusion`, `CumulativeLinkModelQWKHead`) matching the exact executable configurations from the research notebooks.

---

### User Story 5 - Referable DR Screening Calibration & Quantitative Explainability (Priority: P2)

As a clinician or AI evaluation specialist, I want to evaluate Referable Diabetic Retinopathy (RDR, $\text{Grade} \ge 2$) binary screening utility using decision thresholds calibrated strictly on source validation data at a fixed 95% clinical specificity, and evaluate spatial localization quality against IDRiD pixel-level lesion masks using quantitative XAI metrics (Pointing Game, Area-Matched Saliency Recall, pixel AUROC), so that clinical screening accuracy and lesion-grounded attention alignment are rigorously assessed.

**Why this priority**: RDR sensitivity at high specificity is the standard clinical benchmark for deploying automated tele-screening systems, while quantitative XAI provides localization evidence to verify that patch attention focuses on actual microaneurysms and hemorrhages.

**Independent Test**: Can be tested independently by running `tests/test_rdr.py` to ensure the decision threshold is computed exclusively on source validation logits and frozen before applying to target test sets, and running `python scripts/run_xai.py --dry-run`.

**Acceptance Scenarios**:
1. **Given** model predictions on source validation and held-out target test sets, **When** `evaluate_rdr` is called, **Then** the screening threshold is computed solely on source validation to satisfy $\ge 95\%$ specificity, and this frozen threshold is evaluated on target test data to report out-of-distribution sensitivity.
2. **Given** fundus images and corresponding IDRiD ground-truth lesion masks, **When** `scripts/run_xai.py` is executed, **Then** it generates dual-branch Grad-CAM and Attention MIL saliency maps, computing Pointing Game accuracy, Area-Matched Saliency Recall, and pixel AUROC.
3. **Given** any generated XAI documentation or report, **When** reviewed, **Then** it strictly adheres to neutral scientific framing ("localization agreement does not imply causal explanation").

---

### User Story 6 - Public Open-Source Packaging, Security Audit, & Reproducibility (Priority: P3)

As an open-source maintainer and academic reader, I want a clean, secure repository adhering to public open-source standards—including typed Python dataclass configurations (zero YAML), comprehensive English documentation, scrubbed API credentials and personal paths, automated CPU test suites, and clean demonstration notebooks—so that the codebase can be published alongside the paper without security leaks or technical debt.

**Why this priority**: Public release requires zero credential exposure (Kaggle keys, Google Drive paths), clear open-source licensing distinction between code and external dataset licenses, and clean documentation to ensure long-term community trust and reproducibility.

**Independent Test**: Can be tested independently by running repository-wide secret scanning (`grep` for credentials, tokens, and personal paths), executing `pytest tests/` in a clean environment, and validating documentation links in `README.md`.

**Acceptance Scenarios**:
1. **Given** the entire repository tree, **When** scanned for sensitive patterns (`KAGGLE_KEY`, `KAGGLE_USERNAME`, passwords, `/content/drive/MyDrive`, local Windows paths), **Then** zero active or commented-out credentials or personal paths are found.
2. **Given** a new environment, **When** `pytest tests/ -v` is executed on CPU, **Then** all unit tests pass within a few minutes without attempting to download multi-gigabyte datasets or initiate GPU training.
3. **Given** all demonstration notebooks in `notebooks/`, **When** inspected, **Then** cell execution outputs are stripped, all model/trainer implementations are imported from `drdg`, and self-contained tutorials for dataset harmonization, variant training, LODO benchmarking, and XAI inspection are provided.
4. **Given** all discrepancies identified between prose comments and executable code during refactoring, **When** `docs/REFACTOR_AUDIT.md` is inspected, **Then** every item is logged with original location, observed behavior, documentation behavior, and action taken.

---

### Edge Cases

- **Missing Dataset Downloads**: How does the system handle cohorts that cannot be automatically downloaded (e.g., EyePACS)?
  - *Resolution*: Download and preparation scripts clearly fail with descriptive, actionable error messages pointing users to `docs/DATA.md` for manual acquisition procedures.
- **Unbalanced Class Prevalence**: What happens when a held-out dataset has severe class skew (e.g., EyePACS with 73.5% Grade 0 vs. DeepDRiD with 34.0% Grade 0)?
  - *Resolution*: Target test sets preserve 100% natural clinical prevalence. Quota balancing is applied strictly to source training sets during LODO.
- **Zero-Lesion Normal Images in XAI**: How does quantitative XAI handle Grade 0 (no lesion) images during lesion recall evaluation?
  - *Resolution*: Quantitative lesion-mask evaluation is scoped exclusively to annotated lesion subsets (IDRiD segmentation split containing microaneurysms, hemorrhages, exudates).
- **Early Interruption & Checkpoint Recovery**: What happens if a multi-hour LODO fold run is terminated mid-epoch?
  - *Resolution*: `DRTrainer` saves both `latest_checkpoint.pt` (with complete epoch, optimizer, scaler, and scheduler states) and `best_checkpoint.pt`. Passing `--resume` recovers the exact epoch state.
- **Hardware Heterogeneity**: How does the codebase behave on CPU-only machines vs. CUDA GPUs with differing tensor core support (FP16 vs. BF16)?
  - *Resolution*: `drdg.utils.device` and `TrainingConfig` dynamically detect hardware capabilities, defaulting to `float32` on CPU, `float16` on older GPUs, and `bfloat16` when native hardware support is detected.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Core Architecture & Modular Package (`src/drdg`)
- **FR-001**: System MUST provide a top-level installable Python package `drdg` configurable via `pyproject.toml` supporting `pip install -e .`.
- **FR-002**: System MUST implement configuration management using typed Python `dataclasses` in `src/drdg/config.py` (`DataConfig`, `ModelConfig`, `TrainingConfig`, `LocalMILConfig`, `LODOConfig`, `XAIConfig`). Configuration via YAML, JSON, or Hydra is strictly prohibited.
- **FR-003**: System MUST expose CLI parameter overrides via `argparse` across all entry-point scripts in `scripts/`.
- **FR-004**: System MUST define all standard repository directories and file path resolution helpers in `src/drdg/paths.py`.

#### Data Ingestion, Harmonization & Splitting (`drdg.data`)
- **FR-005**: System MUST provide modular dataset downloaders in `src/drdg/data/downloaders/` for DDR, DeepDRiD, IDRiD, IDRiD segmentation, and Messidor-2, with no automatic execution upon module import.
- **FR-006**: System MUST document manual acquisition procedures for EyePACS in `docs/DATA.md` without fabricating automated download code.
- **FR-007**: System MUST provide reusable data preparation functions in `src/drdg/data/preparation/` (`aptos.py`, `deepdrid.py`, `idrid.py`, `idrid_segmentation.py`, `resize.py`).
- **FR-008**: System MUST parse patient identifiers using regex conventions for each cohort in `src/drdg/data/patient_ids.py` and enforce namespacing (`dataset_key::patient_id`) to prevent cross-cohort identifier collisions.
- **FR-009**: System MUST enforce a canonical master metadata schema across all cohorts: `image_id`, `diagnosis`, `dataset_name`, `dataset_key`, `raw_full_path`, `native_path`, `patient_id`, `side`.
- **FR-010**: System MUST validate that all diagnosis labels strictly belong to the integer set $\{0, 1, 2, 3, 4\}$ matching the ordered ICDRS scale.
- **FR-011**: System MUST implement circular non-black retinal ROI cropping, green-channel extraction, and morphological FOV mask computation in `src/drdg/data/fov.py` and `src/drdg/data/preprocessing.py`.
- **FR-012**: System MUST implement native-resolution sliding-window patch extraction, FOV-based foreground filtering, and patch caching in `src/drdg/data/patches.py`.
- **FR-013**: System MUST provide dual-branch dataset handling in `src/drdg/data/datasets.py` (`FundusDualBranchDataset`) and dynamic bag collation in `src/drdg/data/collate.py`.
- **FR-014**: System MUST generate deterministic 6-fold LODO split files in `src/drdg/data/lodo.py` guaranteeing zero patient overlap between source train, source validation, and held-out target test sets.

#### Model Components & Variants (`drdg.models`)
- **FR-015**: System MUST implement global context attention modules in `src/drdg/models/attention/`: Channel Attention, Spatial Attention, CBAM (`cbam.py`), NonLocalBlock2D (`non_local.py`), and QuadrantTokenAggregator (`quadrant_tokens.py`).
- **FR-016**: System MUST implement the Global Context stream (`GlobalContextStream` / `GlobalBranch`) in `src/drdg/models/streams/global_context.py` using ResNet-50 as the feature backbone.
- **FR-017**: System MUST implement the Local MIL stream (`LocalMILBranch`) with Gated Attention Pooling (`GatedAttentionPool`) in `src/drdg/models/streams/local_mil.py`.
- **FR-018**: System MUST implement multi-scale dual-branch feature fusion in `src/drdg/models/fusion.py`.
- **FR-019**: System MUST implement the five required output heads in `src/drdg/models/heads/`: `SoftmaxCEHead`, `SoftmaxQWKHead`, `CORALHead`, `CORNHead`, and `CumulativeLinkModelQWKHead` (CLM).
- **FR-020**: System MUST implement ordinal and QWK loss functions in `src/drdg/losses/` (`ContinuousQWKLoss`, ordinal binary cross-entropy, and CLM negative log-likelihood).
- **FR-021**: System MUST provide paper-aligned model builders in `src/drdg/models/variants/`:
  - `v0.py`: Global ResNet-50 baseline without local MIL.
  - `v1.py`: V0 + Native-resolution Local MIL branch.
  - `v2.py`: V1 + Global relational context/attention package (CBAM, Non-Local, Quadrant Tokens).
  - `v3.py`: V2 + Cumulative Link Model (CLM) ordinal head.
- **FR-022**: System MUST provide unified model factories in `src/drdg/models/factory.py` supporting variant creation and head ablation instantiation.

#### Training Engine (`drdg.training`)
- **FR-023**: System MUST provide a unified, robust training engine in `src/drdg/training/trainer.py` (`DRTrainer`) and `src/drdg/training/head_trainer.py` (`DRHeadTrainer`).
- **FR-024**: System MUST support Automatic Mixed Precision (AMP) with fallback handling across `float32`, `float16`, and `bfloat16`.
- **FR-025**: System MUST support checkpoint saving, resume capability, gradient clipping, early stopping, and QWK warmup scheduling.

#### Evaluation & Metrics (`drdg.evaluation`)
- **FR-026**: System MUST compute Quadratic Weighted Kappa (QWK), overall accuracy, per-grade sensitivity, and Within-1 Accuracy in `src/drdg/evaluation/metrics.py` and `src/drdg/evaluation/grade_metrics.py`.
- **FR-027**: System MUST implement Referable DR (Grade $\ge 2$) binary screening calibration in `src/drdg/evaluation/rdr.py`, ensuring decision thresholds are calibrated exclusively on source validation data at 95% specificity and frozen prior to target testing.
- **FR-028**: System MUST implement paired Wilcoxon signed-rank hypothesis testing for matched LODO cohorts in `src/drdg/evaluation/statistics.py`.
- **FR-029**: System MUST serialize evaluation outputs according to a strict nested schema (`source_validation`, `source_test`, `target_test`) in `src/drdg/evaluation/result_schema.py`.

#### Explainability & Publication Artifacts (`drdg.explainability` & `drdg.visualization`)
- **FR-030**: System MUST implement Grad-CAM for the global branch and Attention MIL patch saliency in `src/drdg/explainability/gradcam.py` and `src/drdg/explainability/mil_saliency.py`.
- **FR-031**: System MUST compute quantitative XAI metrics against IDRiD pixel lesion masks (Pointing Game accuracy, Area-Matched Saliency Recall, pixel AUROC) in `src/drdg/explainability/metrics.py`.
- **FR-032**: System MUST generate publication-ready figures from serialized result files in `src/drdg/visualization/paper_figures.py` and `scripts/export_paper_artifacts.py`.

#### Security, Auditing & Documentation
- **FR-033**: System MUST completely scrub all API credentials, tokens, personal directories, and machine-specific absolute paths from all codebase files, notebooks, and outputs.
- **FR-034**: System MUST provide `.env.example` with empty credential placeholders (`KAGGLE_USERNAME=`, `KAGGLE_KEY=`).
- **FR-035**: System MUST provide a comprehensive audit report in `docs/REFACTOR_AUDIT.md` logging all discrepancies discovered between executable code and prose documentation.
- **FR-036**: System MUST provide comprehensive documentation in `docs/` (`DATA.md`, `ARCHITECTURE.md`, `EXPERIMENTS.md`, `REPRODUCIBILITY.md`, `RESULTS_SCHEMA.md`, `XAI.md`, `CODE_MAP.md`).
- **FR-037**: System MUST provide lightweight, self-contained demonstration notebooks in `notebooks/` (00 to 30) that import from `drdg` with all large cell outputs stripped.
- **FR-038**: System MUST provide fast CPU unit tests in `tests/` covering patient parsing, split leakage prevention, model shapes, ordinal heads, RDR calibration, and result schemas.
- **FR-039**: System MUST provide a GitHub Actions workflow in `.github/workflows/tests.yml` executing CPU tests on pull requests and pushes.

---

### Key Entities

```
+---------------------------------------------------------------------------------+
|                                 MasterRecord                                    |
+---------------------------------------------------------------------------------+
| image_id: str                   | Unique image identifier                       |
| diagnosis: int                  | ICDRS ground-truth grade in {0, 1, 2, 3, 4}   |
| dataset_name: str               | Formal cohort name (e.g. 'APTOS 2019')        |
| dataset_key: str                | Internal slug: 'aptos', 'ddr', etc.          |
| raw_full_path: Path             | Location of raw image                         |
| native_path: Path               | Location of cropped circular ROI image        |
| patient_id: str                 | Namespaced patient ID (dataset_key::orig_id)  |
| side: str                       | Eye lateral identity ('left', 'right', 'unk') |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                                 PatientSplit                                    |
+---------------------------------------------------------------------------------+
| split_id: str                   | Identifier for split configuration            |
| fold_idx: Optional[int]         | LODO fold index in 0..5                       |
| held_out_target: Optional[str]  | Target cohort key for LODO                    |
| train_patient_ids: Set[str]     | Patient IDs in training partition             |
| val_patient_ids: Set[str]       | Patient IDs in validation partition           |
| test_patient_ids: Set[str]      | Patient IDs in evaluation partition           |
| invariant: Disjoint sets        | Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                                 LODOResult                                      |
+---------------------------------------------------------------------------------+
| variant: str                    | Model variant: 'v0', 'v1', 'v2', 'v3'         |
| held_out_dataset: str           | Evaluation cohort key                         |
| fold: int                       | Fold index 0..5                               |
| source_validation: MetricBundle | Validation metrics used strictly for selection|
| source_test: MetricBundle       | Source-domain test benchmark metrics          |
| target_test: MetricBundle       | Zero-shot target generalization metrics       |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                                MetricBundle                                     |
+---------------------------------------------------------------------------------+
| qwk: float                      | Quadratic Weighted Kappa                      |
| accuracy: float                 | Overall classification accuracy               |
| within_1_accuracy: float        | Proportion of predictions within ±1 grade     |
| per_grade_sensitivity: Dict     | Sensitivity per grade 0..4                    |
| rdr_auc: float                  | Area Under ROC Curve for Grade >= 2           |
| rdr_sens_at_95_spec: float      | Sensitivity at frozen 95% specificity threshold
+---------------------------------------------------------------------------------+
```

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of core data, model, training, and evaluation logic is successfully refactored out of the 5 original notebooks into reusable modules in `src/drdg/`.
- **SC-002**: Package installation via `pip install -e .` and package import via `python -c "import drdg"` succeed with zero errors.
- **SC-003**: 100% of unit tests in `tests/` pass on CPU in under 180 seconds, verifying patient ID parsing, patient-level split leakage absence, model variant forward passes, ordinal cutpoint monotonicity, RDR threshold isolation, and result schemas.
- **SC-004**: Zero patient overlap is mathematically verified across all 6 LODO folds (`Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`).
- **SC-005**: All 8 command-line scripts in `scripts/` provide functional `--help` documentation and support non-interactive execution.
- **SC-006**: 100% of credentials, tokens, personal Google Drive paths, and machine-specific local usernames are scrubbed from the repository.
- **SC-007**: RDR screening calibration operates with 0% target label leakage (decision threshold calibrated exclusively on source validation data).
- **SC-008**: All demonstration notebooks in `notebooks/` (00 to 30) have execution cell outputs stripped and contain zero redefined model/trainer class blocks.
- **SC-009**: 100% of discrepancies between prose comments and executable code discovered during refactoring are documented with structured audit records in `docs/REFACTOR_AUDIT.md`.
- **SC-010**: All documentation files (`README.md`, `docs/*.md`, `CITATION.cff`) are written in rigorous, professional English adhering to neutral scientific framing guidelines.

---

## Assumptions

1. **Third-Party Dataset Licensing**: Raw fundus datasets remain the intellectual property of their original clinical providers. The repository will not redistribute raw image binaries, but provides automated downloaders for public subsets where authorized and detailed manual acquisition instructions for others (e.g., EyePACS).
2. **Computational Hardware**: Primary model training and multi-fold LODO benchmarks require CUDA-compatible GPU resources (e.g., NVIDIA RTX 3090, A100, or Colab T4/V100). The package, unit tests, and data metadata utilities must operate fully on CPU.
3. **Reproducibility Determinism**: While random seeds are strictly fixed for Python, NumPy, and PyTorch, cross-hardware CUDA non-determinism (e.g., atomic operations in convolution backpropagation) prevents exact bitwise numerical equivalence across different GPU architectures.
4. **Notebooks as Documentation**: Clean notebooks are intended for visual walkthroughs, smoke testing, and artifact reproduction; automated batch experiments and training runs belong in CLI scripts.
