# Feature Specification: Restore Exact Notebook-to-Python Logic Parity for the DR Research Codebase

**Feature Branch**: `018-restore-notebook-parity`

**Created**: 2026-09-24

**Status**: Draft

**Input**: User description: "Restore Exact Notebook-to-Python Logic Parity for the DR Research Codebase: Audit the repository and correct the entire modular Python implementation under `drdg` so that it faithfully matches the executable logic contained in the 5 canonical research notebooks (`merge-dataset.ipynb`, `merge_dataset_lodo.ipynb`, `main_model (6).ipynb`, `main_model_lodo.ipynb`, `main_model_head_ablation (3).ipynb`). Ensure notebooks remain complete, self-contained historical research artifacts, while `drdg` provides an exact modular pure-Python duplicate with identical semantics, algorithms, and 3-split LODO partitioning."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Canonical Dual-Implementation Model & Notebook Preservation (Priority: P1)

As a Principal Machine Learning Research Engineer and academic peer reviewer, I want the five original research notebooks preserved completely intact and self-contained (retaining all their model classes, data pipelines, training loops, and inline utilities) while `drdg` serves as an independent, modular pure-Python duplicate of that exact logic, so that historical notebook provenance is never destroyed and researchers can verify research claims either by running the canonical Google Colab notebooks or by importing the pure-Python `drdg` package.

**Why this priority**: Modifying original notebooks into thin wrappers breaks provenance, invalidates recorded execution outputs/traces, and violates the fundamental research mandate. The repository must maintain both layers in full fidelity.

**Independent Test**: Can be tested independently by verifying git diffs on the five canonical notebooks (zero lines changed in notebook code cells) while confirming all required models, datasets, losses, and training routines can be imported from `drdg` and executed standalone.

**Acceptance Scenarios**:
1. **Given** the 5 canonical notebooks (`merge-dataset.ipynb`, `merge_dataset_lodo.ipynb`, `main_model (6).ipynb`, `main_model_lodo.ipynb`, `main_model_head_ablation (3).ipynb`), **When** an automated integrity check is run, **Then** all code cells remain byte-for-byte identical to their golden state with zero refactoring into wrapper imports.
2. **Given** any research component implemented in a canonical notebook, **When** corresponding modules in `drdg` are inspected, **Then** `drdg` provides a modular pure-Python implementation of that exact algorithm, parameter set, and control flow.
3. **Given** repository documentation (README, architecture guides, data guides), **When** a user reads the architecture description, **Then** the dual-layer design is explicitly stated: notebooks are complete canonical research artifacts; `drdg` is the modular pure-Python duplicate.

---

### User Story 2 - Exact 3-Split LODO Dataset Generation & Bipartite Partition Parity (Priority: P1)

As a researcher benchmarking cross-domain generalization, I want the Leave-One-Dataset-Out (LODO) data generation and model evaluation pipeline to generate and consume exactly 3 partitions per fold (`train`, `val`, `test`) matching `merge_dataset_lodo.ipynb` and `data/processed/lodo_folds/`, rather than an artificial 4-way split (`source_train`, `source_val`, `source_test`, `target_test`), so that training and evaluation follow the authentic experimental protocol of the paper.

**Why this priority**: Inventing non-existent partitions (`source_test`) or misnaming fold CSVs breaks compatibility with pre-generated data files, alters dataset sample sizes, and invalidates cross-cohort evaluation semantics.

**Independent Test**: Can be tested independently by running `tests/parity/test_lodo_split_parity.py` against `data/processed/lodo_folds/`, asserting exact CSV filenames (`fold_{fold_idx}_{fold_slug}_{split}.csv`), verifying that each fold contains exactly 3 files (`train`, `val`, `test`), and confirming zero patient overlap across splits.

**Acceptance Scenarios**:
1. **Given** the 6 clinical cohorts (APTOS, IDRiD, Messidor-2, DDR, EyePACS, DeepDRiD), **When** LODO split generation is executed, **Then** it produces exactly 18 CSV files (6 folds × 3 splits: `train`, `val`, `test`) in `data/processed/lodo_folds/` following pattern `fold_{fold_idx}_{fold_slug}_{split}.csv`.
2. **Given** any fold `fold_idx`, **When** split compositions are verified, **Then**:
   - `train`: Balanced source pool from 5 remaining cohorts (85% of source patients, balanced via quota sampling).
   - `val`: In-domain source validation from 5 remaining cohorts (15% of source patients, natural prevalence).
   - `test`: 100% of the held-out cohort (natural prevalence).
   - No `source_test` or `target_test.csv` file is generated or expected.
3. **Given** the LODO model training runner (`drdg.experiments.lodo` and `scripts/run_lodo.py`), **When** executed for any fold, **Then** it loads `fold_{fold_idx}_{cohort}_train.csv`, `fold_{fold_idx}_{cohort}_val.csv`, and `fold_{fold_idx}_{cohort}_test.csv`, optimizes on `train`, validates on `val` (calibrating the RDR operating threshold $\tau^*$), freezes $\tau^*$, and evaluates on `test`.

---

### User Story 3 - End-to-End Research Workflow & Execution Dependency Parity (Priority: P1)

As a researcher reproducing the complete experimental study from scratch, I want the repository to document and provide an exact end-to-end execution dependency graph mirroring the original Google Colab workflow (Raw Acquisition $\to$ Total Prep $\to$ Merge Dataset $\to$ Native Pool $\to$ Standard Model / Head Ablation $\to$ LODO Splits $\to$ LODO Training), so that every intermediate artifact has a verified producer and consumer.

**Why this priority**: Workflow mismatches cause silent errors, missing files, or out-of-order execution that destroys research reproducibility.

**Independent Test**: Can be tested independently by walking the producer-consumer dependency table in `docs/DATA_FLOW_AUDIT.md` and validating that every intermediate data file produced by upstream steps satisfies the schema and path requirements of downstream steps.

**Acceptance Scenarios**:
1. **Given** raw datasets staged in `data/raw/`, **When** dataset-specific consolidation is run (`total_aptos.py`, `total_deepdrid.py`, `total_idrid.py`, `total_idrid-segmentation.py`), **Then** it generates the exact consolidated files (`*_total.csv` and `*_total_images/`) required by `merge-dataset.ipynb`.
2. **Given** consolidated dataset files, **When** standard merge execution occurs (`merge-dataset.ipynb` or `scripts/build_master_metadata.py`), **Then** it produces the harmonized native image pool (`data/raw/hybrid/native_cropped/`) and standard split metadata (`train_balanced_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`).
3. **Given** the harmonized native image pool, **When** LODO generation occurs (`merge_dataset_lodo.ipynb` or `scripts/generate_lodo_splits.py`), **Then** it verifies that all native image files exist before generating the 18 frozen fold CSV files.
4. **Given** pre-generated fold CSVs, **When** LODO training runs (`main_model_lodo.ipynb` or `scripts/run_lodo.py`), **Then** it consumes the pre-generated splits directly without recalculating or modifying fold membership.
5. **Given** IDRiD Segmentation data, **When** prepared, **Then** it remains a separate XAI/localization evaluation pathway that does not pollute the 6-cohort disease grading metadata.

---

### User Story 4 - Numerical, Architectural & Training-Step Parity (Priority: P2)

As an ML engineer, I want all neural network modules (CBAM, NonLocalBlock2D, QuadrantTokenAggregator, GlobalContextStream, GatedAttentionPool, LocalMILBranch, FusionNeck, 5 output heads, Continuous QWK loss, CLM NLL) and training routines (AdamW parameter groups, Cosine Annealing, QWK warmup schedule, early stopping) in `drdg` to match the exact mathematical formulation, parameter shapes, tensor semantics, and weight updates of the notebooks, so that models trained with `drdg` produce equivalent scientific results.

**Why this priority**: Subtleties such as gradient clipping behavior, parameter group learning rates (backbone vs new layers), cosine scheduler $T_{max}$, or QWK loss weighting warmup fundamentally determine convergence and QWK performance.

**Independent Test**: Can be tested independently by running deterministic parity tests in `tests/parity/` comparing forward activations, loss scalar values, backward gradients, and one-step parameter updates between notebook implementations and `drdg` modules on identical CPU inputs and weights.

**Acceptance Scenarios**:
1. **Given** identical dummy inputs and model weights, **When** forward passes are executed through Global Branch, Local MIL Branch, Fusion, and each Output Head (`Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, `CLM_QWK`), **Then** max absolute difference between notebook and `drdg` output tensors is $< 1\times 10^{-6}$ in FP32.
2. **Given** identical predictions and integer ground-truth labels, **When** loss and backward gradients are computed for `ContinuousQWKLoss`, `CORAL`, `CORN`, and `CLM_QWK`, **Then** scalar losses match to within $1\times 10^{-6}$ and parameter gradients match to within $1\times 10^{-5}$.
3. **Given** identical initial weights, a mini-batch, and AdamW optimizer, **When** a single training step (forward $\to$ loss $\to$ backward $\to$ clip $\to$ step) is executed, **Then** model parameters after the step match to within numerical precision.
4. **Given** training execution over multiple epochs, **When** the learning rate and QWK loss warmup coefficient $\lambda(t)$ are logged, **Then** their trajectories match the exact notebook schedule step-for-step.

---

### User Story 5 - Code Mapping, Discrepancy Auditing & Transparent Status Reporting (Priority: P3)

As an auditor or reader of the paper, I want comprehensive audit documentation (`docs/NOTEBOOK_CODE_MAP.md`, `docs/REFACTOR_AUDIT.md`, `docs/DATA_FLOW_AUDIT.md`) mapping every notebook symbol to its `drdg` counterpart, detailing every discrepancy identified and resolved, and declaring the final verified integrity status, so that no hidden assumptions or unverified components exist.

**Why this priority**: Open scientific software requires total transparency regarding what has been validated, what was modified for portability/security, and what remains unverified.

**Independent Test**: Can be tested independently by inspecting the audit documents and checking that every research-critical notebook function/class has an explicit status (`EXACT`, `NUMERICALLY_EQUIVALENT`, `STRUCTURALLY_EQUIVALENT`, `PORTABILITY_FIX`), with zero unrecorded changes.

**Acceptance Scenarios**:
1. **Given** all 5 canonical notebooks, **When** `docs/NOTEBOOK_CODE_MAP.md` is reviewed, **Then** every major class, function, loss, and configuration dictionary is mapped to its `drdg` module and symbol with documented parity status.
2. **Given** `docs/REFACTOR_AUDIT.md`, **When** reviewed, **Then** all past discrepancies (including the corrected 4-way vs 3-way LODO split issue, result schema naming, and path handling) are fully documented with root causes and fixes.
3. **Given** the audit completion criteria, **When** all parity tests pass and all discrepancies are resolved, **Then** the final integrity status is set to `VERIFIED` with supporting test execution logs.

---

### Edge Cases

- **Missing Third-Party Cohorts (EyePACS)**: EyePACS has no official automated downloader script. The system must gracefully report missing raw archives with clear manual staging instructions rather than fabricating an invalid download endpoint.
- **Corrupted or Incomplete LODO Folds**: If fewer than 18 fold CSV files are detected in `data/processed/lodo_folds/`, the training runner must halt with a specific error message identifying which `fold_{fold_idx}_{cohort}_{split}.csv` file is missing.
- **Variable Aspect Ratios & Black Borders**: Retinal fundus images with eccentric or non-circular black borders must be properly masked using Otsu thresholding + largest contour detection + bounding box cropping to avoid empty or distorted patches.
- **Extreme Class Imbalance in Rare Cohorts**: During LODO source pool balancing, grade 4 or grade 3 instances may be scarce in certain cohort combinations; the quota balancer must replicate the notebook's replacement/oversampling logic deterministically without crash.
- **Training Interruptions & Checkpoint Recovery**: Resuming from a checkpoint must restore model weights, optimizer states, epoch counter, and best validation QWK without resetting the scheduler or duplicating log entries.
- **Referable DR Threshold Calibration**: If validation predictions result in tied sensitivity/specificity points, the operating threshold $\tau^*$ selection must use the exact tie-breaking rule of `main_model_lodo.ipynb` (first threshold meeting $\ge 80\%$ specificity).

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST strictly preserve all 5 canonical notebooks (`merge-dataset.ipynb`, `merge_dataset_lodo.ipynb`, `main_model (6).ipynb`, `main_model_lodo.ipynb`, `main_model_head_ablation (3).ipynb`) in their original self-contained form without extracting logic into imports or altering code cells.
- **FR-002**: System MUST provide a modular, pip-installable pure-Python package (`drdg`) under `src/drdg/` that faithfully reproduces the complete research logic of the canonical notebooks.
- **FR-003**: System MUST implement Leave-One-Dataset-Out (LODO) split generation in `drdg.data.lodo` that generates exactly 3 CSV files per fold (`train`, `val`, `test`), matching `merge_dataset_lodo.ipynb`.
- **FR-004**: System MUST name LODO fold CSVs strictly according to the pattern `fold_{fold_idx}_{fold_slug}_{split}.csv` (e.g., `fold_0_aptos_train.csv`, `fold_0_aptos_val.csv`, `fold_0_aptos_test.csv`), residing in `data/processed/lodo_folds/`.
- **FR-005**: System MUST eliminate all references to non-existent 4th splits (`source_test`, `target_test.csv`) across data loaders, experiment runners, evaluation schemas, and tests.
- **FR-006**: System MUST enforce bipartite evaluation in LODO experiments: in-domain source validation (`val`) for checkpoint selection and RDR threshold calibration, and out-of-distribution target test (`test`) for final generalization evaluation.
- **FR-007**: System MUST enforce patient-level isolation across all split pipelines, ensuring `Train_Patients ∩ Validation_Patients = ∅` and `Train_Patients ∩ Test_Patients = ∅` for both standard and LODO splits.
- **FR-008**: System MUST implement dataset-specific preparation scripts matching original behavior (`total_aptos.py`, `total_deepdrid.py`, `total_idrid.py`, `total_idrid-segmentation.py`).
- **FR-009**: System MUST replicate the multi-cohort harmonization pipeline from `merge-dataset.ipynb`, generating non-black circular ROI crops in `data/raw/hybrid/native_cropped/` and standard metadata files.
- **FR-010**: System MUST implement the Global Context Stream (`ResNet-50` backbone, `CBAM` convolutional block attention, `NonLocalBlock2D` self-attention, and `QuadrantTokenAggregator` spatial pooling) identical to `main_model (6).ipynb`.
- **FR-011**: System MUST implement the Local MIL Stream (`EfficientNet-B0` backbone, high-resolution native patch extraction, gated attention pooling, and instance-to-bag aggregation) identical to `main_model (6).ipynb`.
- **FR-012**: System MUST implement the Multiscale Fusion Neck (concatenation of global and local embeddings, LayerNorm, Linear projection, GeLU/ReLU activation, dropout) matching notebook dimensions.
- **FR-013**: System MUST implement all five output head formulations:
  - `Softmax_CE`: 5-class nominal classification with Cross-Entropy.
  - `Softmax_QWK`: 5-class classification with hybrid Cross-Entropy + Continuous QWK loss.
  - `CORAL`: Consistent Rank Logits ordinal head with binary cross-entropy.
  - `CORN`: Conditional Ordinal Representations for Neural Networks.
  - `CLM_QWK`: Cumulative Link Model (probit/logistic ordinal regression) with cumulative link NLL + Continuous QWK loss.
- **FR-014**: System MUST implement the Continuous QWK loss with exact matrix normalization, quadratic penalty weighting, and numerical stability epsilons identical to notebook cell implementations.
- **FR-015**: System MUST implement exact definitions for model ablation variants in LODO:
  - `V0`: Global Context Stream only (Softmax CE).
  - `V1`: Dual-Branch (Global ResNet-50 + Local MIL EfficientNet-B0) with Softmax CE.
  - `V2`: Dual-Branch with Global Relational Context (CBAM + Non-Local + Quadrant Tokens) with Softmax CE.
  - `V3`: Full Proposed Architecture: Dual-Branch with Relational Context + Cumulative Link Model (CLM) + Continuous QWK Loss.
- **FR-016**: System MUST implement the exact two-tier optimizer configuration (differential learning rates for pretrained backbones vs newly initialized attention/heads) and Cosine Annealing scheduler.
- **FR-017**: System MUST replicate the linear QWK warmup schedule $\lambda(t)$ matching the notebook's step/epoch stepping behavior.
- **FR-018**: System MUST implement patient-level early stopping monitoring validation QWK with checkpoint saving and restore-best logic.
- **FR-019**: System MUST evaluate Referable Diabetic Retinopathy (RDR, Grade $\ge 2$) by selecting operating threshold $\tau^*$ on source validation data to meet target specificity ($\ge 80\%$) and evaluating sensitivity on held-out test data without label leakage.
- **FR-020**: System MUST serialize evaluation metrics to JSON adhering to the bipartite schema (`in_domain_val` and `ood_test`), including per-grade sensitivity, overall accuracy, Within-1 accuracy, QWK, and RDR metrics.
- **FR-021**: System MUST implement Explainable AI (XAI) modules (Grad-CAM on global feature maps and attention weight splatting on local patches) matching the notebook implementation.
- **FR-022**: System MUST provide parity test suites in `tests/parity/` covering patient parsing, split generation, FOV cropping, patch extraction, forward passes, custom losses, backward gradients, and one-step training updates.
- **FR-023**: System MUST generate `docs/NOTEBOOK_CODE_MAP.md` mapping every notebook symbol to its `drdg` counterpart with status tags.
- **FR-024**: System MUST generate `docs/DATA_FLOW_AUDIT.md` detailing the producer-consumer dependency chain for all raw, intermediate, and processed artifacts.
- **FR-025**: System MUST generate `docs/REFACTOR_AUDIT.md` providing a comprehensive report on all audited items, discrepancies resolved, and the final integrity certification status.

---

### Key Entities *(include if feature involves data)*

- **CanonicalNotebook**: A primary source-of-truth Jupyter notebook containing executable research code, markdown documentation, and historical outputs.
- **LODOFold**: A six-fold cross-domain evaluation unit comprising 5 source cohorts and 1 held-out target cohort, partitioned into exactly 3 sets: `train`, `val`, `test`.
- **FundusImageRecord**: A harmonized metadata record identifying an eye image, containing `image_id`, `patient_id`, `dataset_key`, `side`, `diagnosis` (0-4), `raw_path`, and `native_path`.
- **PatchBag**: A collection of high-resolution native-resolution crops extracted from the retinal FOV, characterized by patch size (224×224), primary/fallback stride, and FOV foreground occupancy threshold.
- **DualBranchModel**: A PyTorch neural network combining a Global Context Stream (256×256 or 512×512) and a Local MIL Stream (bag of 224×224 patches) fused through a projection neck into an ordinal output head.
- **LODOFoldResult**: A structured evaluation artifact recording fold configuration, training history, in-domain validation metrics, and out-of-distribution test metrics.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% preservation of all 5 canonical notebooks: zero lines of code modified, removed, or refactored into external imports.
- **SC-002**: 100% of LODO fold definitions adhere to the 3-split schema (`train`, `val`, `test`), producing exactly 18 CSV files in `data/processed/lodo_folds/` with zero missing partitions.
- **SC-003**: 0% patient leakage across all generated data partitions: automated assertion `Train_Patients ∩ Val_Patients = ∅` and `Train_Patients ∩ Test_Patients = ∅` passes across all 6 folds and standard splits.
- **SC-004**: Numerical forward-pass parity between notebook components and `drdg` modules: maximum absolute difference $< 1\times 10^{-6}$ in FP32 on CPU across all attention blocks, pooling layers, and output heads.
- **SC-005**: Loss and gradient parity: custom loss values (`ContinuousQWKLoss`, `CLM_QWK`, `CORAL`, `CORN`) match notebook calculations to within $1\times 10^{-6}$, and backward gradients match to within $1\times 10^{-5}$.
- **SC-006**: One-step deterministic optimizer update parity: model weights after a single forward-backward-step sequence match between notebook and `drdg` reference implementations.
- **SC-007**: 100% of CLI reproduction scripts (`run_lodo.py`, `run_head_ablation.py`, `build_master_metadata.py`, `generate_lodo_splits.py`) execute in dry-run mode without unhandled exceptions or missing file errors.
- **SC-008**: Comprehensive code mapping in `docs/NOTEBOOK_CODE_MAP.md` covering 100% of research-critical symbols across all 5 canonical notebooks.
- **SC-009**: Complete data flow audit in `docs/DATA_FLOW_AUDIT.md` establishing a verified producer-consumer link for every artifact from raw download to final result JSON.
- **SC-010**: Final integrity certification in `docs/REFACTOR_AUDIT.md` achieves status `VERIFIED` with zero unaddressed CRITICAL or HIGH discrepancies.

---

## Assumptions

- **Raw Data Availability**: Researchers reproducing experiments locally will either download datasets using provided scripts (`ddr.py`, `deep-drid.py`, `idrid.py`, `messidor.py`) or manually place proprietary cohorts (EyePACS, APTOS) into `data/raw/` as documented.
- **Environment & Hardware**: Deterministic parity tests are designed to execute on standard CPU architectures under PyTorch $\ge 2.0$, while full training workflows support CUDA acceleration with optional automatic mixed precision (AMP).
- **Dual Implementation Policy**: Users understand that the repository intentionally maintains both complete Jupyter notebooks (for Colab execution and provenance) and the `drdg` Python package (for automated CLI pipelines and modular development).
- **Directory Structure Stability**: Data directories follow the project root layout (`data/raw/`, `data/processed/`, `results/`, `checkpoints/`), and all scripts resolve paths relative to the project root or via configuration parameters.
