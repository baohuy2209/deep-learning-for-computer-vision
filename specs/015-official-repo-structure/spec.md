# Feature Specification: Official Repository Architecture & End-to-End Pipeline for ICCPR Paper

**Feature Branch**: `015-official-repo-structure`

**Created**: 2026-09-19

**Status**: Draft

**Input**: User description: "từ những file thí nghiệm chính này (@[notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb], @[notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb], @[notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb], @[notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb], @[notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb], @[notebooks/diabetic_retinopathy/pipeline/model/visualize_main_model.ipynb]), giúp tôi thiết kế phần tổ chức cho repo chính thức nha. Đồng thời bạn đọc các file phụ trong src/ rồi là các thư mục khác liên quan, nói chung đọc toàn bộ các file trong project hiện tại nè, rồi lên cho tôi cách tổ chức thư mục cho repo chính thức cho bài 'Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression' cho hội nghị ICCPR sắp tới của tui nhe, lên cách tổ chức thư mục cho đầy đủ từ các bước thực hiện ban đầu của tui, bạn tận dụng lại source trong scratch nha để tránh sinh ra thêm code và đồng thời các source mà thực thi xem các cell thì bạn không cần phải hỏi quyền của tui, tuyệt đối không được thay đổi bất kì logic nào nhá"

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reproducible Data Pipeline & Harmonization (Priority: P1)

As a researcher or conference reviewer evaluating the ICCPR paper, I want a clean, unified command-line and programmatic pipeline to preprocess, harmonize, and split the 6 raw retinal fundus datasets (APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS), so that I can reproduce the exact zero-leakage patient-level splits and multi-source balanced training pools without manual notebook execution.

**Why this priority**: Correct and leakage-free data preparation is the foundational prerequisite for all experimental claims in the paper. Any data discrepancy invalidates benchmark results.

**Independent Test**: Can be verified by running the preprocessing and split generation module on sample or full datasets and confirming:
1. All images are cropped to fundus circular ROI without downsampling distortion into `native_cropped/`.
2. Patient IDs are properly namespaced (`dataset_key::patient_id`) preventing cross-cohort collision.
3. 70/15/15 stratified splits and 6-Fold LODO splits have zero patient overlap between train, val, and test sets.

**Acceptance Scenarios**:
1. **Given** raw dataset directories in `data/raw/`, **When** the user executes `python -m dr_multiscale_mil.data.preprocess` or `scripts/01_preprocess_data.sh`, **Then** all raw images are cropped to ROI and saved to `data/raw/hybrid/native_cropped/` with high-quality JPEG compression.
2. **Given** native cropped images and metadata CSVs, **When** the user executes `python -m dr_multiscale_mil.data.make_splits --mode lodo`, **Then** 18 deterministic LODO split CSVs and `lodo_splits_summary.csv` are generated in `data/processed/lodo_folds/` with zero patient leakage asserted.

---

### User Story 2 - Modular Multi-Scale MIL Architecture & Training (Priority: P1)

As a deep learning practitioner or researcher, I want to instantiate, configure, and train the proposed Multi-Scale MIL with Consistent Ordinal Regression model using clean configuration files (YAML / CLI), so that I can train the champion model or any baseline without modifying internal model definitions.

**Why this priority**: The dual-branch Multi-Scale MIL network with Consistent Ordinal Regression is the primary scientific contribution of the ICCPR paper.

**Independent Test**: Can be verified by running a single training epoch or dry-run validation on the in-domain benchmark with `python -m dr_multiscale_mil.train --config configs/in_domain_main_model.yaml` and verifying forward/backward pass, loss computation (Consistent Ordinal Loss), and QWK metric tracking.

**Acceptance Scenarios**:
1. **Given** a valid configuration file `configs/in_domain_main_model.yaml`, **When** the user starts training via `python -m dr_multiscale_mil.train`, **Then** the model initializes Global Context stream (512x512) and Local Patch stream, extracts high-resolution patches, performs Attention MIL pooling, applies Consistent Ordinal Head, and logs QWK, Kappa, and AUC per epoch.
2. **Given** validation loss and QWK computed each epoch, **When** validation QWK improves, **Then** checkpoint weights are safely saved to `checkpoints/checkpoints_main/best_model_qwk.pt` with full optimizer state.

---

### User Story 3 - 6-Fold Leave-One-Dataset-Out (LODO) Generalization Benchmark (Priority: P2)

As a researcher investigating clinical domain shift, I want to execute the full 6-fold LODO benchmark using an automated runner script, so that I can systematically evaluate out-of-distribution (OOD) generalization when each of the 6 cohorts is held out as an unseen clinical test center.

**Why this priority**: LODO validation is the core experimental pillar supporting the paper's title: "Deconstructing Domain Shift in Diabetic Retinopathy Grading".

**Independent Test**: Can be verified by running `scripts/03_run_lodo_benchmark.sh --fold 0` (or all folds 0 to 5) and ensuring that held-out test evaluation outputs natural clinical prevalence metrics and domain shift gap analysis.

**Acceptance Scenarios**:
1. **Given** 6 pre-generated LODO fold CSVs, **When** the user triggers `python -m dr_multiscale_mil.train_lodo --fold 0`, **Then** fold 0 (APTOS held-out) trains on the remaining 5 cohorts (85% train with quota balancing, 15% validation with natural prevalence) and evaluates on 100% of APTOS without any leakage.
2. **Given** completion of all 6 folds, **When** the user runs `python -m dr_multiscale_mil.evaluate --mode lodo_summary`, **Then** an aggregated domain shift comparison table is exported to `results/tables/lodo_benchmark_results.csv`.

---

### User Story 4 - Head & Architectural Ablation Suite (Priority: P2)

As an academic reviewer, I want to reproduce the ablation studies comparing different prediction heads (Nominal Softmax, Standard Ordinal CORAL, Consistent Ordinal Logits, Regression) and preprocessing strategies (Raw RGB, Ben Graham Green, CLAHE Green), so that I can verify the individual performance gains reported in the paper tables.

**Why this priority**: Demonstrates that each architectural component contributes meaningfully to the final clinical performance and justifies design choices.

**Independent Test**: Can be verified by running `scripts/04_run_head_ablation.sh` and inspecting the generated comparison metrics across the 4 classification heads.

**Acceptance Scenarios**:
1. **Given** the head ablation configuration, **When** `python -m dr_multiscale_mil.train_ablation --ablation head --head consistent_ordinal` is executed, **Then** the model trains using the specified head and outputs comparative QWK, MAE, and Off-by-One error rates to `results/tables/head_ablation_summary.csv`.

---

### User Story 5 - Publication Figures & Explainable AI (XAI) Generation (Priority: P3)

As an author preparing the final ICCPR camera-ready manuscript, I want a single script to generate all 6 publication figures (Dual-stream architecture diagram, Dataset cohort distribution, Head ablation calibration curves, LODO radar charts, Grad-CAM/MIL attention heatmaps, and t-SNE feature embeddings), so that all figures match conference formatting standards (300 DPI, vector/PDF/PNG).

**Why this priority**: Required for conference publication and demonstrates visual interpretability of pathological retinal features (microaneurysms, hemorrhages, exudates).

**Independent Test**: Can be verified by running `python -m dr_multiscale_mil.visualize --all` or `scripts/05_generate_paper_figures.sh` and checking that high-resolution figure files exist in `results/figures/`.

**Acceptance Scenarios**:
1. **Given** trained model checkpoints in `checkpoints/`, **When** the user executes `python -m dr_multiscale_mil.visualize --figures 1,2,3,4,5,6`, **Then** all 6 publication figures are rendered and saved to `results/figures/` in both PNG (300 DPI) and PDF vector formats.

---

### Edge Cases

- **Missing Raw Datasets**: If a user only has a subset of the 6 datasets (e.g., only APTOS and IDRiD), the data preparation pipeline must gracefully warn and allow running in-domain experiments on available datasets without crashing.
- **Hardware Heterogeneity**: Dynamic device detection must seamlessly handle single-GPU (CUDA), multi-GPU (DistributedDataParallel / DataParallel), Apple Silicon (MPS), and CPU environments, setting DataLoader worker counts and pin_memory flags appropriately.
- **Google Colab vs Local Workstation**: Dynamic path resolution must detect `google.colab` environment, handle Google Drive mounting (`/content/drive/MyDrive/...`), and fallback to relative repository paths on local workstations without hardcoded paths.
- **Corrupted / Truncated Image Files**: During dataset preprocessing, images that fail `cv2.imread` must be logged to an error report (`results/logs/corrupted_images.log`) rather than terminating batch execution.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The repository MUST be organized into a clean, modular Python package `dr_multiscale_mil` (under `src/` or root) adhering to standard research software engineering practices:
  - `dr_multiscale_mil/data/`: Dataset harmonization, ROI auto-cropping, data loaders, and patient-level split generation.
  - `dr_multiscale_mil/models/`: Multi-Scale MIL dual-stream architecture, backbones (EfficientNet, DenseNet, ResNet, ViT), attention modules (CBAM, MSAG), and MIL poolers.
  - `dr_multiscale_mil/losses/`: Consistent Ordinal Loss, CORAL loss, QWK loss, and nominal cross-entropy.
  - `dr_multiscale_mil/training/`: Training engine, mixed precision handler, validation loops, early stopping, and checkpointing.
  - `dr_multiscale_mil/evaluation/`: Clinical metrics computation (QWK, Cohen's Kappa, Macro AUC, Sensitivity/Specificity at referable DR threshold >= 2, ECE).
  - `dr_multiscale_mil/visualization/`: Grad-CAM, MIL patch attention heatmaps, t-SNE/UMAP domain alignment, and paper figure generators.
- **FR-002**: System MUST strictly preserve and reuse existing verified implementations from `notebooks/diabetic_retinopathy/pipeline/full/` (`data_loader.py`, `losses_and_heads.py`, `models.py`, `preprocessing.py`, `trainer.py`, `visualization.py`) and `scratch/extracted_defs_from_main_model.py`. Under NO circumstances shall any mathematical logic, loss formulations, or model architectures be altered or rewritten.
- **FR-003**: The repository MUST provide standard configuration files (`configs/`) in YAML format separating:
  - Dataset paths and preprocessing parameters.
  - Model hyperparameters (backbone, patch size, number of patches, attention type, prediction head).
  - Training hyperparameters (batch size, learning rate, scheduler, weight decay, epochs).
  - Benchmark settings (in-domain 70/15/15 vs 6-Fold LODO).
- **FR-004**: System MUST provide unified, executable command-line interfaces (CLI):
  - `preprocess.py` / `dr_multiscale_mil.data.preprocess`: Crop raw fundus images to `native_cropped/`.
  - `generate_splits.py` / `dr_multiscale_mil.data.make_splits`: Create in-domain and LODO CSV splits.
  - `train.py`: Train in-domain model.
  - `train_lodo.py`: Train 6-fold LODO benchmark.
  - `evaluate.py`: Evaluate checkpoints on test sets and output clinical metric summaries.
  - `visualize.py`: Generate ICCPR paper figures.
- **FR-005**: System MUST provide automation shell scripts (`scripts/`) for one-line end-to-end reproduction:
  - `scripts/01_preprocess_data.sh`
  - `scripts/02_train_in_domain.sh`
  - `scripts/03_run_lodo_benchmark.sh`
  - `scripts/04_run_head_ablation.sh`
  - `scripts/05_generate_paper_figures.sh`
- **FR-006**: System MUST maintain demonstration and tutorial Jupyter Notebooks in `notebooks/`:
  - `notebooks/01_data_pipeline_and_lodo_splits.ipynb`
  - `notebooks/02_train_multiscale_mil_model.ipynb`
  - `notebooks/03_head_ablation_experiments.ipynb`
  - `notebooks/04_lodo_generalization_benchmark.ipynb`
  - `notebooks/05_xai_and_publication_figures.ipynb`
  - All notebooks MUST support direct execution on Google Colab with one-click Drive mounting and GPU acceleration.
- **FR-007**: System MUST provide complete documentation in `docs/` and root `README.md`:
  - `README.md`: Paper abstract, architecture diagram, benchmark leaderboard, quick start, installation, and citations.
  - `docs/dataset_preparation.md`: Step-by-step instructions to download the 6 public datasets (APTOS, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS).
  - `docs/reproduce_experiments.md`: Exact commands to reproduce all tables and figures in the ICCPR paper.
  - `docs/colab_guide.md`: Instructions for running experiments on Google Colab using free or Pro T4/A100 instances.

---

### Key Entities

- **CohortDataset**: Represents one of the 6 retinal fundus datasets (APTOS, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS), including its raw images, metadata CSV, patient ID parsing rule, eye side parsing rule, and integer DR severity labels (0 to 4).
- **MultiScaleMILModel**: The dual-stream deep neural network comprising:
  - Global Context Branch (processes whole-fundus image at 512x512 with Green channel enhancement).
  - Local Patch Branch (extracts and processes high-resolution native patches).
  - Attention MIL Pooling Module (aggregates patch representations with learnable weights).
  - Consistent Ordinal Head (outputs monotonically consistent cumulative logits for DR grades 0-4).
- **LODOBenchmark**: Represents the 6-fold Leave-One-Dataset-Out cross-validation scheme where in fold $k$, cohort $k$ is held out as an unperturbed OOD test set, while the remaining 5 cohorts are partitioned into 85% quota-balanced train and 15% natural prevalence validation sets.
- **ClinicalMetrics**: Encapsulates clinical evaluation metrics:
  - Quadratic Weighted Kappa (QWK) — primary ranking metric.
  - Macro-averaged AUC-ROC.
  - Referable DR Sensitivity and Specificity (DR Grade >= 2).
  - Off-by-one Accuracy & Mean Absolute Error (MAE).
  - Expected Calibration Error (ECE).

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: **Zero Code Logic Regression**: 100% of mathematical operations, model weights, loss functions, and data transformations match the existing verified implementations in `notebooks/diabetic_retinopathy/pipeline/full/` and `scratch/`.
- **SC-002**: **One-Command Reproducibility**: Any user can execute the entire pipeline from raw data to evaluation tables using standardized CLI commands and shell scripts in under 3 commands.
- **SC-003**: **Zero Patient Leakage**: 100% of generated splits across in-domain and 6 LODO folds strictly satisfy the zero-leakage invariant ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$) at the patient level.
- **SC-004**: **Publication Asset Generation**: All 6 ICCPR publication figures and 3 result tables are automatically rendered to `results/figures/` and `results/tables/` matching publication camera-ready standards (vector PDF and 300 DPI PNG).
- **SC-005**: **Colab & Workstation Interoperability**: Scripts and notebooks execute without modification on both local Linux/Windows workstations and Google Colab environments.

---

## Assumptions

- **Raw Datasets**: Raw fundus image archives are available in `data/raw/` or uploaded to Google Drive as compressed archives.
- **Computing Environment**: Experiments can run on standard consumer/research GPUs with >= 12GB VRAM (e.g., RTX 3060/3080/4090 or Colab T4/A100) using mixed precision (FP16).
- **Existing Assets**: Pre-extracted Python definitions from `scratch/extracted_defs_from_main_model.py` and `notebooks/diabetic_retinopathy/pipeline/full/` serve as the golden source of truth for all modules.
- **Licensing & Governance**: Code adheres to the Research Constitution rules in `.specify/memory/constitution.md` (ordinal classification focus, strict file integrity, no fallbacks/placeholders, deterministic seeds).
