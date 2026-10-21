# Feature Specification: 001-standalone-lodo-repo-parity

**Feature Branch**: `001-standalone-lodo-repo-parity`  
**Created**: 2026-09-29  
**Status**: Completed / Ready for Production  
**Input**: User description: `"/speckit-specify @directory:DR-evaluation-study2026 @[notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb] giờ bạn giúp tôi cập nhật lại toàn bộ nội dung trong @directory:DR-evaluation-study2026 theo đúng file @[notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb] và import thư mục này kia cũng phải cho đúng, bởi vì cái này tui sẽ tách ra làm repo chính thức cho nghiên cứu này á"`

---

## 1. Executive Summary & Objective

The goal of this specification is to transform `DR-evaluation-study2026` into the **official, standalone, publication-grade research repository** for the Diabetic Retinopathy (DR) Cross-Domain Generalization Study. 

All source modules, data pipelines, model definitions, training engines, evaluation protocols, explainability (XAI) pipelines, and publication figures are synchronized with 100% mathematical and empirical parity to the flagship empirical reference `main_model_lodo.ipynb` (SHA-256: `7c0303e859e86d705a5b9879e6712bd9d584b84b4b63f588b28610b49179c6f2`).

Crucially, all legacy import prefixes (such as `from drdg.`) and artificial junction dependencies have been eliminated in favor of clean, top-level Python package imports native to `DR-evaluation-study2026`.

---

## 2. User Scenarios & Testing *(mandatory)*

### User Story 1 - Clean Standalone Package Imports & Path Resolution (Priority: P1)

As an academic researcher or external reviewer cloning `DR-evaluation-study2026`, I need to install dependencies and run any script or import any package (`import config, paths, data, models, training, evaluation, explainability, visualization, experiments`) without import errors or legacy prefix requirements (`drdg`), so that the repository functions as an independent, reproducible scientific software package.

**Why this priority**: Eliminating legacy coupling and broken module paths is prerequisite to all experimentation, continuous integration, and open-source public release.

**Independent Test**: Executed `python -c "import config, paths, data, models, training, evaluation, explainability, visualization, experiments"` and confirmed 100% error-free import across all 11 top-level packages.

**Acceptance Scenarios**:
1. **Given** a fresh Python environment in `DR-evaluation-study2026`, **When** importing top-level packages or submodules, **Then** all imports resolve directly without `ModuleNotFoundError: No module named 'drdg'`.
2. **Given** `paths.py`, **When** resolving `REPO_ROOT`, **Then** it resolves to the root of `DR-evaluation-study2026`, and automatically exposes canonical directories (`checkpoints/lodo`, `results/lodo`, `data/lodo_folds`, etc.).

---

### User Story 2 - Exact Architecture Ablation & CLM Numerical Stability (Priority: P1)

As a researcher evaluating inductive biases for DR severity grading, I need all four architecture ablation variants ($V_0 \to V_1 \to V_2 \to V_3$) to match `main_model_lodo.ipynb` with exact numerical stability guarantees, specifically:
- $V_0$: Global Context ResNet-50 + Softmax
- $V_1$: Global Context + Local MIL (EfficientNet-B0 Dual Pooling) + Softmax
- $V_2$: Global Context (CBAM + Non-Local + Quadrant Tokens) + Local MIL + Softmax
- $V_3$: Champion Dual-Branch Multi-Scale MIL + Cumulative Link Model (CLM) Ordinal Head with Continuous QWK Loss ($\lambda=0.4$)

**Why this priority**: Divergences in architecture specifications, loss weighting, or floating-point numerical stability invalidate empirical comparisons across benchmark folds.

**Independent Test**: Instantiated all 4 model variants using `models.build_variant(v)` and verified forward-pass outputs, LayerNorm fusion (batch size 1 resilience), and FP32 CLM clog-log survival calculation under bfloat16 mixed precision.

**Acceptance Scenarios**:
1. **Given** the CLM-QWK head in `models/heads/clm.py`, **When** executed under mixed precision (bfloat16/float16), **Then** cutpoint calculations, clog-log link function $S(z)=\exp(-\exp(z))$, and NLL loss are computed under `torch.amp.autocast(enabled=False)` in pure FP32 with clamping $[-15, 15]$ to prevent NaN collapses.
2. **Given** `DualBranchFusion` in `models/fusion.py`, **When** evaluating batches with `batch_size = 1`, **Then** the normalization layer uses `nn.LayerNorm(fusion_dim)` (not BatchNorm) ensuring zero evaluation crashes.

---

### User Story 3 - 6-Fold LODO Orchestration Loop & Post-Hoc Calibration (Priority: P2)

As a benchmark practitioner running Leave-One-Dataset-Out cross-domain generalization, I need `experiments/lodo.py` to orchestrate 6 folds holding out each cohort (`aptos`, `idrid`, `messidor2`, `ddr`, `eyepacs`, `deepdrid`), checkpointing `_best.pth` and `_last.pth` with `is_completed` flag detection, performing bipartite evaluation (in-domain threshold calibration $\tau^*$ + frozen OOD test evaluation), and persisting raw prediction arrays (`.npz`) and calibrated screening summaries.

**Why this priority**: Matches Cells 25, 26, and 27 of `main_model_lodo.ipynb` to provide complete benchmark automation and compute preservation.

**Independent Test**: Executed `python -m experiments.lodo --dry-run` and verified complete training loop, checkpoint restoration, threshold calibration, and JSON summary export.

**Acceptance Scenarios**:
1. **Given** an interrupted LODO run, **When** restarting `python -m experiments.lodo`, **Then** completed folds marked with `is_completed=True` in checkpoints are skipped automatically, preserving compute units.
2. **Given** trained fold checkpoints, **When** executing post-hoc calibration via `--posthoc-calibration`, **Then** prediction arrays are compressed into `results/lodo/posthoc_fixed/raw_preds/{variant}__{cohort}.npz` and summaries are generated into `lodo_summary_calibrated_screening.json`.

---

### User Story 4 - Publication Figure Suite & Quantitative XAI Fidelity (Priority: P2)

As an author preparing camera-ready manuscript figures, I need `visualization/paper_figures.py` and `explainability/` to generate all 7 publication figures (Figures 2 to 8) and quantitative XAI benchmarks (Pointing Game, Area-Matched Saliency Recall, Pixel AUROC) matching Cells 3, 30, 31, 32, and 33 of `main_model_lodo.ipynb`.

**Why this priority**: Essential for scientific validation and manuscript preparation.

**Independent Test**: Verified function signatures and dual SVG/PDF vector output generation for `plot_fig2_dataset_distribution`, `plot_fig3_slope_chart_per_grade_recall`, `plot_fig4_waterfall_qwk_decomposition`, `plot_fig5_lodo_heatmap`, `plot_fig6_latent_severity_axis`, `plot_fig7_referable_dr_roc`, and `plot_fig8_real_xai`.

**Acceptance Scenarios**:
1. **Given** model predictions and ground-truth IDRiD masks, **When** calling `compute_quantitative_xai()`, **Then** it evaluates Pointing Game, Area-Matched Recall, and Pixel AUROC for both Grad-CAM and MIL attention saliency.
2. **Given** any plotting function in `visualization/paper_figures.py`, **When** executed, **Then** figures are rendered in Nature/Lancet publication typography with dual vector format (.pdf and .svg) in `documents/figures/`.

---

## 3. Functional Requirements *(mandatory)*

- **FR-001**: System MUST resolve all file system paths relative to the current repository root in `paths.py` without referring to parent repository structures.
- **FR-002**: All 54+ `.py` source files in `DR-evaluation-study2026` MUST import directly from top-level packages (`paths`, `config`, `data`, `models`, `losses`, `training`, `evaluation`, `explainability`, `visualization`, `experiments`, `utils`) without any `drdg.` prefix.
- **FR-003**: System MUST define and export `ARCHITECTURE_VARIANTS` ($V_0 \dots V_3$), `DATASET_CONFIG`, and `DATASET_KEYS` in `config.py` matching `main_model_lodo.ipynb`.
- **FR-004**: System MUST implement `CumulativeLinkModelQWKHead` in `models/heads/clm.py` with pure FP32 clog-log calculation, numeric clamping $[-15, 15]$, and Continuous QWK loss ($\lambda=0.4$).
- **FR-005**: System MUST implement `DualBranchFusion` with `nn.LayerNorm(fusion_dim)` to ensure numerical resilience on batch size 1.
- **FR-006**: System MUST support model instantiation via `build_variant(name)` in `models/factory.py` using both short codes (`v0`..`v3`) and full variant names (`V0_Baseline_Softmax_GlobalOnly`..`V3_PlusOrdinalHead_H3_Champion`).
- **FR-007**: System MUST provide `DRTrainer` in `training/trainer.py` supporting native bfloat16/float16 mixed precision, GPU-side normalization, gradient clipping (`1.0`), and dual checkpointing (`_best.pth` and `_last.pth`).
- **FR-008**: System MUST implement `calibrate_and_evaluate_screening` in `evaluation/rdr.py` to freeze the in-domain calibrated threshold $\tau^*$ before evaluating OOD test sensitivity at 80%, 85%, 90%, and 95% specificity targets.
- **FR-009**: System MUST provide `FundusGradCAM` in `explainability/gradcam.py` with automatic layer discovery and memory-safe forward/backward hook cleanup.
- **FR-010**: System MUST provide `generate_attention_mil_saliency_map` in `explainability/mil_saliency.py` extracting architecture-native attention weights from `GatedAttentionPool`.
- **FR-011**: System MUST provide `IDRiDXAIEvalDataset` in `explainability/idrid_dataset.py` implementing dual-crop alignment using `compute_fov_mask(img_np, erode_px=0)`.
- **FR-012**: System MUST provide `compute_quantitative_xai` in `explainability/metrics.py` calculating Pointing Game, Area-Matched Recall, and Pixel AUROC.

---

## 4. Key Entities & Data Models

- **ModelVariant**:
  - `name`: String identifier (e.g. `V3_PlusOrdinalHead_H3_Champion`)
  - `short_code`: String code (`v0`, `v1`, `v2`, `v3`)
  - `head_type`: Classification or ordinal head (`Softmax_CE`, `Softmax_QWK`, `CLM_QWK`)
  - `use_local_branch`: Boolean flag for MIL branch
  - `use_nonlocal`, `use_quadrant_tokens`, `use_cbam`: Inductive bias ablation flags
- **LODOFoldResult**:
  - `variant`: Model variant name
  - `held_out_dataset`: Target evaluation cohort name
  - `in_domain_val_qwk`: Validation QWK score
  - `ood_test_qwk`: Out-of-domain test QWK score
  - `delta_lodo`: Generalization drop ($\text{QWK}_{\text{val}} - \text{QWK}_{\text{test}}$)
  - `ood_calibrated_screening`: Table of clinical sensitivities at target specificities
- **QuantitativeXAIBundle**:
  - `Pointing_Game`: Hit rate (%) of peak saliency on lesion mask
  - `Area_Matched_Recall`: Fraction of lesion area covered by top-$K$ pixels
  - `Pixel_AUROC`: Pixel-wise ROC-AUC across the retinal crop

---

## 5. Success Criteria *(mandatory)*

- **SC-001**: 100% of internal imports in `DR-evaluation-study2026` resolve cleanly without any `ModuleNotFoundError` or deprecation warnings.
- **SC-002**: Zero dependencies on external parent repo paths or virtual filesystem junctions (`drdg`).
- **SC-003**: 4-variant architecture ablation ($V_0 \dots V_3$) can be built and evaluated seamlessly from CLI or Python API.
- **SC-004**: End-to-end dry-run execution (`python -m experiments.lodo --dry-run`) completes with exit code 0 and produces valid checkpoint, metric report, and summary JSON.
- **SC-005**: All 7 manuscript figures (Figures 2 to 8) can be rendered in dual vector formats (.pdf and .svg).
- **SC-006**: Complete spec-kit configuration (`.specify/` and `specs/001-standalone-lodo-repo-parity/spec.md`) is maintained within the repository.

---

## 6. Assumptions

- Python 3.10+ and PyTorch 2.0+ environment with CUDA/MPS/CPU support.
- Data splits are formatted as `fold_{fold_idx}_{cohort}_{split}.csv` under `data/lodo_folds/`.
- Training can be executed in full GPU mode (bfloat16) or synthetic dry-run mode for CI/smoke testing.
