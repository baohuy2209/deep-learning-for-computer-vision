# Implementation Plan: Restore Exact Notebook-to-Python Logic Parity (`drdg`)

**Branch**: `018-restore-notebook-parity` | **Date**: 2026-09-24 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/018-restore-notebook-parity/spec.md`

---

## Summary

Restore exact, verified logic, numerical, and workflow parity between the 5 canonical Jupyter research notebooks and the modular pure-Python package `drdg` under `src/drdg/`. 

This plan enacts the intended dual-implementation architecture:
1. The 5 canonical notebooks remain completely self-contained, unaltered, and executable research artifacts (preserving all original classes, data loaders, training loops, and inline utilities).
2. The `drdg` package provides an independent, modular pure-Python duplicate with identical semantics, algorithms, and tensor operations.
3. The LODO partition design is strictly corrected to the authentic 3-split schema (`train`, `val`, `test` across 18 CSV files in `data/processed/lodo_folds/`), permanently eliminating fabricated 4-way partitions (`source_test`, `target_test.csv`).
4. End-to-end data dependencies are audited and documented in `docs/DATA_FLOW_AUDIT.md`, symbol mappings in `docs/NOTEBOOK_CODE_MAP.md`, and parity verification status in `docs/REFACTOR_AUDIT.md`.
5. Automated behavioral parity tests in `tests/parity/` prove equivalence for preprocessing, FOV masks, patch extraction, forward passes, custom losses, backward gradients, and one-step training updates.

---

## Technical Context

**Language/Version**: Python 3.10+ (tested on Python 3.10, 3.11, and 3.12 on Linux, Windows, and Google Colab).

**Primary Dependencies**:
- PyTorch ($\ge 2.0.0$) & Torchvision ($\ge 0.15.0$)
- NumPy ($\ge 1.24.0$) & Pandas ($\ge 2.0.0$)
- Scikit-learn ($\ge 1.2.0$) & SciPy ($\ge 1.10.0$)
- OpenCV-Python (`cv2`) & Pillow (`PIL`)
- Pytest (for CPU testing suite)
- Matplotlib & Seaborn (for publication figures)

**Storage**:
- Raw datasets staged in `data/raw/` (read-only per Constitution Principle III).
- Harmonized native crops in `data/raw/hybrid/native_cropped/` (matching notebook output).
- Pre-generated frozen LODO folds in `data/processed/lodo_folds/` (18 CSV files: `fold_{0-5}_{cohort}_{train,val,test}.csv`).
- Serialized checkpoints in `checkpoints/` and structured evaluation JSONs in `results/`.

**Testing**: Pytest CPU behavioral parity test suite (`tests/parity/`) asserting FP32 max absolute error $< 1\times 10^{-6}$ against notebook golden references, zero patient leakage, and 1-step optimizer update equivalence.

**Target Platform**: Windows 10/11, Ubuntu 20.04/22.04 LTS, and Google Colab (Linux CUDA).

**Project Type**: Academic Research Library, CLI Suite, and Accompanying Replication Package.

**Performance Goals**:
- All CPU parity tests execute in $< 120$ seconds.
- Memory consumption during patch extraction and bag collation $< 4$ GB RAM on CPU.
- Package installation via `pip install -e .` completes cleanly with zero circular imports.

**Constraints**:
- **Absolute Notebook Preservation**: Zero modifications to code cells of the 5 canonical notebooks.
- **Zero Fabricated Partitions**: Strictly 3 splits (`train`, `val`, `test`) per LODO fold. No `source_test`.
- **Numerical & Algorithmic Parity**: Zero tolerance for arbitrary hyperparameter changes or loss deviations.
- **Zero YAML**: Configuration strictly via typed Python dataclasses and CLI `argparse`.
- **Zero Credential Leaks**: Strict adherence to security standards (scrubbed API keys, tokens, personal paths).

**Scale/Scope**:
- 5 Canonical Notebooks audited.
- 6 Cohorts harmonized ($N = 55{,}570$ images, $35{,}474$ patients).
- 4 Architectural Variants (V0 Global, V1 Local MIL, V2 Context, V3 Ordinal CLM).
- 5 Output Heads (Softmax CE, Softmax QWK, CORAL, CORN, CLM QWK).
- 18 Frozen LODO Fold CSVs.
- 25 Discrete Parity Test Modules.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

- [x] **Principle I: Research Context & Methodology**: The problem is strictly modeled as Ordinal Classification optimizing for Quadratic Weighted Kappa (QWK). The dual-branch architecture (Global context + Local MIL) and XAI explainability are preserved in full fidelity.
- [x] **Principle II: Strict File & Code Integrity (DO NOT MODIFY RULE)**: The 5 canonical notebooks are explicitly protected from alteration, refactoring, or deletion. They remain 100% self-contained research artifacts.
- [x] **Principle III: Strict Directory Path Adherence**: All paths strictly adhere to `deep-learning-for-computer-vision/`, `data/raw/`, `data/processed/`, `notebooks/diabetic_retinopathy/`, `src/drdg/`.
- [x] **Principle IV: Code Generation & Reproducibility**: No boilerplate `TODO` or `pass` in critical paths; explicit random seeding across PyTorch, NumPy, and standard random; environment dependencies fully declared.
- [x] **Principle V: Strict Preservation, Zero-Regression & Minimal-Scope Directives**: Existing verified logic is preserved; refactoring only corrects divergences from notebook truth without introducing unprompted structural redesigns.

---

## Project Structure

### Documentation (this feature)

```text
specs/018-restore-notebook-parity/
├── plan.md              # This implementation plan
├── research.md          # Phase 0: Architectural & parity decisions
├── data-model.md        # Phase 1: Data schemas, split layouts, entities
├── quickstart.md        # Phase 1: Validation and execution guide
├── contracts/           # Phase 1: CLI and library API contracts
│   ├── cli_contracts.md
│   ├── lodo_schema_contract.md
│   └── model_api_contract.md
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code & Test Layout

```text
deep-learning-for-computer-vision/
├── notebooks/diabetic_retinopathy/pipeline/  # CANONICAL REFERENCE (100% UNTOUCHED)
│   ├── merge-dataset/
│   │   ├── merge-dataset.ipynb
│   │   └── merge_dataset_lodo.ipynb
│   └── model/
│       ├── main_model (6).ipynb
│       ├── main_model_lodo.ipynb
│       └── main_model_head_ablation (3).ipynb
│
├── src/drdg/                                # MODULAR PURE-PYTHON PACKAGE
│   ├── __init__.py
│   ├── config.py                            # Typed dataclass configs
│   ├── paths.py                             # Path constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collation.py                     # Bag dynamic batching
│   │   ├── dataset.py                       # FundusDualBranchDataset
│   │   ├── harmonization.py                 # Multi-cohort metadata unification
│   │   ├── lodo.py                          # 3-split LODO generator (train, val, test)
│   │   ├── patch.py                         # Native patch extraction
│   │   ├── patient.py                       # Patient ID regex & namespacing
│   │   ├── preprocessing.py                 # Retinal FOV non-black crop
│   │   └── splits.py                        # Stratified patient-level splits
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py                     # CBAM, NonLocalBlock2D, QuadrantTokenAggregator
│   │   ├── dual_branch.py                   # Full DualBranchFundusModel
│   │   ├── fusion.py                        # Multiscale FusionNeck
│   │   ├── global_stream.py                 # ResNet-50 GlobalContextStream
│   │   ├── heads.py                         # 5 Output Heads (Softmax, CORAL, CORN, CLM)
│   │   ├── local_mil.py                     # EfficientNet-B0 LocalMILBranch
│   │   └── variants.py                      # V0, V1, V2, V3 builders
│   ├── training/
│   │   ├── __init__.py
│   │   ├── early_stopping.py                # Patience-based QWK early stopping
│   │   ├── engine.py                        # Epoch training & validation loops
│   │   ├── losses.py                        # ContinuousQWKLoss, CORAL, CORN, CLM
│   │   ├── optimizer.py                     # Two-tier AdamW parameter groups
│   │   ├── scheduler.py                     # CosineAnnealingLR & QWK Warmup
│   │   └── trainer.py                       # Complete DualBranchTrainer
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── lodo.py                          # Bipartite LODO evaluator
│   │   ├── metrics.py                       # QWK, Sensitivity, Within-1, Accuracy
│   │   ├── rdr.py                           # Referable DR threshold calibration
│   │   └── result_schema.py                 # Bipartite LODOFoldResult serialization
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── head_ablation.py                 # 5-head ablation runner
│   │   ├── lodo.py                          # 6-fold LODO runner
│   │   └── standard.py                      # Standard train/val/test runner
│   └── xai/
│       ├── __init__.py
│       ├── gradcam.py                       # Global stream Grad-CAM
│       ├── localization.py                  # Local patch attention splatting
│       └── metrics.py                       # Pointing game, Area-matched recall
│
├── tests/
│   ├── parity/                              # BEHAVIORAL PARITY TEST SUITE
│   │   ├── test_patient_id_parity.py
│   │   ├── test_preprocessing_parity.py
│   │   ├── test_fov_parity.py
│   │   ├── test_patch_parity.py
│   │   ├── test_collate_parity.py
│   │   ├── test_lodo_split_parity.py
│   │   ├── test_attention_parity.py
│   │   ├── test_global_branch_parity.py
│   │   ├── test_local_mil_parity.py
│   │   ├── test_fusion_parity.py
│   │   ├── test_heads_parity.py
│   │   ├── test_losses_parity.py
│   │   ├── test_variant_parity.py
│   │   ├── test_training_step_parity.py
│   │   ├── test_optimizer_parity.py
│   │   ├── test_scheduler_parity.py
│   │   ├── test_early_stopping_parity.py
│   │   ├── test_checkpoint_parity.py
│   │   ├── test_metrics_parity.py
│   │   ├── test_rdr_parity.py
│   │   └── test_xai_parity.py
│   └── ...
│
├── docs/
│   ├── NOTEBOOK_CODE_MAP.md                 # Symbol-by-symbol mapping table
│   ├── DATA_FLOW_AUDIT.md                   # Producer-consumer artifact matrix
│   ├── REFACTOR_AUDIT.md                    # Comprehensive discrepancy & verification report
│   ├── ARCHITECTURE.md                      # Dual-layer architecture description
│   ├── DATA.md                              # Data pipeline & LODO fold layout
│   └── EXPERIMENTS.md                       # Protocol reproduction guide
│
└── scripts/                                 # CLI ENTRY POINTS
    ├── download_dataset.py
    ├── prepare_dataset.py
    ├── build_master_metadata.py
    ├── generate_lodo_splits.py
    ├── run_lodo.py
    └── run_head_ablation.py
```

---

## Phases & Execution Order

### Phase 0: Outline & Research
- Audit the 5 canonical notebooks to identify every research-critical symbol, hyperparameter, and tensor operation.
- Resolve architectural decisions in `specs/018-restore-notebook-parity/research.md`:
  - 3-split LODO schema (`train`, `val`, `test`).
  - Dual-layer coexistence (notebooks intact, `drdg` modular duplicate).
  - Tolerances and fixture design for deterministic CPU parity tests.
  - End-to-end data flow dependency graph.

### Phase 1: Design & Contracts
- Define exact entity schemas and directory structures in `data-model.md`.
- Formulate CLI and API contracts in `contracts/`.
- Provide concrete end-to-end execution guide in `quickstart.md`.
- Re-verify Constitution compliance post-design.

### Phase 2: Implementation & Audit (Planned for `/speckit-tasks` and `/speckit-implement`)
- **Stage 1: LODO Schema Cleanup & Verification**: Audit `data/lodo.py`, `experiments/lodo.py`, `evaluation/lodo.py`, and `evaluation/result_schema.py` to ensure zero references to `source_test` or 4-way splits exist.
- **Stage 2: Comprehensive Code Map Generation**: Programmatically extract notebook symbols and build `docs/NOTEBOOK_CODE_MAP.md`.
- **Stage 3: Data Flow Audit**: Trace every input and output file from acquisition to evaluation in `docs/DATA_FLOW_AUDIT.md`.
- **Stage 4: Parity Test Suite Construction**: Build `tests/parity/` suite with deterministic fixtures testing all layers.
- **Stage 5: Algorithmic & Numerical Correction**: Fix any discrepancies uncovered during parity testing.
- **Stage 6: Documentation Reconstruction**: Rewrite `README.md`, `docs/ARCHITECTURE.md`, `docs/DATA.md`, `docs/EXPERIMENTS.md`, and `docs/REFACTOR_AUDIT.md`.
- **Stage 7: Final Verification**: Execute full parity test suite, run dry-run CLIs, and certify status as `VERIFIED`.
