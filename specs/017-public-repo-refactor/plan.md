# Implementation Plan: Public Academic Repository Refactor (`drdg`)

**Branch**: `017-public-repo-refactor` | **Date**: 2026-09-23 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/017-public-repo-refactor/spec.md`

---

## Summary

Refactor the notebook-centric diabetic retinopathy research codebase into an open-source, reproducible, public GitHub repository supporting the paper *"Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression"*.

The refactoring extracts all algorithmic, data, model, loss, training, and evaluation logic from 5 monolithic notebooks into a clean Python package (`drdg` under `src/drdg/`) with typed Python dataclass configurations (strictly zero YAML), CLI entry points, CPU pytest verification, comprehensive documentation, scrubbed credentials, and demonstration notebooks importing directly from `drdg`.

---

## Technical Context

**Language/Version**: Python 3.10+ (tested on Python 3.10 and 3.11).

**Primary Dependencies**:
- PyTorch ($\ge 2.0.0$) & Torchvision ($\ge 0.15.0$)
- NumPy ($\ge 1.24.0$) & Pandas ($\ge 2.0.0$)
- Scikit-learn ($\ge 1.2.0$) & SciPy ($\ge 1.10.0$)
- OpenCV-Python (`cv2`) & Pillow (`PIL`)
- Kagglehub (for automated dataset acquisition)
- Pytest (for CPU testing suite)
- Matplotlib & Seaborn (for publication figures)

**Storage**:
- File-based raw datasets (`data/raw/`), preprocessed circular FOV images (`data/processed/`), deterministic split CSVs, serialized model checkpoints (`.pt`), and structured JSON/CSV results (`results/`).

**Testing**: Pytest CPU test suite (`tests/`) covering patient parsing, split leakage prevention, model tensor shapes, ordinal cutpoints, RDR calibration, and result schemas.

**Target Platform**: Linux (Ubuntu 20.04/22.04 LTS), Windows 10/11, and Google Colab environments.

**Project Type**: Academic Research Library, CLI Suite, and Accompanying Replication Package.

**Performance Goals**:
- Package installation via `pip install -e .` in $< 30$ seconds.
- CPU unit test suite passes in $< 180$ seconds.
- Data loading with multi-threaded dynamic MIL bag collation achieves $\ge 80$ images/sec on standard SSDs.

**Constraints**:
- **Zero YAML / Hydra**: Configuration strictly via typed Python dataclasses (`src/drdg/config.py`) and CLI `argparse`.
- **Zero Credential Leaks**: Full secret scrubbing of API keys, tokens, and personal directories.
- **Numerical & Algorithmic Preservation**: Exact preservation of original research algorithms; discrepancies logged in `docs/REFACTOR_AUDIT.md`.
- **Neutral Framing**: Prohibition of subjective marketing words ("champion", "superior", or causal claims for XAI).

**Scale/Scope**:
- 6 cohorts harmonized ($N = 55{,}570$ images, $35{,}474$ patients).
- 4 architectural variants (V0 baseline, V1 Local MIL, V2 Global Context, V3 Ordinal CLM).
- 5 output heads (Softmax CE, Softmax QWK, CORAL, CORN, CLM QWK).
- 6-fold LODO cross-domain benchmark.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Constitutional Principle | Status | Evaluation & Compliance Notes |
| :--- | :---: | :--- |
| **I. Research Context & Methodology** | ✅ PASS | Strictly treats DR grading as an ordinal classification task ($0 < 1 < 2 < 3 < 4$), optimizes for QWK (`ContinuousQWKLoss`, CLM), preserves dual-branch high-res patch + global context streams, and supports quantitative XAI. |
| **II. Strict File & Code Integrity** | ✅ PASS | All existing legacy notebooks in `notebooks/diabetic_retinopathy/` and scripts in `src/` remain as untouched references. The refactored package is built cleanly in `src/drdg/` and clean demonstration notebooks in `notebooks/`. |
| **III. Strict Directory Path Adherence** | ✅ PASS | Strictly respects project root layout (`data/raw/`, `data/processed/`, `notebooks/`, `src/drdg/`, `scripts/`, `docs/`, `tests/`). Prohibits hardcoded personal machine paths. |
| **IV. Code Generation & Reproducibility** | ✅ PASS | Full functional code without `pass` or `TODO` placeholders in critical paths. Deterministic seed management for Python, NumPy, and PyTorch across all runners. |
| **V. Strict Preservation & Zero-Regression** | ✅ PASS | Preserves exact mathematical definitions of all feature streams, attention layers, losses, and metric computations. Any prose-vs-code discrepancies are formally audited in `docs/REFACTOR_AUDIT.md`. |

---

## Project Structure

### Documentation (this feature)

```text
specs/017-public-repo-refactor/
├── spec.md                  # Feature specification (/speckit-specify output)
├── plan.md                  # This implementation plan (/speckit-plan output)
├── research.md              # Phase 0 architectural decisions & technical context
├── data-model.md            # Phase 1 data entities, schemas, and invariants
├── quickstart.md            # Phase 1 runnable verification and execution guide
├── contracts/               # Phase 1 interface contracts
│   ├── cli_contracts.md     # Command line argument schemas for all 8 scripts
│   ├── model_contracts.md   # Unified model forward pass & stream interfaces
│   └── result_schemas.md    # JSON/CSV schemas for LODO, ablation, and XAI
├── checklists/
│   └── requirements.md      # Specification quality checklist
└── tasks.md                 # Phase 2 task breakdown (/speckit-tasks output)
```

### Source Code Architecture (Target Repository Layout)

```text
deconstructing-dr-domain-shift/
│
├── README.md                           # Professional academic overview & reproduction guide
├── LICENSE                             # MIT Open Source License (code only)
├── CITATION.cff                        # Paper citation metadata
├── CONTRIBUTING.md                     # Open source contribution guidelines
├── pyproject.toml                      # Package build configuration (pip install -e .)
├── requirements.txt                    # Minimal verified runtime dependencies
├── .gitignore                          # Strict ignore rules (datasets, checkpoints, cache)
├── .env.example                        # Empty credential placeholders
│
├── .github/
│   └── workflows/
│       └── tests.yml                   # Automated CPU unit test workflow
│
├── docs/
│   ├── DATA.md                         # 6 cohorts, manual EyePACS steps, licenses
│   ├── ARCHITECTURE.md                 # Streams, attention, fusion, tensor shapes
│   ├── EXPERIMENTS.md                  # In-domain, 6-Fold LODO, head ablation
│   ├── REPRODUCIBILITY.md              # Seeds, AMP, hardware notes, determinism
│   ├── RESULTS_SCHEMA.md               # Strict separation of val and test schemas
│   ├── XAI.md                          # Grad-CAM, MIL saliency, IDRiD lesion masks
│   ├── CODE_MAP.md                     # Legacy notebook -> src/drdg migration map
│   └── REFACTOR_AUDIT.md               # Mandatory audit of all observed discrepancies
│
├── notebooks/
│   ├── README.md                       # Clean notebook index and usage instructions
│   ├── 00_dataset_harmonization.ipynb  # Data harmonization demonstration
│   ├── 01_lodo_split_generation.ipynb  # Deterministic LODO fold generation
│   ├── 02_dataset_audit.ipynb          # Integrity and leakage verification
│   ├── 10_v0_global_baseline.ipynb     # V0 ResNet-50 baseline walkthrough
│   ├── 11_v1_local_mil.ipynb           # V1 Local MIL stream walkthrough
│   ├── 12_v2_global_context.ipynb      # V2 Relational context walkthrough
│   ├── 13_v3_ordinal_clm.ipynb         # V3 Cumulative Link Model walkthrough
│   ├── 20_lodo_benchmark.ipynb         # 6-Fold LODO evaluation inspector
│   ├── 21_head_ablation.ipynb          # Controlled 5-head benchmark inspector
│   ├── 22_grade_specific_analysis.ipynb# Per-grade sensitivity & within-1 accuracy
│   ├── 23_rdr_screening.ipynb          # RDR screening threshold calibration
│   ├── 24_xai_idrid.ipynb              # IDRiD quantitative lesion XAI
│   └── 30_paper_figures.ipynb          # Publication figure generation
│
├── src/
│   └── drdg/                           # The clean Python research package
│       ├── __init__.py                 # Version and top-level exports
│       ├── config.py                   # Typed Python dataclasses (NO YAML)
│       ├── paths.py                    # Repository path resolution
│       │
│       ├── data/                       # Ingestion, preprocessing, and datasets
│       │   ├── __init__.py
│       │   ├── registry.py             # Cohort metadata registry
│       │   ├── metadata.py             # Master metadata schema validation
│       │   ├── patient_ids.py          # Regex patient parsing & namespacing
│       │   ├── harmonization.py        # 6-cohort harmonization engine
│       │   ├── preprocessing.py        # Circular FOV cropping & green channel
│       │   ├── transforms.py           # Augmentations & tensor normalization
│       │   ├── fov.py                  # Morphological FOV mask computation
│       │   ├── patches.py              # Native sliding-window patch extractor
│       │   ├── datasets.py             # FundusDualBranchDataset implementation
│       │   ├── collate.py              # Dynamic MIL bag collation
│       │   ├── balancing.py            # Training-only quota balancing
│       │   ├── splits.py               # Patient-level stratified splitting
│       │   ├── lodo.py                 # Deterministic 6-fold LODO generator
│       │   ├── downloaders/            # Automated acquisition scripts
│       │   │   ├── __init__.py
│       │   │   ├── ddr.py
│       │   │   ├── deepdrid.py
│       │   │   ├── idrid.py
│       │   │   ├── idrid_segmentation.py
│       │   │   └── messidor2.py
│       │   └── preparation/            # Standalone preparation functions
│       │       ├── __init__.py
│       │       ├── aptos.py
│       │       ├── deepdrid.py
│       │       ├── idrid.py
│       │       ├── idrid_segmentation.py
│       │       └── resize.py
│       │
│       ├── models/                     # Architectures, streams, and heads
│       │   ├── __init__.py
│       │   ├── attention/              # Attention mechanism modules
│       │   │   ├── __init__.py
│       │   │   ├── cbam.py             # Channel & Spatial attention (CBAM)
│       │   │   ├── non_local.py        # Non-Local Block 2D
│       │   │   └── quadrant_tokens.py  # Quadrant Token Aggregator
│       │   ├── streams/                # Multi-scale feature streams
│       │   │   ├── __init__.py
│       │   │   ├── global_context.py   # ResNet-50 + stage-4 attention stream
│       │   │   └── local_mil.py        # Patch backbone + Gated Attention Pool
│       │   ├── heads/                  # Standardized output heads
│       │   │   ├── __init__.py
│       │   │   ├── softmax.py          # Softmax_CE & Softmax_QWK
│       │   │   ├── coral.py            # Consistent Rank Logits (CORAL)
│       │   │   ├── corn.py             # Conditional Ordinal Neural Network
│       │   │   └── clm.py              # Cumulative Link Model (CLM)
│       │   ├── fusion.py               # DualBranchFusion layer
│       │   ├── full_model.py           # DRFullModel & ResNet50AblationModel
│       │   ├── factory.py              # Unified model instantiation factory
│       │   └── variants/               # Paper-aligned architecture builders
│       │       ├── __init__.py
│       │       ├── v0.py               # Global ResNet-50 baseline
│       │       ├── v1.py               # V0 + Native Local MIL
│       │       ├── v2.py               # V1 + Global relational context
│       │       └── v3.py               # V2 + Cumulative Link Model
│       │
│       ├── losses/                     # Objective functions
│       │   ├── __init__.py
│       │   ├── qwk.py                  # Continuous Quadratic Weighted Kappa Loss
│       │   └── ordinal.py              # Binary rank loss & CLM likelihood
│       │
│       ├── training/                   # Robust training engines
│       │   ├── __init__.py
│       │   ├── trainer.py              # DRTrainer (AMP, gradient clip, early stop)
│       │   ├── head_trainer.py         # DRHeadTrainer for controlled ablation
│       │   ├── checkpoint.py           # Checkpoint save/resume helpers
│       │   └── optimization.py         # Optimizer & scheduler builders
│       │
│       ├── evaluation/                 # Metrics & evaluation protocols
│       │   ├── __init__.py
│       │   ├── metrics.py              # QWK, accuracy, confusion matrix
│       │   ├── grade_metrics.py        # Per-grade sensitivity & within-1 acc
│       │   ├── ordinal_metrics.py      # Severity consistency & distance metrics
│       │   ├── rdr.py                  # Source-val threshold calibration & RDR
│       │   ├── statistics.py           # Paired Wilcoxon signed-rank tests
│       │   ├── lodo.py                 # LODO evaluation engine
│       │   └── result_schema.py        # Tripartite metric serialization
│       │
│       ├── experiments/                # High-level orchestration
│       │   ├── __init__.py
│       │   ├── lodo.py                 # Multi-fold LODO runner
│       │   ├── head_ablation.py        # 5-head benchmark runner
│       │   └── xai.py                  # Quantitative XAI runner
│       │
│       ├── explainability/             # Spatial localization & XAI
│       │   ├── __init__.py
│       │   ├── gradcam.py              # Global branch Grad-CAM
│       │   ├── mil_saliency.py         # Local MIL attention saliency mapping
│       │   ├── idrid_dataset.py        # IDRiD segmentation mask loader
│       │   └── metrics.py              # Pointing Game, Area Recall, pixel AUROC
│       │
│       ├── visualization/              # Publication artifacts
│       │   ├── __init__.py
│       │   └── paper_figures.py        # Figure plotting & export
│       │
│       └── utils/                      # Common utilities
│           ├── __init__.py
│           ├── seed.py                 # Seed control (Python, NumPy, PyTorch)
│           ├── logging.py              # Formatted console & file logger
│           ├── device.py               # Device & precision detection
│           └── io.py                   # Atomic JSON and CSV I/O
│
├── scripts/                            # CLI executable entry points
│   ├── download_dataset.py             # Dataset downloader
│   ├── prepare_dataset.py              # Cohort crop & metadata preparer
│   ├── build_master_metadata.py        # Master metadata aggregator
│   ├── generate_lodo_splits.py         # 6-Fold LODO generator
│   ├── train_variant.py                # Single variant trainer
│   ├── run_lodo.py                     # 6-Fold LODO runner
│   ├── run_head_ablation.py            # Controlled head benchmark runner
│   ├── run_xai.py                      # Quantitative XAI runner
│   └── export_paper_artifacts.py       # Paper figures & tables exporter
│
├── data/                               # Dataset directory (git-ignored)
│   ├── README.md                       # Data acquisition & layout guide
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── results/                            # Machine-readable evaluation outputs
│   ├── README.md
│   ├── lodo/
│   │   └── .gitkeep
│   ├── head_ablation/
│   │   └── .gitkeep
│   └── xai/
│       └── .gitkeep
│
└── tests/                              # Fast CPU unit testing suite
    ├── test_patient_ids.py             # Regex parsing across all 6 cohorts
    ├── test_patient_splits.py          # Disjoint patient split validation
    ├── test_lodo_splits.py             # Target cohort exclusion & zero-leakage
    ├── test_fov.py                     # Morphological FOV mask verification
    ├── test_patch_extraction.py        # Sliding-window patch tensor shapes
    ├── test_model_variants.py          # V0..V3 instantiation & forward passes
    ├── test_ordinal_heads.py           # Monotonic cutpoints & output shapes
    ├── test_rdr.py                     # Source-val threshold freezing isolation
    └── test_result_schema.py           # Tripartite JSON schema validation
```

---

## Complexity Tracking

*No constitutional violations identified. The design strictly preserves numerical and algorithmic behavior while modularizing into a standard Python package.*
