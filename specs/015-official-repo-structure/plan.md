# Implementation Plan: Official Repository Architecture & End-to-End Pipeline for ICCPR Paper

**Branch**: `015-official-repo-structure` | **Date**: 2026-09-19 | **Spec**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/015-official-repo-structure/spec.md)

**Input**: Feature specification from `specs/015-official-repo-structure/spec.md`  
**Paper Title**: *Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression* (ICCPR Conference)

---

## Summary

Design and establish the official, publication-ready repository architecture for the ICCPR submission. The architecture organizes the end-to-end research workflow—from multi-cohort fundus data acquisition and zero-leakage patient splitting to Multi-Scale Multiple Instance Learning (MIL) model training with Consistent Ordinal Regression, 6-Fold Leave-One-Dataset-Out (LODO) cross-validation, head ablations, and 300 DPI publication figure generation. All modules reuse existing verified implementations from `notebooks/diabetic_retinopathy/pipeline/full/` and `scratch/extracted_defs_from_main_model.py` with strictly zero alteration of mathematical, algorithmic, or loss logic.

---

## Technical Context

**Language/Version**: Python 3.10+ / 3.12 (standard research environment)  
**Primary Dependencies**: PyTorch $\ge 2.0$, torchvision, timm, albumentations, opencv-python, scikit-learn, pandas, numpy, matplotlib, seaborn, tqdm, pyyaml  
**Storage**: Local workstation storage / Google Drive mount (`data/raw/`, `data/processed/`, `checkpoints/`, `results/`)  
**Testing**: pytest, automated zero-leakage patient intersection assertions, forward/backward pass shape checks  
**Target Platform**: Ubuntu Linux / Windows 11 / Google Colab (CUDA 11.8/12.x GPU with FP16 AMP support; CPU graceful fallback)  
**Project Type**: Deep Learning Research Library, CLI Experiment Runner & Publication Artifact Generator  
**Performance Goals**: Multi-threaded preprocessing ($\sim 80+$ it/s), FP16 mixed precision training, sub-15 minute paper figure generation  
**Constraints**:
- Strict Preservation Directive (Constitution Principle V): zero changes to existing math, loss functions, network layers, or preprocessing formulas.
- Ordinal Classification Requirement (Constitution Principle I): primary optimization and early stopping target is Quadratic Weighted Kappa (QWK).
- Zero Patient Leakage: $Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$ across all in-domain and LODO splits.  
**Scale/Scope**: 6 retinal fundus cohorts (55,570 images, 35,474 unique patients), 6-fold LODO benchmark (18 CSV splits), 4 prediction head ablations, 6 publication figures.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Plan Status | Notes |
|---|---|---|---|
| **I. Research Context & Methodology** | Ordinal classification (0-4), QWK optimization, Dual-branch framework (Global + Local patch), XAI integration | **PASS** | Fully integrated in `MultiScaleMILNet`, Consistent Ordinal Head, and Grad-CAM/MIL attention visualizations. |
| **II. Strict File & Code Integrity** | Never overwrite, modify, or delete verified existing files; generate new modular files | **PASS** | Existing notebooks remain intact as reference; new package `src/dr_multiscale_mil` imports verified logic cleanly. |
| **III. Strict Directory Path Adherence** | Adhere strictly to project root `deep-learning-for-computer-vision/`, `data/raw/`, `data/processed/`, `src/`, etc. | **PASS** | All paths strictly map to approved directory structure with dynamic environment resolution. |
| **IV. Code Generation & Reproducibility** | No fallbacks/placeholders (`TODO`/`pass`), deterministic seeds (`SEED=42`), dependency tracking | **PASS** | Zero boilerplate; all seeds fixed at 42; dependencies documented in `requirement.txt`. |
| **V. Strict Preservation & Zero-Regression** | Zero alteration of verified logic, formulas, or signatures | **PASS** | Reuses pre-extracted definitions from `scratch/extracted_defs_from_main_model.py` and `pipeline/full/`. |

---

## Project Structure

### Documentation (this feature)

```text
specs/015-official-repo-structure/
├── spec.md              # Feature specification
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (technical decisions & rationale)
├── data-model.md        # Phase 1 output (entities, data structures, state machine)
├── quickstart.md        # Phase 1 output (step-by-step reproduction guide)
├── contracts/           # Phase 1 output (CLI and Config schemas)
│   ├── cli-contract.md
│   └── config-schema.md
├── checklists/          # Quality validation
│   └── requirements.md
└── tasks.md             # Phase 2 output (/speckit-tasks command)
```

### Source Code (repository root)

```text
deep-learning-for-computer-vision/
├── configs/                                 # YAML experiment configurations
│   ├── data/
│   │   ├── datasets.yaml
│   │   └── lodo_folds.yaml
│   ├── model/
│   │   ├── multiscale_mil_efficientnet.yaml
│   │   └── baselines.yaml
│   ├── train/
│   │   ├── in_domain.yaml
│   │   └── lodo_benchmark.yaml
│   └── ablations/
│       ├── head_ablation.yaml
│       └── preprocessing_ablation.yaml
│
├── src/
│   └── dr_multiscale_mil/                   # Core Python Research Package
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── harmonization.py             # Dataset schema mapping & patient ID namespacing
│       │   ├── roi_crop.py                  # Auto-crop circular fundus ROI
│       │   ├── split_generator.py           # In-domain & LODO patient-level split generator
│       │   ├── augmentations.py             # Albumentations pipelines for Global & Local streams
│       │   └── dataset.py                   # PyTorch FundusDataset & DataLoader
│       ├── models/
│       │   ├── __init__.py
│       │   ├── backbones.py                 # EfficientNet, DenseNet, ResNet feature extractors
│       │   ├── attention.py                 # CBAM, Spatial/Channel Attention, MSAG
│       │   ├── mil_pooling.py               # Attention MIL Pooling Module
│       │   ├── heads.py                     # Consistent Ordinal Head, CORAL Head, Softmax Head
│       │   └── multiscale_mil_net.py        # Dual-branch MultiScaleMILNet
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── ordinal_loss.py              # Consistent Ordinal Loss & CORAL Loss
│       │   └── qwk_loss.py                  # Soft-QWK & classification losses
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py                   # AMP FP16 engine, CosineAnnealing, optimizer
│       │   └── early_stopping.py            # Early stopping monitored on validation QWK
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py                   # QWK, Cohen's Kappa, AUC, Referable Sens/Spec, ECE
│       │   └── domain_shift.py              # Cross-domain degradation quantification
│       └── visualization/
│           ├── __init__.py
│           ├── gradcam.py                   # Global stream Grad-CAM
│           ├── mil_attention_map.py         # Local patch attention weight heatmaps
│           ├── embeddings.py                # t-SNE / UMAP domain alignment visualization
│           └── paper_figures.py             # Automated Figure 1-6 generation (300 DPI / PDF)
│
├── scripts/                                 # Reproduction Shell Scripts
│   ├── 01_preprocess_data.sh
│   ├── 02_make_splits.sh
│   ├── 03_train_in_domain.sh
│   ├── 04_run_lodo_benchmark.sh
│   ├── 05_run_head_ablation.sh
│   └── 06_generate_figures.sh
│
├── notebooks/                               # Self-Contained Google Colab Notebooks
│   ├── 01_data_harmonization_and_lodo.ipynb
│   ├── 02_train_main_model.ipynb
│   ├── 03_head_ablation_analysis.ipynb
│   ├── 04_lodo_generalization_eval.ipynb
│   └── 05_xai_and_publication_figures.ipynb
│
├── checkpoints/                             # Model weights & training artifacts
│   ├── checkpoints_main/
│   ├── checkpoints_lodo/
│   └── checkpoints_head_ablation/
│
├── results/                                 # Output evaluation artifacts
│   ├── figures/                             # Fig 1 to Fig 6 (PNG 300 DPI + PDF)
│   ├── tables/                              # Benchmark CSV tables
│   └── logs/                                # Training run logs & audit reports
│
└── docs/                                    # Documentation
    ├── dataset_setup.md
    ├── reproduce_experiments.md
    └── colab_guide.md
```

**Structure Decision**: Selected standard Python package hierarchy (`src/dr_multiscale_mil/`) paired with declarative YAML configurations (`configs/`) and single-command shell scripts (`scripts/`). This structure guarantees maximum reproducibility for reviewers, allows seamless installation (`pip install -e .`), and maintains strict separation between library code, experimental configs, and output figures/tables.

---

## Complexity Tracking

> **Constitution Check passed with zero violations. No unjustified complexity introduced.**
