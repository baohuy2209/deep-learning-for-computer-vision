# Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Cumulative Link Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI Tests](https://img.shields.io/badge/CI-passing-brightgreen.svg)](.github/workflows/tests.yml)

Official Python implementation and reproducibility repository accompanying the paper:  
**"Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale Multiple Instance Learning with Cumulative Link Models"**

---

## 📌 Abstract

Diabetic Retinopathy (DR) grading under real-world clinical deployment faces substantial domain shift caused by heterogeneous photographic sensors, variable illumination, diverse optical fields of view, and distinct patient demographics. We present a multi-scale architecture that couples a **Global Context Stream** (ResNet-50 enhanced with CBAM, Non-Local attention, and Quadrant token pooling) with a **Local Multiple Instance Learning (MIL) Stream** (EfficientNet-B0 instance encoder with learnable Gated Attention dual pooling over high-resolution retinal patches). To align the network's predictions with clinical disease progression ($0 < 1 < 2 < 3 < 4$), we formulate classification through a **Cumulative Link Model (CLM)** head with softplus monotonic cutpoints and a complementary log-log link function, optimized with Continuous Quadratic Weighted Kappa (QWK) loss.

Evaluated on a multi-center benchmark of six clinical cohorts (**APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS**; comprising $N = 55{,}570$ images across $35{,}474$ patients) under a strict patient-isolated **6-Fold Leave-One-Dataset-Out (LODO)** cross-validation protocol, the framework establishes robust cross-domain generalization and demonstrates high spatial alignment against expert pixel lesion ground truth from IDRiD.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/baohuy2209/deep-learning-for-computer-vision.git
cd deep-learning-for-computer-vision

# Create virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\Activate.ps1

# Install core package in editable mode and dependencies
pip install -e .
pip install -r requirements.txt
```

### 2. Verify Installation with Unit Tests
```bash
pytest tests/ -v
```

---

## 🔬 End-to-End Experimental Workflow

All experiments can be executed via modular CLI scripts or explored through interactive Jupyter notebooks.

### Step 1: Data Acquisition & Preprocessing
```bash
# Set Kaggle credentials (if downloading Kaggle-hosted cohorts)
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Download raw cohorts (or place manually in data/raw/<cohort>)
python scripts/download_dataset.py --cohort aptos

# Execute circular Field-of-View (FOV) cropping & 512x512 standardization
python scripts/prepare_dataset.py --all

# Generate unified master metadata CSV
python scripts/build_master_metadata.py

# Generate strict patient-isolated 6-fold LODO splits
python scripts/generate_lodo_splits.py
```

### Step 2: 6-Fold LODO Cross-Domain Generalization Benchmark
```bash
# Run proposed variant v3 on held-out target cohort (e.g. APTOS 2019)
python scripts/run_lodo.py --variant v3 --held-out aptos --device cuda

# Verify pipeline integrity with CPU dry run
python scripts/run_lodo.py --variant v3 --held-out aptos --dry-run --device cpu

# Available model variants: v0 (Global), v1 (MIL), v2 (Context), v3 (Proposed)
# Available held-out cohorts: aptos, ddr, deepdrid, idrid, messidor2, eyepacs
```

### Step 3: Controlled Output-Head & Loss Ablation Suite
```bash
# Benchmark all 5 prediction heads under standardized ResNet-50 features
python scripts/run_head_ablation.py --all-heads --device cuda

# Benchmark individual head (softmax_ce, softmax_qwk, coral, corn, clm_qwk)
python scripts/run_head_ablation.py --head clm_qwk --device cuda
```

### Step 4: Quantitative Explainability (XAI) against IDRiD Pixel Masks
```bash
# Benchmark Grad-CAM and MIL Attention against pixel lesion ground truth
python scripts/run_xai.py --checkpoint checkpoints/v3_best.pt --variant v3 --device cuda

# Dry-run validation
python scripts/run_xai.py --variant v3 --dry-run --device cpu
```

### Step 5: Export Publication Figures & LaTeX Tables
```bash
# Compile results into 300 DPI vector PDF figures and LaTeX tables
python scripts/export_paper_artifacts.py --results-dir results/ --output-dir results/paper_artifacts/
```

---

## 🏛️ Dual-Layer Repository Architecture

This repository adopts a strict **dual-layer architecture** designed to satisfy both interactive research provenance and high-performance modular engineering:

```
+-----------------------------------------------------------------------------+
|                          CANONICAL RESEARCH LAYER                            |
|                                                                             |
|  * 5 Golden Notebooks remain 100% self-contained, complete, and runnable.    |
|  * Self-contained: Includes raw data ingestion, model classes, losses, etc. |
|  * NEVER converted into thin wrappers; zero imports from `drdg` package.    |
+-----------------------------------------------------------------------------+
                                      |
                     Exact Mathematical & Logic Parity
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                          MODULAR PYTHON PACKAGE                             |
|                                                                             |
|  * `src/drdg`: Independent, testable, CLI-accessible Python package.       |
|  * Strict numerical, tensor, and pipeline equivalence verified by pytest.   |
|  * Clean CLI scripts for automated cluster sweeps and CI/CD pipelines.     |
+-----------------------------------------------------------------------------+
```

---

## 📂 Repository Organization

```
deep-learning-for-computer-vision/
├── src/drdg/                    # Core Modular Python Research Package (`drdg`)
│   ├── config.py                # Strongly-typed @dataclass experiment configurations (Zero YAML)
│   ├── paths.py                 # Centralized filesystem path resolution
│   ├── data/                    # Multi-cohort loading, FOV cropping, patching, LODO splits
│   ├── models/                  # ResNet-50, EfficientNet-B0, Attention (CBAM, NonLocal), Heads
│   ├── training/                # Trainer engine with AMP, early stopping & safe checkpointing
│   ├── evaluation/              # Bipartite LODO evaluation, QWK, RDR screening, statistics
│   ├── explainability/          # Grad-CAM, MIL attention maps, IDRiD pixel evaluator
│   ├── visualization/           # Publication figure generation utilities
│   └── utils/                   # Device detection, logging, seed control, I/O
├── notebooks/diabetic_retinopathy/pipeline/  # 5 Golden Canonical Research Notebooks
│   ├── merge-dataset/merge-dataset.ipynb     # Multi-cohort consolidation & circular FOV crop
│   ├── merge-dataset/merge_dataset_lodo.ipynb# 6-Fold 3-split LODO generator (18 CSVs)
│   ├── model/main_model (6).ipynb            # Proposed dual-stream model & QWK loss
│   ├── model/main_model_lodo.ipynb           # 6-fold LODO cross-domain benchmark
│   └── model/main_model_head_ablation (3).ipynb # 5-head classification/ordinal ablation
├── scripts/                     # Executable CLI experiment runners
│   ├── download_dataset.py      # Automated dataset downloader
│   ├── prepare_dataset.py       # FOV crop & 512x512 standardization
│   ├── build_master_metadata.py # Unified metadata compiler
│   ├── generate_lodo_splits.py  # 18-file patient-isolated LODO split generator
│   ├── train_variant.py         # Single model training runner
│   ├── run_lodo.py              # 6-fold bipartite LODO benchmark runner
│   ├── run_head_ablation.py     # 5-head controlled ablation runner
│   ├── run_xai.py               # Quantitative XAI evaluation runner
│   ├── verify_notebook_integrity.py # Automated golden notebook integrity checker
│   └── export_paper_artifacts.py# 300 DPI figures & LaTeX tables exporter
├── data/                        # Raw & processed data directory
│   ├── raw/                     # Original cohort downloads
│   ├── processed/native_cropped/# Native circular FOV cropped fundus images
│   └── processed/lodo_folds/    # 18 patient-isolated LODO split CSV files
├── results/                     # Serialized JSON metrics, CSVs & paper artifacts
├── docs/                        # Comprehensive technical documentation & audits
│   ├── NOTEBOOK_CODE_MAP.md     # 100% symbol mapping between notebooks and drdg
│   ├── DATA_FLOW_AUDIT.md       # 6-stage producer-consumer data flow audit
│   ├── REFACTOR_AUDIT.md        # Code vs documentation discrepancy audit (VERIFIED)
│   ├── DATA.md                  # Cohort overview, licensing, FOV & patient splits
│   ├── ARCHITECTURE.md          # Network architecture, tensor shapes, CLM math
│   ├── EXPERIMENTS.md           # Reproduction commands & evaluation protocols
│   ├── REPRODUCIBILITY.md       # Determinism, seed control, hardware benchmarks
│   ├── RESULTS_SCHEMA.md        # Bipartite JSON results serialization schema
│   └── XAI.md                   # Explainability methodology & quantitative validation
├── tests/                       # Complete automated pytest suite
│   ├── parity/                  # 49 numerical, gradient, and tensor parity tests
│   ├── test_imports.py          # Clean import validation
│   ├── test_lodo_splits.py      # 18 CSV schema & zero-leakage tests
│   └── test_result_schema.py    # Bipartite LODOFoldResult validation
├── .github/workflows/           # Automated GitHub Actions CI workflow
├── CITATION.cff                 # Citation metadata
├── CONTRIBUTING.md              # Open-source contribution guidelines
└── LICENSE                      # MIT License
```

---

## 📚 Technical Documentation Index

For in-depth explanations of data engineering, mathematical formulations, and experimental methodologies:
- [`docs/NOTEBOOK_CODE_MAP.md`](docs/NOTEBOOK_CODE_MAP.md): Complete 1-to-1 map between notebook code symbols and `src/drdg` modules.
- [`docs/DATA_FLOW_AUDIT.md`](docs/DATA_FLOW_AUDIT.md): Producer-consumer matrix across the 6-stage research pipeline.
- [`docs/REFACTOR_AUDIT.md`](docs/REFACTOR_AUDIT.md): Comprehensive discrepancy audit certifying final status **VERIFIED**.
- [`docs/DATA.md`](docs/DATA.md): Details on the 6 clinical cohorts, device specifications, FOV extraction, and 18 LODO fold CSVs.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Mathematical derivation of the Cumulative Link Model, clog-log link, and attention blocks.
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): Detailed benchmarking protocols, metric formulations, and statistical significance testing.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): Hardware specifications, PyTorch 2.6 compatibility notes, and deterministic PRNG controls.
- [`docs/RESULTS_SCHEMA.md`](docs/RESULTS_SCHEMA.md): Formal JSON schemas ensuring bipartite `in_domain_val` and `ood_test` reporting.
- [`docs/XAI.md`](docs/XAI.md): Methodology for Grad-CAM, MIL attention density mapping, and Pointing Game/Recall evaluation against IDRiD masks.

---

## 📖 Citation

If you find this codebase or research useful in your work, please cite:

```bibtex
@article{drdg2026multiscale,
  title={Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale Multiple Instance Learning with Cumulative Link Models},
  author={Research Authors},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
