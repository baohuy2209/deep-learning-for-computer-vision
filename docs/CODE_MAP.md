# Code Migration & Architecture Map

This document maps legacy exploratory files, notebooks, and scripts to their modular, reproducible counterparts in the `drdg` package and CLI suites.

---

## 1. High-Level Package Architecture

The refactored package `drdg` (`src/drdg/`) is structured into decoupled, single-responsibility modules:

```
src/drdg/
├── config.py                  # Strongly-typed @dataclass configuration schemas
├── paths.py                   # Dynamic filesystem path resolution from repo root
├── data/                      # Multi-cohort loading, FOV cropping, patching, splits
│   ├── balancing.py           # Class weighting and balanced samplers
│   ├── collate.py             # Dual-branch batch collation
│   ├── datasets.py            # RetinalFundusDataset & MILBatchDataset
│   ├── fov.py                 # Otsu/luminance circular FOV mask extraction
│   ├── metadata.py            # Master metadata construction & registry
│   ├── patches.py             # High-resolution patch extraction (tensor & numpy)
│   ├── patient_ids.py         # Cohort-specific patient ID extraction regexes
│   ├── preprocessing.py       # Standardized resize, padding, normalization
│   ├── splits.py              # Zero-leakage patient-stratified split generation
│   ├── transforms.py          # PyTorch / Albumentations data augmentation
│   └── downloaders/           # Automated Kaggle & HTTP dataset downloaders
├── models/                    # Neural network components & model variants
│   ├── attention/             # CBAM, Non-Local, and Quadrant token pooling
│   ├── feature_streams/       # Global ResNet-50 and Local MIL EfficientNet-B0
│   ├── fusion.py              # Cross-scale multimodal residual fusion
│   ├── heads/                 # Softmax, CORAL, CORN, Cumulative Link Model (CLM)
│   ├── losses/                # Continuous QWK, ordinal log-likelihood, CORAL/CORN
│   ├── variants/              # v0 (Global), v1 (MIL), v2 (Context), v3 (Ordinal CLM)
│   └── factory.py             # build_variant, build_ablation_model
├── training/                  # Training loops, optimization, checkpointing
│   ├── trainer.py             # DualBranchTrainer with AMP & early stopping
│   ├── head_trainer.py        # Isolated head ablation trainer
│   └── checkpoint.py          # PyTorch 2.6 safe checkpoint serialization
├── evaluation/                # Clinical metrics & statistical significance
│   ├── grade_metrics.py       # 5-class QWK, Accuracy, Within-1, Off-by-2+
│   ├── ordinal_metrics.py     # Cumulative link metrics & confusion matrices
│   ├── rdr.py                 # RDR threshold calibration (>=95% spec) & evaluation
│   ├── result_schema.py       # LODOFoldResult & MetricBundle JSON schemas
│   └── statistics.py          # Pairwise Wilcoxon signed-rank significance tests
├── explainability/            # XAI saliency & validation against IDRiD masks
│   ├── gradcam.py             # Global branch Grad-CAM extractor
│   ├── mil_saliency.py        # Local MIL attention continuous density mapping
│   ├── idrid_dataset.py       # IDRiD pixel lesion mask loader (MA, HE, EX, SE)
│   └── metrics.py             # Pointing Game, Area-Matched Recall, Pixel AUROC
├── visualization/             # 300 DPI publication figures & LaTeX tables
│   └── paper_figures.py       # Figures 2-5 generation & table generators
└── utils/                     # Device detection, logging, seed control, I/O
    ├── device.py              # CUDA/MPS/CPU detection & AMP dtype selection
    ├── io.py                  # JSON/CSV readers and writers
    ├── logging.py             # Standardized formatted logger
    └── seed.py                # Deterministic seed control across all PRNGs
```

---

## 2. Legacy-to-Modern Migration Table

| Legacy File / Script | Refactored `drdg` Module | Refactored CLI Script | Refactored Notebook | Description |
|---|---|---|---|---|
| `src/download-dataset/*.py` | `drdg.data.downloaders.*` | `scripts/download_dataset.py` | `00_dataset_harmonization.ipynb` | Automated dataset acquisition & Kaggle credential handling |
| `notebooks/diabetic_retinopathy/00_*.ipynb` | `drdg.data.preprocessing`, `drdg.data.fov` | `scripts/prepare_dataset.py` | `00_dataset_harmonization.ipynb` | Circular FOV cropping and 512x512 standardization |
| `notebooks/diabetic_retinopathy/01_*.ipynb` | `drdg.data.splits`, `drdg.data.patient_ids` | `scripts/generate_lodo_splits.py` | `01_lodo_split_generation.ipynb` | Patient ID extraction & zero-leakage LODO splitting |
| `notebooks/diabetic_retinopathy/02_*.ipynb` | `drdg.models.feature_streams.global_context` | `scripts/train_variant.py --variant v0` | `10_v0_global_baseline.ipynb` | Global ResNet-50 baseline implementation |
| `notebooks/diabetic_retinopathy/03_*.ipynb` | `drdg.models.feature_streams.local_mil` | `scripts/train_variant.py --variant v1` | `11_v1_local_mil.ipynb` | Patch extraction & Gated Attention MIL stream |
| `notebooks/diabetic_retinopathy/04_*.ipynb` | `drdg.models.attention.*` | `scripts/train_variant.py --variant v2` | `12_v2_global_context.ipynb` | CBAM, Non-Local, and Quadrant token attention |
| `notebooks/diabetic_retinopathy/05_*.ipynb` | `drdg.models.heads.clm`, `drdg.models.losses.clm` | `scripts/train_variant.py --variant v3` | `13_v3_ordinal_clm.ipynb` | Cumulative Link Model (CLM) ordinal regression head |
| `notebooks/diabetic_retinopathy/06_*.ipynb` | `drdg.experiments.lodo` | `scripts/run_lodo.py` | `20_lodo_benchmark.ipynb` | 6-Fold Leave-One-Dataset-Out cross-domain evaluation |
| `notebooks/diabetic_retinopathy/07_*.ipynb` | `drdg.experiments.head_ablation` | `scripts/run_head_ablation.py` | `21_head_ablation.ipynb` | Controlled 5-head comparison suite |
| `notebooks/diabetic_retinopathy/08_*.ipynb` | `drdg.evaluation.rdr` | `scripts/run_lodo.py` | `23_rdr_screening.ipynb` | RDR screening calibration at 95% specificity |
| `notebooks/diabetic_retinopathy/09_*.ipynb` | `drdg.explainability.*` | `scripts/run_xai.py` | `24_xai_idrid.ipynb` | Grad-CAM, MIL attention & IDRiD lesion mask benchmark |
| Exploratory plotting scripts | `drdg.visualization.paper_figures` | `scripts/export_paper_artifacts.py` | `30_paper_figures.ipynb` | 300 DPI vector PDF figures & LaTeX table export |
