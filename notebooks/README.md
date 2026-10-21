# Reproducibility & Demonstration Notebooks

This directory contains standalone, reproducible Jupyter notebooks demonstrating data harmonization, zero-leakage cohort splitting, architectural variants, controlled ablations, clinical screening, explainability, and paper figure generation.

All notebooks are designed to run cleanly on standard Python environments with optional GPU acceleration, and cell execution outputs are stripped to ensure clean version control.

---

## Notebook Index

### Data Pipeline & Cohort Harmonization (00–09)
- [`00_dataset_harmonization.ipynb`](00_dataset_harmonization.ipynb): Standardizes the 6 distinct fundus datasets (APTOS, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS) into unified ICDRS 5-class labels and standardized metadata schemas.
- [`01_lodo_split_generation.ipynb`](01_lodo_split_generation.ipynb): Generates strict patient-isolated Leave-One-Dataset-Out (LODO) 6-fold splits, verifying $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, and $\text{source} \cap \text{target} = \emptyset$.
- [`02_dataset_audit.ipynb`](02_dataset_audit.ipynb): Audits class balances, missing image files, duplicate patient records, and field-of-view (FOV) properties across all harmonized cohorts.

### Model Architecture Evolution & Verification (10–19)
- [`10_v0_global_baseline.ipynb`](10_v0_global_baseline.ipynb): Implements and validates Variant `v0` (Standard ResNet-50 global baseline with cross-entropy classification).
- [`11_v1_local_mil.ipynb`](11_v1_local_mil.ipynb): Implements and validates Variant `v1` (Dual-branch MIL combining global ResNet-50 with local patch EfficientNet-B0 and Gated Attention pooling).
- [`12_v2_global_context.ipynb`](12_v2_global_context.ipynb): Implements and validates Variant `v2` (Dual-branch MIL enhanced with Global Context Attention: CBAM, Non-Local, and Quadrant token pooling).
- [`13_v3_ordinal_clm.ipynb`](13_v3_ordinal_clm.ipynb): Implements and validates Variant `v3` (Proposed model: Global Context + Local MIL + Cumulative Link Model ordinal head with complementary log-log link and continuous QWK loss).

### Evaluation, Benchmarks & Ablation Studies (20–29)
- [`20_lodo_benchmark.ipynb`](20_lodo_benchmark.ipynb): Executes and visualizes the full 6-fold Leave-One-Dataset-Out cross-domain generalization benchmark across all 6 cohorts.
- [`21_head_ablation.ipynb`](21_head_ablation.ipynb): Executes a controlled ablation of 5 output heads (Softmax CE, Softmax QWK, CORAL, CORN, CLM QWK) on standardized ResNet-50 features ($D=512$).
- [`22_grade_specific_analysis.ipynb`](22_grade_specific_analysis.ipynb): Evaluates grade-specific performance, confusion matrices, and off-by-1 vs off-by-2+ error distributions across cohorts.
- [`23_rdr_screening.ipynb`](23_rdr_screening.ipynb): Calibrates Referable DR (ICDRS $\ge 2$) decision thresholds ($\tau^*$) at $\ge 95\%$ specificity strictly on source validation data, and evaluates out-of-domain target generalization.
- [`24_xai_idrid.ipynb`](24_xai_idrid.ipynb): Quantitatively benchmarks Global Grad-CAM and Local MIL Gated Attention saliency maps against expert IDRiD pixel lesion ground truth (Pointing Game, Saliency Recall, Pixel AUROC).

### Paper Artifacts & Publication Figures (30)
- [`30_paper_figures.ipynb`](30_paper_figures.ipynb): Generates high-resolution publication figures (300 DPI vector PDF and raster PNG) and LaTeX summary tables matching the submitted manuscript.

---

## Legacy Reference Code
Legacy exploratory notebooks developed during initial model prototyping are preserved intact in `notebooks/diabetic_retinopathy/` for provenance and historical reference. All new reproducible workflows use the clean `drdg` package.
