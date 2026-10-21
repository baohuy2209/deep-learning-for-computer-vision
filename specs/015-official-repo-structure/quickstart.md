# Quickstart Validation Guide: Official ICCPR Repository

**Feature**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/015-official-repo-structure/spec.md) | **Date**: 2026-09-19  
**Paper**: *Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression* (ICCPR Conference)

---

## 1. Prerequisites & Environment Setup

```bash
# Clone and enter the repository
git clone https://github.com/baohuy2209/deep-learning-for-computer-vision.git
cd deep-learning-for-computer-vision

# Create Python virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt
# Or for Windows workstation:
pip install -r requirement-win.txt
```

---

## 2. Step 1: Data Preprocessing & LODO Split Generation

```bash
# 1. Auto-crop fundus circular ROI (remove black borders) across all raw cohorts
bash scripts/01_preprocess_data.sh

# 2. Generate In-Domain (70/15/15) and 6-Fold LODO Splits with zero patient leakage
bash scripts/02_make_splits.sh
```

**Expected Outcome**:
- Cropped images saved to `data/raw/hybrid/native_cropped/`
- 18 LODO split CSV files + `lodo_splits_summary.csv` generated in `data/processed/lodo_folds/`
- Zero patient leakage verified ($Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$)

---

## 3. Step 2: In-Domain Champion Model Training

```bash
# Train the Multi-Scale MIL network with Consistent Ordinal Regression
bash scripts/03_train_in_domain.sh
```

**Expected Outcome**:
- Training runs with Mixed Precision (AMP FP16)
- Checkpoint saved to `checkpoints/checkpoints_main/best_model_qwk.pt`
- Training curves and metric logs saved to `results/in_domain/`

---

## 4. Step 3: 6-Fold LODO Generalization Benchmark

```bash
# Train and evaluate all 6 LODO folds (held-out out-of-distribution evaluation)
bash scripts/04_run_lodo_benchmark.sh
```

**Expected Outcome**:
- Checkpoints saved for each fold under `checkpoints/checkpoints_lodo/`
- Aggregated OOD performance metrics exported to `results/tables/lodo_benchmark_results.csv`
- Domain shift gap metrics computed comparing in-domain vs OOD performance

---

## 5. Step 4: Generate Publication Figures & Tables

```bash
# Generate all Figures (1 to 6) and tables for the ICCPR manuscript
bash scripts/06_generate_figures.sh
```

**Expected Outcome**:
- High-resolution publication figures saved to `results/figures/` (300 DPI PNG + vector PDF):
  - `fig1_pipeline_architecture.pdf`
  - `fig2_dataset_cohort_distribution.pdf`
  - `fig3_head_ablation_calibration.pdf`
  - `fig4_lodo_domain_shift_radar.pdf`
  - `fig5_gradcam_mil_attention_xai.pdf`
  - `fig6_tsne_domain_alignment.pdf`
