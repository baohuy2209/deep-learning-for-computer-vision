# Reproducing Experiments: ICCPR Paper

This document provides exact commands to reproduce all tables and figures presented in:
*Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression* (ICCPR).

---

## 1. Quick Reproduction Pipeline

```bash
# Step 1: Preprocess raw fundus images (ROI crop)
bash scripts/01_preprocess_data.sh

# Step 2: Generate zero-leakage in-domain (70/15/15) and 6-Fold LODO splits
bash scripts/02_make_splits.sh

# Step 3: Train champion model on in-domain benchmark
bash scripts/03_train_in_domain.sh

# Step 4: Run 6-Fold Leave-One-Dataset-Out (LODO) benchmark
bash scripts/04_run_lodo_benchmark.sh

# Step 5: Run prediction head ablation study
bash scripts/05_run_head_ablation.sh

# Step 6: Render all publication figures (1 to 6) and tables
bash scripts/06_generate_figures.sh
```

---

## 2. Expected Results & Output Locations

- **Table 1: In-Domain Benchmark Comparison**: `results/tables/in_domain_eval.json`
- **Table 2: 6-Fold LODO Out-of-Distribution Generalization**: `results/tables/lodo_benchmark_results.csv`
- **Table 3: Prediction Head Ablations (Softmax vs CORAL vs Consistent Ordinal)**: `results/tables/head_ablation_summary.csv`
- **Figures 1 to 6 (PNG 300 DPI & Vector PDF)**: `results/figures/`
