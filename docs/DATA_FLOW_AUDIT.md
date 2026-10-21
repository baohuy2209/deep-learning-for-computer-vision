# Data Flow and Producer-Consumer Dependency Audit

This document records the exact data-dependency graph, directory layout, and producer-consumer relationships of all data artifacts across the research pipeline.

---

## 1. End-to-End Workflow Graph

```text
STAGE 1: RAW DATA ACQUISITION
   - Automated scripts: ddr.py, deep-drid.py, idrid.py, messidor.py
   - Manual placement: EyePACS, APTOS
   Output: data/raw/<cohort_key>/
      ↓
STAGE 2: DATASET-SPECIFIC CONSOLIDATION
   - Scripts: total_aptos.py, total_deepdrid.py, total_idrid.py, total_idrid-segmentation.py
   Output: *_total.csv, *_total_images/
      ↓
STAGE 3: MULTI-COHORT HARMONIZATION (merge-dataset.ipynb)
   - Harmonization, non-black circular ROI crop, patient-level 70/15/15 split, quota balancing
   Outputs:
     - data/raw/hybrid/native_cropped/<cohort_key>/
     - data/raw/hybrid/train_balanced_metadata.csv
     - data/raw/hybrid/val_metadata.csv
     - data/raw/hybrid/test_metadata.csv
      ↓
      ├───────────────────────────────────┬──────────────────────────────────────┐
      ↓                                   ↓                                      ↓
STAGE 4A: STANDARD MAIN MODEL       STAGE 4B: HEAD ABLATION             STAGE 5: LODO GENERATION
   (main_model (6).ipynb)              (main_model_head_ablation (3))      (merge_dataset_lodo.ipynb)
   Consumes:                           Consumes:                           Consumes:
   - native_cropped/                   - native_cropped/                   - native_cropped/
   - train_balanced_metadata.csv       - train_balanced_metadata.csv       - multi-cohort metadata
   - val_metadata.csv                  - val_metadata.csv                  Produces:
   - test_metadata.csv                 - test_metadata.csv                 - data/processed/lodo_folds/
                                                                             fold_{0-5}_{cohort}_{split}.csv
                                                                             (18 CSV files)
                                                                                  ↓
                                                                        STAGE 6: LODO MODEL EXPERIMENTS
                                                                           (main_model_lodo.ipynb)
                                                                           Consumes:
                                                                           - native_cropped/
                                                                           - 18 frozen fold CSV files
                                                                           Produces:
                                                                           - checkpoints/lodo_v*_best.pth
                                                                           - results/lodo/*.json
```

---

## 2. Producer-Consumer Artifact Matrix

| Artifact Path | Producer | Consumer | Research Semantics | Verified Status |
|---|---|---|---|:---:|
| `data/raw/aptos/aptos_total.csv` | `total_aptos.py` | `merge-dataset.ipynb`, `merge_dataset_lodo.ipynb` | Consolidated APTOS labels & image paths | EXACT |
| `data/raw/aptos/aptos_total_images/` | `total_aptos.py` | `merge-dataset.ipynb`, `merge_dataset_lodo.ipynb` | Consolidated raw APTOS images | EXACT |
| `data/raw/ddr/DR_grading.csv` | `ddr.py` (download) | `merge-dataset.ipynb`, `merge_dataset_lodo.ipynb` | Raw DDR grading manifest | EXACT |
| `data/raw/deepdrid/deepdrid_total.csv` | `total_deepdrid.py` | `merge-dataset.ipynb`, `merge_dataset_lodo.ipynb` | Consolidated DeepDRiD labels | EXACT |
| `data/raw/idrid/idrid_total.csv` | `total_idrid.py` | `merge-dataset.ipynb`, `merge_dataset_lodo.ipynb` | Consolidated IDRiD grading labels | EXACT |
| `data/raw/idrid_segmentation/` | `total_idrid-segmentation.py` | XAI evaluation (`xai_idrid.py`) | Lesion segmentation masks (XAI-only) | ISOLATED |
| `data/raw/messidor2preprocess/messidor_data.csv` | `messidor.py` | `merge-dataset.ipynb`, `merge_dataset_lodo.ipynb` | Messidor-2 clinical labels | EXACT |
| `data/raw/eyepacs/trainLabels.csv` | Manual acquisition | `merge-dataset.ipynb`, `merge_dataset_lodo.ipynb` | EyePACS 35k label manifest | EXACT |
| `data/raw/hybrid/native_cropped/` | `merge-dataset.ipynb` | `main_model`, `head_ablation`, `lodo_folds`, `lodo_training` | Harmonized non-black circular ROI crops | EXACT |
| `data/raw/hybrid/train_balanced_metadata.csv` | `merge-dataset.ipynb` | `main_model (6).ipynb`, `head_ablation (3).ipynb` | Standard 70% balanced training metadata | EXACT |
| `data/raw/hybrid/val_metadata.csv` | `merge-dataset.ipynb` | `main_model (6).ipynb`, `head_ablation (3).ipynb` | Standard 15% in-domain validation metadata | EXACT |
| `data/raw/hybrid/test_metadata.csv` | `merge-dataset.ipynb` | `main_model (6).ipynb`, `head_ablation (3).ipynb` | Standard 15% held-out test metadata | EXACT |
| `data/processed/lodo_folds/fold_*_train.csv` | `merge_dataset_lodo.ipynb` | `main_model_lodo.ipynb`, `scripts/run_lodo.py` | 85% source pool, quota-balanced (6 files) | EXACT |
| `data/processed/lodo_folds/fold_*_val.csv` | `merge_dataset_lodo.ipynb` | `main_model_lodo.ipynb`, `scripts/run_lodo.py` | 15% source pool, natural prevalence (6 files) | EXACT |
| `data/processed/lodo_folds/fold_*_test.csv` | `merge_dataset_lodo.ipynb` | `main_model_lodo.ipynb`, `scripts/run_lodo.py` | 100% held-out cohort test (6 files) | EXACT |
| `data/processed/lodo_folds/lodo_splits_summary.csv` | `merge_dataset_lodo.ipynb` | Audit & verification | Patient & image counts per fold | EXACT |
| `checkpoints/lodo_{variant}_fold_{idx}_{cohort}_best.pth` | `main_model_lodo.ipynb`, `run_lodo.py` | Evaluation & publication figures | Best checkpoint selected on source validation QWK | EXACT |
| `results/lodo/{variant}_fold{idx}_{cohort}.json` | `main_model_lodo.ipynb`, `run_lodo.py` | Summary tables & publication figures | Bipartite evaluation metrics (`in_domain_val`, `ood_test`) | EXACT |

---

## 3. Separation of Independent Data Pathways

### 3.1 Grading Cohorts vs. IDRiD Lesion Segmentation
The repository enforces strict separation between:
1. **The 6 Diagnostic Grading Cohorts** (APTOS, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS): These cohorts drive the multi-cohort merge and the 6-fold LODO cross-domain benchmark.
2. **IDRiD Segmentation Dataset**: Preprocessed via `total_idrid-segmentation.py` into `data/raw/idrid_segmentation/`. It contains pixel-level lesion masks (microaneurysms, hemorrhages, hard exudates, soft exudates) and is consumed strictly by XAI evaluation modules (`tests/test_xai_pointing_game.py`, `drdg.explainability`). It is never mixed into grading metadata.

### 3.2 Standard Split vs. LODO Split Isolation
- Standard experiments (`main_model (6).ipynb`, `main_model_head_ablation (3).ipynb`) consume `data/raw/hybrid/{train_balanced,val,test}_metadata.csv`.
- LODO experiments (`main_model_lodo.ipynb`, `scripts/run_lodo.py`) consume the 18 files in `data/processed/lodo_folds/`.
- Neither workflow dynamically overwrites or regenerates the other's splits during model training.
