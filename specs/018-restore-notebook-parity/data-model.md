# Phase 1 Data Model: Exact Notebook-to-Python Logic Parity

**Feature**: `018-restore-notebook-parity` | **Date**: 2026-09-24

This document defines the data structures, metadata schemas, directory hierarchies, and artifact relationships required to maintain exact parity between the canonical research notebooks and the modular `drdg` implementation.

---

## 1. Physical Directory Hierarchy

```text
data/
├── raw/                                    # Read-only raw cohort archives
│   ├── aptos/
│   │   ├── aptos_total.csv                 # Consolidated APTOS metadata
│   │   └── aptos_total_images/             # Consolidated APTOS raw images
│   ├── ddr/
│   │   ├── DR_grading.csv
│   │   └── DR_grading/
│   ├── deepdrid/
│   │   ├── deepdrid_total.csv
│   │   └── deepdrid_total_images/
│   ├── idrid/
│   │   ├── idrid_total.csv
│   │   └── idrid_total_images/
│   ├── idrid_segmentation/                 # Separate XAI lesion evaluation data
│   │   ├── idrid_segmentation_total.csv
│   │   ├── idrid_segmentation_total_images/
│   │   └── idrid_segmentation_total_masks/
│   ├── messidor2preprocess/
│   │   ├── messidor_data.csv
│   │   └── messidor2preprocess/
│   ├── eyepacs/
│   │   ├── trainLabels.csv
│   │   └── train/
│   └── hybrid/                             # Harmonized outputs from merge-dataset.ipynb
│       ├── native_cropped/                 # Non-black circular ROI crops
│       │   ├── aptos/
│       │   ├── ddr/
│       │   ├── deepdrid/
│       │   ├── eyepacs/
│       │   ├── idrid/
│       │   └── messidor2/
│       ├── train_balanced_metadata.csv     # Standard 70% balanced training split
│       ├── val_metadata.csv                # Standard 15% validation split
│       └── test_metadata.csv               # Standard 15% test split
│
└── processed/
    └── lodo_folds/                         # 18 Frozen LODO fold CSVs from merge_dataset_lodo.ipynb
        ├── fold_0_aptos_train.csv
        ├── fold_0_aptos_val.csv
        ├── fold_0_aptos_test.csv
        ├── fold_1_idrid_train.csv
        ├── fold_1_idrid_val.csv
        ├── fold_1_idrid_test.csv
        ├── fold_2_messidor2_train.csv
        ├── fold_2_messidor2_val.csv
        ├── fold_2_messidor2_test.csv
        ├── fold_3_ddr_train.csv
        ├── fold_3_ddr_val.csv
        ├── fold_3_ddr_test.csv
        ├── fold_4_eyepacs_train.csv
        ├── fold_4_eyepacs_val.csv
        ├── fold_4_eyepacs_test.csv
        ├── fold_5_deepdrid_train.csv
        ├── fold_5_deepdrid_val.csv
        ├── fold_5_deepdrid_test.csv
        ├── lodo_splits_summary.csv
        └── lodo_folds.zip
```

---

## 2. Core Entities

### Entity 1: `HarmonizedMetadataRecord`
Represents an individual fundus photograph across any of the six clinical cohorts.

| Field Name | Type | Description | Example |
|---|---|---|---|
| `image_id` | `str` | Unique image identifier | `aptos_total_images_0001` |
| `diagnosis` | `int` | Diabetic retinopathy grade ($0 \dots 4$) | `2` |
| `dataset_name` | `str` | Display name of the cohort | `APTOS 2019` |
| `dataset_key` | `str` | Normalized canonical slug | `aptos` |
| `raw_full_path` | `str` | Absolute/relative path to raw image | `data/raw/aptos/aptos_total_images/000c1434d8d7.png` |
| `native_path` | `str` | Path to cropped native ROI image | `data/raw/hybrid/native_cropped/aptos/000c1434d8d7.png` |
| `patient_id` | `str` | Namespaced patient identifier (`{dataset_key}::{parsed_id}`) | `aptos::000c1434d8d7` |
| `side` | `str` | Laterality (`left`, `right`, or `unknown`) | `left` |

**Validation Rules**:
- `diagnosis` $\in \{0, 1, 2, 3, 4\}$.
- `dataset_key` $\in \{\text{"aptos"}, \text{"idrid"}, \text{"messidor2"}, \text{"ddr"}, \text{"eyepacs"}, \text{"deepdrid"}\}$.
- `patient_id` must start with `{dataset_key}::` to prevent cross-cohort namespace collision.

---

### Entity 2: `LODOFold`
Represents one fold of the 6-Fold Leave-One-Dataset-Out cross-domain benchmark.

| Field Name | Type | Description |
|---|---|---|
| `fold_idx` | `int` | Fold index $0 \dots 5$ |
| `target_cohort` | `str` | Canonical slug of the held-out cohort |
| `fold_slug` | `str` | Slug used in CSV filenames (matches target_cohort) |
| `source_cohorts` | `List[str]` | List of the 5 training cohorts |
| `train_df` | `DataFrame` | 85% source patients, balanced via quota sampling $(1500, 2500)$ per grade |
| `val_df` | `DataFrame` | 15% source patients, natural prevalence |
| `test_df` | `DataFrame` | 100% held-out cohort patients, natural prevalence |

**Partition Isolation Assertions (Zero Leakage)**:
- $\text{Patients}(\text{train\_df}) \cap \text{Patients}(\text{val\_df}) = \emptyset$
- $\text{Patients}(\text{train\_df}) \cap \text{Patients}(\text{test\_df}) = \emptyset$
- $\text{Patients}(\text{val\_df}) \cap \text{Patients}(\text{test\_df}) = \emptyset$

---

### Entity 3: `PatchBag`
Represents high-resolution native patches extracted from an eye's non-black retinal circular ROI.

| Attribute | Value / Shape | Description |
|---|---|---|
| `patch_size` | $224 \times 224$ pixels | High-resolution patch dimension |
| `primary_stride` | $112$ pixels | Sliding-window hop size |
| `fallback_stride` | $56$ pixels | Adaptive hop size if $< K_{\min}$ valid patches found |
| `min_fov_ratio` | $0.60$ | Minimum foreground non-black pixel occupancy |
| `train_k_patches` | $12$ | Uniform random subsample during training |
| `eval_max_patches` | $36$ | Maximum sequential patches during validation/test |
| `bag_tensor_shape` | $(B, K_{\max}, 3, 224, 224)$ | Collation batch tensor with padding |
| `patch_mask_shape`| $(B, K_{\max})$ | Boolean validity mask ($1 = \text{valid}, 0 = \text{pad}$) |

---

### Entity 4: `LODOFoldResult`
Structured JSON output format for each completed fold evaluation.

```json
{
  "fold": 1,
  "held_out_dataset": "aptos",
  "fold_slug": "aptos",
  "variant": "v3",
  "best_epoch": 14,
  "in_domain_val": {
    "qwk": 0.885,
    "accuracy": 0.842,
    "within_1": 0.985,
    "per_grade_recall": [0.95, 0.75, 0.82, 0.88, 0.82],
    "rdr_operating_threshold": 0.42,
    "rdr_sensitivity": 0.94,
    "rdr_specificity": 0.82
  },
  "ood_test": {
    "qwk": 0.832,
    "accuracy": 0.801,
    "within_1": 0.965,
    "per_grade_recall": [0.91, 0.70, 0.78, 0.85, 0.79],
    "rdr_operating_threshold": 0.42,
    "rdr_sensitivity": 0.91,
    "rdr_specificity": 0.80,
    "rdr_auroc": 0.932
  },
  "delta_lodo": -0.053
}
```

---

## 3. LODO Fold Cohort Index Mapping

Matching `merge_dataset_lodo.ipynb` and `main_model_lodo.ipynb`:

| Fold Index | Held-Out Target Cohort | Fold Slug | Source Cohorts (Training & Validation Pool) |
|:---:|:---:|:---:|:---|
| 0 | APTOS 2019 | `aptos` | IDRiD, Messidor-2, DDR, EyePACS, DeepDRiD |
| 1 | IDRiD | `idrid` | APTOS, Messidor-2, DDR, EyePACS, DeepDRiD |
| 2 | Messidor-2 | `messidor2` | APTOS, IDRiD, DDR, EyePACS, DeepDRiD |
| 3 | DDR | `ddr` | APTOS, IDRiD, Messidor-2, EyePACS, DeepDRiD |
| 4 | EyePACS | `eyepacs` | APTOS, IDRiD, Messidor-2, DDR, DeepDRiD |
| 5 | DeepDRiD | `deepdrid` | APTOS, IDRiD, Messidor-2, DDR, EyePACS |
