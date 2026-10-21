# Schema Contract: 6-Fold Leave-One-Dataset-Out (LODO) Metadata

**Feature**: `018-restore-notebook-parity` | **Date**: 2026-09-24

This document defines the strict file naming, partition structure, and column schema contract for LODO split metadata matching `merge_dataset_lodo.ipynb`.

---

## 1. File Naming Contract

All LODO split CSV files must reside in:
`data/processed/lodo_folds/`

The file naming convention is:
`fold_{fold_idx}_{fold_slug}_{split}.csv`

Where:
- `fold_idx` $\in \{0, 1, 2, 3, 4, 5\}$
- `fold_slug` $\in \{\text{"aptos"}, \text{"idrid"}, \text{"messidor2"}, \text{"ddr"}, \text{"eyepacs"}, \text{"deepdrid"}\}$
- `split` $\in \{\text{"train"}, \text{"val"}, \text{"test"}\}$

### Complete Inventory (18 CSV Files)
```text
fold_0_aptos_train.csv
fold_0_aptos_val.csv
fold_0_aptos_test.csv
fold_1_idrid_train.csv
fold_1_idrid_val.csv
fold_1_idrid_test.csv
fold_2_messidor2_train.csv
fold_2_messidor2_val.csv
fold_2_messidor2_test.csv
fold_3_ddr_train.csv
fold_3_ddr_val.csv
fold_3_ddr_test.csv
fold_4_eyepacs_train.csv
fold_4_eyepacs_val.csv
fold_4_eyepacs_test.csv
fold_5_deepdrid_train.csv
fold_5_deepdrid_val.csv
fold_5_deepdrid_test.csv
```

Plus summary:
`lodo_splits_summary.csv`

---

## 2. Prohibited File Names (Legacy Non-Compliant Schemas)

The following naming patterns are **STRICTLY PROHIBITED** and must never be generated or expected by `drdg`:
- `fold_{cohort}_source_train.csv` (INVALID)
- `fold_{cohort}_source_val.csv` (INVALID)
- `fold_{cohort}_source_test.csv` (INVALID - no source_test partition exists)
- `fold_{cohort}_target_test.csv` (INVALID)

---

## 3. CSV Column Schema

Every fold CSV must contain at minimum the following columns:

| Column Name | Data Type | Nullable | Description |
|---|---|---|---|
| `image_id` | `string` | No | Unique image ID (e.g. `aptos_total_images_0001`) |
| `diagnosis` | `int64` | No | Integer DR severity level ($0, 1, 2, 3, 4$) |
| `dataset_name` | `string` | No | Cohort display name (e.g. `APTOS 2019`) |
| `dataset_key` | `string` | No | Canonical cohort slug (`aptos`, `ddr`, etc.) |
| `raw_full_path` | `string` | No | Relative or absolute path to raw image |
| `native_path` | `string` | No | Relative or absolute path to cropped ROI image |
| `patient_id` | `string` | No | Namespaced patient ID (`{dataset_key}::{id}`) |
| `side` | `string` | Yes | Eye laterality (`left`, `right`, `unknown`) |
| `fold` | `int64` | No | Fold index ($0 \dots 5$) |

---

## 4. Zero Patient Leakage Guarantee

For any given fold $k \in \{0, \dots, 5\}$:
Let $T = \text{Unique Patients}(\text{fold\_k\_*\_train.csv})$
Let $V = \text{Unique Patients}(\text{fold\_k\_*\_val.csv})$
Let $E = \text{Unique Patients}(\text{fold\_k\_*\_test.csv})$

The following set invariants must hold:
$$T \cap V = \emptyset$$
$$T \cap E = \emptyset$$
$$V \cap E = \emptyset$$
$$E \cap \left( \bigcup_{c \in \text{Source Cohorts}} \text{Patients}(c) \right) = \emptyset$$
