# Data Directory Overview

This directory houses the raw fundus photography datasets, preprocessed standard images, harmonized master metadata, and patient-stratified Leave-One-Dataset-Out (LODO) split definitions.

---

## Directory Organization

```
data/
├── raw/                         # Untouched images from original repository downloads
│   ├── aptos2019/
│   ├── ddr/
│   ├── deepdrid/
│   ├── idrid/
│   ├── idrid_segmentation/      # Pixel-level binary lesion ground truth (MA, HE, EX, SE)
│   ├── messidor2/
│   └── eyepacs/
├── processed/                   # Standardized 512x512 FOV-cropped and padded images
│   ├── aptos2019/
│   ├── ddr/
│   ├── deepdrid/
│   ├── idrid/
│   ├── messidor2/
│   └── eyepacs/
└── metadata/                    # Unified annotations and zero-leakage partitions
    ├── master_metadata.csv      # Complete record of all 55,570 harmonized images
    ├── dataset_audit.json       # Class distributions and integrity audit metrics
    └── splits/                  # 6-Fold patient-isolated LODO split configurations
        ├── lodo_fold_aptos.json
        ├── lodo_fold_ddr.json
        ├── lodo_fold_deepdrid.json
        ├── lodo_fold_idrid.json
        ├── lodo_fold_messidor2.json
        └── lodo_fold_eyepacs.json
```

---

## Quick Setup

To populate this directory with datasets and generate split files:

1. **Download Raw Data**:
   ```bash
   python scripts/download_dataset.py --cohort aptos
   ```
2. **Execute Circular FOV Preprocessing**:
   ```bash
   python scripts/prepare_dataset.py --all
   ```
3. **Build Unified Master Metadata**:
   ```bash
   python scripts/build_master_metadata.py
   ```
4. **Generate Strict Patient-Isolated LODO Splits**:
   ```bash
   python scripts/generate_lodo_splits.py
   ```

For detailed licensing, field-of-view specifications, and patient identifier extraction rules, see [`docs/DATA.md`](../docs/DATA.md).
