# Data Pipeline & Cohort Harmonization

This document provides a comprehensive guide to data acquisition, licensing, directory layout, preprocessing, and patient-isolated cohort partitioning across the six clinical fundus imaging cohorts utilized in this benchmark.

---

## 1. Multi-Cohort Overview

The benchmark harmonizes $N = 55{,}570$ digital color fundus photographs ($35{,}474$ patients) spanning six distinct healthcare systems, photographic devices, optical fields of view (FOV), and regional demographics:

| Cohort | Source / Region | Primary Device & FOV | Total Images (Patients) | Grade Distribution ($G_0 / G_1 / G_2 / G_3 / G_4$) | Clinical Grading Protocol | Access & License |
|---|---|---|---|---|---|---|
| **APTOS 2019** | Aravind Eye Hospital, India | Topcon / Zeiss ($45^\circ$) | 3,662 (3,662) | 49.3% / 10.1% / 27.3% / 5.3% / 8.1% | Single-grader ICDRS | Kaggle Competition License (Research) |
| **DDR** | Multi-center, China | Canon CR-2 / Topcon ($45^\circ$) | 12,522 (12,522) | 50.0% / 5.0% / 35.8% / 1.9% / 7.3% | Multi-expert consensus ICDRS | Academic Open Research License |
| **DeepDRiD** | Shanghai, China | Topcon TRC-NW400 ($45^\circ$) | 2,000 (500) | 34.0% / 17.8% / 19.6% / 19.6% / 9.0% | 2-field paired stereoscopic (macula/disc) | ISBI Challenge License |
| **IDRiD** | Eye Clinic, India | Kowa VX-10$\alpha$ ($50^\circ$) | 516 (516) | 34.9% / 5.0% / 32.6% / 14.3% / 13.2% | Clinical consensus + Pixel lesion masks | IEEE Dataport / Academic |
| **Messidor-2** | Multi-center, France | Topcon TRC NW6 ($45^\circ$) | 1,748 (874) | 58.1% / 15.4% / 19.9% / 4.3% / 2.3% | ICDRS adjudicated by ophthalmologists | Messidor-2 Research Agreement |
| **EyePACS** | Primary Care Clinics, USA | Multi-device telemedical ($45^\circ$) | 35,126 (17,563) | 73.5% / 7.0% / 15.1% / 2.5% / 2.0% | Certified tele-retinal reading network | Kaggle / EyePACS Research License |

---

## 2. Directory Layout

All data directories are anchored dynamically relative to the package root defined in `drdg.paths`:

```
data/
├── raw/                                 # Raw images downloaded from official repositories
│   ├── aptos2019/
│   │   ├── train_images/
│   │   └── train.csv
│   ├── ddr/
│   ├── deepdrid/
│   ├── idrid/
│   ├── idrid_segmentation/              # Pixel lesion masks (MA, HE, EX, SE)
│   ├── messidor2/
│   └── eyepacs/
├── processed/
│   ├── native_cropped/                  # Preprocessed circular FOV cropped images (PNG)
│   │   ├── 001_left.png
│   │   └── ...
│   └── lodo_folds/                      # 18 patient-isolated LODO split CSV files
│       ├── fold_0_aptos_train.csv
│       ├── fold_0_aptos_val.csv
│       ├── fold_0_aptos_test.csv
│       ├── ...
│       ├── fold_5_deepdrid_test.csv
│       └── lodo_splits_summary.csv
├── metadata/                            # Unified metadata
│   ├── master_metadata.csv              # Unified record of all 55,570 images
│   ├── train_balanced_metadata.csv      # Quota-balanced development pool
│   └── dataset_audit.json               # Class frequencies and cohort integrity checks
```

---

## 3. Automated Data Acquisition

Datasets hosted on Kaggle or public repositories can be downloaded using the unified CLI tool:

```bash
# Verify credentials are set in environment
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Download a specific cohort
python scripts/download_dataset.py --cohort aptos
python scripts/download_dataset.py --cohort idrid
python scripts/download_dataset.py --cohort idrid_segmentation

# Or download all available cohorts
python scripts/download_dataset.py --all
```

---

## 4. Preprocessing & Harmonization Pipeline

Raw fundus images vary significantly in resolution ($1{,}500 \times 1{,}000$ to $4{,}288 \times 2{,}848$), circular mask boundaries, and color illumination. The pipeline executes:

1. **Circular FOV Masking**: `drdg.data.fov.crop_to_fov` segments the illuminated retina from background black margins via thresholding and bounding box extraction.
2. **Standardized Resizing**: Standardized to $512 \times 512$ with aspect-ratio preservation and reflection padding.
3. **Local Patch Extraction**: Dynamically samples $K=16$ high-resolution patches ($128 \times 128$) from informative retinal zones for the local MIL stream (`drdg.data.patches.extract_patches_numpy`).

Execute batch preprocessing:
```bash
python scripts/prepare_dataset.py --cohort aptos --target-size 512
python scripts/prepare_dataset.py --all
```

---

## 5. Master Metadata Construction

To eliminate disparate naming conventions and labels across cohorts, `scripts/build_master_metadata.py` ingests raw annotations and unifies them into a single schema:

```bash
python scripts/build_master_metadata.py
```

Standardized Master Metadata Columns:
- `image_id`: Unique identifier string.
- `patient_id`: Extracted patient or eye identifier ensuring multi-image isolation.
- `cohort`: Name of cohort (`aptos`, `ddr`, `deepdrid`, `idrid`, `messidor2`, `eyepacs`).
- `dr_grade`: Standardized integer grade $\in \{0, 1, 2, 3, 4\}$.
- `file_path`: Relative path to processed image file.
- `view_field`: Field identifier (`macula`, `optic_disc`, or `center`).

---

## 6. Zero-Leakage Patient-Isolated LODO Partitions

A critical flaw in naive medical ML is random image-level splitting where left and right eyes, or serial views of the same patient, appear in both training and test partitions. 

`drdg` enforces strict patient-level isolation using `drdg.data.patient_ids.extract_patient_id` and `drdg.data.splits.split_patient_stratified`.

### Partition Invariants:
1. **Source Training Set** (`train`): Formed from 5 source cohorts (80% of source patients).
2. **Source Validation Set** (`val`): Formed from 5 source cohorts (20% of source patients).
   $$\text{Patients}(\mathcal{D}_{\text{train}}^{\text{source}}) \cap \text{Patients}(\mathcal{D}_{\text{val}}^{\text{source}}) = \emptyset$$
3. **Target Test Set** (`test`): 100% of images from the 6th held-out cohort.
   $$\text{Patients}(\mathcal{D}_{\text{source}}) \cap \text{Patients}(\mathcal{D}_{\text{target}}) = \emptyset$$

### Storage & Files:
Each of the 6 folds produces exactly 3 CSV files in `data/processed/lodo_folds/`:
- `fold_{fold_idx}_{cohort}_train.csv`
- `fold_{fold_idx}_{cohort}_val.csv`
- `fold_{fold_idx}_{cohort}_test.csv`
along with `lodo_splits_summary.csv` summarizing image counts and patient integrity across all 18 files.

Generate and verify all 18 LODO split files:
```bash
python scripts/generate_lodo_splits.py
```
Verification assertions in `tests/test_lodo_splits.py` and `tests/parity/test_lodo_split_parity.py` automatically validate that set intersections between patient partitions evaluate to empty sets.
