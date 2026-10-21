# Data Model & Schema Specification: DR Preprocessing & Harmonization Pipeline

## 1. Core Data Entities

### HarmonizedMasterRecord
Represents the unified master table before splitting and preprocessing.

| Field Name | Type | Description | Constraints / Valid Values |
| :--- | :--- | :--- | :--- |
| `image_id` | `str` | Standardized image file identifier | Unique non-null string |
| `diagnosis` | `int` | ICDRS DR severity grade | Integer $\in \{0, 1, 2, 3, 4\}$ |
| `dataset_name` | `str` | Origin dataset name | One of: `APTOS`, `DDR`, `DeepDRiD`, `IDRiD`, `Messidor-2`, `EyePACS` |
| `raw_full_path` | `str` | Absolute/relative path to raw image | Existing image file path |

---

### DatasetSplitRecord
Represents records after leak-free 70/15/15 stratified partitioning.

| Field Name | Type | Description | Constraints / Valid Values |
| :--- | :--- | :--- | :--- |
| `image_id` | `str` | Standardized image file identifier | Non-null string |
| `diagnosis` | `int` | ICDRS DR severity grade | Integer $\in \{0, 1, 2, 3, 4\}$ |
| `dataset_name` | `str` | Origin dataset name | String |
| `split` | `str` | Partition assignment | One of: `"train"`, `"val"`, `"test"` |
| `is_augmented` | `bool` | Flag indicating synthetic oversampling | `False` for raw images, `True` for augmented train samples |

---

### ProcessedImageArtifact
Represents preprocessed image files written to disk.

| Field Name | Type | Description | Path Location |
| :--- | :--- | :--- | :--- |
| `image_id` | `str` | Image identifier | String |
| `resolution` | `str` | Spatial dimensions | `"300x300"` or `"512x512"` |
| `split` | `str` | Partition group | `"train"`, `"val"`, or `"test"` |
| `processed_path_300` | `str` | File path to 300x300 image | `data/processed/hybrid/300x300/{split}/{diagnosis}/{image_id}.png` |
| `processed_path_512` | `str` | File path to 512x512 image | `data/processed/hybrid/512x512/{split}/{diagnosis}/{image_id}.png` |

## 2. Pipeline Entity Data Flow

```mermaid
flowchart TD
    A[6 Raw Datasets (53,977 images)] -->|Step 1: Harmonize| B[Master Harmonized DataFrame]
    B -->|Step 2: Stratified 70/15/15 Split| C[Train Set: 37,783]
    B -->|Step 2: Stratified 70/15/15 Split| D[Val Set: 8,097]
    B -->|Step 2: Stratified 70/15/15 Split| E[Test Set: 8,097]
    
    C -->|Step 3-4: Crop + CLAHE| F[Preprocessed Train Set]
    F -->|Step 5: Undersample C0,C1,C2 & Augment C3,C4| G[Balanced Train Set: 10,000 images]
    
    D -->|Step 3-4: Crop + CLAHE| H[Preprocessed Val Set: 8,097 raw]
    E -->|Step 3-4: Crop + CLAHE| I[Preprocessed Test Set: 8,097 raw]
    
    G -->|Save 300x300 & 512x512| J[data/processed/hybrid/300x300 & 512x512]
    H -->|Save 300x300 & 512x512| J
    I -->|Save 300x300 & 512x512| J
```

## 3. Data Balancing Target Specifications (Train Set ONLY)

| DR Grade | Original Train Count (~70%) | Balancing Operation | Target Train Count |
| :--- | :--- | :--- | :--- |
| **Grade 0** | ~24,511 | Random Undersample | **2,000** |
| **Grade 1** | ~2,733 | Random Undersample | **2,000** |
| **Grade 2** | ~7,924 | Random Undersample | **2,000** |
| **Grade 3** | ~1,163 | Albumentations Oversample | **2,000** |
| **Grade 4** | ~1,452 | Albumentations Oversample | **2,000** |
| **Total** | **37,783** | **Harmonized & Balanced** | **10,000** |
