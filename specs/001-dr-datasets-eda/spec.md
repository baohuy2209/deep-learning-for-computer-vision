# Feature Specification: Comprehensive Exploratory Data Analysis (EDA) for 6 DR Datasets

**Feature Branch**: `001-dr-datasets-eda`

**Created**: 2026-08-09

**Status**: Draft

**Input**: User description: "/speckit-specify # TASK: COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA) FOR 6 DR DATASETS"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Multi-Dataset Standardization and Integrity Auditing (Priority: P1)

As a computer vision researcher, I want to load 6 Diabetic Retinopathy (DR) datasets (APTOS, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS), inspect their schema, standardize column labels (`image_id`, `level`), and run sanity checks for duplicate IDs, missing labels, and invalid DR grades (outside 0–4), so that I can ensure raw data cleanliness before model training.

**Why this priority**: Raw multi-source datasets have inconsistent schemas and potential corruption/missing values. Establishing clean, standardized data tables is essential for all subsequent analysis and training pipelines.

**Independent Test**: Can be fully verified by running the data ingestion and validation functions on each raw CSV in `data/raw/` to produce standardized DataFrames and audit logs without errors.

**Acceptance Scenarios**:

1. **Given** 6 raw DR dataset paths in `data/raw/`, **When** the dataset ingestion script executes, **Then** all label CSVs are loaded, column names are mapped to standard `image_id` and `level` fields, and basic structural summaries (`.head()`, `.info()`) are displayed.
2. **Given** standardized label DataFrames, **When** the sanity check function runs, **Then** it accurately reports duplicate image IDs, null/NaN label values, and out-of-bounds DR grades outside the valid range [0, 1, 2, 3, 4].

---

### User Story 2 - Class Distribution & Eye Metadata Analysis (Priority: P2)

As a medical AI researcher, I want to analyze patient eye side metadata (Left vs. Right eye) and visualize class imbalance across individual datasets as well as the combined hybrid dataset using bar charts and percentage pie charts, so that I can evaluate bias and select appropriate sampling strategies (e.g., weighted loss or sampler).

**Why this priority**: DR severity imbalance across datasets significantly impacts Ordinal Classification performance and QWK metrics. Understanding per-dataset and aggregated class proportions guides loss function design.

**Independent Test**: Can be tested independently by computing DR level frequencies and rendering side-by-side distribution plots for both individual and merged datasets.

**Acceptance Scenarios**:

1. **Given** datasets with lateral eye naming conventions (e.g., EyePACS), **When** patient/eye metadata parsing executes, **Then** `patient_id` and `side` (left/right) are extracted and unique patient counts are summarized.
2. **Given** standardized DR labels across all 6 datasets, **When** class distribution visualization triggers, **Then** both counts and percentage pie charts are plotted for individual datasets and the merged hybrid dataset.

---

### User Story 3 - Visual Sampling, Resolution Distribution & Quality Auditing (Priority: P3)

As a deep learning engineer, I want to sample representative fundus images for each DR level (0–4), analyze image resolutions (Width, Height, Aspect Ratio), and assess image quality metrics (grayscale brightness and Laplacian variance sharpness) across datasets, so that I can determine necessary pre-processing steps (e.g., CLAHE, resizing, blur filtering).

**Why this priority**: Image size variations and quality domain shifts (brightness differences, motion blur) across datasets directly inform image preprocessing and normalization strategies.

**Independent Test**: Can be tested independently by randomly sampling images from each dataset, loading via PIL/OpenCV, and computing resolution scatter plots and brightness/sharpness overlapping histograms.

**Acceptance Scenarios**:

1. **Given** image paths for each DR grade, **When** visual sampling executes, **Then** a 1x5 subplot grid displays one valid image per DR level (0 to 4) per dataset, handling missing images gracefully with warnings.
2. **Given** 100 randomly sampled fundus images per dataset, **When** resolution and quality metrics are computed, **Then** scatter/KDE plots of dimensions and overlapping histograms ($\alpha=0.5$) of brightness and Laplacian sharpness are generated with clinical explanatory markdowns.

---

### Edge Cases

- What happens when a CSV file path or image directory specified in the configuration dictionary does not exist on disk? (System must log a warning and skip the dataset without crashing).
- How does the system handle image loading errors caused by corrupted image files or missing filenames referenced in the CSV? (System must catch `FileNotFoundError` or image decode errors, issue a warning with the missing image ID, and continue execution).
- What happens when a dataset has 0 images for a specific DR grade (e.g., grade 4 in a small dataset)? (System must display a blank placeholder or warning in the 1x5 grid for that missing grade).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define a centralized configuration dictionary mapping each dataset name (APTOS, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS) to its respective raw image folder path and CSV label file path relative to `data/raw/`.
- **FR-002**: System MUST dynamically handle relative directory resolution to work seamlessly whether executed from the project root or from `notebooks/diabetic_retinopathy/`.
- **FR-003**: System MUST load label CSV files and standardize column names into unified `image_id` and `level` (or `diagnosis`) schema for downstream processing.
- **FR-004**: System MUST perform automated data integrity audits checking for: duplicate `image_id` entries, missing/NaN values in the label column, and invalid DR severity grades outside $[0, 1, 2, 3, 4]$.
- **FR-005**: System MUST parse patient ID and eye lateral orientation (left vs. right) from image identifiers where available (e.g., EyePACS) and report unique patient counts and eye distribution.
- **FR-006**: System MUST calculate exact DR level counts and plot both Bar Charts (counts) and Pie Charts (percentages) for each of the 6 datasets.
- **FR-007**: System MUST generate a 1x5 grid of image subplots displaying one representative fundus image per DR severity level ($0, 1, 2, 3, 4$) for each dataset.
- **FR-008**: System MUST randomly sample 100 images per dataset to compute Width, Height, and Aspect Ratio, displaying resolution distributions via Scatter or KDE plots.
- **FR-009**: System MUST compute Average Brightness (mean grayscale intensity) and Image Sharpness (OpenCV Laplacian variance `cv2.Laplacian(img, cv2.CV_64F).var()`) on the sampled images.
- **FR-010**: System MUST render overlapping histograms ($\alpha=0.5$) comparing Brightness (0–255) and Sharpness distributions across all datasets.
- **FR-011**: System MUST simulate a combined 6-dataset hybrid pool and plot the aggregated DR class distribution and severe imbalance ratio.
- **FR-012**: System MUST include clinical markdown annotations discussing the biological/medical implications of domain shifts, CLAHE contrast enhancement, and motion blur filtering.

### Key Entities

- **Dataset Configuration Entry**: Represents a dataset's metadata, holding name, root raw directory path, CSV label path, raw ID column name, and raw label column name.
- **Standardized Retinopathy Record**: Unified record containing `dataset_name`, `image_id`, `patient_id` (optional), `eye_side` (optional), `level` (integer 0–4), `image_width`, `image_height`, `aspect_ratio`, `brightness`, and `sharpness_score`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the 6 target DR datasets are audited and standardized automatically upon notebook execution.
- **SC-002**: Data sanity checks identify 100% of duplicate IDs, missing labels, and out-of-range DR grades across all raw CSVs.
- **SC-003**: EDA visualization pipeline completes execution producing all required figures (bar plots, pie charts, 1x5 image grids, scatter/KDE resolution plots, and overlapping quality histograms) without unhandled exceptions.
- **SC-004**: Execution of resolution and quality sampling (100 images per dataset, 600 total) completes in under 60 seconds.

## Assumptions

- Target datasets (APTOS, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS) are present under `data/raw/` in their respective directory paths.
- Images are in standard formats (.png, .jpg, .jpeg, .tif) readable by PIL and OpenCV.
- Python execution environment has standard data science libraries installed (`pandas`, `numpy`, `matplotlib`, `seaborn`, `opencv-python`, `pillow`, `tqdm`).
- Missing images or corrupted files are non-fatal anomalies that should be logged as warnings rather than crashing the pipeline.
