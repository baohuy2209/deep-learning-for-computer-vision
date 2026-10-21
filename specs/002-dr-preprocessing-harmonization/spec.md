# Feature Specification: Robust Image Preprocessing & Harmonization Pipeline for 6-Dataset DR Hybrid Pool

**Feature Branch**: `002-dr-preprocessing-harmonization`

**Created**: 2026-08-09

**Status**: Draft

**Input**: User description: "/speckit-specify # TASK: ROBUST IMAGE PREPROCESSING & HARMONIZATION PIPELINE FOR 6-DATASET DR HYBRID POOL"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Multi-Center Dataset Harmonization & Leak-Free Stratified Splitting (Priority: P1)

As a computer vision researcher, I want to merge 6 distinct Diabetic Retinopathy datasets (APTOS 2019, EyePACS, Messidor-2, IDRiD, DeepDRiD, DDR) into a unified master DataFrame standardized to International Clinical Diabetic Retinopathy Severity Scale (ICDRS 0–4) labels, and perform a 70/15/15 Stratified Train/Val/Test split before any image transformation or sampling, so that data leakage is strictly prevented and validation/testing reflect unadulterated real-world distributions.

**Why this priority**: Data leakage from performing image augmentation or balancing prior to splitting invalidates model evaluation. Stratified splitting must serve as the uncompromised foundation for the entire training and evaluation pipeline.

**Independent Test**: Can be tested by running the harmonization and stratified split functions on raw label tables, confirming identical class distribution percentages across Train (70%), Val (15%), and Test (15%) splits without overlap in patient or image IDs.

**Acceptance Scenarios**:

1. **Given** label DataFrames from the 6 target datasets, **When** harmonization runs, **Then** all rows are concatenated into a single master DataFrame with standardized columns (`image_id`, `diagnosis`, `dataset_name`).
2. **Given** the master harmonized DataFrame, **When** stratified splitting executes with `stratify=y`, **Then** exactly 70% of samples are assigned to Train, 15% to Validation, and 15% to Test, preserving original class proportions across all three partitions.

---

### User Story 2 - Automated ROI Auto-Cropping & Dual-Resolution Image Enhancement (Priority: P2)

As a deep learning engineer, I want to auto-crop uninformative black background borders around retina fundus images using Ben Graham's thresholding technique, apply Green Channel extraction/blending, CLAHE contrast enhancement, and Gaussian blur noise reduction, and output dual-resolution image sets (`300x300` and `512x512`), so that downstream dual-branch networks receive high-quality, standardized inputs.

**Why this priority**: Raw fundus photographs contain up to 50% non-diagnostic black borders and severe cross-camera illumination variations. Automated ROI cropping and CLAHE contrast enhancement expose critical lesions (microaneurysms, hemorrhages).

**Independent Test**: Can be verified by processing a sample batch of raw images, confirming that black borders are removed, images are successfully exported at both `300x300` and `512x512` resolution, and contrast-enhanced fundus details are clearly visible.

**Acceptance Scenarios**:

1. **Given** raw color fundus photographs, **When** ROI auto-cropping executes, **Then** black peripheral margins are detected and trimmed to isolate the circular retinal region.
2. **Given** cropped fundus regions, **When** quality enhancement runs, **Then** CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gaussian blur are applied, and processed images are saved in dual resolution folders (`300x300` and `512x512`) under `data/processed/hybrid/`.

---

### User Story 3 - Targeted Training Set Balancing & Albumentations Augmentation (Priority: P3)

As a machine learning researcher, I want to apply class balancing *strictly to the 70% Training Set* by undersampling dominant classes (Class 0, Class 1, Class 2) and oversampling/augmenting minority classes (Class 3, Class 4) using `albumentations` transformations (flips, rotations up to 90°, brightness/contrast shifts, zoom/shear) to achieve exactly 2,000 images per class (10,000 total training images), while leaving Validation and Test sets untouched.

**Why this priority**: Severe class imbalance (64.9% Class 0 vs 3.1% Class 3) causes deep models to collapse towards predicting normal cases. Target balancing forces equal gradient representation during training without distorting validation metrics.

**Independent Test**: Can be verified by auditing class counts in the augmented Training set to confirm exactly 2,000 images per grade (10,000 total), while verifying that Validation and Test set image counts remain unaugmented.

**Acceptance Scenarios**:

1. **Given** the 70% Train partition, **When** class balancing executes, **Then** Class 0, Class 1, and Class 2 are undersampled to 2,000 images each, and Class 3 and Class 4 are oversampled/augmented via Albumentations to reach 2,000 images each.
2. **Given** Validation and Test partitions, **When** the pipeline executes, **Then** zero augmentation or undersampling is applied to Val/Test sets.

---

### User Story 4 - Pixel Normalization & Pipeline Notebook Deliverable (Priority: P4)

As a research assistant, I want the entire pipeline implemented in a clean Jupyter Notebook at `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb` with pixel normalization (0 to 1 scaling and ImageNet/Train set Z-score standardization) and comprehensive markdown explanations of clinical and technical trade-offs, so that experiments can be easily reproduced and audited.

**Why this priority**: Reproducible research requires clear code structure, saved dataset split metadata, normalized tensors, and clinical justifications for auditability.

**Independent Test**: Can be verified by executing `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb` to completion, checking that pixel tensor values fall within expected normalized bounds, and verifying saved metadata CSVs.

**Acceptance Scenarios**:

1. **Given** preprocessed images, **When** normalization executes, **Then** pixel values are scaled to $[0, 1]$ or standard Z-score $[-1, 1]$.
2. **Given** the completed pipeline, **When** the notebook is saved, **Then** it resides at `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb` with markdown annotations explaining clinical rationale.

---

### Edge Cases

- What happens if an image has extreme vignetting or non-standard aspect ratio where threshold cropping fails? (System must fallback to center square cropping without crashing).
- What happens if an oversampled image generates duplicate augmented filenames? (System must append unique index hashes to augmented image filenames).
- What happens if disk space is insufficient for both 300x300 and 512x512 image sets? (System must log disk usage warnings and verify directory existence before writing).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST harmonize 6 Diabetic Retinopathy datasets (APTOS 2019, EyePACS, Messidor-2, IDRiD, DeepDRiD, DDR) into a unified master DataFrame with standardized columns (`image_id`, `diagnosis`, `dataset_name`).
- **FR-002**: System MUST map all dataset labels to the 5-level International Clinical Diabetic Retinopathy Severity Scale (ICDRS grades 0 to 4).
- **FR-003**: System MUST perform a Stratified Train/Val/Test split (70% Train, 15% Validation, 15% Test) using `stratify=y` *prior* to any data augmentation or balancing.
- **FR-004**: System MUST maintain the Validation (15%) and Test (15%) sets in their raw, real-world class distributions without any undersampling or synthetic augmentation.
- **FR-005**: System MUST implement Ben Graham's / contour-based ROI auto-cropping using OpenCV to eliminate non-diagnostic black borders around the fundus circular region.
- **FR-006**: System MUST generate preprocessed images in dual spatial resolutions: `300x300` pixels and `512x512` pixels.
- **FR-007**: System MUST apply Green Channel extraction/blending to mitigate inter-camera illumination variance.
- **FR-008**: System MUST apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance fine microvascular lesions (microaneurysms, hard exudates, hemorrhages).
- **FR-009**: System MUST apply Gaussian Blur filtering to reduce high-frequency image acquisition noise.
- **FR-010**: System MUST apply Albumentations augmentation transformations (Random Horizontal/Vertical Flip, Random 90° Rotation, ShiftScaleRotate, Random Brightness/Contrast) strictly to minority training samples.
- **FR-011**: System MUST balance the 70% Training set to achieve exactly 2,000 images per DR grade ($0, 1, 2, 3, 4$), yielding a total balanced training pool of 10,000 images.
- **FR-012**: System MUST normalize image pixel intensity values to $[0, 1]$ or standard Z-score $[-1, 1]$ using ImageNet or Train set statistics.
- **FR-013**: System MUST save preprocessed output images in `data/processed/hybrid/300x300/` and `data/processed/hybrid/512x512/` (and reference `data/raw/hybrid/` if requested).
- **FR-014**: System MUST deliver the complete executable code in a Jupyter Notebook at `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb` with detailed clinical and technical markdown annotations.

### Key Entities

- **HarmonizedMasterRecord**: Record holding `image_id`, `diagnosis` (0–4), `dataset_name`, `raw_file_path`, and `split_group` (`train`, `val`, `test`).
- **PreprocessingConfig**: Configuration object specifying spatial target sizes (`(300, 300)` and `(512, 512)`), CLAHE clip limit, Gaussian kernel size, and target training class count ($2,000$).
- **AugmentationPipeline**: Albumentations transformation composition applied dynamically or offline to minority training samples.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of 53,977 raw images are harmonized and partitioned into 70/15/15 Train/Val/Test splits with identical class proportions across partitions.
- **SC-002**: ROI auto-cropping and image enhancement pipeline processes images producing clean `300x300` and `512x512` outputs without border artifacts.
- **SC-003**: Augmented Training pool achieves exactly 2,000 images per DR level (10,000 total training samples), while Validation and Test set counts remain unaugmented.
- **SC-004**: Pipeline execution notebook at `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb` runs end-to-end without unhandled exceptions.

## Assumptions

- Python environment includes `opencv-python`, `albumentations`, `scikit-learn`, `pandas`, `numpy`, `pillow`, `matplotlib`, and `tqdm`.
- Input image paths from the 6 datasets under `data/raw/` are accessible as verified during EDA.
- Output directory `data/processed/hybrid/` (and `data/raw/hybrid/`) has write permissions.
