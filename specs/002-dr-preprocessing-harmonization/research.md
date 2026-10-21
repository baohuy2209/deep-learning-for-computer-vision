# Research & Technical Decisions: DR Preprocessing & Harmonization Pipeline

## 1. ROI Auto-Cropping & Ben Graham's Technique

- **Decision**: Combine OpenCV Otsu/binary thresholding with bounding rectangle contour extraction. Fall back to Ben Graham's center crop (cropping to circle radius using grayscale thresholding) if contour area is $<30\%$ of total image area.
- **Rationale**: Retinal fundus photographs captured across multi-center cameras (Topcon, Canon, Zeiss) feature varying peripheral black margins. Auto-cropping removes up to 50% non-diagnostic black pixels, reducing memory overhead and focusing model attention purely on the spherical retinal vascular tree.
- **Alternatives Considered**:
  - Fixed aspect ratio cropping (fails due to varying camera sensor aspect ratios across the 6 datasets).
  - Manual bounding box annotation (unfeasible for 53,977 images).

## 2. Leak-Free 70/15/15 Stratified Partitioning Strategy

- **Decision**: Perform `sklearn.model_selection.train_test_split` with `stratify=y` on the 53,977 harmonized master dataset *before* any cropping, CLAHE enhancement, or Albumentations oversampling.
- **Rationale**: Performing synthetic oversampling or data transformation prior to train/test splitting causes data leakage (nearly identical augmented versions of training images appearing in validation or test sets). Keeping Val (15%) and Test (15%) completely untouched ensures unbiased, real-world generalization metrics.

## 3. Targeted Training Set Class Balancing

- **Decision**: Apply class balancing *strictly to the 70% Training split* targeting exactly 2,000 images per DR grade ($0, 1, 2, 3, 4$), yielding a total balanced training pool of 10,000 images.
  - **Class 0 (No DR - ~24,500 images)**: Randomly undersample to 2,000 images.
  - **Class 2 (Moderate - ~7,900 images)**: Randomly undersample to 2,000 images.
  - **Class 1 (Mild - ~2,700 images)**: Randomly undersample to 2,000 images.
  - **Class 3 (Severe - ~1,160 images)**: Oversample/Augment to 2,000 images using Albumentations.
  - **Class 4 (Proliferative - ~1,450 images)**: Oversample/Augment to 2,000 images using Albumentations.
- **Rationale**: Prevents model bias toward Class 0 (64.9%) and ensures balanced gradient updates across all severity levels during QWK loss optimization, while keeping total training set size at a computationally efficient 10,000 images.

## 4. Multi-Resolution Processing (300x300 & 512x512)

- **Decision**: Save preprocessed image outputs at both `300x300` pixels and `512x512` pixels under `data/processed/hybrid/300x300/` and `data/processed/hybrid/512x512/` (and link `data/raw/hybrid/`).
- **Rationale**: `300x300` resolution enables rapid prototyping, baseline model selection, and global context branch processing, while `512x512` resolution preserves high-frequency microaneurysms and neovascularization details essential for the high-resolution local patch branch.

## 5. CLAHE & Image Quality Enhancement Parameters

- **Decision**:
  - Green Channel extraction / weighted blending: $0.299R + 0.587G + 0.114B$ (Green channel exhibits maximum contrast for vascular lesions).
  - CLAHE parameters: `clipLimit=2.0`, `tileGridSize=(8, 8)`.
  - Gaussian Blur kernel: `(3, 3)`, $\sigma=0$.
- **Rationale**: Standardizes contrast across heterogeneous light sources without introducing artificial boundary artifacts or exaggerating high-frequency noise.
