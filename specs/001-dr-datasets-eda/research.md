# Research & Technical Decisions: 6 DR Datasets EDA

## 1. Directory Path & Relative Execution Resolution

- **Decision**: Implement a dynamic base path resolver function `get_base_path()` using `os.path.exists()` to automatically locate `data/raw/` whether the notebook is executed from project root (`deep-learning-for-computer-vision/`) or from `notebooks/diabetic_retinopathy/`.
- **Rationale**: Researchers may launch Jupyter servers from either directory. Hardcoded relative paths break when execution context changes.
- **Alternatives Considered**:
  - Hardcoding `../../data/raw/`: Fails when run from root.
  - Hardcoding `data/raw/`: Fails when run from nested `notebooks/` subfolder.

## 2. Dynamic Dataset Schema & Label Standardization Strategy

- **Decision**: Define a central Python dictionary `DATASET_CONFIG` mapping dataset keys to `name`, `image_dir`, `csv_path`, `id_col`, `label_col`, and `id_suffix` (if image extension needs appending).
- **Rationale**: Raw datasets use varying header names (e.g., `id_code` vs `image` vs `name`, `diagnosis` vs `level` vs `DR_grade`). Centralizing schema configuration allows easy updating without modifying core EDA functions (DRY principle).
- **Alternatives Considered**: Writing custom loading functions per dataset (rejected due to code duplication and maintenance burden).

## 3. Sharpness & Brightness Metric Algorithms

- **Decision**:
  - **Sharpness**: OpenCV Variance of Laplacian (`cv2.Laplacian(gray_img, cv2.CV_64F).var()`).
  - **Brightness**: Mean intensity of 8-bit grayscale image (`np.mean(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY))`).
- **Rationale**: Laplacian variance is computationally efficient and standard in CV literature for estimating focus blur. Mean grayscale intensity directly measures global exposure across multi-site cameras.
- **Alternatives Considered**:
  - Tenengrad gradient for sharpness (more expensive computation with negligible accuracy improvement for EDA).
  - HSV Value channel for brightness (computationally equivalent to grayscale conversion for RGB).

## 4. Exception Handling for Missing / Corrupted Images

- **Decision**: Wrap PIL/OpenCV image opening calls inside `try-except (FileNotFoundError, UnidentifiedImageError, Exception)` blocks. Log warning messages and skip missing files rather than stopping execution.
- **Rationale**: Real-world DR datasets frequently contain corrupted images or mismatched CSV references. EDA notebooks must be resilient and execute cleanly to completion.

## 5. Notebook Storage & Output Specification

- **Decision**: Save executable notebook script as `notebooks/diabetic_retinopathy/eda_6_datasets.ipynb` (and `# %%` annotated Python script).
- **Rationale**: Complies with Constitution Rule 3 (`notebooks/diabetic_retinopathy/`) and Rule 2 (non-destructive creation of new files).
