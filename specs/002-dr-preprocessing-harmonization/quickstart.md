# Quickstart & End-to-End Validation Guide: DR Preprocessing & Harmonization Pipeline

## Prerequisites

- Python 3.10+ environment initialized.
- Standardized raw CSVs and image folders present under `data/raw/`.
- Required Python libraries:
  ```bash
  pip install opencv-python albumentations scikit-learn pandas numpy pillow matplotlib tqdm jupyter
  ```

## Output Directory Structure

The preprocessing pipeline populates the following target directory paths:
- `data/processed/hybrid/300x300/train/` (10,000 balanced images, 2,000 per class)
- `data/processed/hybrid/300x300/val/` (~8,097 raw images)
- `data/processed/hybrid/300x300/test/` (~8,097 raw images)
- `data/processed/hybrid/512x512/train/` (10,000 balanced images, 2,000 per class)
- `data/processed/hybrid/512x512/val/` (~8,097 raw images)
- `data/processed/hybrid/512x512/test/` (~8,097 raw images)
- `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb`

## Step-by-Step Validation Scenarios

### Scenario 1: Notebook Execution

```bash
jupyter execute notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb
```
*Expected Result*: Complete execution without errors, generating 70/15/15 stratified split tables, saving 300x300 and 512x512 preprocessed images to `data/processed/hybrid/`, and verifying 10,000 total training samples.

### Scenario 2: Validation of Split & Class Balance Metadata

```python
import pandas as pd

train_df = pd.read_csv("data/processed/hybrid/train_split.csv")
print(train_df["diagnosis"].value_counts())
# Expected: Exactly 2,000 images for each diagnosis level 0, 1, 2, 3, 4
```

## Verification Checklist

- [ ] Harmonization merges all 6 datasets without missing rows (53,977 total).
- [ ] Stratified 70/15/15 split maintains identical class ratios across Train, Val, and Test.
- [ ] Auto-cropping removes peripheral black margins from fundus images.
- [ ] CLAHE and Gaussian blur enhance retinal vessel details and microaneurysms.
- [ ] Training set achieves exactly 2,000 images per DR grade (10,000 total images).
- [ ] Validation and Test sets remain raw and unaugmented.
- [ ] Images are successfully exported at both `300x300` and `512x512` resolution.
