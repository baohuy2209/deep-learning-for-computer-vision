# Quickstart & End-to-End Validation Guide: 6 DR Datasets EDA

## Prerequisites

- Python 3.10+ environment installed.
- Raw datasets present in `data/raw/`:
  - `data/raw/aptos`
  - `data/raw/ddr`
  - `data/raw/deepdrid`
  - `data/raw/idrid`
  - `data/raw/messidor2preprocess`
  - `data/raw/zipEyepacs`
- Required Python libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn opencv-python pillow tqdm jupyter
  ```

## Execution Environment

The EDA notebook is designed to execute seamlessly from either location:
1. Root directory: `deep-learning-for-computer-vision/`
2. Subdirectory: `notebooks/diabetic_retinopathy/`

## Step-by-Step Validation Scenarios

### Scenario 1: Notebook Execution from Root

```bash
jupyter execute notebooks/diabetic_retinopathy/eda_6_datasets.ipynb
```
*Expected Result*: Notebook runs through all 8 steps without throwing exceptions, generating all required plots and summaries.

### Scenario 2: Verification of Standalone Python Script (`# %%` cells)

```bash
python notebooks/diabetic_retinopathy/eda_script.py
```
*Expected Result*: Data loading, sanity auditing, metric extraction, and plot rendering execute cleanly with output printed to terminal and figures displayed.

## Verification Checklist

- [ ] Path resolver correctly identifies `data/raw/`.
- [ ] Schema standardization maps all raw CSV headers to `image_id` and `level`.
- [ ] Duplicate IDs, missing NaNs, and out-of-range DR grades are accurately reported.
- [ ] 1x5 image grid plots 5 sample images per dataset (or logs warnings for missing files).
- [ ] 100-sample resolution scatter plots and aspect ratio metrics are displayed.
- [ ] Overlapping histograms ($\alpha=0.5$) for brightness and Laplacian sharpness render across datasets.
- [ ] Aggregated class distribution bar chart illustrates overall hybrid dataset imbalance.
