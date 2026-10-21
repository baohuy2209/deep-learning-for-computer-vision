# Contract Specification: Preprocessing & Harmonization Pipeline Functions

## 1. Master Dataset Harmonization Contract

```python
def merge_and_harmonize_datasets(
    dataset_dfs: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Concatenates 6 dataset DataFrames, standardizing columns to 'image_id', 'diagnosis', 'dataset_name'.
    
    Returns:
        pd.DataFrame: Harmonized master table containing all 53,977 records.
    """
```

## 2. Stratified Leak-Free Split Contract

```python
def create_stratified_splits(
    df: pd.DataFrame, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15, 
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified Train/Val/Test splitting using stratify=y before augmentation.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
```

## 3. Geometric ROI Auto-Cropping Contract

```python
def crop_fundus_roi(
    img: np.ndarray, 
    tolerance: int = 7
) -> np.ndarray:
    """
    Detects and crops non-diagnostic black borders around the fundus circular ROI.
    """
```

## 4. CLAHE & Image Quality Enhancement Contract

```python
def enhance_fundus_image(
    img_rgb: np.ndarray, 
    clip_limit: float = 2.0, 
    tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Extracts Green channel, applies CLAHE contrast enhancement, and Gaussian blur filtering.
    """
```

## 5. Albumentations Augmentation & Training Set Balancing Contract

```python
def balance_and_augment_train_set(
    train_df: pd.DataFrame, 
    target_per_class: int = 2000, 
    output_dir_300: str = "data/processed/hybrid/300x300/train",
    output_dir_512: str = "data/processed/hybrid/512x512/train",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Undersamples dominant classes (0, 1, 2) and oversamples/augments minority classes (3, 4)
    using Albumentations to produce exactly target_per_class images per grade (10,000 total).
    
    Returns:
        pd.DataFrame: Balanced training set record table.
    """
```

## 6. Pixel Normalization Contract

```python
def normalize_image_tensor(
    img: np.ndarray, 
    mode: str = "imagenet"
) -> np.ndarray:
    """
    Normalizes pixel values to [0, 1] or standard Z-score using ImageNet / Train set mean & std.
    """
```
