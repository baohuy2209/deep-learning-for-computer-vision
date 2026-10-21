# Contract Specification: EDA Pipeline Functions & Signatures

## 1. Directory Path Resolver Contract

```python
def resolve_raw_data_dir() -> str:
    """
    Dynamically resolves the absolute or relative path to `data/raw/`
    depending on current working directory.
    
    Returns:
        str: Relative path to `data/raw/` (e.g. 'data/raw' or '../../data/raw')
    """
```

## 2. Dataset Ingestion & Standardization Contract

```python
def load_and_standardize_dataset(
    config: dict, 
    base_raw_dir: str
) -> tuple[pd.DataFrame, dict]:
    """
    Loads raw CSV label file, renames columns to 'image_id' and 'level',
    runs sanity audits, and extracts lateral eye metadata.
    
    Args:
        config (dict): Dataset configuration mapping.
        base_raw_dir (str): Base path to raw data directory.
        
    Returns:
        tuple[pd.DataFrame, dict]: 
            - Standardized DataFrame with columns ['image_id', 'level', 'patient_id', 'side'].
            - Audit summary dict with keys: 'duplicates', 'missing_values', 'invalid_labels'.
    """
```

## 3. Class Distribution Plotting Contract

```python
def plot_dataset_class_distribution(
    df: pd.DataFrame, 
    dataset_name: str
) -> None:
    """
    Renders side-by-side Bar Chart (counts) and Pie Chart (percentages)
    for DR levels 0 through 4.
    """
```

## 4. Visual Sampling Contract

```python
def plot_sample_images_per_level(
    df: pd.DataFrame, 
    img_dir: str, 
    dataset_name: str,
    id_suffix: str = ""
) -> None:
    """
    Renders a 1x5 grid of image subplots showing 1 randomly selected image per DR level (0-4).
    Catches FileNotFoundError or PIL decode errors gracefully.
    """
```

## 5. Resolution & Image Quality Metrics Contract

```python
def compute_image_metrics_sample(
    df: pd.DataFrame, 
    img_dir: str, 
    dataset_name: str, 
    sample_size: int = 100,
    id_suffix: str = ""
) -> pd.DataFrame:
    """
    Randomly samples images to extract Width, Height, Aspect Ratio,
    Grayscale Brightness, and OpenCV Laplacian Sharpness Variance.
    
    Returns:
        pd.DataFrame: Metric table per sampled image.
    """
```

## 6. Overlapping Quality Distribution Contract

```python
def plot_quality_distributions(
    metrics_df: pd.DataFrame
) -> None:
    """
    Plots overlapping histograms (alpha=0.5) for Brightness and Sharpness
    across all 6 datasets.
    """
```
