"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - DATA LOADER MODULE
================================================================================
Core Research Focus: Deconstructing Domain Shift in DR Severity Grading.
Features:
  - Centralized multi-dataset configuration (APTOS, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS).
  - Dynamic path resolution across local and remote environments.
  - Strict Patient-Level Grouping (GroupKFold) to prevent cross-eye data leakage.
  - Leave-One-Dataset-Out (LODO) Generator for out-of-domain (OOD) generalization testing.
  - Class imbalance mitigation via WeightedRandomSampler & inverse frequency weighting.
  - PyTorch Dataset supporting dual-branch inputs (Global Image + Local MIL Patches).
================================================================================
"""

import os
import re
import random
import logging
from typing import Dict, Any, Tuple, Generator, Optional, List, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Setup module logger
logger = logging.getLogger("DRDataLoader")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s | %(asctime)s | DRDataLoader] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ==============================================================================
# 1. DYNAMIC BASE DIRECTORY RESOLUTION
# ==============================================================================

def get_base_raw_dir(custom_path: Optional[str] = None) -> str:
    """
    Dynamically discovers the absolute path to the `data/raw/` directory.
    Checks multiple potential relative locations based on working directory.
    
    Args:
        custom_path: Explicit override path if provided.
        
    Returns:
        str: Absolute path to `data/raw/`.
        
    Raises:
        FileNotFoundError: If `data/raw/` directory cannot be found.
    """
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)
        
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, "data", "raw"),
        os.path.join(cwd, "..", "data", "raw"),
        os.path.join(cwd, "..", "..", "data", "raw"),
        os.path.join(cwd, "..", "..", "..", "data", "raw"),
        os.path.join(cwd, "..", "..", "..", "..", "data", "raw"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "raw"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "raw")
    ]
    
    for candidate in candidates:
        norm_path = os.path.abspath(candidate)
        if os.path.exists(norm_path) and os.path.isdir(norm_path):
            logger.info(f"Resolved raw data directory at: {norm_path}")
            return norm_path
            
    raise FileNotFoundError(
        f"[ERROR] Could not resolve `data/raw/` relative to current directory '{cwd}'. "
        f"Please pass `custom_path` to `get_base_raw_dir()`."
    )


# ==============================================================================
# 2. DATASET CONFIGURATION DICTIONARY
# ==============================================================================

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": os.path.join("aptos", "aptos_total.csv"),
        "img_dir": os.path.join("aptos", "aptos_total_images"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png",
        "nested_patient": False
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": os.path.join("ddr", "DR_grading.csv"),
        "img_dir": os.path.join("ddr", "DR_grading", "DR_grading"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg",
        "nested_patient": False
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": os.path.join("deepdrid", "deepdrid_total.csv"),
        "img_dir": os.path.join("deepdrid", "deepdrid_total_images"),
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "id_suffix": ".jpg",
        "nested_patient": True
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": os.path.join("idrid", "idrid_total.csv"),
        "img_dir": os.path.join("idrid", "idrid_total_images"),
        "id_col": "image_id",
        "label_col": "diagnosis",
        "id_suffix": ".jpg",
        "nested_patient": False
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": os.path.join("messidor2preprocess", "messidor_data.csv"),
        "img_dir": os.path.join("messidor2preprocess", "messidor-2", "messidor-2", "preprocess"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": "",
        "nested_patient": False
    },
    "EyePACS": {
        "name": "EyePACS (Kaggle)",
        "csv_path": os.path.join("zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
        "img_dir": os.path.join("zipEyepacs", "train"),
        "id_col": "image",
        "label_col": "level",
        "id_suffix": ".jpeg",
        "nested_patient": False
    }
}


# ==============================================================================
# 3. METADATA PARSING & DATASET STANDARDIZATION
# ==============================================================================

def parse_patient_and_side(image_id: str, dataset_key: str) -> Tuple[str, str]:
    """
    Parses unique patient_id and lateral eye side ('left', 'right', 'unknown')
    from image identifiers across different dataset naming conventions.
    
    Args:
        image_id: Raw image string identifier from CSV.
        dataset_key: Name key of the dataset.
        
    Returns:
        Tuple[str, str]: (patient_id, lateral_side)
    """
    img_str = str(image_id).strip()
    img_lower = img_str.lower()
    
    # 1. Lateral eye side resolution
    side = "unknown"
    if "left" in img_lower or "_l" in img_lower or img_lower.endswith("l"):
        side = "left"
    elif "right" in img_lower or "_r" in img_lower or img_lower.endswith("r"):
        side = "right"
        
    # 2. Patient ID extraction
    if dataset_key == "DeepDRiD":
        # Format: '<patient_id>_<side><view>' (e.g., '1_l1' -> patient '1')
        patient_id = img_str.split("_")[0]
    elif dataset_key == "EyePACS":
        # Format: '<patient_id>_left' or '<patient_id>_right'
        patient_id = img_str.split("_")[0]
    elif dataset_key == "Messidor-2":
        # Messidor image codes often contain date prefix + sequence (e.g., 20051020_43808_0100_PP)
        parts = img_str.split("_")
        patient_id = parts[0] + "_" + parts[1] if len(parts) >= 2 else img_str
    elif dataset_key == "IDRiD":
        # IDRiD images are typically distinct patient cases
        patient_id = img_str.replace("test", "").replace(".jpg", "")
    elif dataset_key == "DDR":
        # DDR images: unique image code per patient
        patient_id = img_str.split(".")[0]
    else:
        # APTOS: Each id_code represents a unique subject session
        patient_id = img_str
        
    # Prefix patient_id with dataset key to ensure global uniqueness when merging
    global_patient_id = f"{dataset_key}_{patient_id}"
    return global_patient_id, side


def load_single_dataset(key: str, cfg: Dict[str, Any], base_dir: str) -> pd.DataFrame:
    """
    Loads, cleans, and standardizes a single DR dataset into a uniform schema:
      ['image_id', 'level', 'dataset', 'image_file', 'full_path', 'patient_id', 'side']
      
    Args:
        key: Dataset key name (e.g., 'APTOS', 'DDR').
        cfg: Configuration dictionary entry for the dataset.
        base_dir: Base directory path containing raw datasets.
        
    Returns:
        pd.DataFrame: Cleaned, standardized DataFrame with verified image paths.
    """
    full_csv_path = os.path.join(base_dir, cfg["csv_path"])
    full_img_dir = os.path.join(base_dir, cfg["img_dir"])
    
    # EyePACS fallback check for single vs nested trainLabels.csv
    if key == "EyePACS" and not os.path.exists(full_csv_path):
        alt_csv = os.path.join(base_dir, "zipEyepacs", "trainLabels.csv")
        if os.path.exists(alt_csv):
            full_csv_path = alt_csv
            
    if not os.path.exists(full_csv_path):
        logger.warning(f"[{key}] CSV file not found at: {full_csv_path}. Skipping dataset.")
        return pd.DataFrame()
        
    if not os.path.exists(full_img_dir):
        logger.warning(f"[{key}] Image directory not found at: {full_img_dir}. Skipping dataset.")
        return pd.DataFrame()
        
    raw_df = pd.read_csv(full_csv_path)
    
    # Locate label column with fallback matching
    label_col = cfg["label_col"]
    if label_col not in raw_df.columns:
        matching = [c for c in raw_df.columns if any(k in c.lower() for k in ["dr", "level", "diagnosis", "grade"])]
        if matching:
            label_col = matching[0]
            logger.info(f"[{key}] Matched alternative label column: '{label_col}'")
        else:
            raise KeyError(f"[{key}] Could not find label column '{cfg['label_col']}' in CSV.")
            
    id_col = cfg["id_col"]
    if id_col not in raw_df.columns:
        matching_id = [c for c in raw_df.columns if any(k in c.lower() for k in ["id", "image", "file", "name"])]
        if matching_id:
            id_col = matching_id[0]
        else:
            raise KeyError(f"[{key}] Could not find ID column '{cfg['id_col']}' in CSV.")
            
    df = raw_df[[id_col, label_col]].copy()
    df.columns = ["image_id", "level"]
    df["dataset"] = cfg["name"]
    df["dataset_key"] = key
    
    # Clean and cast DR ordinal levels (0 to 4)
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df = df.dropna(subset=["level"]).copy()
    df["level"] = df["level"].astype(int)
    df = df[df["level"].isin([0, 1, 2, 3, 4])].copy()
    
    # Standardize image filename with suffix
    suffix = cfg.get("id_suffix", "")
    if suffix:
        df["image_file"] = df["image_id"].astype(str).apply(
            lambda x: x if x.endswith(suffix) else f"{x}{suffix}"
        )
    else:
        df["image_file"] = df["image_id"].astype(str)
        
    # Construct verified absolute image paths (handling nested patient subdirectories)
    if cfg.get("nested_patient", False):
        df["full_path"] = df.apply(
            lambda row: os.path.join(full_img_dir, str(row["image_id"]).split("_")[0], row["image_file"]),
            axis=1
        )
    else:
        df["full_path"] = df["image_file"].apply(lambda f: os.path.join(full_img_dir, f))
        
    # Extract Patient ID & Eye Lateral Side
    parsed_meta = df["image_id"].apply(lambda img_id: parse_patient_and_side(img_id, key))
    df["patient_id"] = [m[0] for m in parsed_meta]
    df["side"] = [m[1] for m in parsed_meta]
    
    # Audit file existence
    file_exists = df["full_path"].apply(os.path.exists)
    missing_count = (~file_exists).sum()
    if missing_count > 0:
        logger.warning(f"[{key}] {missing_count}/{len(df)} image files missing on disk! Filtering missing entries.")
        df = df[file_exists].copy()
        
    logger.info(f"[{key}] Loaded {len(df)} verified images | Unique Patients: {df['patient_id'].nunique()}")
    return df.reset_index(drop=True)


def load_all_datasets(base_dir: Optional[str] = None, config: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all configured datasets into a dictionary of DataFrames.
    
    Args:
        base_dir: Path to raw data directory. Auto-resolved if None.
        config: Custom dataset configuration dictionary. Defaults to DATASET_CONFIG.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping dataset key -> clean DataFrame.
    """
    if base_dir is None:
        base_dir = get_base_raw_dir()
    if config is None:
        config = DATASET_CONFIG
        
    datasets = {}
    for key, cfg in config.items():
        df = load_single_dataset(key, cfg, base_dir)
        if not df.empty:
            datasets[key] = df
            
    logger.info(f"Successfully loaded {len(datasets)} datasets: {list(datasets.keys())}")
    return datasets


# ==============================================================================
# 4. PATIENT-LEVEL SPLITTING & LODO GENERATOR
# ==============================================================================

def split_train_val_by_patient(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a merged DataFrame into Train and Validation subsets using GroupShuffleSplit
    grouped strictly by `patient_id` to prevent cross-eye leakage between eyes of the same patient.
    
    Args:
        df: Input DataFrame containing 'patient_id' column.
        val_ratio: Proportion of patients allocated to validation (default 15%).
        random_state: Reproducibility random seed.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    
    # Assert zero patient overlap
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    overlap = train_patients.intersection(val_patients)
    assert len(overlap) == 0, f"[ERROR] Data leakage detected! Overlapping patients: {len(overlap)}"
    
    return train_df, val_df


def leave_one_dataset_out_generator(
    dataset_dfs: Dict[str, pd.DataFrame],
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str], None, None]:
    """
    Leave-One-Dataset-Out (LODO) Generator for evaluating Out-Of-Domain (OOD) Domain Shift.
    Yields (train_df, val_df, test_df, ood_dataset_name) across N iterations,
    where each dataset serves as the isolated OOD test set once.
    
    Args:
        dataset_dfs: Dictionary of loaded dataset DataFrames.
        val_ratio: Validation ratio for in-domain patient-level splitting.
        random_state: Random seed for group splitting.
        
    Yields:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
          (In-Domain Train DataFrame, In-Domain Val DataFrame, OOD Test DataFrame, OOD Dataset Key)
    """
    dataset_keys = list(dataset_dfs.keys())
    total_rounds = len(dataset_keys)
    
    for i, test_key in enumerate(dataset_keys, 1):
        logger.info(f"\n=======================================================")
        logger.info(f"LODO FOLD [{i}/{total_rounds}]: OOD Test Set -> '{test_key}'")
        logger.info(f"=======================================================")
        
        # 1. Isolated Out-Of-Domain Test Set
        test_df = dataset_dfs[test_key].copy().reset_index(drop=True)
        
        # 2. Pool remaining 5 in-domain training datasets
        train_pool_keys = [k for k in dataset_keys if k != test_key]
        train_pool_dfs = [dataset_dfs[k] for k in train_pool_keys]
        merged_train_df = pd.concat(train_pool_dfs, axis=0, ignore_index=True)
        
        # 3. Patient-aware Train / Val split on the in-domain training pool
        train_df, val_df = split_train_val_by_patient(
            merged_train_df, val_ratio=val_ratio, random_state=random_state
        )
        
        logger.info(
            f"Train Size: {len(train_df)} images ({train_df['patient_id'].nunique()} patients) | "
            f"Val Size: {len(val_df)} images ({val_df['patient_id'].nunique()} patients) | "
            f"OOD Test Size: {len(test_df)} images ({test_df['patient_id'].nunique()} patients)"
        )
        
        yield train_df, val_df, test_df, test_key


# ==============================================================================
# 5. CLASS IMBALANCE MITIGATION & WEIGHTED SAMPLING
# ==============================================================================

def compute_class_weights(df: pd.DataFrame, num_classes: int = 5, beta: float = 0.999) -> torch.Tensor:
    """
    Computes Effective Number of Samples (ENS) / Inverse Frequency Class Weights
    for loss function re-weighting in severe class imbalance.
    
    Args:
        df: Training DataFrame containing 'level' column.
        num_classes: Number of ordinal severity classes (0 to 4).
        beta: Class-balanced hyperparameter (0.999 for medical datasets).
        
    Returns:
        torch.Tensor: Normalized tensor of shape (num_classes,).
    """
    class_counts = df["level"].value_counts().to_dict()
    weights = []
    for c in range(num_classes):
        count = class_counts.get(c, 1)
        effective_num = (1.0 - beta) / (1.0 - (beta ** count) + 1e-8)
        weights.append(effective_num)
        
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    weight_tensor = weight_tensor / weight_tensor.sum() * num_classes
    return weight_tensor


def create_weighted_sampler(df: pd.DataFrame, num_classes: int = 5) -> WeightedRandomSampler:
    """
    Creates a PyTorch WeightedRandomSampler to balance minority DR grades (Severe & Proliferative)
    during training batch sampling.
    
    Args:
        df: Training DataFrame containing 'level' column.
        num_classes: Number of ordinal severity classes (5).
        
    Returns:
        WeightedRandomSampler: PyTorch sampler instance.
    """
    class_counts = df["level"].value_counts().to_dict()
    class_weights = {c: 1.0 / (class_counts.get(c, 1) ** 0.5) for c in range(num_classes)}
    
    sample_weights = [class_weights[int(label)] for label in df["level"]]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ==============================================================================
# 6. PYTORCH DATASET (GLOBAL + LOCAL MIL DUAL-BRANCH)
# ==============================================================================

class DRMultiScaleDataset(Dataset):
    """
    PyTorch Dataset supporting Dual-Branch Multi-scale input representations:
      - Global Branch: 300x300 resized full fundus image.
      - Local MIL Branch: K=16 sliding window grid patches of size 128x128.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        global_transform: Optional[Any] = None,
        patch_transform: Optional[Any] = None,
        num_patches: int = 16,
        patch_size: int = 128,
        global_size: int = 300,
        is_training: bool = False
    ):
        """
        Args:
            df: DataFrame containing 'full_path' and 'level' columns.
            global_transform: Transform pipeline applied to global image.
            patch_transform: Transform pipeline applied to each MIL patch.
            num_patches: Number of local patches K to extract (default 16).
            patch_size: Resolution of each square patch (default 128).
            global_size: Resolution of global context image (default 300).
            is_training: Whether data is in training mode (enables random patch jitter).
        """
        self.df = df.reset_index(drop=True)
        self.global_transform = global_transform
        self.patch_transform = patch_transform
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.global_size = global_size
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.df)

    def _extract_grid_patches(self, img: Image.Image) -> torch.Tensor:
        """
        Extracts K=16 local high-resolution patches using a uniform grid sliding window,
        filtering out dark background border areas.
        
        Args:
            img: PIL Image object.
            
        Returns:
            torch.Tensor: Tensor of shape (K, 3, patch_size, patch_size).
        """
        w, h = img.size
        grid_dim = int(np.ceil(np.sqrt(self.num_patches)))  # 4x4 grid for K=16
        step_x = max(1, (w - self.patch_size) // max(1, grid_dim - 1))
        step_y = max(1, (h - self.patch_size) // max(1, grid_dim - 1))
        
        patches = []
        for i in range(grid_dim):
            for j in range(grid_dim):
                left = min(w - self.patch_size, i * step_x)
                top = min(h - self.patch_size, j * step_y)
                
                # Apply slight random perturbation during training
                if self.is_training:
                    left += random.randint(-4, 4)
                    top += random.randint(-4, 4)
                    left = max(0, min(w - self.patch_size, left))
                    top = max(0, min(h - self.patch_size, top))
                    
                patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
                
                if self.patch_transform:
                    patch_tensor = self.patch_transform(patch)
                else:
                    patch_tensor = torch.from_numpy(np.array(patch)).permute(2, 0, 1).float() / 255.0
                    
                patches.append(patch_tensor)
                if len(patches) == self.num_patches:
                    break
            if len(patches) == self.num_patches:
                break
                
        # Pad with duplicate patches if less than K
        while len(patches) < self.num_patches:
            patches.append(patches[-1] if patches else torch.zeros(3, self.patch_size, self.patch_size))
            
        return torch.stack(patches, dim=0)  # Shape: (K, 3, H, W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row["full_path"]
        label = int(row["level"])
        
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image at '{img_path}': {e}. Returning blank placeholder.")
            pil_img = Image.new("RGB", (self.global_size, self.global_size), (0, 0, 0))
            
        # 1. Global Branch Image
        if self.global_transform:
            global_tensor = self.global_transform(pil_img)
        else:
            resized = pil_img.resize((self.global_size, self.global_size), Image.BILINEAR)
            global_tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
            
        # 2. Local MIL Branch Patches
        local_patches = self._extract_grid_patches(pil_img)
        
        return {
            "global_img": global_tensor,                      # (3, 300, 300)
            "local_patches": local_patches,                   # (K=16, 3, 128, 128)
            "label": torch.tensor(label, dtype=torch.long),   # Integer label in [0..4]
            "dataset": row["dataset"],
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"])
        }


# ==============================================================================
# 7. DATALOADER BUILDER
# ==============================================================================

def get_dr_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    global_train_transform: Any,
    global_eval_transform: Any,
    patch_train_transform: Any,
    patch_eval_transform: Any,
    batch_size: int = 16,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    num_patches: int = 16,
    patch_size: int = 128,
    global_size: int = 300
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for Training, Validation, and OOD Testing sets.
    
    Args:
        train_df: Training DataFrame.
        val_df: In-domain Validation DataFrame.
        test_df: Out-of-domain Testing DataFrame.
        global_train_transform: Augmentation pipeline for global training image.
        global_eval_transform: Deterministic pipeline for global eval image.
        patch_train_transform: Augmentation pipeline for local MIL patches.
        patch_eval_transform: Deterministic pipeline for local MIL patches.
        batch_size: Mini-batch size.
        num_workers: CPU worker threads.
        use_weighted_sampler: Whether to apply class balancing WeightedRandomSampler.
        num_patches: Number of MIL patches per image (default 16).
        patch_size: Patch resolution (default 128).
        global_size: Global resolution (default 300).
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    train_dataset = DRMultiScaleDataset(
        df=train_df,
        global_transform=global_train_transform,
        patch_transform=patch_train_transform,
        num_patches=num_patches,
        patch_size=patch_size,
        global_size=global_size,
        is_training=True
    )
    
    val_dataset = DRMultiScaleDataset(
        df=val_df,
        global_transform=global_eval_transform,
        patch_transform=patch_eval_transform,
        num_patches=num_patches,
        patch_size=patch_size,
        global_size=global_size,
        is_training=False
    )
    
    test_dataset = DRMultiScaleDataset(
        df=test_df,
        global_transform=global_eval_transform,
        patch_transform=patch_eval_transform,
        num_patches=num_patches,
        patch_size=patch_size,
        global_size=global_size,
        is_training=False
    )
    
    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader
