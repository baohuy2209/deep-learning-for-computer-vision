# %% [markdown]
# # Robust Image Preprocessing, Harmonization & Balancing Pipeline for 6-Dataset DR Hybrid Pool
# 
# **Author:** Expert Computer Vision & Medical AI Assistant  
# **Dataset Scope:** APTOS 2019, EyePACS, Messidor-2, IDRiD, DeepDRiD, DDR (53,977 raw images).  
# **Target Outputs:** Dual resolutions (`300x300` & `512x512`), 70/15/15 leak-free stratified splits, CLAHE contrast enhancement, and targeted training set class balancing (exactly 2,000 images/class = 10,000 total train images).

# %%
import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# Random seeds for research reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"[INFO] Initialized preprocessing pipeline with SEED={SEED}")

# %% [markdown]
# ## 1. Dynamic Path Resolution & Configuration Dictionary

# %%
def get_base_paths() -> tuple[str, str]:
    """
    Dynamically resolves relative paths to `data/raw/` and `data/processed/`
    based on current working directory.
    """
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "data", "raw")):
        raw_dir = os.path.join(cwd, "data", "raw")
        proc_dir = os.path.join(cwd, "data", "processed")
    elif os.path.exists(os.path.join(cwd, "..", "..", "..", "data", "raw")):
        raw_dir = os.path.abspath(os.path.join(cwd, "..", "..", "..", "data", "raw"))
        proc_dir = os.path.abspath(os.path.join(cwd, "..", "..", "..", "data", "processed"))
    elif os.path.exists(os.path.join(cwd, "..", "data", "raw")):
        raw_dir = os.path.abspath(os.path.join(cwd, "..", "data", "raw"))
        proc_dir = os.path.abspath(os.path.join(cwd, "..", "data", "processed"))
    else:
        raise FileNotFoundError("[ERROR] Cannot locate `data/raw/` relative to " + cwd)

    os.makedirs(proc_dir, exist_ok=True)
    return raw_dir, proc_dir

BASE_RAW_DIR, BASE_PROCESSED_DIR = get_base_paths()
HYBRID_PROCESSED_300 = os.path.join(BASE_PROCESSED_DIR, "hybrid", "300x300")
HYBRID_PROCESSED_512 = os.path.join(BASE_PROCESSED_DIR, "hybrid", "512x512")
HYBRID_RAW_REF = os.path.join(BASE_RAW_DIR, "hybrid")

os.makedirs(HYBRID_PROCESSED_300, exist_ok=True)
os.makedirs(HYBRID_PROCESSED_512, exist_ok=True)
os.makedirs(HYBRID_RAW_REF, exist_ok=True)

print(f"[INFO] Raw Dir: {BASE_RAW_DIR}")
print(f"[INFO] Processed 300x300 Dir: {HYBRID_PROCESSED_300}")
print(f"[INFO] Processed 512x512 Dir: {HYBRID_PROCESSED_512}")

# Raw dataset configuration dictionary
DATASET_CONFIG = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": os.path.join("aptos", "train_1.csv"),
        "img_dir": os.path.join("aptos", "train_images"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "DDR": {
        "name": "DDR",
        "csv_path": os.path.join("ddr", "DR_grading.csv"),
        "img_dir": os.path.join("ddr", "DR_grading", "DR_grading"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ""
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": os.path.join("deepdrid", "DeepDRiD-master", "regular_fundus_images", "regular-fundus-training", "regular-fundus-training.csv"),
        "img_dir": os.path.join("deepdrid", "DeepDRiD-master", "regular_fundus_images", "regular-fundus-training", "Images"),
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "id_suffix": ".jpg"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": os.path.join("idrid", "idrid_labels.csv"),
        "img_dir": os.path.join("idrid", "Imagenes"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": os.path.join("messidor2preprocess", "messidor_data.csv"),
        "img_dir": os.path.join("messidor2preprocess", "messidor-2"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ""
    },
    "EyePACS": {
        "name": "EyePACS",
        "csv_path": os.path.join("zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
        "img_dir": os.path.join("zipEyepacs", "train"),
        "id_col": "image",
        "label_col": "level",
        "id_suffix": ".jpeg"
    }
}

# Fallback check for EyePACS CSV path if nested directory varies
eyepacs_alt_csv = os.path.join(BASE_RAW_DIR, "zipEyepacs", "trainLabels.csv")
if not os.path.exists(os.path.join(BASE_RAW_DIR, DATASET_CONFIG["EyePACS"]["csv_path"])) and os.path.isfile(eyepacs_alt_csv):
    DATASET_CONFIG["EyePACS"]["csv_path"] = os.path.join("zipEyepacs", "trainLabels.csv")

# %% [markdown]
# ## 2. Step 1: Master Dataset Harmonization & Schema Standardization
# 
# **Clinical Justification:** All 6 datasets map to the International Clinical Diabetic Retinopathy Severity Scale (ICDRS 0-4).
# Merging multi-center datasets exposes the network to diverse camera sensors (Topcon, Canon, Zeiss) and illumination conditions, improving cross-domain generalizability.

# %%
def load_and_standardize_single_dataset(key: str, cfg: dict, base_dir: str) -> pd.DataFrame:
    full_csv_path = os.path.join(base_dir, cfg["csv_path"])
    full_img_dir = os.path.join(base_dir, cfg["img_dir"])
    
    if not os.path.exists(full_csv_path):
        print(f"[WARNING] Skipping {key} - CSV not found: {full_csv_path}")
        return pd.DataFrame()
        
    raw_df = pd.read_csv(full_csv_path)
    label_col = cfg["label_col"]
    if label_col not in raw_df.columns:
        possible = [c for c in raw_df.columns if "dr" in c.lower() or "level" in c.lower() or "diagnosis" in c.lower()]
        if possible:
            label_col = possible[0]
            
    df = raw_df[[cfg["id_col"], label_col]].copy()
    df.columns = ["image_id", "diagnosis"]
    df["dataset_name"] = cfg["name"]
    
    # Ensure correct suffix
    if cfg["id_suffix"]:
        df["image_file"] = df["image_id"].astype(str).apply(lambda x: x if x.endswith(cfg["id_suffix"]) else x + cfg["id_suffix"])
    else:
        df["image_file"] = df["image_id"].astype(str)
        
    df["raw_full_path"] = df["image_file"].apply(lambda f: os.path.join(full_img_dir, f))
    
    # Clean non-integer labels
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["diagnosis"].isin([0, 1, 2, 3, 4])]
    
    return df

def merge_and_harmonize_all_datasets(config_dict: dict, base_dir: str) -> pd.DataFrame:
    dfs = []
    for key, cfg in config_dict.items():
        df_sub = load_and_standardize_single_dataset(key, cfg, base_dir)
        if not df_sub.empty:
            dfs.append(df_sub)
            print(f"[HARMONIZED] {cfg['name']}: {len(df_sub):,} records")
            
    master_df = pd.concat(dfs, ignore_index=True)
    print(f"\n[SUCCESS] Master Harmonized Dataset: {len(master_df):,} total images across {master_df['dataset_name'].nunique()} sources.")
    return master_df

master_df = merge_and_harmonize_all_datasets(DATASET_CONFIG, BASE_RAW_DIR)

# %% [markdown]
# ## 3. Step 2: Stratified Leak-Free Train/Val/Test Split (70% / 15% / 15%)
# 
# **Critical Methodological Requirement:**
# The 70/15/15 split MUST occur *before* any oversampling or data augmentation.
# Applying augmentation prior to splitting causes severe data leakage (augmented variations of training images bleeding into validation or test sets).
# Validation (15%) and Test (15%) sets remain untouched in their raw real-world class distributions.

# %%
def create_stratified_splits(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # Step 1: Split into Train (70%) and Temp (30%)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df, 
        test_size=temp_ratio, 
        stratify=df["diagnosis"], 
        random_state=random_state
    )
    
    # Step 2: Split Temp (30%) evenly into Val (15%) and Test (15%)
    val_relative_ratio = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1.0 - val_relative_ratio), 
        stratify=temp_df["diagnosis"], 
        random_state=random_state
    )
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    print("=== STRATIFIED SPLIT SUMMARY (LEAK-FREE) ===")
    print(f"Train Set (70%): {len(train_df):,} images")
    print(f"Val Set   (15%): {len(val_df):,} images")
    print(f"Test Set  (15%): {len(test_df):,} images")
    
    return train_df, val_df, test_df

train_df_raw, val_df_raw, test_df_raw = create_stratified_splits(master_df, random_state=SEED)

# %% [markdown]
# ## 4. Step 3 & Step 4: Geometric Preprocessing & Image Enhancement
# 
# **1. Auto-Crop ROI:** Thresholding technique to remove non-diagnostic black borders around the fundus circular region.  
# **2. Green Channel Extraction / Blending:** Highlights retinal microvascular structures (hemorrhages, microaneurysms exhibit high contrast in the green spectrum).  
# **3. CLAHE Contrast Enhancement:** Contrast Limited Adaptive Histogram Equalization (`clipLimit=2.0`, `tileGridSize=(8,8)`).  
# **4. Gaussian Blur Noise Reduction:** Removes high-frequency camera noise.

# %%
def crop_fundus_roi(img_bgr: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """
    Auto-crops uninformative black borders around fundus photograph.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
        
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance
    
    if not np.any(mask):
        return img_bgr
        
    # Crop to bounding box of non-black pixels
    row_mask = np.any(mask, axis=1)
    col_mask = np.any(mask, axis=0)
    ymin, ymax = np.where(row_mask)[0][[0, -1]]
    xmin, xmax = np.where(col_mask)[0][[0, -1]]
    
    cropped = img_bgr[ymin:ymax+1, xmin:xmax+1]
    return cropped if cropped.size > 0 else img_bgr

def enhance_fundus_image(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extracts Green channel weighted blending, applies CLAHE, and Gaussian Blur.
    """
    # Convert RGB to BGR for OpenCV processing
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Auto-Crop ROI
    cropped_bgr = crop_fundus_roi(bgr)
    
    # 2. Green Channel weighted blend
    b, g, r = cv2.split(cropped_bgr)
    # Blend green channel heavily with subtle gray balance
    green_blended = cv2.addWeighted(g, 0.75, cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY), 0.25, 0)
    
    # 3. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_g = clahe.apply(green_blended)
    
    # Reconstruct 3-channel RGB image (using enhanced green for all channels or merging)
    enhanced_bgr = cv2.merge([b, enhanced_g, r])
    
    # 4. Gaussian Blur Noise Reduction
    blurred_bgr = cv2.GaussianBlur(enhanced_bgr, (3, 3), 0)
    
    return cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)

# %% [markdown]
# ## 5. Step 5: Albumentations Augmentation & Targeted Training Set Balancing
# 
# **Balancing Strategy (Applied STRICTLY to 70% Training Set):**
# - Target Count: Exactly 2,000 images per DR severity grade (10,000 total training images).
# - Class 0, Class 1, Class 2: Randomly undersampled to 2,000 images.
# - Class 3, Class 4: Oversampled & augmented via Albumentations to 2,000 images.
# - Validation (8,097) & Test (8,097) sets: Left untouched in raw distribution.

# %%
train_augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
])

def balance_and_preprocess_split(
    split_name: str, 
    df: pd.DataFrame, 
    target_per_class: int = 2000, 
    is_train: bool = False
) -> pd.DataFrame:
    
    records = []
    
    for grade in range(5):
        grade_subset = df[df["diagnosis"] == grade].copy()
        # Filter for rows whose raw image file actually exists on disk
        valid_mask = grade_subset["raw_full_path"].apply(lambda p: os.path.isfile(p))
        grade_subset_existing = grade_subset[valid_mask].reset_index(drop=True)
        current_count = len(grade_subset_existing)
        
        if current_count == 0:
            print(f"[WARNING] No existing image files found for grade {grade} in split '{split_name}'.")
            continue

        if is_train:
            if current_count >= target_per_class:
                selected_subset = grade_subset_existing.sample(target_per_class, random_state=SEED).reset_index(drop=True)
                aug_multiplier = 1
            else:
                selected_subset = grade_subset_existing.copy()
                aug_multiplier = (target_per_class // current_count) + 1
        else:
            selected_subset = grade_subset_existing.copy()
            aug_multiplier = 1
            
        count_saved = 0
        
        # Loop until target_per_class is reached for train set
        while is_train and count_saved < target_per_class:
            for idx, row in selected_subset.iterrows():
                if count_saved >= target_per_class:
                    break
                
                img_path = row["raw_full_path"]
                try:
                    raw_img = Image.open(img_path).convert("RGB")
                    img_np = np.array(raw_img)
                except Exception:
                    continue

                if count_saved < current_count and not is_train:
                    processed_np = enhance_fundus_image(img_np)
                    is_aug = False
                elif count_saved < len(selected_subset) and current_count >= target_per_class:
                    processed_np = enhance_fundus_image(img_np)
                    is_aug = False
                else:
                    enhanced = enhance_fundus_image(img_np)
                    augmented = train_augmentor(image=enhanced)
                    processed_np = augmented["image"]
                    is_aug = True

                out_name = f"{split_name}_g{grade}_{row['image_id']}_c{count_saved}.png"
                out_name_clean = re.sub(r'[^\w\.-]', '_', out_name)
                
                path_300 = os.path.join(HYBRID_PROCESSED_300, split_name, str(grade), out_name_clean)
                path_512 = os.path.join(HYBRID_PROCESSED_512, split_name, str(grade), out_name_clean)
                
                os.makedirs(os.path.dirname(path_300), exist_ok=True)
                os.makedirs(os.path.dirname(path_512), exist_ok=True)
                
                img_pil = Image.fromarray(processed_np)
                img_pil.resize((300, 300), Image.Resampling.BILINEAR).save(path_300)
                img_pil.resize((512, 512), Image.Resampling.BILINEAR).save(path_512)
                
                records.append({
                    "split": split_name,
                    "diagnosis": grade,
                    "image_id": row["image_id"],
                    "dataset_name": row["dataset_name"],
                    "path_300": path_300,
                    "path_512": path_512,
                    "is_augmented": is_aug
                })
                
                count_saved += 1

        if not is_train:
            for idx, row in selected_subset.iterrows():
                img_path = row["raw_full_path"]
                try:
                    raw_img = Image.open(img_path).convert("RGB")
                    img_np = np.array(raw_img)
                except Exception:
                    continue

                processed_np = enhance_fundus_image(img_np)
                out_name = f"{split_name}_g{grade}_{row['image_id']}.png"
                out_name_clean = re.sub(r'[^\w\.-]', '_', out_name)
                
                path_300 = os.path.join(HYBRID_PROCESSED_300, split_name, str(grade), out_name_clean)
                path_512 = os.path.join(HYBRID_PROCESSED_512, split_name, str(grade), out_name_clean)
                
                os.makedirs(os.path.dirname(path_300), exist_ok=True)
                os.makedirs(os.path.dirname(path_512), exist_ok=True)
                
                img_pil = Image.fromarray(processed_np)
                img_pil.resize((300, 300), Image.Resampling.BILINEAR).save(path_300)
                img_pil.resize((512, 512), Image.Resampling.BILINEAR).save(path_512)
                
                records.append({
                    "split": split_name,
                    "diagnosis": grade,
                    "image_id": row["image_id"],
                    "dataset_name": row["dataset_name"],
                    "path_300": path_300,
                    "path_512": path_512,
                    "is_augmented": False
                })
                
                count_saved += 1
                if is_train and count_saved >= target_per_class:
                    break
                    
    return pd.DataFrame(records)

# Execute preprocessing & balancing on Train, Val, and Test splits
print("\n=== PROCESSING 70% BALANCED TRAIN SET (TARGET: 2,000 PER CLASS) ===")
train_processed_df = balance_and_preprocess_split("train", train_df_raw, target_per_class=2000, is_train=True)

print("\n=== PROCESSING 15% UNTOUCHED VALIDATION SET ===")
val_processed_df = balance_and_preprocess_split("val", val_df_raw, is_train=False)

print("\n=== PROCESSING 15% UNTOUCHED TEST SET ===")
test_processed_df = balance_and_preprocess_split("test", test_df_raw, is_train=False)

# %% [markdown]
# ## 6. Step 6: Pixel Normalization & Pipeline Verification

# %%
def normalize_image_tensor(img_np: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """
    Normalizes pixel values to [0, 1] or standard ImageNet Z-score.
    """
    img_float = img_np.astype(np.float32) / 255.0
    if mode == "minmax":
        return img_float
    elif mode == "zscore":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (img_float - mean) / std
    return img_float

# Save processed metadata tables
train_processed_df.to_csv(os.path.join(BASE_PROCESSED_DIR, "hybrid", "train_balanced_metadata.csv"), index=False)
val_processed_df.to_csv(os.path.join(BASE_PROCESSED_DIR, "hybrid", "val_raw_metadata.csv"), index=False)
test_processed_df.to_csv(os.path.join(BASE_PROCESSED_DIR, "hybrid", "test_raw_metadata.csv"), index=False)

print("\n=== FINAL PREPROCESSING & BALANCING SUMMARY ===")
print("Balanced Train Set (300x300 & 512x512):")
print(train_processed_df["diagnosis"].value_counts().sort_index())

print("\nUntouched Validation Set:")
print(val_processed_df["diagnosis"].value_counts().sort_index())

print("\nUntouched Test Set:")
print(test_processed_df["diagnosis"].value_counts().sort_index())
