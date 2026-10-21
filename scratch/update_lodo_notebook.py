# -*- coding: utf-8 -*-
"""
Script to update main_model_lodo.ipynb with:
1. Google Drive mount & exact unzipping directly from '/content/drive/MyDrive/native_cropped.zip' to '/content/dataset_local' without fallback.
2. Standardized paths and 100 epochs configuration (matching main_model (6).ipynb).
3. Robust Master cohort ingestion without synthetic fallbacks.
4. Per-fold metadata CSV exports (folds/fold_{k}_{cohort}_train.csv, val.csv, test.csv) following merge-dataset.ipynb logic.
5. 3-way preprocessing caching: local NPZ for train/val, zipped test cache stored on Google Drive (test_cache_{cohort}.zip) and auto-restored.
6. Intelligent checkpoint resumption & Compute Unit (CU) protection: completed folds are marked with is_completed=True and skipped on reconnect.
7. Post-hoc threshold calibration and all HD_1, HD_2, HD_3, HD_4 directives.
"""

import json
import ast
import sys
from pathlib import Path

NOTEBOOK_PATH = r"notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"

def build_cell_02():
    return """import os
import sys
import time
import re
import json
import logging
import random
import zipfile
import shutil
import warnings
from pathlib import Path, PureWindowsPath
from typing import List, Tuple, Union, Dict, Optional, Any
from functools import partial

# --- Deep Learning & PyTorch Core ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

# --- Scientific Computing & Image Processing ---
import cv2
cv2.setNumThreads(0)  # Mandatory per HD_3: prevent OpenCV multi-threading contention in DataLoader workers
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from tqdm.auto import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    accuracy_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    auc,
)

# --- Visualization & Academic Plotting ---
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch

warnings.filterwarnings("ignore")

# Setup logger with singleton handler
logger = logging.getLogger("LODOTrainer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s | %(asctime)s | LODOTrainer] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Set random seed for research reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
print("[INFO] Environment initialized successfully. Seed locked at 42.")
"""

def build_cell_04():
    return """# ==============================================================================
# 1.1 AUTOMATIC GOOGLE DRIVE MOUNTING (COLAB OR LOCAL)
# ==============================================================================
import os
import time

try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        print('Connecting to Google Drive...')
        drive.mount('/content/drive')
    else:
        print('Google Drive already mounted at /content/drive.')
except ImportError:
    print('Running in local environment (google.colab not available). Drive mount skipped.')

# ==============================================================================
# 1.2 AUTOMATED DATASET UNPACKING TO COLAB LOCAL FAST SSD (/content/dataset_local)
# ==============================================================================
# Exact standard path from main_model (6).ipynb - NO FALLBACK
zip_source = '/content/drive/MyDrive/native_cropped.zip'   # Path on Google Drive
extract_dest = '/content/dataset_local'                   # Local fast SSD directory on Colab

if os.path.exists(zip_source):
    if not os.path.exists(extract_dest) or len(os.listdir(extract_dest)) == 0:
        os.makedirs(extract_dest, exist_ok=True)
        print(f"Extracting dataset archive '{zip_source}' to '{extract_dest}'...")
        t0 = time.time()
        # Fast shell decompression
        get_ipython().system('unzip -q "$zip_source" -d "$extract_dest"')
        print(f"-> Extraction completed in {time.time() - t0:.1f} seconds!")
        print(f"-> Directory structure at '{extract_dest}':")
        get_ipython().system('ls -lh "$extract_dest"')
    else:
        print(f"-> Dataset directory '{extract_dest}' already populated ({len(os.listdir(extract_dest))} items). Skipping extraction.")
else:
    print(f"[INFO] zip_source '{zip_source}' not found or running in local environment.")
    if os.path.exists(extract_dest):
        print(f"-> Found existing local dataset directory at '{extract_dest}'.")
    else:
        print(f"-> Note: Ensure dataset metadata and images are located at '{extract_dest}' or update DATA_ROOT.")

# ==============================================================================
# 1.3 ROOT DIRECTORIES & PATH SYSTEM CONFIGURATION
# ==============================================================================
DATA_ROOT = '/content/dataset_local'
RESULT_ROOT = '/content/drive/MyDrive/Luận Văn Tốt Nghiệp 2026 - 2027/code/notebooks/diabetic_retinopathy/pipeline/model'

# If Drive path does not exist (e.g. running locally or path shifted), use safe fallback to results
if not os.path.exists(RESULT_ROOT) and not os.path.exists('/content/drive/MyDrive'):
    RESULT_ROOT = './results/lodo'

CHECKPOINT_DIR_LODO = os.path.join(RESULT_ROOT, "checkpoints_lodo")
RESULTS_DIR_LODO    = os.path.join(RESULT_ROOT, "results_lodo")
FOLDS_DIR           = os.path.join(RESULTS_DIR_LODO, "folds")
CACHE_DIR_LODO      = os.path.join("/content", "cache_fundus_dual_branch")
TEST_CACHE_DRIVE_DIR = os.path.join(RESULT_ROOT, "test_cache_zips")
FIGURES_DIR         = os.path.join(RESULT_ROOT, "figures_publication")

os.makedirs(CHECKPOINT_DIR_LODO, exist_ok=True)
os.makedirs(RESULTS_DIR_LODO, exist_ok=True)
os.makedirs(FOLDS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR_LODO, exist_ok=True)
os.makedirs(TEST_CACHE_DRIVE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set SMOKE_TEST = False for official 6-fold LODO evaluation across all 6 datasets
SMOKE_TEST = False

IMG_SIZE = 512
PATCH_SIZE = 128
PATCH_STRIDE = 96
PATCH_FOV_THRESH = 0.5
BATCH_SIZE = 16  # Optimized for Colab GPU (15GB+ VRAM)
QWK_WARMUP_EPOCHS = 1 if SMOKE_TEST else 15
LODO_EPOCHS = 3 if SMOKE_TEST else 100  # Elevated to 100 epochs matching main_model!
TARGET_QWK_WEIGHT = 0.4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6
PATIENCE_EARLY_STOPPING = 15
PATIENCE_REDUCE_LR = 4
REDUCE_LR_FACTOR = 0.5
GRAD_CLIP = 1.0
NUM_WORKERS = min(8, os.cpu_count() or 4) if os.name != 'nt' else 0

image_root_dir = f"{DATA_ROOT}/native_cropped" if os.path.exists(f"{DATA_ROOT}/native_cropped") else DATA_ROOT

# 6 Raw International Datasets Configuration
DATASET_CONFIG = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": "aptos/aptos_total.csv",
        "id_col": "id_code",
        "label_col": "diagnosis"
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": "ddr/DR_grading.csv",
        "id_col": "id_code",
        "label_col": "diagnosis"
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": "deepdrid/deepdrid_total.csv",
        "id_col": "image_id",
        "label_col": "patient_DR_Level"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": "idrid/idrid_total.csv",
        "id_col": "image_id",
        "label_col": "diagnosis"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": "messidor2preprocess/messidor_data.csv",
        "id_col": "id_code",
        "label_col": "diagnosis"
    },
    "EyePACS": {
        "name": "EyePACS (Kaggle)",
        "csv_path": "zipEyepacs/trainLabels.csv/trainLabels.csv",
        "id_col": "image",
        "label_col": "level"
    }
}

DATASET_KEYS = list(DATASET_CONFIG.keys())
print(f"Total LODO Cohorts Configured: {len(DATASET_KEYS)} datasets -> {DATASET_KEYS}")
print(f"Execution Mode: {'SMOKE TEST (Sanity check)' if SMOKE_TEST else 'FULL 6-FOLD LODO RUN (Production)'}")
print(f"Epochs per Fold: {LODO_EPOCHS} | Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")
print(f"Drive Results Directory: '{RESULT_ROOT}'")
print(f"Drive Test Cache Archive Directory: '{TEST_CACHE_DRIVE_DIR}'")
"""

def build_cell_07():
    return """# ==============================================================================
# 2.2 MASTER COHORT INGESTION & HARMONIZATION (Local & Colab Cross-Platform)
# ==============================================================================
def load_and_harmonize_master_cohorts() -> pd.DataFrame:
    \"\"\"
    Loads harmonized master dataset metadata across all 6 clinical cohorts.
    Strictly enforces real dataset paths (no synthetic fallback).
    Checks Colab local SSD (/content/dataset_local/), repo root, and drive paths.
    \"\"\"
    candidate_roots = [
        DATA_ROOT,
        f"{DATA_ROOT}/hybrid",
        "/content/dataset_local",
        "/content/dataset_local/hybrid",
        "./data/raw/hybrid",
        "../../../../data/raw/hybrid",
        "/content/drive/MyDrive/Luận Văn Tốt Nghiệp 2026 - 2027/code/data/raw/hybrid"
    ]

    master_df = None
    for root in candidate_roots:
        if not os.path.exists(root):
            continue
        p_train = os.path.join(root, "train_balanced_metadata.csv")
        if not os.path.exists(p_train):
            p_train = os.path.join(root, "train_metadata.csv")
        p_val   = os.path.join(root, "val_metadata.csv")
        p_test  = os.path.join(root, "test_metadata.csv")
        if os.path.exists(p_train) and os.path.exists(p_val) and os.path.exists(p_test):
            print(f"[INFO] Ingesting pre-harmonized master dataset from: '{root}'")
            dfs = [pd.read_csv(p) for p in [p_train, p_val, p_test]]
            master_df = pd.concat(dfs, ignore_index=True)
            break

    # Search for master single CSV
    if master_df is None:
        for root in candidate_roots:
            if not os.path.exists(root):
                continue
            p_master = os.path.join(root, "master_metadata.csv")
            if os.path.exists(p_master):
                print(f"[INFO] Loading master metadata from: '{p_master}'")
                master_df = pd.read_csv(p_master)
                break

    # If smoke test is requested and sample exists
    if master_df is None and SMOKE_TEST:
        sample_candidates = [
            f"{DATA_ROOT}/raw/samples/hybrid/sample_50_metadata.csv",
            "./data/raw/samples/hybrid/sample_50_metadata.csv",
            "../../../../data/raw/samples/hybrid/sample_50_metadata.csv"
        ]
        for p_sample in sample_candidates:
            if os.path.exists(p_sample):
                print(f"[WARN] Ingesting fallback sample dataset for SMOKE TEST: '{p_sample}'")
                master_df = pd.read_csv(p_sample)
                break

    if master_df is None:
        raise FileNotFoundError(
            f"[ERROR] Could not locate valid dataset metadata CSV in candidate roots: {candidate_roots}.\\n"
            "Please ensure dataset archive is unzipped at '/content/dataset_local' or metadata CSVs exist."
        )

    # Standardize column types and guarantee patient_id / side with cohort prefix (HD_2)
    if "patient_id" not in master_df.columns or master_df["patient_id"].isnull().any():
        pids, sides = [], []
        for _, r in master_df.iterrows():
            d_key = str(r.get("dataset_key", r.get("dataset_name", "DR")))
            pid, side = parse_patient_and_side(str(r["image_id"]), dataset_key=d_key)
            pids.append(pid)
            sides.append(side)
        master_df["patient_id"] = pids
        master_df["side"] = sides

    # Normalize native_path column if running on Colab where dataset is relocated
    colab_native_root = f"{DATA_ROOT}/native_cropped" if os.path.exists(f"{DATA_ROOT}/native_cropped") else DATA_ROOT
    def remap_native_path(p):
        p_str = str(p).replace("\\\\", "/")
        b_name = Path(p_str).name
        # Check if file exists directly
        if os.path.exists(p_str):
            return p_str
        # Check in colab_native_root
        d_key = p_str.split("/")[-2] if len(p_str.split("/")) >= 2 else ""
        colab_p = os.path.join(colab_native_root, d_key, b_name) if d_key else os.path.join(colab_native_root, b_name)
        if os.path.exists(colab_p):
            return colab_p
        # Search directly in colab_native_root
        candidate = os.path.join(colab_native_root, b_name)
        if os.path.exists(candidate):
            return candidate
        return p_str

    master_df["native_path"] = master_df["native_path"].apply(remap_native_path)

    # Ensure integer labels [0, 4]
    master_df["diagnosis"] = pd.to_numeric(master_df["diagnosis"], errors="coerce").fillna(0).astype(int)
    master_df = master_df[master_df["diagnosis"].isin([0, 1, 2, 3, 4])].copy().reset_index(drop=True)

    print(f"[INFO] Ingested Master Dataset: {len(master_df):,} images across {master_df['dataset_key'].nunique()} clinical cohorts.")
    print(f"       Unique Patients: {master_df['patient_id'].nunique():,}")
    return master_df

master_df = load_and_harmonize_master_cohorts()
print(\"\\nMaster DataFrame Summary by Cohort & Clinical Severity:\")
print(pd.crosstab(master_df[\"dataset_key\"], master_df[\"diagnosis\"]))
"""

def build_cell_08():
    return """# ==============================================================================
# 2.3 SOURCE-AWARE TRAINING BALANCING & 85/15 SPLIT (Fixes HD_2 Bug 3)
# ==============================================================================
def create_balanced_training_pool(
    df_train_5_sources: pd.DataFrame,
    target_range: Tuple[int, int] = (1500, 2500),
    seed: int = 42
) -> pd.DataFrame:
    \"\"\"
    Applies multi-source quota balancing per disease grade on the 5 training datasets.
    Prevents large cohorts (e.g. EyePACS) from overwhelming smaller cohorts (e.g. IDRiD).
    Safely filters remaining samples by native_path to prevent index mismatch duplication (HD_2).
    \"\"\"
    balanced_dfs = []
    np.random.seed(seed)

    for grade in range(5):
        grade_subset = df_train_5_sources[df_train_5_sources["diagnosis"] == grade].copy().reset_index(drop=True)
        current_count = len(grade_subset)

        if current_count == 0:
            continue

        datasets_in_grade = grade_subset["dataset_key"].unique()
        num_datasets = len(datasets_in_grade)

        if current_count > target_range[1]:
            target_count = int(np.random.RandomState(seed + grade).randint(target_range[0], target_range[1] + 1))
            base_quota = max(1, target_count // max(1, num_datasets))

            sampled_datasets = []
            leftover_quota = 0
            eligible_datasets = []

            for ds_name in datasets_in_grade:
                ds_subset = grade_subset[grade_subset["dataset_key"] == ds_name]
                ds_count = len(ds_subset)

                if ds_count <= base_quota:
                    sampled_datasets.append(ds_subset)
                    leftover_quota += (base_quota - ds_count)
                else:
                    eligible_datasets.append((ds_name, ds_subset, ds_count))

            if eligible_datasets:
                quota_per_large_ds = base_quota + (leftover_quota // len(eligible_datasets))
                for ds_name, ds_subset, ds_count in eligible_datasets:
                    take_count = min(quota_per_large_ds, ds_count)
                    ds_sampled = ds_subset.sample(take_count, random_state=seed)
                    sampled_datasets.append(ds_sampled)

            selected_subset = pd.concat(sampled_datasets, ignore_index=True)
            if len(selected_subset) > target_count:
                selected_subset = selected_subset.sample(target_count, random_state=seed).reset_index(drop=True)
            elif len(selected_subset) < target_count:
                # Top up from remaining using native_path matching (fixing HD_2 Page 3 bug)
                selected_paths = set(selected_subset["native_path"].dropna().tolist())
                remaining = grade_subset[~grade_subset["native_path"].isin(selected_paths)]
                if not remaining.empty:
                    extra_needed = min(target_count - len(selected_subset), len(remaining))
                    extra_samples = remaining.sample(extra_needed, random_state=seed)
                    selected_subset = pd.concat([selected_subset, extra_samples], ignore_index=True)
        else:
            selected_subset = grade_subset.copy()

        balanced_dfs.append(selected_subset)

    master_balanced = pd.concat(balanced_dfs, ignore_index=True)
    return master_balanced


def split_5_dataset_training_pool(
    df_pool: pd.DataFrame,
    train_ratio: float = 0.85,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    \"\"\"
    Splits the 5-dataset pool into Train (85%) and In-Domain Val (15%) using GroupShuffleSplit on patient_id.
    Strictly guarantees zero patient-level overlap between train and validation splits (HD_2).
    \"\"\"
    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_pool, groups=df_pool["patient_id"]))

    train_df = df_pool.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df_pool.iloc[val_idx].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    return train_df, val_df
"""

def build_cell_10():
    return """# ==============================================================================
# 3.1 ROBUST RETINAL FOV & SPATIAL ANCHOR COORDINATE COMPUTATION (Fixes HD_3)
# ==============================================================================
def apply_green_channel_ablation(
    img_rgb: np.ndarray,
    ablation_mode: str = "ben_graham_green"
) -> np.ndarray:
    \"\"\"
    Applies fundus image preprocessing ablations centered on the Green channel.
    The Green channel provides the highest contrast for retinal vascular micro-lesions.
    \"\"\"
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    b, g, r = cv2.split(denoised)

    if ablation_mode == "ben_graham_green":
        # Ben Graham method: subtract local illumination background
        blurred = cv2.GaussianBlur(g, (0, 0), 30)
        adjusted = cv2.addWeighted(g, 4, blurred, -4, 128)
        processed = cv2.merge([adjusted, adjusted, adjusted])
    elif ablation_mode == "clahe_green":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_g = clahe.apply(g)
        processed = cv2.merge([enhanced_g, enhanced_g, enhanced_g])
    else:
        processed = cv2.merge([g, g, g])

    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)


def get_global_augmentations(img_size: int = IMG_SIZE, is_train: bool = True) -> A.Compose:
    \"\"\"
    Constructs the Albumentations transformation pipeline for the Global Context branch.
    Resizes whole-fundus images to fixed dimensions (IMG_SIZE x IMG_SIZE).
    \"\"\"
    if is_train:
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR)
        ])


def compute_fov_mask(img_np: np.ndarray, erode_px: int = 15) -> np.ndarray:
    \"\"\"
    Computes the true retinal Field-of-View (FOV) binary mask for a SINGLE fundus image.
    IMPORTANT (HD_3): Must be computed on the RAW RGB fundus image BEFORE Ben Graham preprocessing.
    \"\"\"
    if img_np.ndim == 3:
        gray = img_np[:, :, 1] if img_np.shape[2] == 3 else img_np[:, :, 0]
    else:
        gray = img_np

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fov_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        fov_mask = cleaned

    if erode_px > 0:
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        fov_mask = cv2.erode(fov_mask, kernel_erode)

    return (fov_mask > 0).astype(np.uint8)


def compute_valid_anchors(
    img_shape: Tuple[int, int],
    fov_mask: np.ndarray,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5
) -> np.ndarray:
    \"\"\"Calculates 2D top-left coordinate anchors [y0, x0] passing the FOV threshold (stride=96 per HD_3).\"\"\"
    h, w = img_shape[:2]
    p = patch_size

    if np.any(fov_mask):
        coords = np.argwhere(fov_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    else:
        y_min, x_min, y_max, x_max = 0, 0, h, w

    content_h = max(1, y_max - y_min)
    content_w = max(1, x_max - x_min)

    def get_coords(curr_stride: int):
        s = max(1, curr_stride)
        y_starts = list(range(y_min, y_min + content_h - p + 1, s)) if content_h > p else [y_min]
        x_starts = list(range(x_min, x_min + content_w - p + 1, s)) if content_w > p else [x_min]

        if content_h > p and (y_min + content_h - p) not in y_starts:
            y_starts.append(y_min + content_h - p)
        if content_w > p and (x_min + content_w - p) not in x_starts:
            x_starts.append(x_min + content_w - p)

        valid_list = []
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = min(y0 + p, h), min(x0 + p, w)
                mask_patch = fov_mask[y0:y1, x0:x1]
                fov_ratio = mask_patch.mean() if mask_patch.size > 0 else 0.0
                if fov_ratio >= fov_thresh:
                    valid_list.append([y0, x0])
        return valid_list

    anchors = get_coords(stride)
    if len(anchors) < 32 and stride > 48 and min(content_h, content_w) > p:
        dense_anchors = get_coords(48)
        if len(dense_anchors) > len(anchors):
            anchors = dense_anchors

    if len(anchors) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        anchors = [[cy, cx]]

    return np.array(anchors, dtype=np.int32)


def precompute_dataset_cache(
    df: pd.DataFrame,
    img_root: Optional[str] = None,
    cache_dir: str = "./cache",
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
    ablation_mode: str = "ben_graham_green"
) -> Dict[str, int]:
    \"\"\"
    Offline precomputes and saves compressed NPZ cache files containing:
      - 'img': Preprocessed image array (processed_u8, uint8)
      - 'anchors': Valid patch anchor top-left coordinates (N, 2)
    Eliminates redundant CPU calculations across 100+ training epochs (HD_3).
    \"\"\"
    os.makedirs(cache_dir, exist_ok=True)
    records = df.to_dict("records")

    def process_single(row):
        dataset_key = str(row.get("dataset_key", "default"))
        native_path_str = str(row.get("native_path", row.get("raw_full_path", "")))
        img_id = PureWindowsPath(native_path_str).stem if native_path_str else str(row.get("image_id", "sample"))

        target_dir = os.path.join(cache_dir, dataset_key)
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, f"{img_id}.npz")

        if os.path.exists(out_path):
            return True

        if img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
            img_path = os.path.join(img_root, dataset_key, PureWindowsPath(native_path_str).name)
        else:
            img_path = native_path_str

        bgr = cv2.imread(img_path)
        if bgr is None:
            return False

        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fov_mask = compute_fov_mask(img_rgb, erode_px=15)
        processed_u8 = apply_green_channel_ablation(img_rgb, ablation_mode=ablation_mode)
        valid_anchors = compute_valid_anchors(
            img_rgb.shape[:2], fov_mask, patch_size=patch_size, stride=stride, fov_thresh=fov_thresh
        )

        np.savez_compressed(out_path, img=processed_u8, anchors=valid_anchors)
        return True

    success_count = 0
    fail_count = 0

    print(f"[INFO] Precomputing NPZ cache for {len(records)} samples -> '{cache_dir}'...")
    for r in tqdm(records, desc="Precomputing NPZ Cache", leave=False):
        res = process_single(r)
        if res:
            success_count += 1
        else:
            fail_count += 1

    print(f"[INFO] Cache generation completed: {success_count} succeeded, {fail_count} failed.")
    return {"success": success_count, "failed": fail_count}


def ensure_test_cache_on_drive(
    test_df: pd.DataFrame,
    held_out_key: str,
    local_cache_dir: str,
    drive_cache_dir: str,
    img_root: Optional[str] = None
) -> str:
    \"\"\"
    3-Way Preprocessing Caching Engine (Held-Out Test Set):
    1. Checks if 'test_cache_{held_out_key.lower()}.zip' already exists in Google Drive.
       If found, downloads/extracts directly into local_cache_dir, skipping CPU calculations.
    2. If not found on Drive, precomputes NPZ files from raw images into local_cache_dir,
       compresses the cohort cache into a .zip archive, and persists it to Google Drive.
    \"\"\"
    os.makedirs(drive_cache_dir, exist_ok=True)
    os.makedirs(local_cache_dir, exist_ok=True)
    zip_drive_path = os.path.join(drive_cache_dir, f"test_cache_{held_out_key.lower()}.zip")
    cohort_local_cache = os.path.join(local_cache_dir, held_out_key)

    # 1. Existing Zip on Google Drive -> Instant Restore
    if os.path.exists(zip_drive_path):
        print(f"[CACHE] Found existing Test Cache Zip on Google Drive: '{zip_drive_path}'")
        if not os.path.exists(cohort_local_cache) or len(os.listdir(cohort_local_cache)) == 0:
            print(f"  -> Restoring test cache into '{cohort_local_cache}'...")
            t0 = time.time()
            with zipfile.ZipFile(zip_drive_path, 'r') as zf:
                zf.extractall(local_cache_dir)
            print(f"  -> Restored test cache in {time.time() - t0:.1f}s!")
        else:
            print(f"  -> Local test cache already populated ({len(os.listdir(cohort_local_cache))} items).")
        return cohort_local_cache

    # 2. Compute NPZ from raw images
    print(f"[CACHE] Generating new precomputed test cache for '{held_out_key}' ({len(test_df)} images)...")
    precompute_dataset_cache(
        df=test_df,
        img_root=img_root,
        cache_dir=local_cache_dir,
        patch_size=128,
        stride=96,
        fov_thresh=0.5,
        ablation_mode="ben_graham_green"
    )

    # 3. Compress and persist to Google Drive
    if os.path.exists(cohort_local_cache) and len(os.listdir(cohort_local_cache)) > 0:
        print(f"  -> Compressing test cache and persisting to Drive: '{zip_drive_path}'...")
        t0 = time.time()
        temp_zip = os.path.join("/content", f"temp_test_cache_{held_out_key.lower()}.zip") if os.path.exists("/content") else f"./temp_{held_out_key.lower()}.zip"
        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(cohort_local_cache):
                for f in files:
                    full_f = os.path.join(root, f)
                    rel_f = os.path.relpath(full_f, local_cache_dir)
                    zf.write(full_f, rel_f)
        shutil.copy2(temp_zip, zip_drive_path)
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        print(f"  -> Saved test cache archive to Google Drive in {time.time() - t0:.1f}s!")

    return cohort_local_cache


def extract_mil_patches_from_anchors(
    img_np: np.ndarray,
    anchors: np.ndarray,
    patch_size: int = 128,
    mode: str = "train",
    num_sample: Union[int, Tuple[int, int]] = 48,
    max_patches_eval: int = 256
) -> torch.Tensor:
    \"\"\"Slices patches using precomputed coordinate anchors [y0, x0].\"\"\"
    h, w, _ = img_np.shape
    p = patch_size
    n_anchors = len(anchors)

    if mode == "train":
        if isinstance(num_sample, (tuple, list)):
            m_target = int(num_sample[0]) if num_sample[0] == num_sample[1] else int(np.random.randint(num_sample[0], num_sample[1] + 1))
        else:
            m_target = int(num_sample)
        if n_anchors < m_target:
            idx = np.random.choice(n_anchors, size=m_target, replace=True)
        else:
            idx = np.random.choice(n_anchors, size=m_target, replace=False)
        selected = anchors[idx]
    else:
        max_eval = min(n_anchors, max_patches_eval)
        idx = np.linspace(0, n_anchors - 1, max_eval).astype(int)
        selected = anchors[idx]

    patches = []
    for y0, x0 in selected:
        y1, x1 = min(y0 + p, h), min(x0 + p, w)
        patch = img_np[y0:y1, x0:x1]
        if patch.shape[0] < p or patch.shape[1] < p:
            pad_h = max(0, p - patch.shape[0])
            pad_w = max(0, p - patch.shape[1])
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        patches.append(patch)

    patches_np = np.stack(patches, axis=0)
    return torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()


def trainer_compatible_collate_fn(
    batch,
    mode: str = "train",
    num_sample: Union[int, Tuple[int, int]] = (32, 64),
    patch_size: int = 128,
    max_patches_eval: int = 256
):
    \"\"\"Collate function transmitting raw uint8 tensors across shared memory IPC (HD_3).\"\"\"
    global_imgs = torch.stack([item["global_image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)

    if mode == "train":
        if isinstance(num_sample, (tuple, list)):
            batch_m = int(np.random.randint(num_sample[0], num_sample[1] + 1))
        else:
            batch_m = int(num_sample)
    else:
        batch_m = max_patches_eval

    local_patches_list = [
        extract_mil_patches_from_anchors(
            item["local_native_image"],
            item["valid_anchors"],
            patch_size=patch_size,
            mode=mode,
            num_sample=batch_m,
            max_patches_eval=max_patches_eval
        )
        for item in batch
    ]

    if mode == "train":
        local_patches = torch.stack(local_patches_list, dim=0)
        return {
            "global_img": global_imgs,
            "local_patches": local_patches,
            "label": labels,
        }

    n_max = max(p.shape[0] for p in local_patches_list)
    padded, masks = [], []
    for p in local_patches_list:
        n = p.shape[0]
        if n < n_max:
            pad = torch.zeros((n_max - n, *p.shape[1:]), dtype=p.dtype)
            p = torch.cat([p, pad], dim=0)
        mask = torch.zeros(n_max, dtype=torch.bool)
        mask[:n] = True
        padded.append(p)
        masks.append(mask)

    local_patches = torch.stack(padded, dim=0)
    patch_mask = torch.stack(masks, dim=0)

    return {
        "global_img": global_imgs,
        "local_patches": local_patches,
        "patch_mask": patch_mask,
        "label": labels,
    }
"""

def build_cell_19():
    return """# ==============================================================================
# 5.2 DRTRAINER ENGINE OPTIMIZED FOR NVIDIA A100 WITH SMART RESUMPTION (HD_3 & HD_4)
# ==============================================================================
class DRTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "./checkpoints_lodo",
        qwk_warmup_epochs: int = 15
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.use_amp = use_amp and (self.device_type == "cuda")
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.qwk_warmup_epochs = qwk_warmup_epochs
        os.makedirs(checkpoint_dir, exist_ok=True)

        # GPU-side ImageNet Normalization Constants for high-throughput uint8 tensors (HD_3)
        self.mean_global = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std_global = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.mean_local = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.std_local = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 1, 3, 1, 1)

        if self.device_type == "cuda" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.use_scaler = False
        else:
            self.amp_dtype = torch.float16
            self.use_scaler = self.use_amp

        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_scaler)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets, all_probs = [], [], []

        pbar = tqdm(dataloader, desc="Training (AMP)", leave=False, unit="batch")
        for batch in pbar:
            global_img = batch["global_img"].to(self.device, non_blocking=True)
            local_patches = batch["local_patches"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            if global_img.dtype == torch.uint8:
                global_img = global_img.float().div_(255.0).sub_(self.mean_global).div_(self.std_global)
            if local_patches.dtype == torch.uint8:
                if local_patches.ndim == 5:
                    local_patches = local_patches.float().div_(255.0).sub_(self.mean_local).div_(self.std_local)
                else:
                    local_patches = local_patches.float().div_(255.0).sub_(self.mean_global).div_(self.std_global)

            patch_mask = batch.get("patch_mask", None)
            if patch_mask is not None:
                patch_mask = patch_mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                if patch_mask is not None:
                    try:
                        outputs = self.model(global_img, local_patches, patch_mask=patch_mask)
                    except TypeError:
                        outputs = self.model(global_img, local_patches)
                else:
                    outputs = self.model(global_img, local_patches)

                if hasattr(self.model, "compute_loss"):
                    loss = self.model.compute_loss(outputs, targets, class_weights=self.class_weights)
                elif hasattr(self.model, "head") and hasattr(self.model.head, "loss_fn"):
                    loss = self.model.head.loss_fn(outputs, targets, class_weights=self.class_weights)
                else:
                    loss = outputs["loss"]

            if self.use_scaler:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item() * len(targets)
            all_preds.extend(outputs["preds"].detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            if "probs" in outputs:
                all_probs.extend(outputs["probs"].detach().cpu().numpy())

        n_samples = len(all_targets)
        metrics = calculate_dr_metrics(all_targets, all_preds, y_probs=np.array(all_probs) if all_probs else None)
        metrics["Loss"] = total_loss / max(1, n_samples)
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets, all_probs, all_rank_probs = [], [], [], []

        pbar = tqdm(dataloader, desc="Eval", leave=False, unit="batch")
        for batch in pbar:
            global_img = batch["global_img"].to(self.device, non_blocking=True)
            local_patches = batch["local_patches"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            if global_img.dtype == torch.uint8:
                global_img = global_img.float().div_(255.0).sub_(self.mean_global).div_(self.std_global)
            if local_patches.dtype == torch.uint8:
                if local_patches.ndim == 5:
                    local_patches = local_patches.float().div_(255.0).sub_(self.mean_local).div_(self.std_local)
                else:
                    local_patches = local_patches.float().div_(255.0).sub_(self.mean_global).div_(self.std_global)

            patch_mask = batch.get("patch_mask", None)
            if patch_mask is not None:
                patch_mask = patch_mask.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                if patch_mask is not None:
                    try:
                        outputs = self.model(global_img, local_patches, patch_mask=patch_mask)
                    except TypeError:
                        outputs = self.model(global_img, local_patches)
                else:
                    outputs = self.model(global_img, local_patches)

                if hasattr(self.model, "compute_loss"):
                    loss = self.model.compute_loss(outputs, targets, class_weights=self.class_weights)
                elif hasattr(self.model, "head") and hasattr(self.model.head, "loss_fn"):
                    loss = self.model.head.loss_fn(outputs, targets, class_weights=self.class_weights)
                else:
                    loss = outputs["loss"]

            total_loss += loss.item() * len(targets)
            all_preds.extend(outputs["preds"].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if "probs" in outputs:
                all_probs.extend(outputs["probs"].cpu().numpy())

            rank_probs = extract_rank_probs(outputs)
            if rank_probs is not None:
                all_rank_probs.extend(rank_probs.cpu().numpy())

        n_samples = len(all_targets)
        eval_loss = total_loss / max(1, n_samples)
        probs_np = np.array(all_probs) if all_probs else None
        metrics = calculate_dr_metrics(all_targets, all_preds, y_probs=probs_np)
        metrics["Loss"] = eval_loss
        metrics["probs"] = probs_np

        if all_rank_probs:
            metrics["Monotonicity_Violations"] = rank_monotonicity_violations(np.array(all_rank_probs))
        else:
            metrics["Monotonicity_Violations"] = None

        return metrics, np.array(all_targets), np.array(all_preds)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        model_name: str = "dr_model",
        patience_early_stopping: int = 15,
        patience_reduce_lr: int = 4,
        reduce_lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        resume_checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        \"\"\"
        Executes training with ModelCheckpoint, EarlyStopping, and Smart Resumption.
        Prevents redundant re-training when fold is already marked completed (saving Colab CUs).
        \"\"\"
        best_val_qwk = -1.0
        best_val_loss = float("inf")
        start_epoch = 1
        is_completed = False

        epochs_no_improve_es = 0
        epochs_no_improve_lr = 0
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_qwk": [], "val_qwk": [],
            "val_within_1": [], "val_ref_sens": [],
            "val_ref_sens_at_spec95": [],
            "val_ref_sens_at_spec90": [],
            "val_ref_sens_at_spec85": [],
            "val_ref_sens_at_spec80": [],
            "val_ref_auc": []
        }

        using_external_scheduler = self.lr_scheduler is not None
        is_plateau_scheduler = using_external_scheduler and isinstance(
            self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pth")
        last_checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_last.pth")

        # --- RESUME TRAINING & CU PROTECTION WORKFLOW ---
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            logger.info(f"Inspecting checkpoint: '{resume_checkpoint_path}'")
            ckpt = torch.load(resume_checkpoint_path, map_location=self.device, weights_only=False)

            # 1. SMART CHECK: IF FOLD IS ALREADY MARKED COMPLETED -> DO NOT RETRAIN!
            if ckpt.get("is_completed", False) or ckpt.get("epoch", 0) >= epochs:
                logger.info(f"  -> [COMPLETED] Checkpoint is marked COMPLETED! (Epoch {ckpt.get('epoch', 0)}/{epochs}, Best Val QWK: {ckpt.get('best_val_qwk', 0.0):.4f})")
                logger.info("  -> Restoring model weights and skipping training loop to protect Colab Compute Units.")
                self.model.load_state_dict(ckpt["model_state_dict"])
                best_val_qwk = ckpt.get("best_val_qwk", ckpt.get("val_qwk", -1.0))
                history = ckpt.get("history", history)
                val_metrics = ckpt.get("val_metrics", {})
                test_metrics = ckpt.get("test_metrics", {})

                # If test_metrics were not saved, run one final evaluation on test_loader
                if not test_metrics and test_loader is not None:
                    logger.info("  -> Running evaluation on Held-Out Test set...")
                    test_metrics, _, _ = self.evaluate(test_loader)
                    test_metrics["Val_Test_Gap"] = best_val_qwk - test_metrics["QWK"]

                return {
                    "best_val_qwk": best_val_qwk,
                    "best_checkpoint": best_checkpoint_path if os.path.exists(best_checkpoint_path) else resume_checkpoint_path,
                    "history": history,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "ood_test_metrics": test_metrics,
                    "is_completed": True
                }

            # 2. RESUME ACTIVE TRAINING FROM LAST EPOCH
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.use_scaler and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            if using_external_scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                self.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_qwk = ckpt.get("best_val_qwk", ckpt.get("val_qwk", -1.0))
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", history)
            epochs_no_improve_es = ckpt.get("epochs_no_improve_es", 0)
            epochs_no_improve_lr = ckpt.get("epochs_no_improve_lr", 0)
            logger.info(f"-> Resuming training from Epoch {start_epoch}/{epochs} (Previous Best Val QWK: {best_val_qwk:.4f})")

        logger.info(f"Starting Training from Epoch {start_epoch} to {epochs} on {self.device} ({self.amp_dtype})")

        val_metrics = {}
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            head = getattr(self.model, "head", self.model)
            if hasattr(head, "set_qwk_weight") and hasattr(head, "target_qwk_weight"):
                progress = min(1.0, epoch / max(1, self.qwk_warmup_epochs))
                head.set_qwk_weight(progress * head.target_qwk_weight)

            train_metrics = self.train_epoch(train_loader)
            val_metrics, _, _ = self.evaluate(val_loader)
            elapsed = time.time() - t0

            val_loss = val_metrics["Loss"]
            val_qwk = val_metrics["QWK"]

            if using_external_scheduler:
                if is_plateau_scheduler:
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            history["train_loss"].append(train_metrics["Loss"])
            history["val_loss"].append(val_loss)
            history["train_qwk"].append(train_metrics["QWK"])
            history["val_qwk"].append(val_qwk)
            history["val_within_1"].append(val_metrics["Within_1_Grade_Acc"])
            history["val_ref_sens"].append(val_metrics["Referable_Sensitivity"])
            history.setdefault("val_ref_sens_at_spec95", []).append(val_metrics.get("Ref_Sensitivity_at_Spec95", float('nan')))
            history.setdefault("val_ref_sens_at_spec90", []).append(val_metrics.get("Ref_Sensitivity_at_Spec90", float('nan')))
            history.setdefault("val_ref_sens_at_spec85", []).append(val_metrics.get("Ref_Sensitivity_at_Spec85", float('nan')))
            history.setdefault("val_ref_sens_at_spec80", []).append(val_metrics.get("Ref_Sensitivity_at_Spec80", float('nan')))
            history["val_ref_auc"].append(val_metrics.get("Referable_AUC", float('nan')))

            log_msg = (
                f"[Epoch {epoch:02d}/{epochs:02d} - {elapsed:.1f}s] "
                f"Train Loss: {train_metrics['Loss']:.4f} | Train QWK: {train_metrics['QWK']:.4f} || "
                f"Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f} | "
                f"Val Within-1: {val_metrics['Within_1_Grade_Acc']*100:.1f}% | "
                f"Val Ref-Sens: {val_metrics['Referable_Sensitivity']*100:.1f}% | "
                f"Val Ref-Sens@Spec85: {val_metrics.get('Ref_Sensitivity_at_Spec85', float('nan'))*100:.1f}% | "
                f"Val Ref-AUC: {val_metrics.get('Referable_AUC', float('nan')):.4f} | "
                f"LR: {current_lr:.2e}"
            )
            logger.info(log_msg)

            checkpoint_state = {
                "epoch": epoch,
                "is_completed": (epoch >= epochs),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.use_scaler else None,
                "scheduler_state_dict": self.lr_scheduler.state_dict() if using_external_scheduler else None,
                "val_qwk": val_qwk,
                "best_val_qwk": max(best_val_qwk, val_qwk),
                "best_val_loss": min(best_val_loss, val_loss),
                "history": history,
                "val_metrics": val_metrics,
                "epochs_no_improve_es": epochs_no_improve_es,
                "epochs_no_improve_lr": epochs_no_improve_lr
            }
            torch.save(checkpoint_state, last_checkpoint_path)

            if val_qwk > best_val_qwk:
                best_val_qwk = val_qwk
                epochs_no_improve_es = 0
                torch.save(checkpoint_state, best_checkpoint_path)
                logger.info(f"  -> Saved new Best Model: '{best_checkpoint_path}' (Val QWK: {best_val_qwk:.4f})")
            else:
                epochs_no_improve_es += 1

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_no_improve_lr = 0
            else:
                epochs_no_improve_lr += 1
                if not using_external_scheduler and epochs_no_improve_lr >= patience_reduce_lr:
                    old_lr = current_lr
                    new_lr = max(old_lr * reduce_lr_factor, min_lr)
                    if old_lr > min_lr:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        logger.info(f"  -> [ReduceLROnPlateau] Reduced LR: {old_lr:.2e} -> {new_lr:.2e}")
                    epochs_no_improve_lr = 0

            # Early Stopping on Val QWK (HD_4)
            if epochs_no_improve_es >= patience_early_stopping:
                logger.info(f"  -> [EarlyStopping] Val QWK plateaued for {patience_early_stopping} epochs. Stopping early!")
                is_completed = True
                break

        # Mark fold completed
        is_completed = True

        test_metrics = {}
        if test_loader is not None and os.path.exists(best_checkpoint_path):
            logger.info("Loading Best Model weights for evaluation on Held-Out OOD Test set...")
            ckpt = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            test_metrics, _, _ = self.evaluate(test_loader)

            val_test_gap = best_val_qwk - test_metrics["QWK"]
            test_metrics["Val_Test_Gap"] = val_test_gap

            logger.info(
                f"=== HELD-OUT OOD TEST RESULTS ===\\n"
                f"OOD Test QWK: {test_metrics['QWK']:.4f} | "
                f"Val QWK: {best_val_qwk:.4f} | "
                f"Delta_LODO Gap: {val_test_gap:.4f} | "
                f"OOD Within-1: {test_metrics['Within_1_Grade_Acc']*100:.1f}% | "
                f"OOD Ref-Sens@Spec85: {test_metrics.get('Ref_Sensitivity_at_Spec85', float('nan'))*100:.1f}% | "
                f"OOD Ref-AUC: {test_metrics.get('Referable_AUC', float('nan')):.4f}"
            )

        # Update best checkpoint with final completed state and test metrics
        if os.path.exists(best_checkpoint_path):
            ckpt = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
            ckpt["is_completed"] = True
            ckpt["test_metrics"] = test_metrics
            torch.save(ckpt, best_checkpoint_path)

        if os.path.exists(last_checkpoint_path):
            ckpt_last = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_last["is_completed"] = True
            ckpt_last["test_metrics"] = test_metrics
            torch.save(ckpt_last, last_checkpoint_path)

        return {
            "best_val_qwk": best_val_qwk,
            "best_checkpoint": best_checkpoint_path,
            "history": history,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "ood_test_metrics": test_metrics,
            "is_completed": True
        }
"""

def build_cell_21():
    return """# ==============================================================================
# 6. EXECUTE 6-FOLD LEAVE-ONE-DATASET-OUT (LODO) PIPELINE WITH SMART RESUME & DRIVE CACHE
# ==============================================================================
lodo_results = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_fold_artifacts = {}

summary_csv_path = os.path.join(RESULTS_DIR_LODO, "lodo_results_summary.csv")
summary_json_path = os.path.join(RESULTS_DIR_LODO, "lodo_summary.json")

# Preload existing results summary if resuming across sessions
if os.path.exists(summary_json_path):
    try:
        with open(summary_json_path, "r", encoding="utf-8") as f:
            lodo_results = json.load(f)
        print(f"[INFO] Loaded existing LODO results across {len(lodo_results)} completed folds from '{summary_json_path}'.")
    except Exception as e:
        logger.warning(f"Could not read previous summary JSON: {e}")

print(f"\\n{'='*75}")
print("STARTING 6-FOLD LEAVE-ONE-DATASET-OUT (LODO) GENERALIZATION PROTOCOL")
print(f"{'='*75}\\n")

folds_to_run = DATASET_KEYS[:2] if SMOKE_TEST else DATASET_KEYS

train_collate_fn = partial(trainer_compatible_collate_fn, mode="train", num_sample=(32, 64))
eval_collate_fn = partial(trainer_compatible_collate_fn, mode="eval", max_patches_eval=256)

for fold_idx, held_out_key in enumerate(folds_to_run, 1):
    held_out_name = DATASET_CONFIG[held_out_key]["name"]
    print(f"\\n{'='*75}")
    print(f"LODO FOLD {fold_idx}/{len(folds_to_run)}: Holding out [{held_out_name}] as Unseen OOD Test Set")
    print(f"{'='*75}")

    fold_model_name = f"lodo_fold_{fold_idx}_{held_out_key.lower()}"
    ckpt_path = os.path.join(CHECKPOINT_DIR_LODO, f"{fold_model_name}_best.pth")
    last_ckpt_path = os.path.join(CHECKPOINT_DIR_LODO, f"{fold_model_name}_last.pth")

    # --------------------------------------------------------------------------
    # CHECKPOINT CU PROTECTION CHECK: SKIP COMPLETED FOLDS IN 0 SECONDS
    # --------------------------------------------------------------------------
    if held_out_key in lodo_results and os.path.exists(ckpt_path):
        try:
            ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if ckpt_meta.get("is_completed", False):
                print(f"[SKIP] Fold {fold_idx} ({held_out_name}) has ALREADY COMPLETED training!")
                print(f"  -> In-Domain Val QWK: {lodo_results[held_out_key].get('in_domain_val_qwk', 0.0):.4f} | OOD Test QWK: {lodo_results[held_out_key].get('ood_test_qwk', 0.0):.4f}")
                print(f"  -> Skipping fold execution to protect Google Colab Compute Units (CU).\\n")
                continue
        except Exception as e:
            logger.warning(f"Failed checking checkpoint completion flag for {held_out_name}: {e}")

    # 1. Isolate the Held-out OOD Test Dataset (100% unseen, retains 100% natural clinical prevalence)
    ood_test_df = master_df[master_df["dataset_key"] == held_out_key].copy().reset_index(drop=True)
    ood_test_df["split"] = "test"

    # 2. Pool the remaining 5 datasets for training and in-domain validation
    train_5_df = master_df[master_df["dataset_key"] != held_out_key].copy().reset_index(drop=True)

    # 3. Split 5-dataset pool FIRST into Train Raw (85%) and In-Domain Val (15%) with Patient Grouping (HD_2)
    train_raw_df, val_df = split_5_dataset_training_pool(train_5_df, train_ratio=0.85, random_state=42 + fold_idx)

    # 4. Apply Multi-source quota balancing STRICTLY to the 85% training split (HD_2)
    target_sample_range = (40, 80) if SMOKE_TEST else (1500, 2500)
    train_df = create_balanced_training_pool(train_raw_df, target_range=target_sample_range, seed=42 + fold_idx)

    # 5. PERSIST METADATA CSVs FOR THIS FOLD (Ensures complete clinical auditability)
    p_train_csv = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{held_out_key.lower()}_train.csv")
    p_val_csv   = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{held_out_key.lower()}_val.csv")
    p_test_csv  = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{held_out_key.lower()}_test.csv")

    train_df.to_csv(p_train_csv, index=False)
    val_df.to_csv(p_val_csv, index=False)
    ood_test_df.to_csv(p_test_csv, index=False)

    print(f"  * Saved Fold {fold_idx} Metadata -> '{FOLDS_DIR}'")
    print(f"    - Train (85% Balanced): {len(train_df):,} images | Patients: {train_df['patient_id'].nunique():,}")
    print(f"    - Val (15% Natural)   : {len(val_df):,} images | Patients: {val_df['patient_id'].nunique():,}")
    print(f"    - Held-out OOD Test   : {len(ood_test_df):,} images | Natural Prevalence ({held_out_name})")

    # 6. ENSURE PRECOMPUTED TEST CACHE ON GOOGLE DRIVE
    ensure_test_cache_on_drive(
        test_df=ood_test_df,
        held_out_key=held_out_key,
        local_cache_dir=CACHE_DIR_LODO,
        drive_cache_dir=TEST_CACHE_DRIVE_DIR,
        img_root=image_root_dir
    )

    # Pre-cache training and validation samples
    precompute_dataset_cache(val_df, img_root=image_root_dir, cache_dir=CACHE_DIR_LODO)

    # 7. Build PyTorch Datasets & DataLoaders
    train_ds = FundusDualBranchDataset(train_df, img_size=IMG_SIZE, split="train", cache_dir=CACHE_DIR_LODO, img_root=image_root_dir)
    val_ds   = FundusDualBranchDataset(val_df, img_size=IMG_SIZE, split="val", cache_dir=CACHE_DIR_LODO, img_root=image_root_dir)
    test_ds  = FundusDualBranchDataset(ood_test_df, img_size=IMG_SIZE, split="test", cache_dir=CACHE_DIR_LODO, img_root=image_root_dir)

    num_w = NUM_WORKERS
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_w, collate_fn=train_collate_fn, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_w, collate_fn=eval_collate_fn, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_w, collate_fn=eval_collate_fn, pin_memory=torch.cuda.is_available())

    # 8. Build Dual-Branch Model Architecture & Optimizer
    fold_model = build_model(pretrained=True, use_cbam=True, fusion_dim=512, num_classes=5, qwk_weight=TARGET_QWK_WEIGHT)
    optimizer = torch.optim.AdamW(fold_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=MIN_LR)

    trainer = DRTrainer(
        model=fold_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        use_amp=True,
        checkpoint_dir=CHECKPOINT_DIR_LODO,
        qwk_warmup_epochs=QWK_WARMUP_EPOCHS
    )

    # Resumption priority: last epoch checkpoint -> best checkpoint -> None (fresh run)
    resume_target = last_ckpt_path if os.path.exists(last_ckpt_path) else (ckpt_path if os.path.exists(ckpt_path) else None)

    # 9. Train model or restore completed fold
    fold_history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=LODO_EPOCHS,
        model_name=fold_model_name,
        patience_early_stopping=PATIENCE_EARLY_STOPPING,
        patience_reduce_lr=PATIENCE_REDUCE_LR,
        reduce_lr_factor=REDUCE_LR_FACTOR,
        min_lr=MIN_LR,
        resume_checkpoint_path=resume_target
    )

    in_qwk = fold_history["best_val_qwk"]
    ood_metrics = fold_history.get("ood_test_metrics", {})
    ood_qwk = ood_metrics.get("QWK", 0.0)
    delta_qwk = in_qwk - ood_qwk

    # 10. Post-Hoc Screening Threshold Calibration on Val vs OOD Test (HD_4)
    calib_df = None
    if "probs" in fold_history.get("val_metrics", {}) and "probs" in ood_metrics:
        calib_df = calibrate_and_evaluate_screening(
            val_probs=fold_history["val_metrics"]["probs"],
            val_targets=np.array(val_df["diagnosis"].values),
            test_probs=ood_metrics["probs"],
            test_targets=np.array(ood_test_df["diagnosis"].values),
            target_specificities=(0.80, 0.85, 0.90, 0.95)
        )

    lodo_results[held_out_key] = {
        "fold": fold_idx,
        "held_out_dataset": held_out_name,
        "in_domain_val_qwk": in_qwk,
        "ood_test_qwk": ood_qwk,
        "delta_qwk": delta_qwk,
        "delta_lodo": delta_qwk,
        "ood_accuracy": ood_metrics.get("Accuracy", 0.0) * 100,
        "ood_within_1": ood_metrics.get("Within_1_Grade_Acc", 0.0) * 100,
        "ood_ref_sens": ood_metrics.get("Referable_Sensitivity", 0.0) * 100,
        "ood_ref_sens_spec95": ood_metrics.get("Ref_Sensitivity_at_Spec95", float("nan")) * 100,
        "ood_ref_sens_spec90": ood_metrics.get("Ref_Sensitivity_at_Spec90", float("nan")) * 100,
        "ood_ref_sens_spec85": ood_metrics.get("Ref_Sensitivity_at_Spec85", float("nan")) * 100,
        "ood_ref_sens_spec80": ood_metrics.get("Ref_Sensitivity_at_Spec80", float("nan")) * 100,
        "ood_ref_auc": ood_metrics.get("Referable_AUC", float("nan")),
        "recall_g0": ood_metrics.get("Recall_Grade_0", 0.0) * 100,
        "recall_g1": ood_metrics.get("Recall_Grade_1", 0.0) * 100,
        "recall_g2": ood_metrics.get("Recall_Grade_2", 0.0) * 100,
        "recall_g3": ood_metrics.get("Recall_Grade_3", 0.0) * 100,
        "recall_g4": ood_metrics.get("Recall_Grade_4", 0.0) * 100,
        "monotonicity_violations": (ood_metrics.get("Monotonicity_Violations") or 0.0) * 100,
        "checkpoint_path": fold_history["best_checkpoint"]
    }

    eval_fold_artifacts = {
        "model": fold_model,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "held_out_name": held_out_name,
        "held_out_key": held_out_key,
        "val_metrics": fold_history.get("val_metrics", {}),
        "test_metrics": ood_metrics
    }

    # Save running results after every completed fold
    df_lodo = pd.DataFrame.from_dict(lodo_results, orient="index")
    df_lodo.to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(lodo_results, f, indent=2)

    print(f"\\n [FOLD {fold_idx} RESULTS: {held_out_name}]")
    print(f"  * In-Domain Val QWK           : {in_qwk:.4f}")
    print(f"  * Out-of-Domain (OOD) Test QWK: {ood_qwk:.4f}")
    print(f"  * Domain Shift Drop (Delta)   : {delta_qwk:.4f}")
    print(f"  * OOD Within-1-Grade Accuracy : {lodo_results[held_out_key]['ood_within_1']:.1f}%")
    print(f"  * OOD Ref-Sens @ Spec 85%     : {lodo_results[held_out_key]['ood_ref_sens_spec85']:.1f}% (HD_4 Expected: 79.6%)")
    print(f"  * OOD Referable Continuous AUC: {lodo_results[held_out_key]['ood_ref_auc']:.4f}")

df_lodo = pd.DataFrame.from_dict(lodo_results, orient="index")
df_lodo.to_csv(summary_csv_path, index=False)
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(lodo_results, f, indent=2)

print(f"\\n{'='*75}")
print(f" [SUCCESS] 6-Fold LODO Protocol Completed & Exported to '{RESULTS_DIR_LODO}'")
print(f"  - CSV Summary : {summary_csv_path}")
print(f"  - JSON Summary: {summary_json_path}")
print(f"{'='*75}\\n")
if not df_lodo.empty and "held_out_dataset" in df_lodo.columns:
    cols_to_show = [c for c in ["held_out_dataset", "in_domain_val_qwk", "ood_test_qwk", "delta_lodo", "ood_ref_sens_spec85", "ood_ref_auc"] if c in df_lodo.columns]
    print(df_lodo[cols_to_show].to_string(index=False))
"""

def update_notebook():
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Validate cell count
    print(f"Current cell count: {len(nb['cells'])}")

    # Update cell sources
    updates = {
        2: build_cell_02(),
        4: build_cell_04(),
        7: build_cell_07(),
        8: build_cell_08(),
        10: build_cell_10(),
        19: build_cell_19(),
        21: build_cell_21()
    }

    for cell_idx, new_code in updates.items():
        # Syntax check via ast.parse
        # Strip ipython magic lines for ast parsing
        clean_code_lines = []
        for line in new_code.splitlines(True):
            if line.strip().startswith("get_ipython") or line.strip().startswith("!") or line.strip().startswith("%"):
                clean_code_lines.append("# " + line)
            else:
                clean_code_lines.append(line)
        clean_code = "".join(clean_code_lines)

        try:
            ast.parse(clean_code)
            print(f"  [OK] Syntax check passed for Cell {cell_idx:02d}")
        except SyntaxError as se:
            print(f"  [ERROR] Syntax error in Cell {cell_idx:02d}: {se}")
            sys.exit(1)

        # Convert to Jupyter notebook cell lines
        cell_lines = [line + "\n" for line in new_code.split("\n")]
        # Remove trailing extra newline
        if cell_lines and cell_lines[-1] == "\n":
            cell_lines.pop()

        nb["cells"][cell_idx]["source"] = cell_lines

    # Save updated notebook
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[SUCCESS] Successfully updated {NOTEBOOK_PATH}!")

if __name__ == "__main__":
    update_notebook()
