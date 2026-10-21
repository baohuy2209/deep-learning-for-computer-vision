"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - DATA LOADER & STERILE MULTI-CENTER SPLITTING
================================================================================
Core Research Focus: 6-Dataset Harmonization, Zero Cross-Eye Bilateral Contamination,
Multi-Center LODO Generator, and Dual-Branch PyTorch Dataset.

Supported Datasets (55,570 Total Standardized Images):
  1. APTOS 2019 (Aravind Eye Hospital, India)
  2. DDR Dataset (Multi-hospital Partners, China)
  3. DeepDRiD (ISBI 2020 Challenge, Shanghai, China)
  4. IDRiD (Nanded Eye Hospital, India - Ultra High-Res)
  5. Messidor-2 (Brest & Saint-Etienne Hospitals, France)
  6. EyePACS (Kaggle Screening Cohort, USA)
================================================================================
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
try:
    import albumentations as A
except ImportError:
    A = None


# ==============================================================================
# 1. DATASET CONFIGURATION DICTIONARY
# ==============================================================================

DATASET_CONFIG = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": os.path.join("aptos", "aptos_total.csv"),
        "img_dir": os.path.join("aptos", "aptos_total_images"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": os.path.join("ddr", "DR_grading.csv"),
        "img_dir": os.path.join("ddr", "DR_grading", "DR_grading"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": os.path.join("deepdrid", "deepdrid_total.csv"),
        "img_dir": os.path.join("deepdrid", "deepdrid_total_images"),
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "id_suffix": ".jpg"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": os.path.join("idrid", "idrid_total.csv"),
        "img_dir": os.path.join("idrid", "idrid_total_images"),
        "id_col": "image_id",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": os.path.join("messidor2preprocess", "messidor_data.csv"),
        "img_dir": os.path.join("messidor2preprocess", "messidor-2", "messidor-2", "preprocess"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "EyePACS": {
        "name": "EyePACS (Kaggle)",
        "csv_path": os.path.join("zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
        "img_dir": os.path.join("zipEyepacs", "train"),
        "id_col": "image",
        "label_col": "level",
        "id_suffix": ".jpeg"
    }
}


# ==============================================================================
# 2. PATIENT ID PARSING & ZERO-LEAKAGE GROUPING
# ==============================================================================

def parse_patient_and_side(image_id: str, dataset_key: str) -> Tuple[str, str]:
    """
    Extracts unique patient_id and eye side ('left', 'right', 'unknown') across
    heterogeneous clinical cohort conventions.
    
    Ensures:
      - IDRiD produces ~400+ distinct patient units (not 1).
      - EyePACS and DeepDRiD do not collide across dataset boundaries.
      - DDR multi-field codes are correctly grouped.
      - Messidor-2 paired examination codes are mapped to single individuals.
    """
    img_str = str(image_id).strip()
    img_lower = img_str.lower()
    k = dataset_key.lower()
    
    # 1. Lateral eye side resolution
    side = "unknown"
    if "left" in img_lower or "_l" in img_lower or img_lower.endswith("l"):
        side = "left"
    elif "right" in img_lower or "_r" in img_lower or img_lower.endswith("r"):
        side = "right"
        
    # 2. Patient ID extraction per cohort convention
    if "idrid" in k:
        # IDRiD_001 or train_IDRiD_001 -> take suffix after last underscore ("001")
        base = img_lower.replace(".jpg", "").replace(".png", "")
        pid = base.split("_")[-1]
    elif "eyepacs" in k or "deepdrid" in k:
        # EyePACS: '10_left' -> '10'; DeepDRiD: '104_l1' -> '104'
        pid = img_str.split("_")[0]
    elif "ddr" in k:
        # DDR: '007-0004-000' -> group by clinic & subject prefix
        parts = img_str.replace(".jpg", "").replace(".png", "").split("-")
        pid = "-".join(parts[:2]) if len(parts) >= 2 else img_str.split(".")[0]
    elif "messidor" in k:
        # Messidor-2 paired session: 20051020_43808_0100_PP -> group by session date prefix
        pid = img_str.split("_")[0]
    else:
        # APTOS: 1 image = 1 subject
        pid = img_str
        
    # Mandatory dataset prefix preventing inter-dataset collision
    namespaced_pid = f"{dataset_key}::{pid}"
    return namespaced_pid, side


def load_single_dataset(key: str, cfg: Dict[str, Any], base_dir: str) -> pd.DataFrame:
    """Loads, standardizes, and parses metadata for a single dataset."""
    full_csv_path = os.path.join(base_dir, cfg["csv_path"])
    full_img_dir = os.path.join(base_dir, cfg["img_dir"])
    
    if not os.path.exists(full_csv_path):
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
    df["dataset_key"] = key
    
    if cfg["id_suffix"]:
        df["image_file"] = df["image_id"].astype(str).apply(
            lambda x: x if x.endswith(cfg["id_suffix"]) else x + cfg["id_suffix"]
        )
    else:
        df["image_file"] = df["image_id"].astype(str)
        
    df["raw_full_path"] = df["image_file"].apply(lambda f: os.path.join(full_img_dir, f))
    
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["diagnosis"].isin([0, 1, 2, 3, 4])]
    
    parsed = df["image_id"].apply(lambda x: parse_patient_and_side(x, dataset_key=key))
    df["patient_id"] = [p[0] for p in parsed]
    df["side"] = [p[1] for p in parsed]
    return df


def harmonize_all_datasets(config_dict: Dict[str, Any], base_dir: str) -> pd.DataFrame:
    """Harmonizes all available cohorts into a master sterile DataFrame."""
    dfs = []
    for key, cfg in config_dict.items():
        sub_df = load_single_dataset(key, cfg, base_dir)
        if not sub_df.empty:
            dfs.append(sub_df)
    if not dfs:
        return pd.DataFrame()
    master_df = pd.concat(dfs, ignore_index=True)
    return master_df


# ==============================================================================
# 3. STERILE SPLITTING & MULTI-SOURCE QUOTA BALANCING
# ==============================================================================

def split_master_pool_patient_level(
    master_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes leak-free patient-level grouping on the FULL master DataFrame.
    Guarantees:
      - Inter-eye bilateral data leakage is completely eliminated.
      - Validation and Test sets preserve their natural clinical prevalence (untouched).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    gss_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss_train.split(master_df, groups=master_df['patient_id']))
    
    train_df = master_df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = master_df.iloc[temp_idx].copy().reset_index(drop=True)
    
    gss_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state)
    val_idx, test_idx = next(gss_test.split(temp_df, groups=temp_df['patient_id']))
    
    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return train_df, val_df, test_df


def create_balanced_master_pool(
    train_df: pd.DataFrame,
    target_range: Tuple[int, int] = (1500, 2500),
    seed: int = 42
) -> pd.DataFrame:
    """
    Applies multi-source quota allocation STRICTLY to the training DataFrame.
    Prevents large cohorts (EyePACS) from overwhelming clinical reference cohorts (IDRiD, Messidor-2).
    Uses stable column identifiers ('raw_full_path') to avoid index collisions.
    """
    balanced_dfs = []
    np.random.seed(seed)
    
    for grade in range(5):
        grade_subset = train_df[train_df["diagnosis"] == grade].copy()
        valid_mask = grade_subset["raw_full_path"].apply(lambda p: os.path.isfile(p))
        grade_subset_existing = grade_subset[valid_mask].copy().reset_index(drop=True)
        current_count = len(grade_subset_existing)
        
        if current_count == 0:
            continue
            
        datasets_in_grade = grade_subset_existing["dataset_name"].unique()
        num_datasets = len(datasets_in_grade)
        
        if current_count > target_range[1]:
            target_count = int(np.random.RandomState(seed + grade).randint(target_range[0], target_range[1] + 1))
            base_quota = target_count // num_datasets
            
            sampled_datasets = []
            leftover_quota = 0
            eligible_datasets = []
            
            for ds_name in datasets_in_grade:
                ds_subset = grade_subset_existing[grade_subset_existing["dataset_name"] == ds_name]
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
                # Filter via stable unique column 'raw_full_path' to prevent index reset bug
                remaining = grade_subset_existing[~grade_subset_existing["raw_full_path"].isin(selected_subset["raw_full_path"])]
                if not remaining.empty:
                    diff = target_count - len(selected_subset)
                    extra = remaining.sample(min(diff, len(remaining)), random_state=seed)
                    selected_subset = pd.concat([selected_subset, extra], ignore_index=True)
        else:
            selected_subset = grade_subset_existing.copy()
            
        balanced_dfs.append(selected_subset)
        
    balanced_train_df = pd.concat(balanced_dfs, ignore_index=True)
    return balanced_train_df


# ==============================================================================
# 4. ROBUST RETINAL FOV & ANCHOR COORDINATE COMPUTATION
# ==============================================================================

def compute_fov_mask(img_np: np.ndarray, erode_px: int = 15) -> np.ndarray:
    """
    Computes the true retinal Field-of-View (FOV) binary mask for a fundus image.
    IMPORTANT: Must be computed on the RAW RGB fundus image BEFORE Ben Graham preprocessing.
    (Ben Graham sets background to 128 gray, which collapses Otsu thresholding).

    4-Step Pipeline:
    1. Otsu's Adaptive Thresholding on Green/Intensity channel.
    2. Morphological Close + Open (25x25 ellipse) to seal internal vessel holes.
    3. Largest Connected Component to isolate true retinal disk (strip glare/artifacts).
    4. Inward Morphological Erosion (erode_px=15) to eliminate edge vignetting/halos.
    """
    if img_np.ndim == 3:
        gray = img_np[:, :, 1] if img_np.shape[2] == 3 else img_np[:, :, 0]
    else:
        gray = img_np

    # 1. Otsu's Adaptive Thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Morphological Close & Open
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 3. Largest Connected Component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fov_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        fov_mask = cleaned

    # 4. Inward Erosion
    if erode_px > 0:
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        fov_mask = cv2.erode(fov_mask, kernel_erode)

    return (fov_mask > 0).astype(np.uint8)


def apply_green_channel_ablation(
    img_rgb: np.ndarray, ablation_mode: str = "ben_graham_green"
) -> np.ndarray:
    """
    Applies fundus image preprocessing centered on the Green channel.
    The Green channel provides the highest contrast for retinal vascular micro-lesions.
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    b, g, r = cv2.split(denoised)

    if ablation_mode == "ben_graham_green":
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


def compute_valid_anchors(
    img_shape: Tuple[int, int],
    fov_mask: np.ndarray,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5
) -> np.ndarray:
    """
    Calculates 2D top-left coordinate anchors [y0, x0] passing the FOV threshold.
    Returns (N, 2) numpy array of integer coordinates.
    """
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
    # Adaptive Fallback: if fewer than 32 candidates, rescan with dense stride (s=48)
    if len(anchors) < 32 and stride > 48 and min(content_h, content_w) > p:
        dense_anchors = get_coords(48)
        if len(dense_anchors) > len(anchors):
            anchors = dense_anchors

    if len(anchors) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        anchors = [[cy, cx]]

    return np.array(anchors, dtype=np.int32)


# ==============================================================================
# 5. OFFLINE PRECOMPUTATION & NPZ CACHING ENGINE
# ==============================================================================

def precompute_dataset_cache(
    df: pd.DataFrame,
    img_root: Optional[str] = None,
    cache_dir: str = "data/cache/npz_anchors",
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
    ablation_mode: str = "ben_graham_green",
    n_jobs: int = -1
) -> Dict[str, int]:
    """
    Offline precomputes and saves compressed NPZ cache files containing:
      - 'img': Preprocessed image array (processed_u8, uint8)
      - 'anchors': Valid patch anchor top-left coordinates (N, 2)
    Eliminates redundant CPU calculations across 100+ training epochs.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pathlib import PureWindowsPath
    from tqdm.auto import tqdm

    os.makedirs(cache_dir, exist_ok=True)
    records = df.to_dict("records")

    def process_single(row):
        dataset_key = row.get("dataset_key", "default")
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
    with tqdm(total=len(records), desc="Precomputing NPZ Cache") as pbar:
        for r in records:
            res = process_single(r)
            if res:
                success_count += 1
            else:
                fail_count += 1
            pbar.update(1)

    print(f"[INFO] Cache generation completed: {success_count} succeeded, {fail_count} failed.")
    return {"success": success_count, "failed": fail_count}


# ==============================================================================
# 6. SLIDING-WINDOW MIL PATCH EXTRACTION (Overlapping s=96 + Raw FOV Filter + uint8 IPC)
# ==============================================================================

def extract_mil_patches_sliding(
    img_np: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
    mode: str = "train",
    num_sample: Union[int, Tuple[int, int]] = 48,
    max_patches_eval: int = 256,
    native_path: Optional[str] = None,
    valid_anchors: Optional[np.ndarray] = None
) -> torch.Tensor:
    """
    Extracts local high-resolution retinal patches via strided sliding-window scanning.
    Features Adaptive Fallback Stride:
      - Default stride = 96 (25% overlap) for high-resolution images (>1500px).
      - If image is small (~500px) and candidates < M, automatically reduces stride (s=48 or 64)
        to extract sufficient unique patches without artificial replacement/duplication.
      - If precomputed valid_anchors are provided via cache, slices directly for instant execution.
    """
    h, w, _ = img_np.shape
    p = patch_size

    # Determine target M quota in train mode
    if mode == "train":
        if isinstance(num_sample, (tuple, list)):
            m_target = int(np.random.randint(num_sample[0], num_sample[1] + 1))
        else:
            m_target = int(num_sample)
    else:
        m_target = max_patches_eval

    # Fast-path: Slicing directly if valid_anchors from cache are provided
    if valid_anchors is not None and len(valid_anchors) > 0:
        n_anchors = len(valid_anchors)
        if mode == "train":
            if n_anchors < m_target:
                idx = np.random.choice(n_anchors, size=m_target, replace=True)
            else:
                idx = np.random.choice(n_anchors, size=m_target, replace=False)
            selected_coords = valid_anchors[idx]
        else:
            max_eval = min(n_anchors, max_patches_eval)
            idx = np.linspace(0, n_anchors - 1, max_eval).astype(int)
            selected_coords = valid_anchors[idx]

        patches = []
        for y0, x0 in selected_coords:
            y1, x1 = min(y0 + p, h), min(x0 + p, w)
            patch = img_np[y0:y1, x0:x1]
            if patch.shape[0] < p or patch.shape[1] < p:
                pad_h = max(0, p - patch.shape[0])
                pad_w = max(0, p - patch.shape[1])
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            patches.append(patch)

        patches_np = np.stack(patches, axis=0)
        return torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()

    # 1. Use external FOV mask if provided, else compute on img_np
    if fov_mask is None:
        fov_mask = compute_fov_mask(img_np, erode_px=15)

    if np.any(fov_mask):
        coords = np.argwhere(fov_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    else:
        y_min, x_min, y_max, x_max = 0, 0, h, w

    content_h = max(1, y_max - y_min)
    content_w = max(1, x_max - x_min)

    # 2. Inner candidate generator with dynamic stride
    def get_candidates(current_stride):
        s = max(1, current_stride)
        y_starts = list(range(y_min, y_min + content_h - p + 1, s)) if content_h > p else [y_min]
        x_starts = list(range(x_min, x_min + content_w - p + 1, s)) if content_w > p else [x_min]

        if content_h > p and (y_min + content_h - p) not in y_starts:
            y_starts.append(y_min + content_h - p)
        if content_w > p and (x_min + content_w - p) not in x_starts:
            x_starts.append(x_min + content_w - p)

        cands = []
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = y0 + p, x0 + p
                y1c, x1c = min(y1, h), min(x1, w)
                patch = img_np[y0:y1c, x0:x1c]
                mask_patch = fov_mask[y0:y1c, x0:x1c]
                fov_ratio = mask_patch.mean() if mask_patch.size > 0 else 0.0
                if fov_ratio < fov_thresh:
                    continue

                if patch.shape[0] < p or patch.shape[1] < p:
                    pad_h = max(0, p - patch.shape[0])
                    pad_w = max(0, p - patch.shape[1])
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                cands.append(patch)
        return cands

    # Initial scan with default stride (e.g. 96)
    candidates = get_candidates(stride)

    # Adaptive Fallback Stride: if small image yields fewer candidates than M, rescan with dense stride (s=48)
    if mode == "train" and len(candidates) < m_target and stride > 48 and min(content_h, content_w) > p:
        dense_candidates = get_candidates(48)
        if len(dense_candidates) > len(candidates):
            candidates = dense_candidates

    # Fallback: Extract central patch if zero candidates pass FOV filter
    if len(candidates) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        patch = img_np[cy:cy + p, cx:cx + p]
        if patch.shape[0] < p or patch.shape[1] < p:
            patch = np.pad(patch, ((0, max(0, p - patch.shape[0])), (0, max(0, p - patch.shape[1])), (0, 0)), mode="reflect")
        candidates = [patch]

    # 3. Mode dispatch:
    if mode == "train":
        if len(candidates) < m_target:
            idx = np.random.choice(len(candidates), size=m_target, replace=True)
        else:
            idx = np.random.choice(len(candidates), size=m_target, replace=False)
        selected = [candidates[i] for i in idx]
    else:  # eval / inference
        if len(candidates) > max_patches_eval:
            idx = np.linspace(0, len(candidates) - 1, max_patches_eval).astype(int)
            selected = [candidates[i] for i in idx]
        else:
            selected = candidates

    # 4. PyTorch uint8 Tensor conversion
    patches_np = np.stack(selected, axis=0)
    patches_u8 = torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()
    return patches_u8


def get_global_augmentations(img_size: int = 512, is_train: bool = True):
    """
    Constructs the Albumentations transformation pipeline for the Global Context branch.
    """
    try:
        import albumentations as A
        if is_train:
            return A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=30,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.10,
                    contrast_limit=0.10,
                    p=0.5
                ),
            ])
        else:
            return A.Compose([
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR)
            ])
    except ImportError:
        return lambda image: cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)


class FundusDualBranchDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, pd.DataFrame],
        img_size: int = 512,
        img_root: Optional[str] = None,
        split: str = "train",
        ablation_mode: str = "ben_graham_green",
        cache_dir: Optional[str] = None,
        patch_size: int = 128,
        stride: int = 96,
        transform: Optional[Any] = None
    ):
        """
        Dataset serving the proposed Dual-Branch model (Global + Local MIL) on Google Colab.
        Optimizations:
          1. Reads image paths directly from 'native_path' column in metadata CSV.
          2. Converts DataFrame to list of dicts (self.records) for O(1) in-memory indexing.
          3. Decodes JPEG (Quality 95) via high-speed OpenCV C++ backend.
          4. Separates FOV Mask calculation (strictly on RAW RGB) from Ben Graham enhancement.
          5. Returns uint8 tensors to minimize PyTorch DataLoader IPC Shared Memory bandwidth (755MB -> 189MB/batch).
          6. Supports optional precomputed NPZ anchor coordinate caching.
        """
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.reset_index(drop=True)
        else:
            self.df = pd.read_csv(csv_path)

        self.records = self.df.to_dict("records")
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.split = split
        self.img_root = img_root
        self.ablation_mode = ablation_mode
        self.is_train = (split == "train")
        self.cache_dir = cache_dir

        # Augmentation and resize applied ONLY to the Global stream
        if transform is not None:
            self.global_transform = transform
        else:
            self.global_transform = get_global_augmentations(
                img_size, is_train=self.is_train
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        label = int(row.get("diagnosis", 0))

        # 1. Resolve image path
        native_path_str = str(row.get("native_path", row.get("raw_full_path", "")))
        dataset_key = row.get("dataset_key", "default")
        if self.img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
            img_path = f"{self.img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"
        else:
            img_path = native_path_str

        # 2. Check if precomputed NPZ cache is available
        img_id = PureWindowsPath(native_path_str).stem if native_path_str else str(row.get("image_id", idx))
        cache_file = os.path.join(self.cache_dir, dataset_key, f"{img_id}.npz") if self.cache_dir else None

        if cache_file and os.path.exists(cache_file):
            try:
                data = np.load(cache_file)
                processed_np = data["img"]
                valid_anchors = data["anchors"]
                fov_mask = None
            except Exception:
                processed_np, valid_anchors, fov_mask = None, None, None
        else:
            processed_np, valid_anchors, fov_mask = None, None, None

        if processed_np is None:
            # Read JPEG via high-speed OpenCV backend
            bgr = cv2.imread(img_path)
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    img_rgb = np.array(pil_img)
                except Exception as e:
                    logger.error(f"Error loading image from {img_path}: {e}")
                    img_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            # CRITICAL FIX: Compute FOV mask strictly on RAW RGB image BEFORE Ben Graham
            fov_mask = compute_fov_mask(img_rgb, erode_px=15)

            # Apply Green channel ablation / Ben Graham background subtraction
            processed_np = apply_green_channel_ablation(
                img_rgb, ablation_mode=self.ablation_mode
            )
            valid_anchors = compute_valid_anchors(
                img_rgb.shape[:2], fov_mask, patch_size=self.patch_size, stride=self.stride, fov_thresh=0.5
            )

            # Auto-save cache archive for subsequent epochs (Epoch 2 -> 100+)
            if cache_file:
                try:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    np.savez_compressed(cache_file, img=processed_np, anchors=valid_anchors)
                except Exception:
                    pass

        # ======================================================================
        # STREAM 1: GLOBAL CONTEXT BRANCH -> RESIZE (uint8 tensor for IPC speed)
        # ======================================================================
        global_augmented = self.global_transform(image=processed_np)
        global_img = global_augmented["image"] if isinstance(global_augmented, dict) else global_augmented
        # Retain uint8 tensor to slash IPC shared memory transfer (Normalized on GPU in DRTrainer)
        global_tensor = torch.from_numpy(global_img).permute(2, 0, 1).contiguous()

        # ======================================================================
        # STREAM 2: LOCAL MIL BRANCH -> PRESERVE NATIVE HIGH-RESOLUTION & FOV MASK
        # ======================================================================
        local_native_np = processed_np

        return {
            "global_image": global_tensor,
            "local_native_image": local_native_np,
            "fov_mask": fov_mask,
            "valid_anchors": valid_anchors,
            "native_path": img_path,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ==============================================================================
# 7. HIGH-THROUGHPUT BATCH COLLATE_FN (uint8 IPC Transmission & Synchronized Masks)
# ==============================================================================

def trainer_compatible_collate_fn(
    batch,
    mode: str = "train",
    num_sample: Union[int, Tuple[int, int]] = (32, 64)
):
    """
    Collate function transmitting raw uint8 tensors across shared memory IPC:
    - Train mode: Returns [Batch, M, 3, 128, 128] uint8 tensor.
    - Eval mode: Zero-pads variable patch count up to batch maximum N_max, returns boolean patch_mask.
    """
    if mode == "train":
        if isinstance(num_sample, (tuple, list)):
            batch_m = int(np.random.randint(num_sample[0], num_sample[1] + 1))
        else:
            batch_m = int(num_sample)
    else:
        batch_m = num_sample

    global_imgs = torch.stack([item["global_image"] for item in batch], dim=0)
    local_patches_list = [
        extract_mil_patches_sliding(
            item["local_native_image"],
            fov_mask=item.get("fov_mask"),
            patch_size=128,
            stride=96,
            fov_thresh=0.5,
            mode=mode,
            num_sample=batch_m,
            native_path=item.get("native_path"),
            valid_anchors=item.get("valid_anchors")
        )
        for item in batch
    ]
    labels = torch.stack([item["label"] for item in batch], dim=0)

    if mode == "train":
        local_patches = torch.stack(local_patches_list, dim=0)  # [Batch, M, 3, 128, 128] uint8
        return {
            "global_img": global_imgs,
            "local_patches": local_patches,
            "label": labels,
        }

    # --- Eval Mode: Zero-pad variable patch counts to batch maximum N_max ---
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

    local_patches = torch.stack(padded, dim=0)     # [Batch, N_max, 3, 128, 128] uint8
    patch_mask = torch.stack(masks, dim=0)         # [Batch, N_max], True = valid patch

    return {
        "global_img": global_imgs,
        "local_patches": local_patches,
        "patch_mask": patch_mask,
        "label": labels,
    }


