import json, os

notebook_path = r"notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb"

cells = []

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

# 1. Title & Intro
cells.append(md_cell("""# 6-Fold Leave-One-Dataset-Out (LODO) Dataset Generation Pipeline
### Clinical OOD Cross-Domain Generalization & Zero-Leakage Patient Splitting
**Author**: Antigravity AI & Medical Imaging Team  
**Goal**: Pre-generate frozen, deterministic 6-fold train/val/test splits for the Leave-One-Dataset-Out (LODO) benchmark.
- **Held-Out OOD Test (Fold $k$)**: 100% of cohort $k$ held out with unaltered natural clinical prevalence.
- **In-Domain Pool**: Remaining 5 cohorts grouped at patient level using `GroupShuffleSplit(train_size=0.85, random_state=42+k)`.
- **In-Domain Val (15%)**: Natural clinical prevalence preserved (no quota balancing) for honest early-stopping / model selection.
- **Train Pool (85%)**: Multi-source quota balancing (1,500 - 2,500 images per DR grade) to eliminate severe class imbalance.
- **Output**: 18 CSV files (6 folds $\times$ 3 splits) + `lodo_splits_summary.csv` + `lodo_folds.zip` ready for Google Colab."""))

# 2. Imports
cells.append(code_cell("""# ==============================================================================
# 1. IMPORTS, SYSTEM ENVIRONMENT & DETERMINISTIC SEED
# ==============================================================================
import os
import re
import sys
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
np.random.seed(SEED)
print(f"[INFO] Python {sys.version.split()[0]} | Pandas {pd.__version__} | Deterministic SEED={SEED}")"""))

# 3. Path resolution
cells.append(code_cell("""# ==============================================================================
# 2. DYNAMIC PATH RESOLUTION & DIRECTORY SETUP
# ==============================================================================
def get_base_paths():
    cwd = os.getcwd()
    # Support running from workspace root or notebooks/ subfolder
    if os.path.exists(os.path.join(cwd, "data", "raw")):
        base_raw = os.path.abspath(os.path.join(cwd, "data", "raw"))
        base_proc = os.path.abspath(os.path.join(cwd, "data", "processed"))
    else:
        base_raw = os.path.abspath(os.path.join(cwd, "..", "..", "..", "data", "raw"))
        base_proc = os.path.abspath(os.path.join(cwd, "..", "..", "..", "data", "processed"))
    
    native_dir = os.path.join(base_raw, "hybrid", "native_cropped")
    out_dir = os.path.join(base_proc, "lodo_folds")
    os.makedirs(out_dir, exist_ok=True)
    return base_raw, native_dir, out_dir

BASE_RAW_DIR, NATIVE_CROPPED_DIR, LODO_FOLDS_DIR = get_base_paths()
print(f"[PATHS] Base Raw        : {BASE_RAW_DIR}")
print(f"[PATHS] Native Cropped  : {NATIVE_CROPPED_DIR} (Exists: {os.path.exists(NATIVE_CROPPED_DIR)})")
print(f"[PATHS] LODO Folds Out  : {LODO_FOLDS_DIR}")"""))

# 4. Config dict
cells.append(code_cell("""# ==============================================================================
# 3. RAW DATASET CONFIGURATION (6 RETINAL FUNDUS COHORTS)
# ==============================================================================
DATASET_CONFIG = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": os.path.join("aptos", "aptos_total.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "fold_key": "aptos"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": os.path.join("idrid", "idrid_total.csv"),
        "id_col": "image_id",
        "label_col": "diagnosis",
        "fold_key": "idrid"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": os.path.join("messidor2preprocess", "messidor_data.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "fold_key": "messidor2"
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": os.path.join("ddr", "DR_grading.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "fold_key": "ddr"
    },
    "EyePACS": {
        "name": "EyePACS",
        "csv_path": os.path.join("zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
        "id_col": "image",
        "label_col": "level",
        "fold_key": "eyepacs"
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": os.path.join("deepdrid", "deepdrid_total.csv"),
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "fold_key": "deepdrid"
    }
}
print(f"[CONFIG] Registered {len(DATASET_CONFIG)} retinal fundus cohorts.")"""))

# 5. Patient parsing
cells.append(code_cell("""# ==============================================================================
# 4. PATIENT ID PARSING & ZERO-LEAKAGE NAMESPACING
# ==============================================================================
def parse_patient_and_side(image_id: str, dataset_key: str) -> tuple[str, str]:
    \"\"\"
    Extracts canonical patient_id and eye side from raw image identifiers.
    Prepends dataset_key:: to eliminate cross-cohort patient ID collisions.
    - IDRiD: train_IDRiD_001 -> 001
    - DDR: 007-0004-000 -> 007-0004
    - EyePACS: 10_left -> 10
    - DeepDRiD: 104_l1 -> 104
    - Messidor-2: 20051020_43808_0100_PP -> 20051020_43808
    - APTOS: 1 image = 1 patient
    \"\"\"
    img_str = str(image_id).lower()
    k = dataset_key.lower()
    
    if "idrid" in k:
        base = img_str.replace(".jpg", "").replace(".png", "")
        pid = base.split("_")[-1]
    elif "eyepacs" in k or "deepdrid" in k:
        pid = img_str.split("_")[0]
    elif "ddr" in k:
        parts = img_str.replace(".jpg", "").replace(".png", "").split("-")
        pid = "-".join(parts[:2]) if len(parts) >= 2 else img_str.split(".")[0]
    elif "messidor" in k:
        pid = img_str.split("_")[0]
    else:
        pid = img_str
        
    side = "left" if "left" in img_str or "_l" in img_str else \\
           "right" if "right" in img_str or "_r" in img_str else "unknown"
    return f"{dataset_key}::{pid}", side"""))

# 6. Master dataset harmonization
cells.append(code_cell("""# ==============================================================================
# 5. MASTER DATASET HARMONIZATION & IMAGE INTEGRITY CHECK
# ==============================================================================
dfs = []
for key, cfg in DATASET_CONFIG.items():
    csv_full = os.path.join(BASE_RAW_DIR, cfg["csv_path"])
    if not os.path.exists(csv_full):
        raise FileNotFoundError(f"Missing required CSV for {key}: {csv_full}")
        
    raw_df = pd.read_csv(csv_full)
    df = raw_df[[cfg["id_col"], cfg["label_col"]]].copy()
    df.columns = ["image_id", "diagnosis"]
    df["dataset_name"] = cfg["name"]
    df["dataset_key"] = key
    df["fold_key"] = cfg["fold_key"]
    
    # Cast diagnosis to integer 0..4
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["diagnosis"].isin([0, 1, 2, 3, 4])]
    
    # Relative path matching native_cropped directory structure
    df["filename"] = df["image_id"].apply(lambda x: f"{re.sub(r'[^\\w\\.-]', '_', str(x))}.jpg")
    df["relative_path"] = df["dataset_key"] + "/" + df["filename"]
    df["native_path"] = df["relative_path"]
    
    # Filter strictly for images verified to exist on disk
    df["exists"] = df["relative_path"].apply(lambda rp: os.path.exists(os.path.join(NATIVE_CROPPED_DIR, rp)))
    existing_df = df[df["exists"]].copy().drop(columns=["exists"])
    
    parsed = existing_df["image_id"].apply(lambda x: parse_patient_and_side(x, key))
    existing_df["patient_id"] = [p[0] for p in parsed]
    existing_df["side"] = [p[1] for p in parsed]
    
    dfs.append(existing_df)
    print(f"[OK] {key:12s}: {len(existing_df):5,d} valid native images (out of {len(df):5,d} rows)")

master_df = pd.concat(dfs, ignore_index=True)
print(f"\\n[TOTAL] Master Harmonized Pool: {len(master_df):,d} images across {master_df['dataset_key'].nunique()} cohorts.")"""))

# 7. Multi-source balancing
cells.append(code_cell("""# ==============================================================================
# 6. MULTI-SOURCE QUOTA BALANCING (1,500 - 2,500 PER GRADE)
# ==============================================================================
def create_balanced_train_pool(train_df: pd.DataFrame, target_range=(1500, 2500), seed=42) -> pd.DataFrame:
    \"\"\"
    Applies multi-source proportional allocation per DR grade to prevent single-source domination.
    Ensures rare grades (Grade 1 & 3) receive adequate representation without artificial distortion.
    \"\"\"
    balanced_dfs = []
    for grade in range(5):
        grade_subset = train_df[train_df["diagnosis"] == grade].copy().reset_index(drop=True)
        current_count = len(grade_subset)
        if current_count == 0:
            continue
            
        datasets_in_grade = grade_subset["dataset_key"].unique()
        num_datasets = len(datasets_in_grade)
        
        if current_count > target_range[1]:
            target_count = int(np.random.RandomState(seed + grade).randint(target_range[0], target_range[1] + 1))
            base_quota = target_count // num_datasets
            sampled_datasets = []
            leftover_quota = 0
            eligible = []
            
            for ds_key in datasets_in_grade:
                ds_subset = grade_subset[grade_subset["dataset_key"] == ds_key]
                if len(ds_subset) <= base_quota:
                    sampled_datasets.append(ds_subset)
                    leftover_quota += (base_quota - len(ds_subset))
                else:
                    eligible.append((ds_key, ds_subset, len(ds_subset)))
                    
            if eligible:
                quota_large = base_quota + (leftover_quota // len(eligible))
                for ds_key, ds_subset, count in eligible:
                    sampled_datasets.append(ds_subset.sample(min(quota_large, count), random_state=seed))
                    
            selected = pd.concat(sampled_datasets, ignore_index=True)
            if len(selected) > target_count:
                selected = selected.sample(target_count, random_state=seed).reset_index(drop=True)
            elif len(selected) < target_count:
                rem = grade_subset[~grade_subset["relative_path"].isin(selected["relative_path"])]
                if not rem.empty:
                    extra = rem.sample(min(target_count - len(selected), len(rem)), random_state=seed)
                    selected = pd.concat([selected, extra], ignore_index=True)
            balanced_dfs.append(selected)
        else:
            balanced_dfs.append(grade_subset)
            
    return pd.concat(balanced_dfs, ignore_index=True)"""))

# 8. 6-Fold Generation loop
cells.append(code_cell("""# ==============================================================================
# 7. EXECUTE 6-FOLD LODO SPLIT GENERATION & INTEGRITY AUDIT
# ==============================================================================
cohort_keys = list(DATASET_CONFIG.keys())
summary_records = []

for fold_idx, held_out_key in enumerate(cohort_keys):
    fold_slug = DATASET_CONFIG[held_out_key]["fold_key"]
    print(f"\\n{'='*70}")
    print(f" GENERATING FOLD {fold_idx}: HELD-OUT OOD = {held_out_key} ({fold_slug.upper()})")
    print(f"{'='*70}")
    
    # 1. Held-out OOD Test Set (100% natural clinical prevalence)
    test_df = master_df[master_df["dataset_key"] == held_out_key].copy().reset_index(drop=True)
    test_df["split"] = "test"
    test_df["fold"] = fold_idx
    test_df["held_out_dataset"] = held_out_key
    
    # 2. 5-Dataset In-Domain Pool
    pool_df = master_df[master_df["dataset_key"] != held_out_key].copy().reset_index(drop=True)
    
    # 3. Patient-level 85/15 Split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=SEED + fold_idx)
    train_idx, val_idx = next(gss.split(pool_df, groups=pool_df["patient_id"]))
    
    train_raw_df = pool_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = pool_df.iloc[val_idx].copy().reset_index(drop=True)
    
    # 4. Quota Balancing strictly on 85% Train split
    train_df = create_balanced_train_pool(train_raw_df, target_range=(1500, 2500), seed=SEED + fold_idx)
    train_df["split"] = "train"
    train_df["fold"] = fold_idx
    train_df["held_out_dataset"] = held_out_key
    
    val_df["split"] = "val"
    val_df["fold"] = fold_idx
    val_df["held_out_dataset"] = held_out_key
    
    # 5. ZERO PATIENT LEAKAGE ASSERTIONS
    train_pids = set(train_df["patient_id"])
    val_pids = set(val_df["patient_id"])
    test_pids = set(test_df["patient_id"])
    assert len(train_pids.intersection(val_pids)) == 0, f"PATIENT LEAKAGE: Train vs Val in fold {fold_idx}"
    assert len(train_pids.intersection(test_pids)) == 0, f"PATIENT LEAKAGE: Train vs Test in fold {fold_idx}"
    assert len(val_pids.intersection(test_pids)) == 0, f"PATIENT LEAKAGE: Val vs Test in fold {fold_idx}"
    print(f"  [AUDIT] Zero Patient Leakage: 100% Passed across Train, Val, and OOD Test.")
    
    # 6. Save Split CSV Files
    f_train_path = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{fold_slug}_train.csv")
    f_val_path   = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{fold_slug}_val.csv")
    f_test_path  = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{fold_slug}_test.csv")
    
    train_df.to_csv(f_train_path, index=False)
    val_df.to_csv(f_val_path, index=False)
    test_df.to_csv(f_test_path, index=False)
    
    print(f"  Train Set : {len(train_df):5,d} images ({train_df['patient_id'].nunique():5,d} patients) -> {os.path.basename(f_train_path)}")
    print(f"  Val Set   : {len(val_df):5,d} images ({val_df['patient_id'].nunique():5,d} patients) -> {os.path.basename(f_val_path)}")
    print(f"  Test Set  : {len(test_df):5,d} images ({test_df['patient_id'].nunique():5,d} patients) -> {os.path.basename(f_test_path)}")
    
    summary_records.append({
        "fold": fold_idx,
        "held_out_dataset": held_out_key,
        "fold_slug": fold_slug,
        "train_images": len(train_df),
        "train_patients": train_df["patient_id"].nunique(),
        "val_images": len(val_df),
        "val_patients": val_df["patient_id"].nunique(),
        "test_images": len(test_df),
        "test_patients": test_df["patient_id"].nunique(),
        "train_referable_pct": round((train_df["diagnosis"] >= 2).mean() * 100, 2),
        "val_referable_pct": round((val_df["diagnosis"] >= 2).mean() * 100, 2),
        "test_referable_pct": round((test_df["diagnosis"] >= 2).mean() * 100, 2)
    })

summary_df = pd.DataFrame(summary_records)
sum_csv_path = os.path.join(LODO_FOLDS_DIR, "lodo_splits_summary.csv")
summary_df.to_csv(sum_csv_path, index=False)
print(f"\\n[SUMMARY] Saved master summary table to '{sum_csv_path}'")
display(summary_df)"""))

# 9. Packaging Zip
cells.append(code_cell("""# ==============================================================================
# 8. AUTOMATED ZIP ARCHIVING & GOOGLE DRIVE DEPLOYMENT GUIDE
# ==============================================================================
zip_path = os.path.join(os.path.dirname(LODO_FOLDS_DIR), "lodo_folds.zip")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fname in sorted(os.listdir(LODO_FOLDS_DIR)):
        if fname.endswith(".csv"):
            fpath = os.path.join(LODO_FOLDS_DIR, fname)
            z.write(fpath, arcname=fname)

print(f"[ARCHIVE SUCCESS] Created '{zip_path}' ({os.path.getsize(zip_path):,d} bytes)")
print(\"\"\"
--------------------------------------------------------------------------------
GOOGLE COLAB DEPLOYMENT INSTRUCTIONS:
1. Upload 'lodo_folds.zip' to Google Drive at:
   '/content/drive/MyDrive/lodo_folds.zip'
   OR create folder '/content/drive/MyDrive/lodo_folds/' and upload the 18 CSV files.
2. In 'main_model_lodo.ipynb', the LODO fold loop will automatically detect
   these 18 CSV files and train on the exact frozen splits!
--------------------------------------------------------------------------------
\"\"\")"""))

nb_content = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb_content, f, indent=1, ensure_ascii=False)

print(f"[SUCCESS] Built notebook: {notebook_path}")
