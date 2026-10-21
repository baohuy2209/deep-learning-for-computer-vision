import os, sys, re, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

raw_dir = r"D:\Github\deep-learning-for-computer-vision\data\raw"
native_dir = r"D:\Github\deep-learning-for-computer-vision\data\raw\hybrid\native_cropped"
out_dir = r"D:\Github\deep-learning-for-computer-vision\data\processed\lodo_folds"
os.makedirs(out_dir, exist_ok=True)

DATASET_CONFIG = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": os.path.join(raw_dir, "aptos", "aptos_total.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "fold_key": "aptos"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": os.path.join(raw_dir, "idrid", "idrid_total.csv"),
        "id_col": "image_id",
        "label_col": "diagnosis",
        "fold_key": "idrid"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": os.path.join(raw_dir, "messidor2preprocess", "messidor_data.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "fold_key": "messidor2"
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": os.path.join(raw_dir, "ddr", "DR_grading.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "fold_key": "ddr"
    },
    "EyePACS": {
        "name": "EyePACS",
        "csv_path": os.path.join(raw_dir, "zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
        "id_col": "image",
        "label_col": "level",
        "fold_key": "eyepacs"
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": os.path.join(raw_dir, "deepdrid", "deepdrid_total.csv"),
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "fold_key": "deepdrid"
    }
}

def parse_patient_and_side(image_id: str, dataset_key: str):
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
    side = "left" if "left" in img_str or "_l" in img_str else "right" if "right" in img_str or "_r" in img_str else "unknown"
    return f"{dataset_key}::{pid}", side

# Load and standardize master dataset
dfs = []
for key, cfg in DATASET_CONFIG.items():
    raw_df = pd.read_csv(cfg["csv_path"])
    df = raw_df[[cfg["id_col"], cfg["label_col"]]].copy()
    df.columns = ["image_id", "diagnosis"]
    df["dataset_name"] = cfg["name"]
    df["dataset_key"] = key
    df["fold_key"] = cfg["fold_key"]
    
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["diagnosis"].isin([0, 1, 2, 3, 4])]
    
    df["filename"] = df["image_id"].apply(lambda x: f"{re.sub(r'[^\w\.-]', '_', str(x))}.jpg")
    df["relative_path"] = df["dataset_key"] + "/" + df["filename"]
    df["native_path"] = df["relative_path"]
    
    df["exists"] = df["relative_path"].apply(lambda rp: os.path.exists(os.path.join(native_dir, rp)))
    existing_df = df[df["exists"]].copy().drop(columns=["exists"])
    
    parsed = existing_df["image_id"].apply(lambda x: parse_patient_and_side(x, key))
    existing_df["patient_id"] = [p[0] for p in parsed]
    existing_df["side"] = [p[1] for p in parsed]
    
    dfs.append(existing_df)

master_df = pd.concat(dfs, ignore_index=True)
print(f"Master DF loaded: {len(master_df)} images across {master_df['dataset_key'].nunique()} cohorts.")

def create_balanced_train_pool(train_df: pd.DataFrame, target_range=(1500, 2500), seed=42) -> pd.DataFrame:
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
            
    return pd.concat(balanced_dfs, ignore_index=True)

# LODO 6-Fold Generation
cohort_keys = list(DATASET_CONFIG.keys())
summary_records = []

for fold_idx, held_out_key in enumerate(cohort_keys):
    fold_slug = DATASET_CONFIG[held_out_key]["fold_key"]
    print(f"\n--- Generating Fold {fold_idx}: Held-out OOD = {held_out_key} ({fold_slug}) ---")
    
    # 1. Test set = 100% held-out dataset
    test_df = master_df[master_df["dataset_key"] == held_out_key].copy().reset_index(drop=True)
    test_df["split"] = "test"
    test_df["fold"] = fold_idx
    test_df["held_out_dataset"] = held_out_key
    
    # 2. Train/Val pool = remaining 5 datasets
    pool_df = master_df[master_df["dataset_key"] != held_out_key].copy().reset_index(drop=True)
    
    # 3. Patient-level 85/15 split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=42 + fold_idx)
    train_idx, val_idx = next(gss.split(pool_df, groups=pool_df["patient_id"]))
    
    train_raw_df = pool_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = pool_df.iloc[val_idx].copy().reset_index(drop=True)
    
    # 4. Quota balancing strictly on Train split
    train_df = create_balanced_train_pool(train_raw_df, target_range=(1500, 2500), seed=42 + fold_idx)
    train_df["split"] = "train"
    train_df["fold"] = fold_idx
    train_df["held_out_dataset"] = held_out_key
    
    val_df["split"] = "val"
    val_df["fold"] = fold_idx
    val_df["held_out_dataset"] = held_out_key
    
    # 5. Sanity Checks: Zero Patient Leakage
    train_pids = set(train_df["patient_id"])
    val_pids = set(val_df["patient_id"])
    test_pids = set(test_df["patient_id"])
    assert len(train_pids.intersection(val_pids)) == 0, f"LEAKAGE train/val in fold {fold_idx}"
    assert len(train_pids.intersection(test_pids)) == 0, f"LEAKAGE train/test in fold {fold_idx}"
    assert len(val_pids.intersection(test_pids)) == 0, f"LEAKAGE val/test in fold {fold_idx}"
    
    # 6. Save CSVs
    f_train_path = os.path.join(out_dir, f"fold_{fold_idx}_{fold_slug}_train.csv")
    f_val_path = os.path.join(out_dir, f"fold_{fold_idx}_{fold_slug}_val.csv")
    f_test_path = os.path.join(out_dir, f"fold_{fold_idx}_{fold_slug}_test.csv")
    
    train_df.to_csv(f_train_path, index=False)
    val_df.to_csv(f_val_path, index=False)
    test_df.to_csv(f_test_path, index=False)
    
    print(f"  Train: {len(train_df):5d} images ({train_df['patient_id'].nunique():5d} pts) | Grade dist: {train_df['diagnosis'].value_counts().to_dict()}")
    print(f"  Val  : {len(val_df):5d} images ({val_df['patient_id'].nunique():5d} pts) | Grade dist: {val_df['diagnosis'].value_counts().to_dict()}")
    print(f"  Test : {len(test_df):5d} images ({test_df['patient_id'].nunique():5d} pts) | Grade dist: {test_df['diagnosis'].value_counts().to_dict()}")
    print(f"  [OK] Zero-leakage verified for fold {fold_idx}")
    
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
        "test_referable_pct": round((test_df["diagnosis"] >= 2).mean() * 100, 2),
        "val_referable_pct": round((val_df["diagnosis"] >= 2).mean() * 100, 2),
        "train_referable_pct": round((train_df["diagnosis"] >= 2).mean() * 100, 2)
    })

sum_df = pd.DataFrame(summary_records)
sum_csv_path = os.path.join(out_dir, "lodo_splits_summary.csv")
sum_df.to_csv(sum_csv_path, index=False)
print("\n=== LODO 6-FOLD SPLITS SUMMARY ===")
print(sum_df.to_string(index=False))
