import os, re, sys
import pandas as pd

raw_dir = r"D:\Github\deep-learning-for-computer-vision\data\raw"
native_dir = r"D:\Github\deep-learning-for-computer-vision\data\raw\hybrid\native_cropped"

DATASET_CONFIG = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": os.path.join("aptos", "aptos_total.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": os.path.join("ddr", "DR_grading.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": os.path.join("deepdrid", "deepdrid_total.csv"),
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "id_suffix": ".jpg"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": os.path.join("idrid", "idrid_total.csv"),
        "id_col": "image_id",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": os.path.join("messidor2preprocess", "messidor_data.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "EyePACS": {
        "name": "EyePACS",
        "csv_path": os.path.join("zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
        "id_col": "image",
        "label_col": "level",
        "id_suffix": ".jpeg"
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

dfs = []
for key, cfg in DATASET_CONFIG.items():
    csv_p = os.path.join(raw_dir, cfg["csv_path"])
    raw_df = pd.read_csv(csv_p)
    df = raw_df[[cfg["id_col"], cfg["label_col"]]].copy()
    df.columns = ["image_id", "diagnosis"]
    df["dataset_name"] = cfg["name"]
    df["dataset_key"] = key
    
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["diagnosis"].isin([0, 1, 2, 3, 4])]
    
    # Exact native_path formula from merge-dataset.ipynb
    df["filename"] = df["image_id"].apply(lambda x: f"{re.sub(r'[^\w\.-]', '_', str(x))}.jpg")
    df["relative_path"] = df["dataset_key"] + "/" + df["filename"]
    
    df["exists"] = df["relative_path"].apply(lambda rp: os.path.exists(os.path.join(native_dir, rp)))
    existing_df = df[df["exists"]].copy().drop(columns=["exists"])
    
    parsed = existing_df["image_id"].apply(lambda x: parse_patient_and_side(x, key))
    existing_df["patient_id"] = [p[0] for p in parsed]
    existing_df["side"] = [p[1] for p in parsed]
    
    dfs.append(existing_df)
    print(f"Loaded {key:10s}: {len(existing_df):5d} existing images (out of {len(df)} rows)")

master_df = pd.concat(dfs, ignore_index=True)
print(f"\n[TOTAL] master_df has {len(master_df):,} images across {master_df['dataset_key'].nunique()} cohorts!")
print(master_df["dataset_key"].value_counts())
