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
        "id_suffix": ".jpg",
        "folder_name": "APTOS"
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": os.path.join(raw_dir, "ddr", "DR_grading.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg",
        "folder_name": "DDR"
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": os.path.join(raw_dir, "deepdrid", "deepdrid_total.csv"),
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "id_suffix": ".jpg",
        "folder_name": "DeepDRiD"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": os.path.join(raw_dir, "idrid", "idrid_total.csv"),
        "id_col": "image_id",
        "label_col": "diagnosis",
        "id_suffix": ".jpg",
        "folder_name": "IDRiD"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": os.path.join(raw_dir, "messidor2preprocess", "messidor_data.csv"),
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg",
        "folder_name": "Messidor-2"
    },
    "EyePACS": {
        "name": "EyePACS",
        "csv_path": os.path.join(raw_dir, "zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
        "id_col": "image",
        "label_col": "level",
        "id_suffix": ".jpg",
        "folder_name": "EyePACS"
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
    if not os.path.exists(cfg["csv_path"]):
        print(f"Warning: {key} CSV missing")
        continue
    df = pd.read_csv(cfg["csv_path"])
    label_col = cfg["label_col"]
    id_col = cfg["id_col"]
    df = df[[id_col, label_col]].copy()
    df.columns = ["image_id", "diagnosis"]
    df["dataset_name"] = cfg["name"]
    df["dataset_key"] = key
    df["folder_name"] = cfg["folder_name"]
    
    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df[df["diagnosis"].isin([0, 1, 2, 3, 4])]
    
    # Check image existence in native_cropped
    folder = cfg["folder_name"]
    # standardize filename
    df["filename"] = df["image_id"].apply(lambda x: re.sub(r'[^\w\.-]', '_', str(x)) + ".jpg" if not str(x).endswith(".jpg") else re.sub(r'[^\w\.-]', '_', str(x)))
    df["relative_path"] = df["folder_name"] + "/" + df["filename"]
    df["native_path"] = df["relative_path"]
    
    # Filter only images that exist in native_cropped
    df["exists"] = df["relative_path"].apply(lambda rp: os.path.exists(os.path.join(native_dir, rp)))
    existing_df = df[df["exists"]].copy().drop(columns=["exists"])
    
    parsed = existing_df["image_id"].apply(lambda x: parse_patient_and_side(x, key))
    existing_df["patient_id"] = [p[0] for p in parsed]
    existing_df["side"] = [p[1] for p in parsed]
    
    dfs.append(existing_df)
    print(f"Loaded {key:10s}: {len(existing_df):5d} existing images (out of {len(df)} metadata rows)")

master_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal master_df: {len(master_df)} valid native_cropped images across {master_df['dataset_key'].nunique()} cohorts.")
