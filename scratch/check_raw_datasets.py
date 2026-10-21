import os, re, sys
import pandas as pd

raw_dir = r"D:\Github\deep-learning-for-computer-vision\data\raw"

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
        "img_dir": os.path.join("messidor2preprocess", "messidor_total_images"),
        "id_col": "image_id",
        "label_col": "adjudicated_dr_grade",
        "id_suffix": ".jpg"
    },
    "EyePACS": {
        "name": "EyePACS",
        "csv_path": os.path.join("zipEyepacs", "eyepacs_total.csv"),
        "img_dir": os.path.join("zipEyepacs_train_images", "train"),
        "id_col": "image",
        "label_col": "level",
        "id_suffix": ".jpeg"
    }
}

for k, cfg in DATASET_CONFIG.items():
    csv_p = os.path.join(raw_dir, cfg["csv_path"])
    img_d = os.path.join(raw_dir, cfg["img_dir"])
    csv_exists = os.path.exists(csv_p)
    img_exists = os.path.exists(img_d)
    count = len(pd.read_csv(csv_p)) if csv_exists else 0
    print(f"{k:12s}: CSV={csv_exists} ({count:5d} rows), ImgDir={img_exists}")
