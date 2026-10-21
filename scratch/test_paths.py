import os, pandas as pd

raw_dir = r"D:\Github\deep-learning-for-computer-vision\data\raw"

paths = {
    "APTOS": os.path.join(raw_dir, "aptos", "aptos_total.csv"),
    "DDR": os.path.join(raw_dir, "ddr", "DR_grading.csv"),
    "DeepDRiD": os.path.join(raw_dir, "deepdrid", "deepdrid_total.csv"),
    "IDRiD": os.path.join(raw_dir, "idrid", "idrid_total.csv"),
    "Messidor-2": os.path.join(raw_dir, "messidor2preprocess", "messidor_data.csv"),
    "EyePACS": os.path.join(raw_dir, "zipEyepacs", "trainLabels.csv", "trainLabels.csv"),
}

for k, p in paths.items():
    exists = os.path.exists(p)
    if exists:
        df = pd.read_csv(p)
        print(f"{k:12s}: FOUND ({len(df):6d} rows) | cols={df.columns.tolist()[:4]}")
    else:
        print(f"{k:12s}: NOT FOUND at {p}")
