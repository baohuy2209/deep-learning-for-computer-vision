import os, pandas as pd

h_dir = r"D:\Github\deep-learning-for-computer-vision\data\raw\hybrid"

train_raw = pd.read_csv(os.path.join(h_dir, "train_raw_metadata.csv")) if os.path.exists(os.path.join(h_dir, "train_raw_metadata.csv")) else None
val_raw = pd.read_csv(os.path.join(h_dir, "val_raw_metadata.csv")) if os.path.exists(os.path.join(h_dir, "val_raw_metadata.csv")) else None
test_raw = pd.read_csv(os.path.join(h_dir, "test_raw_metadata.csv")) if os.path.exists(os.path.join(h_dir, "test_raw_metadata.csv")) else None

print("train_raw:", len(train_raw) if train_raw is not None else "None")
print("val_raw  :", len(val_raw) if val_raw is not None else "None")
print("test_raw :", len(test_raw) if test_raw is not None else "None")

if train_raw is not None and val_raw is not None and test_raw is not None:
    full_df = pd.concat([train_raw, val_raw, test_raw], ignore_index=True)
    print("Total reconstructed master_df:", len(full_df))
    print(full_df["dataset_key"].value_counts())
    print("Columns:", full_df.columns.tolist())
    print("Sample native_path:", full_df["native_path"].iloc[0])
