import os
from pathlib import Path

import pandas as pd

# === 1. CẤU HÌNH ĐƯỜNG DẪN ===
BASE_RAW_DIR = Path("../../data/raw/hybrid")
BASE_SAMPLE_DIR = Path("../../data/raw/samples/hybrid")

# File metadata gốc của tập Train
CSV_TRAIN_SOURCE = BASE_RAW_DIR / "val_metadata.csv"
if not CSV_TRAIN_SOURCE.exists():
    CSV_TRAIN_SOURCE = BASE_RAW_DIR / "val_balanced_metadata.csv"

# Đường dẫn file CSV sample đầu ra
OUTPUT_CSV_PATH = BASE_SAMPLE_DIR / "val_sample_50_metadata.csv"
OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

N_SAMPLE_PER_CLASS = 50  # 50 ảnh/lớp x 5 lớp = 250 dòng
SEED = 42

# === 2. ĐỌC DỮ LIỆU & LẤY MẪU THEO CLASS ===
print(f"⏳ Đang đọc dữ liệu từ: {CSV_TRAIN_SOURCE.resolve()}...")
df_train = pd.read_csv(CSV_TRAIN_SOURCE)
df_train["diagnosis"] = df_train["diagnosis"].astype(int)

# Kiểm tra sự tồn tại của file ảnh theo native_path
df_train["_exists"] = df_train["native_path"].apply(
    lambda p: os.path.exists(str(p))
)
df_valid = df_train[df_train["_exists"]].drop(columns=["_exists"])

# Lấy mẫu ngẫu nhiên cho từng lớp
sampled_dfs = []
for grade in sorted(df_valid["diagnosis"].unique()):
    subset = df_valid[df_valid["diagnosis"] == grade]
    n_take = min(len(subset), N_SAMPLE_PER_CLASS)
    sampled = subset.sample(n_take, random_state=SEED)
    sampled_dfs.append(sampled)
    print(f"  • Class {grade}: Lấy {len(sampled)} / {len(subset)} mẫu.")

df_sample = (
    pd.concat(sampled_dfs, ignore_index=True)
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)

# === 3. XUẤT FILE CSV METADATA SAMPLE ===
df_sample.to_csv(OUTPUT_CSV_PATH, index=False)

print("\n" + "=" * 60)
print("✅ TẠO SAMPLE METADATA THÀNH CÔNG (KHÔNG COPY FILE ẢNH)!")
print(f"📄 File CSV: {OUTPUT_CSV_PATH.resolve()}")
print(f"📊 Tổng số mẫu: {len(df_sample)} dòng.")
print(df_sample["diagnosis"].value_counts().sort_index())
print("=" * 60)
