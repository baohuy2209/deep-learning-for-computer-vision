import os
import shutil
import random
import pandas as pd
from pathlib import Path

# === CẤU HÌNH ===
SOURCE_DIR = "../data/raw/aptos/train_images/train_images/"
CSV_PATH = "../data/raw/aptos/train_1.csv"
OUTPUT_DIR = "../data/processed/aptos/sample_200"
N_SAMPLE = 200
SEED = 42

CLASS_NAMES = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR",
}

random.seed(SEED)

df = pd.read_csv(CSV_PATH)
for label, cls_name in CLASS_NAMES.items():
    subset = df[df["diagnosis"] == label]
    imgs = subset["id_code"].tolist()

    if len(imgs) < N_SAMPLE:
        print(f"⚠️  {cls_name}: chỉ có {len(imgs)} ảnh, lấy hết")
        sampled = imgs
    else:
        sampled = random.sample(imgs, N_SAMPLE)

    out_dir = Path(OUTPUT_DIR) / cls_name
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id in sampled:
        src = Path(SOURCE_DIR) / f"{img_id}.png"
        if src.exists():
            shutil.copy2(src, out_dir / src.name)
            count += 1
        else:
            print(f"  ❌ Không tìm thấy: {img_id}.png")

    print(f"✅ {cls_name}: {count} ảnh → {out_dir}")

print(f"\n🎉 Done! Xem tại: {OUTPUT_DIR}")
