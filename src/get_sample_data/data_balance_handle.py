import os
import shutil
import random
import csv
from pathlib import Path
from collections import defaultdict
# === CẤU HÌNH ===
SOURCE_DIR = "../../data/raw/zipEyepacs/train"  # sửa lại đường dẫn
CSV_PATH = "../../data/raw/zipEyepacs/trainLabels.csv/trainLabels.csv"
OUTPUT_DIR = "../../data/raw/zipEyepacs_train_images/full_data"
N_SAMPLE = 10000
IMAGE_NAME="image" # images || idcode 
LEVEL="level" # level || diagnosis
IMAGE_TYPE="jpeg" # png || jpeg 
SEED = 42
random.seed(SEED)
# Map số nhãn -> tên class chuẩn APTOS
LABEL_MAP = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR",
}
class_to_files = defaultdict(list)
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f) 
    for row in reader: 
        id_code = row[IMAGE_NAME].strip()
        diagnosis = int(row[LEVEL].strip())
        class_name = LABEL_MAP[diagnosis]
        img_path = Path(SOURCE_DIR)/f"{id_code}.{IMAGE_TYPE}"
        if img_path.exists():
            class_to_files[class_name].append(img_path)
        else: 
            print(f"⚠️  Không tìm thấy ảnh: {img_path}")
# Bước 2: Lấy N_SAMPLE ảnh cho mỗi class rồi copy 
for class_name, imgs in class_to_files.items():
    if len(imgs) < N_SAMPLE: 
        print(f"{class_name}: chỉ có {len(imgs)} ảnh, lấy hết")
        sampled = imgs 
    else:
        sampled = random.sample(imgs, N_SAMPLE)
    out_dir = Path(OUTPUT_DIR)/class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sampled: 
        shutil.copy(img_path, out_dir / img_path.name)
    print(f"✅ {class_name}: {len(sampled)} ảnh")
print(f"\n🎉 Done! Xem tại: {OUTPUT_DIR}")