import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

# === CẤU HÌNH CHO DEEPDRID ===
# Trỏ vào thư mục Images (nơi chứa các thư mục con 1, 2, 3...)
SOURCE_DIR = "../../data/raw/deepdrid/DeepDRiD-master/regular_fundus_images/regular-fundus-training/Images"
CSV_PATH = "../../data/raw/deepdrid/DeepDRiD-master/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv"
OUTPUT_DIR = "../../data/raw/samples/deepdrid/sample_200"

N_SAMPLE = 200
IMAGE_COL = "image_id"  # Cột chứa tên ảnh (vd: 1_l1)
PATIENT_COL = "patient_id"  # Cột chứa ID bệnh nhân để dò ra thư mục con
LABEL_COL = "patient_DR_Level"  # Có thể đổi thành left_eye_DR_Level nếu muốn nhãn chi tiết từng mắt
IMAGE_TYPE = "jpg"  # DeepDRiD dùng đuôi .jpg
SEED = 42

random.seed(SEED)

# Map số nhãn -> tên class chuẩn
LABEL_MAP = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR",
}

class_to_files = defaultdict(list)

# Bước 1: Đọc CSV và gom nhóm đường dẫn
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_id = row[IMAGE_COL].strip()
        patient_id = row[PATIENT_COL].strip()

        # Bỏ qua các dòng trống hoặc lỗi
        if not image_id or not row[LABEL_COL].strip():
            continue

        diagnosis = int(row[LABEL_COL].strip())
        class_name = LABEL_MAP[diagnosis]

        # Lắp ráp đường dẫn: SOURCE_DIR / patient_id / image_id.jpg
        img_path = Path(SOURCE_DIR) / patient_id / f"{image_id}.{IMAGE_TYPE}"

        if img_path.exists():
            class_to_files[class_name].append(img_path)
        else:
            print(f"⚠️  Không tìm thấy ảnh: {img_path}")

# Bước 2: Lấy N_SAMPLE ảnh cho mỗi class rồi copy
print(
    f"Bắt đầu trích xuất sample cho DeepDRiD (Mục tiêu: {N_SAMPLE} ảnh/lớp)...\n"
)

for class_name, imgs in class_to_files.items():
    if len(imgs) < N_SAMPLE:
        print(
            f"  - {class_name}: chỉ có {len(imgs)} ảnh (Không đủ {N_SAMPLE}), lấy hết toàn bộ."
        )
        sampled = imgs
    else:
        sampled = random.sample(imgs, N_SAMPLE)

    out_dir = Path(OUTPUT_DIR) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sampled:
        shutil.copy(img_path, out_dir / img_path.name)

    print(f"✅ Đã copy xong {class_name}: {len(sampled)} ảnh")

print(f"\n🎉 Done! Xem sample tại: {OUTPUT_DIR}")
