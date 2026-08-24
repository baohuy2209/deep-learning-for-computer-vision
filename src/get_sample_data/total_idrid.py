import os
import shutil
from pathlib import Path

import pandas as pd

# === CẤU HÌNH ĐƯỜNG DẪN ĐÚNG THEO CÂY THƯ MỤC ===
BASE_IDRID_DIR = Path("../../data/raw/idrid/Disease Grading")

TRAIN_IMG_SRC = BASE_IDRID_DIR / "1. Original Images/a. Training Set"
TEST_IMG_SRC = BASE_IDRID_DIR / "1. Original Images/b. Testing Set"

TRAIN_CSV_SRC = (
    BASE_IDRID_DIR
    / "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
)
TEST_CSV_SRC = (
    BASE_IDRID_DIR
    / "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
)

OUTPUT_DIR = Path("../../data/raw/idrid")
OUTPUT_IMG_DIR = OUTPUT_DIR / "idrid_total_images"
OUTPUT_CSV = OUTPUT_DIR / "idrid_total.csv"

OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. ĐỌC VÀ GỘP 2 FILE CSV NHÃN ---
train_df = pd.read_csv(TRAIN_CSV_SRC)
test_df = pd.read_csv(TEST_CSV_SRC)

# Tìm tên cột linh hoạt
label_col_candidates = [
    c
    for c in train_df.columns
    if "grade" in c.lower() or "diagnosis" in c.lower() or "level" in c.lower()
]
true_label_col = (
    label_col_candidates[0] if label_col_candidates else "Retinopathy grade"
)

id_col_candidates = [
    c for c in train_df.columns if "image" in c.lower() or "id" in c.lower()
]
true_id_col = id_col_candidates[0] if id_col_candidates else "Image name"

train_df = train_df[[true_id_col, true_label_col]].copy()
train_df.columns = ["image_id", "diagnosis"]
train_df["subset"] = "train"

test_df = test_df[[true_id_col, true_label_col]].copy()
test_df.columns = ["image_id", "diagnosis"]
test_df["subset"] = "test"

idrid_total_df = pd.concat([train_df, test_df], ignore_index=True)

# --- 2. XỬ LÝ ẢNH VÀ GỘP THẲNG SUBSET VÀO IMAGE_ID ---
new_image_ids = []
new_image_paths = []

for _, row in idrid_total_df.iterrows():
    orig_id = (
        str(row["image_id"]).strip().replace(".jpg", "").replace(".png", "")
    )
    subset = row["subset"]  # 'train' hoặc 'test'

    # Biến image_id thành dạng 'train_IDRiD_001' hoặc 'test_IDRiD_001' để phân biệt vĩnh viễn
    new_id = f"{subset}_{orig_id}"
    orig_filename = f"{orig_id}.jpg"
    new_filename = f"{new_id}.jpg"

    src_folder = TRAIN_IMG_SRC if subset == "train" else TEST_IMG_SRC
    src_path = src_folder / orig_filename
    dest_path = OUTPUT_IMG_DIR / new_filename

    if src_path.exists():
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)
        new_image_ids.append(new_id)
        new_image_paths.append(f"idrid_total_images/{new_filename}")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy ảnh tại {src_path}")
        new_image_ids.append(None)
        new_image_paths.append(None)

# Cập nhật lại DataFrame chuẩn chỉ gồm các cột cần thiết
final_idrid_df = pd.DataFrame(
    {
        "image_id": new_image_ids,
        "diagnosis": idrid_total_df["diagnosis"],
        "image_path": new_image_paths,
    }
).dropna()

# Làm sạch kiểu dữ liệu nhãn
final_idrid_df["diagnosis"] = final_idrid_df["diagnosis"].astype(int)

# Lưu ra file CSV tổng mới của IDRiD
final_idrid_df.to_csv(OUTPUT_CSV, index=False)

print("-" * 50)
print(f"✅ Gộp và chuẩn hóa IDRiD thành công!")
print(f"🖼️ Tổng số ảnh sau gộp: {len(final_idrid_df)}")
print(
    f"📄 File CSV chuẩn gọn tại: {OUTPUT_CSV} (Schema hoàn toàn khớp với các dataset khác)"
)
