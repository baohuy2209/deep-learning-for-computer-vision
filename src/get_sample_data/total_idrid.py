import os
import re
import shutil
from pathlib import Path

import pandas as pd

# === 1. CẤU HÌNH ĐƯỜNG DẪN ===
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


# === 2. HÀM TÌM FILE ẢNH LINH HOẠT ===
def find_image(folder: Path, base_name: str) -> Path | None:
    for ext in [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]:
        p = folder / f"{base_name}{ext}"
        if p.exists():
            return p
    return None


# === 3. ĐỌC VÀ CHUẨN HÓA 2 FILE CSV NHÃN ===
def load_clean_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    id_col = next(
        (c for c in df.columns if "image" in c.lower() or "id" in c.lower()),
        "Image name",
    )
    label_col = next(
        (
            c
            for c in df.columns
            if "grade" in c.lower()
            or "diagnosis" in c.lower()
            or "level" in c.lower()
        ),
        "Retinopathy grade",
    )

    clean_df = df[[id_col, label_col]].copy().dropna()
    clean_df.columns = ["orig_id", "diagnosis"]

    clean_df["orig_id"] = (
        clean_df["orig_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.(jpg|png|jpeg)", "", regex=True)
    )
    clean_df["diagnosis"] = clean_df["diagnosis"].astype(int)

    # Sắp xếp theo số thứ tự gốc
    clean_df["num"] = clean_df["orig_id"].apply(
        lambda x: int(re.findall(r"\d+", x)[-1]) if re.findall(r"\d+", x) else 0
    )
    clean_df = clean_df.sort_values("num").reset_index(drop=True)
    return clean_df


print("⏳ Đang đọc và sắp xếp nhãn dữ liệu...")
train_df = load_clean_csv(TRAIN_CSV_SRC)
test_df = load_clean_csv(TEST_CSV_SRC)

# === 4. ĐÁNH SỐ TĂNG DẦN LIÊN TỤC (TRAIN: 001->413, TEST: 414->516) ===
records = []

# Xử lý tập Train: Giữ nguyên IDRiD_001 -> IDRiD_413
for idx, row in train_df.iterrows():
    orig_id = row["orig_id"]
    new_id = f"IDRiD_{idx + 1:03d}"
    dest_filename = f"{new_id}.jpg"

    src_img = find_image(TRAIN_IMG_SRC, orig_id)
    dest_img = OUTPUT_IMG_DIR / dest_filename

    if src_img and src_img.exists():
        if not dest_img.exists():
            shutil.copy2(src_img, dest_img)
        records.append({"image_id": new_id, "diagnosis": row["diagnosis"]})
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy ảnh train '{orig_id}'")

offset = len(records)  # 413

# Xử lý tập Test: Đánh số tiếp tục từ IDRiD_414 -> IDRiD_516
for idx, row in test_df.iterrows():
    orig_id = row["orig_id"]
    new_id = f"IDRiD_{offset + idx + 1:03d}"
    dest_filename = f"{new_id}.jpg"

    src_img = find_image(TEST_IMG_SRC, orig_id)
    dest_img = OUTPUT_IMG_DIR / dest_filename

    if src_img and src_img.exists():
        if not dest_img.exists():
            shutil.copy2(src_img, dest_img)
        records.append({"image_id": new_id, "diagnosis": row["diagnosis"]})
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy ảnh test '{orig_id}'")

# === 5. XUẤT FILE CSV TỔNG ===
final_df = pd.DataFrame(records)
final_df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("✅ HOÀN TẤT GỘP IDRiD ĐÁNH SỐ TĂNG DẦN LIÊN TỤC (001 -> 516)")
print("=" * 60)
print(f"📊 Tổng số ảnh: {len(final_df)} ảnh")
print(f"   • Train gốc (413 ảnh): IDRiD_001 -> IDRiD_413")
print(f"   • Test gốc  (103 ảnh): IDRiD_414 -> IDRiD_516")
print(f"📁 Thư mục ảnh : {OUTPUT_IMG_DIR.resolve()}")
print(f"📄 File CSV    : {OUTPUT_CSV.resolve()}")
print("\nBảng xem trước 5 dòng đầu:")
print(final_df.head())
print("\nBảng xem trước 5 dòng cuối:")
print(final_df.tail())
print("=" * 60)
