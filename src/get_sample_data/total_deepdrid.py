import os
import shutil
from pathlib import Path

import pandas as pd

# === CẤU HÌNH ĐƯỜNG DẪN GỐC ===
# Thư mục gốc chứa 3 thư mục con của DeepDRiD
BASE_DIR = Path("../../data/raw/deepdrid/DeepDRiD-master/regular_fundus_images")

# Thư mục đích để chứa tất cả ảnh (Ngang hàng với DeepDRiD-master cho gọn)
OUTPUT_IMG_DIR = Path("../../data/raw/deepdrid/deepdrid_total_images")
OUTPUT_CSV = Path("../../data/raw/deepdrid/deepdrid_total.csv")

# Tạo thư mục đích nếu chưa có
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. ĐỌC DỮ LIỆU ---
train_df = pd.read_csv(
    BASE_DIR / "regular-fundus-training/regular-fundus-training.csv"
)
val_df = pd.read_csv(
    BASE_DIR / "regular-fundus-validation/regular-fundus-validation.csv"
)
challenge_df = pd.read_excel(
    BASE_DIR / "Online-Challenge1&2-Evaluation/Challenge1_labels.xlsx"
)

# --- 2. CHUẨN HÓA VÀ LỌC NHÃN ---
# Challenge: Đổi tên cột và tách patient_id
challenge_df = challenge_df.rename(columns={"DR_Levels": "patient_DR_Level"})
challenge_df["patient_id"] = challenge_df["image_id"].apply(
    lambda x: str(x).split("_")[0]
)

# Lọc bỏ các dòng không có nhãn bệnh lý
cols_to_keep = ["image_id", "patient_id", "patient_DR_Level"]
train_clean = train_df.dropna(subset=["patient_DR_Level"])[cols_to_keep].copy()
val_clean = val_df.dropna(subset=["patient_DR_Level"])[cols_to_keep].copy()
challenge_clean = challenge_df.dropna(subset=["patient_DR_Level"])[
    cols_to_keep
].copy()


# --- 3. HÀM XỬ LÝ COPY ẢNH VÀ TẠO ĐƯỜNG DẪN ---
def process_and_copy_images(df, source_subfolder, is_nested=True):
    new_paths = []
    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        pat_id = str(row["patient_id"])
        img_name = f"{img_id}.jpg"

        # Xác định đường dẫn gốc dựa trên cấu trúc lồng nhau hay không
        if is_nested:
            src_path = (
                BASE_DIR / source_subfolder / "Images" / pat_id / img_name
            )
        else:
            src_path = BASE_DIR / source_subfolder / "Images" / img_name

        dest_path = OUTPUT_IMG_DIR / img_name

        # Copy ảnh vật lý
        if src_path.exists():
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
            # Lưu đường dẫn mới vào CSV (Chỉ lưu phần tương đối để code linh hoạt)
            new_paths.append(f"deepdrid_total_images/{img_name}")
        else:
            print(f"⚠️ Lỗi: Không tìm thấy ảnh gốc tại {src_path}")
            new_paths.append(None)  # Đánh dấu lỗi

    df["image_path"] = new_paths
    return df


# --- 4. THỰC THI COPY CHO TỪNG TẬP ---
print("⏳ Đang copy ảnh từ tập Train (Nested)...")
train_clean = process_and_copy_images(
    train_clean, "regular-fundus-training", is_nested=True
)

print("⏳ Đang copy ảnh từ tập Val (Nested)...")
val_clean = process_and_copy_images(
    val_clean, "regular-fundus-validation", is_nested=True
)

print("⏳ Đang copy ảnh từ tập Challenge (Flat)...")
challenge_clean = process_and_copy_images(
    challenge_clean, "Online-Challenge1&2-Evaluation", is_nested=True
)

# Loại bỏ những dòng bị lỗi không tìm thấy ảnh (nếu có)
train_clean = train_clean.dropna(subset=["image_path"])
val_clean = val_clean.dropna(subset=["image_path"])
challenge_clean = challenge_clean.dropna(subset=["image_path"])

# --- 5. GỘP VÀ LƯU CSV ---
deepdrid_total_df = pd.concat(
    [train_clean, val_clean, challenge_clean], ignore_index=True
)
deepdrid_total_df.to_csv(OUTPUT_CSV, index=False)

print("-" * 50)
print(f"✅ Đã gộp và copy xong toàn bộ DeepDRiD!")
print(f"🖼️ Tổng số ảnh thu được: {len(deepdrid_total_df)}")
print(f"📁 Ảnh được gom sạch sẽ tại: {OUTPUT_IMG_DIR}")
print(f"📄 File CSV tổng tại: {OUTPUT_CSV}")
