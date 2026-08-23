import os
import shutil
from pathlib import Path

import pandas as pd

# === CẤU HÌNH ĐƯỜNG DẪN GỐC ===
BASE_DIR = Path("../../data/raw/aptos")
OUTPUT_IMG_DIR = BASE_DIR / "aptos_total_images"  # Thư mục chứa toàn bộ ảnh gộp
OUTPUT_CSV = BASE_DIR / "aptos_total.csv"

# Tạo thư mục đích nếu chưa tồn tại
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Đọc 3 file CSV
train_df = pd.read_csv(BASE_DIR / "train_1.csv")
val_df = pd.read_csv(BASE_DIR / "valid.csv")
test_df = pd.read_csv(BASE_DIR / "test.csv")


# Hàm hỗ trợ copy ảnh và tạo đường dẫn mới
def process_images_and_paths(df, source_subfolder):
    new_paths = []
    for _, row in df.iterrows():
        img_name = str(row["id_code"]) + ".png"
        src_path = BASE_DIR / source_subfolder / img_name
        dest_path = OUTPUT_IMG_DIR / img_name

        # Copy ảnh vật lý (nếu ảnh chưa tồn tại ở thư mục đích để tránh copy đè tốn thời gian)
        if src_path.exists():
            if not dest_path.exists():
                shutil.copy2(src_path, dest_path)
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy ảnh gốc tại {src_path}")

        # Cập nhật đường dẫn mới cho DataFrame
        new_paths.append(f"aptos_total_images/{img_name}")

    df["image_path"] = new_paths
    return df


# 2. Thực hiện copy ảnh và cập nhật DataFrame
print("⏳ Đang copy ảnh từ tập Train...")
train_df = process_images_and_paths(train_df, "train_images/train_images")

# Lưu ý: Bạn kiểm tra lại tên thư mục vật lý của val và test xem có bị lặp 2 lần không nhé
# Mình đang viết theo cấu trúc bạn cung cấp ở tin nhắn trước
print("⏳ Đang copy ảnh từ tập Val...")
val_df = process_images_and_paths(val_df, "val_images/val_images")

print("⏳ Đang copy ảnh từ tập Test...")
test_df = process_images_and_paths(test_df, "test_images/test_images")

# 3. Gộp cả 3 dataframe lại thành 1 cục duy nhất
aptos_total_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# 4. Lưu ra file CSV tổng
aptos_total_df.to_csv(OUTPUT_CSV, index=False)

print("-" * 50)
print(f"✅ Đã gộp và copy xong toàn bộ!")
print(f"🖼️ Tổng số ảnh APTOS hiện tại: {len(aptos_total_df)}")
print(f"📁 Ảnh được lưu tập trung tại: {OUTPUT_IMG_DIR}")
print(f"📄 File CSV tổng được lưu tại: {OUTPUT_CSV}")
