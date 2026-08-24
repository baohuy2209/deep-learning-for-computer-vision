import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ==========================================
# CẤU HÌNH THƯ MỤC
# ==========================================
BASE_HYBRID_DIR = os.path.abspath(
    os.path.join("..", "..", "data", "processed", "hybrid")
)
NATIVE_DIR = os.path.join(BASE_HYBRID_DIR, "native_cropped")

DIR_300 = os.path.join(BASE_HYBRID_DIR, "300x300")
DIR_512 = os.path.join(BASE_HYBRID_DIR, "512x512")

os.makedirs(DIR_300, exist_ok=True)
os.makedirs(DIR_512, exist_ok=True)

# Các file metadata CSV hiện tại của native_cropped
metadata_files = {
    "train": os.path.join(BASE_HYBRID_DIR, "train_balanced_metadata.csv"),
    "val": os.path.join(BASE_HYBRID_DIR, "val_raw_metadata.csv"),
    "test": os.path.join(BASE_HYBRID_DIR, "test_raw_metadata.csv"),
}


# ==========================================
# HÀM ĐỌC METADATA CŨ VÀ RESIZE HÀ LOẠT
# ==========================================
def resize_and_save_using_existing_metadata():
    for split_name, meta_path in metadata_files.items():
        if not os.path.exists(meta_path):
            print(
                f"[WARNING] Không tìm thấy file metadata: {meta_path}. Bỏ qua."
            )
            continue

        df = pd.read_csv(meta_path)

        # Đảm bảo trong df có cột native_path
        if "native_path" not in df.columns:
            print(
                f"[ERROR] File {meta_path} không có cột 'native_path'. Vui lòng kiểm tra lại."
            )
            continue

        print(
            f"\n⏳ Đang resize cho tập [{split_name.upper()}] dựa trên metadata gốc ({len(df):,} ảnh)..."
        )

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            native_path = row["native_path"]

            if not pd.notna(native_path) or not os.path.exists(
                str(native_path)
            ):
                continue

            try:
                # Đọc ảnh gốc từ native_path
                img_pil = Image.open(native_path).convert("RGB")
            except Exception:
                continue

            grade = str(row["diagnosis"])
            file_name = Path(
                native_path
            ).name  # Lấy tên file gốc (ví dụ: train_g0_...png)

            # 1. Lưu bản 300x300 chung cấu trúc thư mục split/grade
            save_dir_300 = os.path.join(DIR_300, split_name, grade)
            os.makedirs(save_dir_300, exist_ok=True)
            path_300 = os.path.join(save_dir_300, file_name)
            if not os.path.exists(path_300):
                img_pil.resize((300, 300), Image.Resampling.BILINEAR).save(
                    path_300
                )

            # 2. Lưu bản 512x512 chung cấu trúc thư mục split/grade
            save_dir_512 = os.path.join(DIR_512, split_name, grade)
            os.makedirs(save_dir_512, exist_ok=True)
            path_512 = os.path.join(save_dir_512, file_name)
            if not os.path.exists(path_512):
                img_pil.resize((512, 512), Image.Resampling.BILINEAR).save(
                    path_512
                )


# Thực thi resize
resize_and_save_using_existing_metadata()
print(
    "\n🎉 HOÀN TẤT RESIZE! Các file metadata cũ giữ nguyên hoàn toàn và sẵn sàng sử dụng."
)
