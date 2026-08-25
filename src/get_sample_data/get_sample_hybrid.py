import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

# === CẤU HÌNH ===
CSV_PATH = "../../data/processed/hybrid/train_balanced_metadata.csv"

SOURCE_DIR = "../../data/processed/hybrid/native_cropped/train"

OUTPUT_DIR = "../../data/raw/samples/hybrid/native_cropped/sample_50"
OUTPUT_CSV_PATH = "../../data/raw/samples/hybrid/native_cropped/sample_50_metadata.csv"

N_SAMPLE = 50
LEVEL = "diagnosis"  # Cột chứa nhãn bệnh lý
SEED = 42
random.seed(SEED)

# Đọc toàn bộ dataframe gốc
df_original = pd.read_csv(CSV_PATH)

class_to_rows = defaultdict(list)

# Bước 1: Đọc metadata và gom nhóm theo class (diagnosis)
for idx, row in df_original.iterrows():
    diagnosis = int(str(row[LEVEL]).strip())
    class_name = str(diagnosis)

    # Lấy tên file ảnh từ cột native_path hoặc tên file gốc
    file_name = Path(row["native_path"]).name

    # Xây dựng đường dẫn nguồn dựa trên SOURCE_DIR và class_name (hoặc cấu trúc thư mục của bạn)
    # Nếu thư mục nguồn của bạn có phân chia theo class_name (vd: train/0/, train/1/):
    img_path = Path(SOURCE_DIR) / class_name / file_name

    # Hoặc nếu SOURCE_DIR chứa thẳng tất cả ảnh không chia folder con, bạn dùng:
    # img_path = Path(SOURCE_DIR) / file_name

    if img_path.exists():
        row_dict = row.to_dict()
        row_dict["valid_img_path"] = img_path
        class_to_rows[class_name].append(row_dict)
    else:
        print(f"⚠️ Không tìm thấy ảnh thực tế tại: {img_path}")

# Bước 2: Lấy N_SAMPLE ảnh cho mỗi class, copy ảnh và xây dựng danh sách metadata mới
sampled_rows = []

for class_name, rows in class_to_rows.items():
    if len(rows) < N_SAMPLE:
        print(f"⚠️ Class {class_name}: chỉ có {len(rows)} ảnh, lấy hết")
        sampled = rows
    else:
        sampled = random.sample(rows, N_SAMPLE)

    out_dir = Path(OUTPUT_DIR) / f"{class_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in sampled:
        src_path = item["valid_img_path"]
        file_name = src_path.name
        dst_path = out_dir / file_name

        # Copy ảnh vào thư mục sample
        shutil.copy(src_path, dst_path)

        # Cập nhật lại đường dẫn trong metadata trỏ thẳng tới thư mục sample mới
        item["native_path"] = str(dst_path)
        # Xóa trường tạm trước khi ghi file
        del item["valid_img_path"]

        sampled_rows.append(item)

    print(f"✅ Class {class_name}: Đã copy {len(sampled)} ảnh thành công.")

# Bước 3: Lưu DataFrame metadata mới cho tập sample
os.makedirs(Path(OUTPUT_CSV_PATH).parent, exist_ok=True)
df_sample = pd.DataFrame(sampled_rows)
df_sample.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n🎉 HOÀN TẤT!")
print(f"📁 Thư mục ảnh sample: {OUTPUT_DIR}")
print(
    f"📄 File CSV metadata sample: {OUTPUT_CSV_PATH} (Đã tổng hợp đầy đủ {len(df_sample)} mẫu)."
)
