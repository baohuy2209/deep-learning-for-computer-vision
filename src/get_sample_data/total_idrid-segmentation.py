import os
import shutil
from pathlib import Path
import pandas as pd

# === 1. ĐỊNH NGHĨA ĐƯỜNG DẪN ===
BASE_DIR = Path("../../data/raw/idrid_segmentation")

# Thư mục đích tổng hợp
DEST_FUNDUS_DIR = BASE_DIR / "idrid_segmentation_total_images"
DEST_MASK_DIR = BASE_DIR / "idrid_segmentation_total_masks"
OUTPUT_CSV = BASE_DIR / "idrid_segmentation_total.csv"

DEST_FUNDUS_DIR.mkdir(parents=True, exist_ok=True)
DEST_MASK_DIR.mkdir(parents=True, exist_ok=True)

# Các thư mục nguồn cần quét (Train + Test)
SUBSET_FOLDERS = [
    BASE_DIR / "DR_Training_Set",
    BASE_DIR / "DR_Testing_Set",
]

# === 2. GỘP TOÀN BỘ ẢNH VÀ MASK VÀO 2 THƯ MỤC TỔNG ===
records = []

for subset in SUBSET_FOLDERS:
    src_fundus = subset / "Fundus Images"
    src_masks = subset / "Combined Masks"

    if not src_masks.exists() or not src_fundus.exists():
        continue

    for mask_path in sorted(src_masks.glob("*.png")):
        image_id = mask_path.stem  # ID dạng: IDRiD_01, IDRiD_02,...
        fundus_path = src_fundus / f"{image_id}.jpg"

        if fundus_path.exists():
            target_fundus = DEST_FUNDUS_DIR / f"{image_id}.jpg"
            target_mask = DEST_MASK_DIR / f"{image_id}.png"

            # Copy ảnh gốc và mask vào thư mục tổng
            if not target_fundus.exists():
                shutil.copy2(fundus_path, target_fundus)
            if not target_mask.exists():
                shutil.copy2(mask_path, target_mask)

            records.append({
                "image_id": image_id,
                "fundus_image": str(target_fundus.resolve()),
                "combined_mask": str(target_mask.resolve()),
            })
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy ảnh tương ứng cho {mask_path.name}")

# === 3. LƯU FILE CSV TỔNG ===
df_total = pd.DataFrame(records)
df_total.to_csv(OUTPUT_CSV, index=False)

print("-" * 50)
print(f"✅ Đã gộp thành công {len(df_total)}/81 cặp ảnh và mask!")
print(f"📁 Thư mục Fundus Images (.jpg) : {DEST_FUNDUS_DIR.resolve()}")
print(f"📁 Thư mục Combined Masks (.png): {DEST_MASK_DIR.resolve()}")
print(f"📄 File CSV tổng hợp            : {OUTPUT_CSV.resolve()}")
print("-" * 50)