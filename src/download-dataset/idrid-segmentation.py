import os
import shutil
from pathlib import Path

import kagglehub

# Kaggle API authentication (read from environment variables or ~/.kaggle/kaggle.json)
kaggle_user = os.environ.get("KAGGLE_USERNAME")
kaggle_key = os.environ.get("KAGGLE_KEY")

# Download dataset from Kaggle
print("Downloading IDRiD Segmentation Dataset...")
cache_path = kagglehub.dataset_download(
    "realhaadkhan/idrid-segmentation-dataset"
)
print(f"Downloaded cache path: {cache_path}")

# 3. Định vị thư mục đích (data/raw/idrid_segmentation)
target_dir = Path("../../data/raw/idrid_segmentation")
target_dir.mkdir(parents=True, exist_ok=True)

# 4. Sao chép dữ liệu từ cache vào thư mục dự án
print(f"⏳ Đang sao chép dữ liệu vào: {target_dir.resolve()}...")
for item in Path(cache_path).iterdir():
    dest = target_dir / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)

print(f"\n✅ Hoàn tất! Dữ liệu đã được lưu tại: {target_dir.resolve()}")

# 5. Kiểm tra nhanh danh sách thư mục / file sau khi tải
print("\n=== CÁC THƯ MỤC / FILE ĐÃ TẢI VỀ ===")
for p in sorted(target_dir.glob("*")):
    print(f" 📁 {p.name}")
