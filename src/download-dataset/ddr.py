import os
import shutil
from pathlib import Path
import kagglehub

# 1. Download dataset về thư mục cache mặc định của kagglehub
cache_path = kagglehub.dataset_download("mariaherrerot/ddrdataset")
print("Đã download về cache:", cache_path)

# 2. Xác định đường dẫn thư mục đích: data/raw/ddr
# (Tính từ gốc dự án)
target_dir = Path("../../data/raw/ddr")
target_dir.mkdir(parents=True, exist_ok=True)

# 3. Copy toàn bộ file/thư mục từ cache sang thư mục target
print(f"Đang di chuyển dữ liệu sang {target_dir}...")
for item in Path(cache_path).iterdir():
    dest = target_dir / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)

print(f"Hoàn thành! Dữ liệu đã được lưu tại: {target_dir.resolve()}")