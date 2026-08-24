import shutil
from pathlib import Path
import kagglehub

# Tải dataset về cache
cache_path = kagglehub.dataset_download("abdullahshafi315/indian-diabetic-retinopathy-image-datasetidrid")

# Đường dẫn thư mục đích: data/raw/idrid
target_dir = Path("../../data/raw/idrid")
# target_dir.mkdir(parents=True, exist_ok=True)

# Sao chép vào thư mục dự án
print(f"Đang chuyển dữ liệu về: {target_dir}...")
for item in Path(cache_path).iterdir():
    dest = target_dir / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)

print(f"✅ Hoàn thành! Dữ liệu đã lưu tại: {target_dir.resolve()}")