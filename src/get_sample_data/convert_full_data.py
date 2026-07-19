import numpy as np
from PIL import Image
from pathlib import Path
import pickle

# === CẤU HÌNH ===
SOURCE_DIR = "../../data/raw/aptos_gaus_filter/gaussian_filtered_images/gaussian_filtered_images"
OUTPUT_FILE = "../../data/processed/aptos/full_gaus/full_dataset.pkl"
IMG_SIZE = (300, 300)

CLASS_ORDER = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

classes = [c for c in CLASS_ORDER if (Path(SOURCE_DIR) / c).is_dir()]
class_to_idx = {cls: i for i, cls in enumerate(classes)}
print("Classes:", class_to_idx)

data, labels = [], []
failed_files = []

for cls in classes:
    cls_path = Path(SOURCE_DIR) / cls
    img_files = sorted(
        list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")) + list(cls_path.glob("*.png"))
    )

    print(f"📂 {cls}: tổng {len(img_files)} ảnh, lấy toàn bộ")

    count = 0
    for img_path in img_files:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            arr = np.array(img)  # (300, 300, 3), giá trị 0-255
            data.append(arr)
            labels.append(class_to_idx[cls])
            count += 1
        except Exception as e:
            failed_files.append((img_path, str(e)))
            print(f"⚠️  Lỗi đọc ảnh {img_path}: {e}")

    print(f"✅ {cls}: {count} ảnh đã xử lý")

data = np.stack(data)      # (N, 300, 300, 3)
labels = np.array(labels)  # (N,)

print(f"\nData shape : {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Phân bố class: {np.bincount(labels)}")
print(f"Giá trị pixel: min={data.min()}, max={data.max()}")
print(f"dtype: {data.dtype}")
if failed_files:
    print(f"\n⚠️  Có {len(failed_files)} ảnh bị lỗi, đã skip.")

# Tạo thư mục output nếu chưa có
out_path = Path(OUTPUT_FILE)
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"data": data, "labels": labels, "class_names": classes}, f)

print(f"\n✅ Đã lưu tại {OUTPUT_FILE}")