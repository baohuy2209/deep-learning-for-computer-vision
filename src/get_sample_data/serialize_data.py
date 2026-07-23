import numpy as np
from PIL import Image
from pathlib import Path
import pickle

SOURCE_DIR = "../../data/raw/aptos_test_images_gaus"
OUTPUT_FILE = "../../data/processed/aptos/test_images_gaus/sample_dataset.pkl"
IMG_SIZE = (300, 300)

CLASS_ORDER = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
classes = [c for c in CLASS_ORDER if (Path(SOURCE_DIR) / c).is_dir()]
class_to_idx = {cls: i for i, cls in enumerate(classes)}
print("Classes:", class_to_idx)

data, labels = [], []

for cls in classes:
    cls_path = Path(SOURCE_DIR) / cls
    img_files = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))

    for img_path in img_files:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img)  # shape: (300, 300, 3), giá trị 0-255
        data.append(arr)
        labels.append(class_to_idx[cls])

data = np.stack(data)  # (N, 300, 300, 3)
labels = np.array(labels)  # (N,)

print(f"Data shape : {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Giá trị pixel: min={data.min()}, max={data.max()}")
print(f"Ví dụ 1 ảnh:\n{data[0]}")  # in thử ma trận số

# Lưu lại
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"data": data, "labels": labels, "class_names": classes}, f)

print(f"\n✅ Đã lưu tại {OUTPUT_FILE}")
