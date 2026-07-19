import pickle
import numpy as np

with open("../../data/processed/aptos/sample_50_gaus/sample_dataset.pkl", "rb") as f:
    d = pickle.load(f)

# 1. Check shape và phân bố
print("Data shape:", d["data"].shape)
print("Phân bố nhãn:", np.bincount(d["labels"]))
print("Class names:", d["class_names"])
# Kỳ vọng: mỗi class ~50 (hoặc ít hơn nếu class đó thiếu ảnh, script đã tự in cảnh báo lúc chạy)

# 2. Check dtype và range pixel hợp lệ
print("dtype:", d["data"].dtype)          # kỳ vọng uint8
print("min/max:", d["data"].min(), d["data"].max())  # kỳ vọng 0-255

# 3. Check không có ảnh toàn đen/toàn trắng (dấu hiệu đọc lỗi)
for i in range(len(d["data"])):
    if d["data"][i].std() < 1:
        print(f"⚠️ Ảnh index {i} (label={d['class_names'][d['labels'][i]]}) gần như đồng màu, nghi lỗi")