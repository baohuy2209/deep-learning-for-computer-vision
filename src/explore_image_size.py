import cv2
import numpy as np
from pathlib import Path

SOURCE_DIR = "../data/raw/zipEyepacs/train/"

# ============================================================
# Đọc kích thước tất cả ảnh
# ============================================================
sizes = []

# for img_path in Path(SOURCE_DIR).rglob("*.jpg"):
#     img = cv2.imread(str(img_path))
#     if img is not None:
#         h, w = img.shape[:2]
#         sizes.append((h, w, img_path))

# for img_path in Path(SOURCE_DIR).rglob("*.png"):
#     img = cv2.imread(str(img_path))
#     if img is not None:
#         h, w = img.shape[:2]
#         sizes.append((h, w, img_path))
# files = list(Path(SOURCE_DIR).rglob("*.jpeg"))

# print(len(files))
# print(files[:5])
for img_path in Path(SOURCE_DIR).rglob("*.jpeg"):
    img = cv2.imread(str(img_path))
    print(str(img_path))
    if img is not None:
        h, w = img.shape[:2]
        sizes.append((h, w, img_path))

# ============================================================
# Tìm lớn nhất & nhỏ nhất
# ============================================================
sizes_sorted = sorted(sizes, key=lambda x: x[0] * x[1])  # sort theo diện tích

smallest = sizes_sorted[0]
largest  = sizes_sorted[-1]

print(f"Tổng số ảnh : {len(sizes)}")
print()
print(f"Nhỏ nhất : {smallest[1]}x{smallest[0]} px")
print(f"  → File : {smallest[2]}")
print()
print(f"Lớn nhất : {largest[1]}x{largest[0]} px")
print(f"  → File : {largest[2]}")

# ============================================================
# Thống kê thêm
# ============================================================
all_h = [s[0] for s in sizes]
all_w = [s[1] for s in sizes]

print(f"\nThống kê chiều cao (H):")
print(f"  Min: {min(all_h)}  |  Max: {max(all_h)}  |  Trung bình: {int(np.mean(all_h))}")
print(f"Thống kê chiều rộng (W):")
print(f"  Min: {min(all_w)}  |  Max: {max(all_w)}  |  Trung bình: {int(np.mean(all_w))}")