import os
import shutil
import random
from pathlib import Path

# === CẤU HÌNH ===
SOURCE_DIR = "../data/raw/aptos_gaus_filter/gaussian_filtered_images/gaussian_filtered_images"  # sửa lại đường dẫn
OUTPUT_DIR = "../data/raw/sample_50"
N_SAMPLE = 50
SEED = 42

random.seed(SEED)

for cls in os.listdir(SOURCE_DIR):
    cls_path = Path(SOURCE_DIR) / cls
    if not cls_path.is_dir():
        continue
    
    imgs = [f for f in cls_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if len(imgs) < N_SAMPLE:
        print(f"⚠️  {cls}: chỉ có {len(imgs)} ảnh, lấy hết")
        sampled = imgs
    else:
        sampled = random.sample(imgs, N_SAMPLE)
    
    out_dir = Path(OUTPUT_DIR) / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in sampled:
        shutil.copy2(img_path, out_dir / img_path.name)
    
    print(f"✅ {cls}: {len(sampled)} ảnh")

print(f"\n🎉 Done! Xem tại: {OUTPUT_DIR}")