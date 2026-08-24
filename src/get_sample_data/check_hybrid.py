import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
BASE_PROCESSED_DIR = "../../data/processed"
# 1. Đường dẫn đến file metadata của tập train đã lưu
meta_csv_path = os.path.join(BASE_PROCESSED_DIR, "hybrid", "train_balanced_metadata.csv")

# Đọc file CSV metadata
if os.path.exists(meta_csv_path):
    df_check = pd.read_csv(meta_csv_path)
    print(f"[INFO] Đã tải thành công metadata. Tổng số dòng trong tập train: {len(df_check):,}")
    
    # 2. Lấy ngẫu nhiên 1 dòng bất kỳ trong tệp hybrid
    sample_row = df_check.sample(n=1, random_state=42).iloc[0]
    
    # Trích xuất thông tin
    img_id = sample_row["image_id"]
    diagnosis = sample_row["diagnosis"]
    dataset_source = sample_row["dataset_name"]
    native_path = sample_row["native_path"]
    patient_id = sample_row["patient_id"]
    
    print("\n" + "="*50)
    print("=== THÔNG TIN CHI TIẾT MẪU ẢNH KIỂM TRA ===")
    print("="*50)
    print(f"• Image ID     : {img_id}")
    print(f"• Patient ID   : {patient_id}")
    print(f"• Nguồn Dataset: {dataset_source}")
    print(f"• Mức độ DR    : Grade {diagnosis}")
    print(f"• Đường dẫn ảnh: {native_path}")
    
    # 3. Đọc ảnh vật lý từ ổ cứng để kiểm tra độ phân giải gốc
    if os.path.exists(native_path):
        img_pil = Image.open(native_path).convert("RGB")
        img_np = np.array(img_pil)
        
        print(f"• Kích thước thực (Native Resolution): {img_np.shape[1]}x{img_np.shape[0]} pixels (W x H)")
        print(f"• Dải giá trị Pixel (Min-Max)      : [{img_np.min()}, {img_np.max()}]")
        
        # 4. Hiển thị ảnh kiểm tra
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.title(f"Dataset: {dataset_source} | Grade: {diagnosis}\nNative Size: {img_np.shape[1]}x{img_np.shape[0]}px", fontsize=11)
        plt.axis("off")
        plt.show()
    else:
        print(f"[ERROR] Không tìm thấy file ảnh tại đường dẫn: {native_path}")
else:
    print(f"[WARNING] Không tìm thấy file metadata tại: {meta_csv_path}. Bạn hãy chạy cell lưu metadata trước nhé!")