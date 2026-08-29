"""
==============================================================================
Fast Hybrid Dataset Preprocessing & Split Generator
==============================================================================
Implements 3 Core Optimizations recommended for fast & scalable dataset prep:
1. Multi-threaded / Multi-core Parallel reading, cropping, and writing with `joblib.Parallel(n_jobs=8)`.
2. High-quality JPEG 95 encoding (instead of PNG) to accelerate I/O and save disk space on 3000x2000 fundus images.
3. Fast DataFrame creation using `list of dict` (instead of slow `df.iterrows()`), generating all split & fold CSVs in seconds.
==============================================================================
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import GroupShuffleSplit
from joblib import Parallel, delayed
from tqdm import tqdm

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & DATASET
# ==============================================================================
BASE_RAW_DIR = Path("data/raw")
HYBRID_RAW_DIR = BASE_RAW_DIR / "hybrid"
NATIVE_CROPPED_DIR = HYBRID_RAW_DIR / "native_cropped"
FOLDS_DIR = HYBRID_RAW_DIR / "folds"

NATIVE_CROPPED_DIR.mkdir(parents=True, exist_ok=True)
FOLDS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

DATASET_CONFIG = {
    "APTOS": {
        "name": "APTOS 2019",
        "csv_path": "aptos/aptos_total_images/train.csv",
        "img_dir": "aptos/aptos_total_images",
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "DDR": {
        "name": "DDR Dataset",
        "csv_path": "ddr/DR_grading.csv",
        "img_dir": "ddr/DR_grading/DR_grading",
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "DeepDRiD": {
        "name": "DeepDRiD",
        "csv_path": "deepdrid/deepdrid_total.csv",
        "img_dir": "deepdrid/deepdrid_total_images",
        "id_col": "image_id",
        "label_col": "patient_DR_Level",
        "id_suffix": ".jpg"
    },
    "IDRiD": {
        "name": "IDRiD",
        "csv_path": "idrid/idrid_total.csv",
        "img_dir": "idrid/idrid_total_images",
        "id_col": "image_id",
        "label_col": "diagnosis",
        "id_suffix": ".jpg"
    },
    "Messidor-2": {
        "name": "Messidor-2",
        "csv_path": "messidor2preprocess/messidor_data.csv",
        "img_dir": "messidor2preprocess/messidor-2/messidor-2/preprocess",
        "id_col": "id_code",
        "label_col": "diagnosis",
        "id_suffix": ".png"
    },
    "EyePACS": {
        "name": "EyePACS (Kaggle)",
        "csv_path": "zipEyepacs/trainLabels.csv/trainLabels.csv",
        "img_dir": "zipEyepacs/train",
        "id_col": "image",
        "label_col": "level",
        "id_suffix": ".jpeg"
    }
}


# ==============================================================================
# 2. HÀM TỐI ƯU CROP ROI VÀ LƯU JPEG QUALITY 95 (TỐI ƯU 1 & 2)
# ==============================================================================
def crop_fundus_roi(img_bgr: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """
    Tự động loại bỏ viền đen không chẩn đoán xung quanh ảnh đáy mắt.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
        
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance
    
    if not np.any(mask):
        return img_bgr
        
    row_mask = np.any(mask, axis=1)
    col_mask = np.any(mask, axis=0)
    ymin, ymax = np.where(row_mask)[0][[0, -1]]
    xmin, xmax = np.where(col_mask)[0][[0, -1]]
    
    cropped = img_bgr[ymin:ymax+1, xmin:xmax+1]
    return cropped if cropped.size > 0 else img_bgr


def process_and_save_image(task: Tuple[str, str]) -> bool:
    """
    Đọc ảnh gốc, crop viền đen và lưu định dạng JPEG Quality 95.
    Lưu JPEG 95 nhanh gấp 10-20 lần so với PNG lossless trên ảnh lớn 3000x2000.
    """
    raw_path, native_path = task
    if os.path.exists(native_path):
        return True
    try:
        raw_img = cv2.imread(raw_path)
        if raw_img is None:
            return False
        cropped = crop_fundus_roi(raw_img)
        os.makedirs(os.path.dirname(native_path), exist_ok=True)
        # TỐI ƯU 2: Ghi file JPEG Quality 95 (chất lượng y khoa, tối ưu tốc độ encode)
        cv2.imwrite(native_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return True
    except Exception:
        return False


# ==============================================================================
# 3. TRÍCH XUẤT PATIENT_ID & DÙNG LIST OF DICT ĐỂ TẠO METADATA (TỐI ƯU 3)
# ==============================================================================
def parse_patient_and_side(image_id: str, dataset_key: str) -> Tuple[str, str]:
    img_str = str(image_id).lower()
    k = dataset_key.lower()
    
    if "idrid" in k:
        base = img_str.replace(".jpg", "").replace(".png", "")
        pid = base.split("_")[-1]
    elif "eyepacs" in k or "deepdrid" in k:
        pid = img_str.split("_")[0]
    elif "ddr" in k:
        parts = img_str.replace(".jpg", "").replace(".png", "").split("-")
        pid = "-".join(parts[:2]) if len(parts) >= 2 else img_str.split(".")[0]
    elif "messidor" in k:
        pid = img_str.split("_")[0]
    else:
        pid = img_str
        
    side = "left" if "left" in img_str or "_l" in img_str else \
           "right" if "right" in img_str or "_r" in img_str else "unknown"
    return f"{dataset_key}::{pid}", side


def build_master_dataframe_fast(config_dict: dict, base_dir: Path) -> pd.DataFrame:
    """
    TỐI ƯU 3: Bỏ hoàn toàn .iterrows() hay append DataFrame.
    Gom toàn bộ dữ liệu vào `records` (list of dict) rồi tạo DataFrame 1 lần ở cuối.
    """
    records: List[Dict] = []
    
    for key, cfg in config_dict.items():
        csv_file = base_dir / cfg["csv_path"]
        img_folder = base_dir / cfg["img_dir"]
        
        if not csv_file.exists():
            print(f"[WARNING] Bỏ qua {key} - không tìm thấy CSV: {csv_file}")
            continue
            
        raw_df = pd.read_csv(csv_file)
        label_col = cfg["label_col"]
        if label_col not in raw_df.columns:
            matches = [c for c in raw_df.columns if "dr" in c.lower() or "level" in c.lower() or "diagnosis" in c.lower()]
            if matches:
                label_col = matches[0]
                
        # Duyệt nhanh theo numpy / zip thay vì iterrows()
        id_values = raw_df[cfg["id_col"]].astype(str).tolist()
        label_values = pd.to_numeric(raw_df[label_col], errors="coerce").tolist()
        suffix = cfg.get("id_suffix", "")
        
        for img_id, diag in zip(id_values, label_values):
            if pd.isna(diag) or int(diag) not in [0, 1, 2, 3, 4]:
                continue
            
            img_file = img_id if img_id.endswith(suffix) else img_id + suffix
            raw_path = str((img_folder / img_file).resolve())
            
            # Tên file đích chuẩn hóa
            clean_id = re.sub(r'[^\w\.-]', '_', str(img_id))
            native_filename = f"{clean_id}.jpg"
            native_path = str((NATIVE_CROPPED_DIR / key / native_filename).resolve())
            
            pid, side = parse_patient_and_side(img_id, dataset_key=key)
            
            records.append({
                "image_id": img_id,
                "diagnosis": int(diag),
                "dataset_name": cfg["name"],
                "dataset_key": key,
                "image_file": img_file,
                "raw_full_path": raw_path,
                "patient_id": pid,
                "side": side,
                "native_path": native_path
            })
            
        print(f"  • Đã tải metadata cho {cfg['name']}: {len(records)} tổng số mẫu...")

    # Tạo DataFrame duy nhất 1 lần
    master_df = pd.DataFrame(records)
    print(f"\n✅ Đã tạo Master DataFrame thành công với {len(master_df):,} ảnh từ {master_df['dataset_name'].nunique()} nguồn dữ liệu!")
    return master_df


# ==============================================================================
# 4. CHIA SPLIT 70/15/15 LEAK-FREE & XUẤT 6 FILE CSV METADATA TRONG VÀI GIÂY
# ==============================================================================
def split_and_export_metadata(df: pd.DataFrame, random_state: int = 42):
    """
    Thực hiện GroupShuffleSplit ở cấp độ patient_id chống rò rỉ dữ liệu (Leak-free).
    Sinh toàn bộ các file CSV metadata (train, val, test, fold) chỉ trong vài giây.
    """
    print("\n⏳ Đang phân chia 70/15/15 ở cấp độ bệnh nhân (Patient-Level GroupSplit)...")
    
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    train_idx, temp_idx = next(gss_train.split(df, groups=df['patient_id']))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)
    
    gss_test = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    val_idx, test_idx = next(gss_test.split(temp_df, groups=temp_df['patient_id']))
    
    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    # 1. Gộp toàn bộ vào file fold hoàn chỉnh
    fold_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    fold_path = FOLDS_DIR / "fold_internal_70_15_15.csv"
    fold_df.to_csv(fold_path, index=False)
    
    # 2. Xuất các file split độc lập
    train_df.to_csv(HYBRID_RAW_DIR / "train_metadata.csv", index=False)
    val_df.to_csv(HYBRID_RAW_DIR / "val_metadata.csv", index=False)
    test_df.to_csv(HYBRID_RAW_DIR / "test_metadata.csv", index=False)
    
    print("\n" + "=" * 60)
    print("✅ ĐÃ XUẤT XONG CÁC FILE CSV METADATA (MẤT VÀI GIÂY)!")
    print(f"📄 Fold CSV: {fold_path}")
    print(f"📄 Train CSV: {HYBRID_RAW_DIR / 'train_metadata.csv'} ({len(train_df):,} mẫu)")
    print(f"📄 Val CSV:   {HYBRID_RAW_DIR / 'val_metadata.csv'} ({len(val_df):,} mẫu)")
    print(f"📄 Test CSV:  {HYBRID_RAW_DIR / 'test_metadata.csv'} ({len(test_df):,} mẫu)")
    print("=" * 60)


if __name__ == "__main__":
    print(">>> BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ NHANH CHO DATASET HYBRID...")
    
    # 1. Tạo Master DataFrame siêu tốc bằng List of Dict
    master_df = build_master_dataframe_fast(DATASET_CONFIG, BASE_RAW_DIR)
    
    # 2. TỐI ƯU 1: Xử lý song song n_jobs=8 vòng lặp đọc-crop-ghi JPEG 95
    tasks = list(zip(master_df["raw_full_path"], master_df["native_path"]))
    num_jobs = min(8, os.cpu_count() or 4)
    print(f"\n[INFO] Khởi chạy joblib.Parallel với n_jobs={num_jobs} (Đọc - Crop - Ghi JPEG 95)...")
    
    results = Parallel(n_jobs=num_jobs, prefer="threads", batch_size=64)(
        delayed(process_and_save_image)(task) for task in tqdm(tasks, desc="Fast Native Cropping")
    )
    
    # 3. Lọc các mẫu hợp lệ tồn tại trên đĩa
    master_df["_exists"] = master_df["native_path"].apply(os.path.exists)
    valid_master_df = master_df[master_df["_exists"]].drop(columns=["_exists"]).reset_index(drop=True)
    print(f"✅ Đã xử lý & lưu {len(valid_master_df):,} ảnh vào {NATIVE_CROPPED_DIR}")
    
    # 4. Xuất các file CSV metadata & fold 70/15/15 chỉ trong vài giây
    split_and_export_metadata(valid_master_df, random_state=SEED)
