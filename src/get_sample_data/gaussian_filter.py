import os
import cv2
from pathlib import Path
from tqdm import (
    tqdm,
)  # Thư viện để hiện thanh tiến trình cho đẹp (pip install tqdm)

# === CẤU HÌNH ===
# Thư mục test bạn vừa sinh ra từ script trước
INPUT_DIR = "../../data/raw/aptos_test_images"
# Thư mục mới chứa ảnh test đã qua xử lý Gaussian
OUTPUT_DIR = "../../data/raw/aptos_test_images_gaus"

# LƯU Ý QUAN TRỌNG: Kích thước Kernel này (15, 15) bắt buộc
# phải giống y hệt với thông số bạn đã dùng cho tập Train!
K_SIZE = (15, 15)


def apply_gaussian_to_test_set():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    # Tạo thư mục gốc đầu ra nếu chưa có
    output_path.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách các thư mục class bên trong aptos_test_images
    # (No_DR, Mild, Moderate, Severe, Proliferate_DR)
    class_folders = [f.name for f in os.scandir(input_path) if f.is_dir()]

    if not class_folders:
        print(f"⚠️ Không tìm thấy thư mục class nào trong {INPUT_DIR}")
        return

    print(
        f"🚀 Bắt đầu xử lý Gaussian Filter {K_SIZE} cho các class: {class_folders}\n"
    )

    for class_name in class_folders:
        in_class_dir = input_path / class_name
        out_class_dir = output_path / class_name

        # Tạo thư mục con tương ứng ở thư mục đích
        out_class_dir.mkdir(parents=True, exist_ok=True)

        # Lấy tất cả các file .png trong thư mục class hiện tại
        image_files = list(in_class_dir.glob("*.png"))

        print(f"📁 Đang xử lý class [{class_name}] ({len(image_files)} ảnh)...")

        # Dùng tqdm để hiển thị thanh tiến trình
        for img_path in tqdm(image_files, desc=class_name):
            # Đọc ảnh gốc
            img = cv2.imread(str(img_path))

            if img is not None:
                # Áp dụng bộ lọc Gaussian
                blurred = cv2.GaussianBlur(img, K_SIZE, 0)

                # Tạo đường dẫn lưu file mới
                save_path = out_class_dir / img_path.name

                # Lưu ảnh đã xử lý
                cv2.imwrite(str(save_path), blurred)
            else:
                print(f"⚠️ Lỗi không thể đọc ảnh: {img_path}")

    print(
        f"\n🎉 Done! Toàn bộ ảnh test đã được làm mờ và lưu tại: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    apply_gaussian_to_test_set()
