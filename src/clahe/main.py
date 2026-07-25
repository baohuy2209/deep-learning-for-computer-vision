import cv2
import os
import glob
from tqdm import tqdm

def apply_clahe_to_fundus(image_path, output_path, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Hàm xử lý CLAHE cho một ảnh duy nhất theo chuẩn ảnh y tế.
    """
    # 1. Đọc ảnh bằng OpenCV (OpenCV mặc định đọc theo định dạng BGR)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return False

    # 2. Chuyển đổi từ hệ màu BGR sang LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 3. Tách 3 kênh L, A, B
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 4. Khởi tạo bộ lọc CLAHE
    # clipLimit: Ngưỡng giới hạn độ tương phản (tránh làm nhiễu hạt bị phóng đại quá mức)
    # tileGridSize: Kích thước ô vuông cắt nhỏ bức ảnh để cân bằng sáng
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    # 5. Áp dụng CLAHE ĐỘC QUYỀN lên kênh L (Độ sáng)
    cl = clahe.apply(l_channel)

    # 6. Gộp kênh L đã xử lý với kênh A và B nguyên bản
    merged_lab = cv2.merge((cl, a_channel, b_channel))

    # 7. Chuyển ngược lại từ LAB về BGR để lưu
    final_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # 8. Lưu ảnh ra thư mục đích
    cv2.imwrite(output_path, final_img)
    return True

def process_dataset_with_clahe(input_folder, output_folder):
    """
    Hàm quét toàn bộ thư mục và xử lý CLAHE hàng loạt.
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Đã tạo thư mục đầu ra: {output_folder}")

    # Lấy danh sách tất cả các ảnh (hỗ trợ png, jpg, jpeg)
    search_path = os.path.join(input_folder, "*.[pj][pn][g]") # Bắt cả jpg, png, jpeg
    image_paths = glob.glob(search_path)
    
    if len(image_paths) == 0:
        print("Không tìm thấy ảnh nào trong thư mục đầu vào!")
        return

    print(f"Bắt đầu xử lý CLAHE cho {len(image_paths)} bức ảnh...")

    # Chạy vòng lặp với thanh tiến trình tqdm
    for img_path in tqdm(image_paths, desc="Đang xử lý"):
        # Lấy tên file gốc (VD: 'image_001.png')
        filename = os.path.basename(img_path)
        
        # Đường dẫn lưu file mới
        save_path = os.path.join(output_folder, filename)
        
        # Gọi hàm xử lý
        apply_clahe_to_fundus(img_path, save_path)
        
    print("\n✅ Hoàn tất quá trình xử lý CLAHE!")

# ==========================================
# 3. CHẠY THỰC TẾ
# ==========================================
# (Bạn cần thay đổi đường dẫn này cho khớp với máy của bạn)
INPUT_DIR = "../../../data/raw/aptos/images/" 
OUTPUT_DIR = "../../../data/processed/aptos/clahe_images/"

# Gọi hàm
process_dataset_with_clahe(INPUT_DIR, OUTPUT_DIR)