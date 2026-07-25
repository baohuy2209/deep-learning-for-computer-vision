import cv2
import numpy as np
from skimage.filters import frangi
import matplotlib.pyplot as plt

def extract_dr_lesions(image_path):
    """
    Trích xuất các đặc trưng tổn thương từ ảnh đáy mắt.
    Trả về: Ảnh gốc, Mạch máu, Tiết dịch sáng, Vi phình mạch/Xuất huyết
    """
    # 1. ĐỌC ẢNH VÀ TÁCH KÊNH GREEN
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)  # OpenCV đọc theo BGR
    
    # 2. ÁP DỤNG CLAHE LÊN KÊNH GREEN ĐỂ TĂNG TƯƠNG PHẢN
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g)
    
    # 3. TRÍCH XUẤT MẠCH MÁU (FRANGI FILTER)
    # Frangi yêu cầu ảnh dạng float (0 -> 1)
    g_float = g_clahe.astype(float) / 255.0
    
    # Thông số sigmas quy định độ dày của mạch máu cần tìm
    vessels = frangi(g_float, sigmas=range(1, 4, 1), black_ridges=True)
    # Scale lại về 0-255 để dễ hiển thị
    vessels_img = (vessels / np.max(vessels) * 255).astype(np.uint8)

    # Khởi tạo kernel hình elip (phù hợp với hình dáng tổn thương mắt)
    # Kích thước 15x15 pixel (bạn có thể điều chỉnh theo độ phân giải ảnh thực tế)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # 4. TRÍCH XUẤT TIẾT DỊCH CỨNG / BÔNG TIẾT (MORPHOLOGICAL TOP-HAT)
    # Top-hat giữ lại các chi tiết SÁNG hơn nền (nhỏ hơn kernel)
    exudates = cv2.morphologyEx(g_clahe, cv2.MORPH_TOPHAT, kernel)
    
    # 5. TRÍCH XUẤT VI PHÌNH MẠCH / XUẤT HUYẾT (MORPHOLOGICAL BOTTOM-HAT)
    # Bottom-hat giữ lại các chi tiết TỐI hơn nền (nhỏ hơn kernel)
    hemorrhages = cv2.morphologyEx(g_clahe, cv2.MORPH_BLACKHAT, kernel)

    return img, vessels_img, exudates, hemorrhages

# ==========================================
# TEST THỬ VÀ VẼ BIỂU ĐỒ TRỰC QUAN
# ==========================================
# (Bạn đổi đường dẫn này thành một file ảnh bất kỳ trong dataset của bạn)
test_image_path = "../../../data/raw/aptos/images/0a0906214cf4.png" 

img, vessels, exudates, hemorrhages = extract_dr_lesions(test_image_path)

# Hiển thị kết quả bằng Matplotlib
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("1. Ảnh Gốc (Original)")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(vessels, cmap='gray')
plt.title("2. Mạch máu (Frangi Filter)")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(exudates, cmap='gray')
plt.title("3. Tiết dịch (Top-Hat)")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(hemorrhages, cmap='gray')
plt.title("4. Xuất huyết (Bottom-Hat)")
plt.axis('off')

plt.tight_layout()
plt.show()