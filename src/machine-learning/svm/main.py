import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
with open("../../../data/processed/aptos/sample_50/sample_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X           = data['data']        # shape: (N, 300, 300, 3)
y           = data['labels']      # shape: (N,) - Giữ nguyên mảng số nguyên, KHÔNG to_categorical
class_names = data['class_names'] 

# ==========================================
# 2. TIỀN XỬ LÝ CHO SVM (BẮT BUỘC)
# ==========================================
print("Đang tiến hành tiền xử lý dữ liệu cho SVM...")

# 2.1. Duỗi phẳng ảnh (Flatten)
N = X.shape[0]
X_flatten = X.reshape(N, -1) # Chuyển từ (N, 300, 300, 3) thành (N, 270000)
print(f"Kích thước sau khi duỗi: {X_flatten.shape}")

# 2.2. Chia tập Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X_flatten, y, test_size=0.2, random_state=42, stratify=y
)

# 2.3. Chuẩn hóa thang đo (StandardScaler)
# Quan trọng: Chỉ fit trên tập Train, sau đó transform cho cả Train và Test để tránh rò rỉ dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2.4. Giảm chiều dữ liệu bằng PCA (Chống tràn RAM)
# n_components=0.95 nghĩa là PCA sẽ tự tìm số lượng chiều sao cho giữ lại được 95% thông tin gốc của ảnh
print("Đang chạy thuật toán nén PCA...")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Số lượng đặc trưng (features) đã giảm từ {X_flatten.shape[1]} xuống còn {X_train_pca.shape[1]}!")

# ==========================================
# 3. KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH SVM
# ==========================================
print("\nBắt đầu huấn luyện mô hình SVM (Có thể mất một lúc)...")
start_time = time.time()

# Sử dụng kernel 'rbf' (Radial basis function) là hiệu quả nhất cho dữ liệu phi tuyến tính như ảnh
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train model
svm_model.fit(X_train_pca, y_train)

end_time = time.time()
print(f"✅ Đã huấn luyện xong! Thời gian: {end_time - start_time:.2f} giây")

# ==========================================
# 4. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
print("\nĐang dự đoán tập Test...")
y_pred = svm_model.predict(X_test_pca)

# In báo cáo kết quả chi tiết
acc = accuracy_score(y_test, y_pred)
print(f"\n🏆 ĐỘ CHÍNH XÁC TỔNG THỂ (Accuracy): {acc * 100:.2f}%\n")

print("--- BÁO CÁO PHÂN LOẠI CHI TIẾT ---")
print(classification_report(y_test, y_pred, target_names=class_names))

print("--- MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
print(confusion_matrix(y_test, y_pred))