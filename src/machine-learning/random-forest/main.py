import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
with open("../../../data/processed/aptos/sample_50/sample_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X           = data['data']        
y           = data['labels']      
class_names = data['class_names'] 

# ==========================================
# 2. TIỀN XỬ LÝ (GIỮ NGUYÊN PIPELINE)
# ==========================================
print("Đang tiến hành tiền xử lý dữ liệu cho Random Forest...")

# Duỗi phẳng ảnh (Flatten)
N = X.shape[0]
X_flatten = X.reshape(N, -1) 
print(f"Kích thước sau khi duỗi: {X_flatten.shape}")

# Chia tập Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X_flatten, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa thang đo (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Giảm chiều dữ liệu bằng PCA 
print("Đang chạy thuật toán nén PCA...")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Số lượng đặc trưng (features) đã giảm từ {X_flatten.shape[1]} xuống còn {X_train_pca.shape[1]}!")

# ==========================================
# 3. KHỞI TẠO VÀ HUẤN LUYỆN RANDOM FOREST
# ==========================================
print("\nBắt đầu huấn luyện mô hình Random Forest...")
start_time = time.time()

# Cấu hình Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,        # Số lượng cây quyết định (càng nhiều càng ổn định, nhưng train sẽ lâu hơn)
    max_depth=None,          # Cho phép cây mọc tự do đến khi phân tách hoàn toàn các nút
    min_samples_split=2,     # Số lượng mẫu tối thiểu để tách một nút
    n_jobs=-1,               # Sử dụng toàn bộ sức mạnh CPU (tất cả các luồng) để chạy song song
    random_state=42,
    class_weight='balanced'  # Tự động cân bằng trọng số nếu dữ liệu các mức độ DR bị lệch (imbalanced)
)

# Train model
rf_model.fit(X_train_pca, y_train)

end_time = time.time()
print(f"✅ Đã huấn luyện xong! Thời gian: {end_time - start_time:.2f} giây")

# ==========================================
# 4. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
print("\nĐang dự đoán tập Test...")
y_pred = rf_model.predict(X_test_pca)

acc = accuracy_score(y_test, y_pred)
print(f"\n🏆 ĐỘ CHÍNH XÁC TỔNG THỂ (Accuracy): {acc * 100:.2f}%\n")

print("--- BÁO CÁO PHÂN LOẠI CHI TIẾT ---")
print(classification_report(y_test, y_pred, target_names=class_names))

print("--- MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
print(confusion_matrix(y_test, y_pred))